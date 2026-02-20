import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional

# --- 1. CONFIGURATION & IMPORTS ---
# Предполагаем, что вы уже скачали файлы токенизатора или используете библиотеку TTS
# from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
# Для примера сделаем заглушку, если импорт не сработает,
# но в реальном коде раскомментируйте импорт выше.

try:
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
except ImportError:
    print("Warning: TTS library not found. Using MockTokenizer for demonstration.")


    class VoiceBpeTokenizer:
        def __init__(self, vocab_file=None): self.pad_id = 0

        def encode(self, text, lang="en"): return [1, 2, 3]  # Dummy implementation

        def __call__(self, text, lang="en"): return self.encode(text, lang)


class Config:
    sample_rate = 24000  # XTTS standard
    n_mels = 80
    n_fft = 2048
    hop_length = 256
    win_length = 1024

    # Model dims
    d_model = 256
    n_heads = 4
    n_layers = 4  # Маленький "студент" (у XTTS их десятки)
    encoder_dim = 512  # Размерность вектора спикера от XTTS

    batch_size = 8
    lr = 1e-4
    epochs = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"


# --- 2. DATA PROCESSING ---

class MelSpectrogramProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.sample_rate,
            n_fft=Config.n_fft,
            win_length=Config.win_length,
            hop_length=Config.hop_length,
            n_mels=Config.n_mels,
            power=2.0,
            normalized=False
        )

    def forward(self, audio):
        # audio: [1, T]
        mel = self.mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.squeeze(0).transpose(0, 1)  # [Frames, n_mels]


class PodcastDistillDataset(Dataset):
    def __init__(self, root_dir, tokenizer, teacher_model=None):
        """
        teacher_model: нужен, если мы хотим заранее кэшировать эмбеддинги спикера,
        чтобы не гонять тяжелую модель каждый раз в обучении.
        """
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.mel_proc = MelSpectrogramProcessor()
        self.samples = []
        self.teacher_model = teacher_model  # Ссылка на XTTS для извлечения фич

        print(f"Scanning directory: {root_dir}...")
        folders = [f for f in self.root_dir.iterdir() if f.is_dir()]

        for folder in folders:
            json_files = list(folder.glob("*.json"))
            if not json_files: continue

            with open(json_files[0], 'r', encoding='utf-8') as f:

                metadata_list = json.load(f)

            for i, entry in enumerate(metadata_list):
                audio_filename = f"{folder.name}_{i}.mp3"
                audio_path = folder / audio_filename
                # В цикле обхода metadata_list
                if len(entry["text"]) > 182:
                    # print(f"Skipping long sample: {len(entry['text'])} chars")
                    continue
                if audio_path.exists():
                    self.samples.append({
                        "text": entry["text"],
                        "audio_path": str(audio_path),
                        "speaker_id": entry.get("speaker", "unknown")
                    })

        print(f"Dataset loaded: {len(self.samples)} samples found.")

    def __len__(self):
        return len(self.samples)

    def load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != Config.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, Config.sample_rate)
            waveform = resampler(waveform)
        return waveform

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Обработка текста
        # XTTS токенизатор
        token_ids = self.tokenizer.encode(sample["text"], lang="ru")

        # 2. Обработка аудио
        waveform = self.load_audio(sample["audio_path"])
        mel_spec = self.mel_proc(waveform)  # [Frames, 80]

        # 3. Speaker Conditioning (Zero-Shot component)
        # В реальной дистилляции мы бы прогнали аудио через XTTS Encoder,
        # чтобы получить вектор стиля. Здесь мы либо загружаем его (если прекэширован),
        # либо будем вычислять в collate_fn. Для примера вернем сырое аудио референса.

        return {
            "text_ids": torch.tensor(token_ids, dtype=torch.long),
            "mel_target": mel_spec,
            "ref_audio": waveform.squeeze(0),  # Для вычисления эмбеддинга
            "text_raw": sample["text"]
        }


def collate_fn(batch):
    # Паддинг для батчинга
    text_ids = [b['text_ids'] for b in batch]
    mels = [b['mel_target'] for b in batch]
    ref_audios = [b['ref_audio'] for b in batch]

    text_padded = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=0)
    mels_padded = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True,
                                                  padding_value=-11.51)  # log(1e-5) approx padding

    # Создаем маски длины
    text_lens = torch.tensor([t.size(0) for t in text_ids])
    mel_lens = torch.tensor([m.size(0) for m in mels])

    return {
        "text": text_padded,
        "text_lens": text_lens,
        "mel": mels_padded,
        "mel_lens": mel_lens,
        "ref_audios": ref_audios  # List of tensors
    }


# --- 3. ARCHITECTURE (The Student) ---
# Student должен быть:
# 1. Acoustic Model (Text + Speaker Emb -> Mel Spec)
# 2. Non-autoregressive (быстрее) или simple autoregressive (проще учить на малых данных)
# Для простоты и стабильности реализуем легкий Transformer-TTS.

class SpeakerIntegrator(nn.Module):
    """Интегрирует вектор спикера в текст"""

    def __init__(self, d_model, speaker_dim):
        super().__init__()
        self.projection = nn.Linear(speaker_dim, d_model)

    def forward(self, x, speaker_emb):
        # x: [B, T, d_model]
        # speaker_emb: [B, speaker_dim]
        spk_proj = self.projection(speaker_emb).unsqueeze(1)  # [B, 1, d_model]
        return x + spk_proj  # Additive conditioning (просто и эффективно)


class StudentModel(nn.Module):
    def __init__(self, vocab_size, speaker_dim=Config.encoder_dim):
        super().__init__()
        self.d_model = Config.d_model

        # 1. Text Encoder
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.max_seq_len = 4096
        self.pos_encoding = self._generate_positional_encoding(self.max_seq_len, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=Config.n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=Config.n_layers)

        # 2. Speaker Conditioning Mechanism (The "Cloning" part)
        self.speaker_integrator = SpeakerIntegrator(self.d_model, speaker_dim)

        # 3. Decoder (Predicts Mel)
        # В полноценном non-autoregressive нужен Duration Predictor.
        # Для упрощения сделаем простой декодер, который принимает Text+Spk и выдает Mel
        # (NB: Без duration predictor мы предполагаем апсэмплинг.
        # В продакшене здесь нужен Variance Adapter из FastSpeech2).

        # Простая реализация: LSTM/Conv слои для "растягивания" времени или
        # использование CrossAttention на позиционные энкодинги мелов.

        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=Config.n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=Config.n_layers)

        self.mel_proj = nn.Linear(self.d_model, Config.n_mels)
        self.stop_proj = nn.Linear(self.d_model, 1)  # Stop token prediction

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, text, speaker_emb, mel_target=None):
        # Text Encoding
        x = self.embedding(text)  # [B, T_text, D]
        # Add Positional
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)

        # Inject Speaker Style
        x = self.speaker_integrator(x, speaker_emb)
        memory = self.encoder(x)

        # Decoding (Training mode with Teacher Forcing)
        # На вход декодеру подаем Mel (сдвинутый) или Positional Embeddings, если inference
        if mel_target is not None:
            # Проецируем входной мел обратно в размерность модели
            # В реальности нужен Prenet, здесь упрощаем
            # Обычно: Mel -> Linear -> ReLU -> Linear -> Decoder
            # Мы используем memory как key/value, а таргет как query?
            # Для простоты Transformer-TTS: Target Mels нужны для teacher forcing

            # Создаем dummy target input (или используем правильный masking)
            tgt_seq_len = mel_target.size(1)
            tgt = torch.zeros(mel_target.size(0), tgt_seq_len, self.d_model).to(mel_target.device)
            tgt = tgt + self.pos_encoding[:, :tgt_seq_len, :].to(tgt.device)

            output = self.decoder(tgt, memory)
        else:
            # Inference loop (simplified)
            output = memory  # Placeholder logic

        mel_out = self.mel_proj(output)
        return mel_out


# --- 4. PRETRAINED INTERFACES (Teacher Components) ---

class TeacherInterface:
    def __init__(self, device):
        self.device = device
        print("Loading XTTS Teacher (Configuration only for Speaker Encoder)...")
        # Здесь должен быть код загрузки реального XTTS
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        config = XttsConfig()
        config.load_json("C:/Users/light/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir="C:/Users/light/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2")
        self.model.to(device)
        pass

    def get_speaker_embedding(self, audio_list):
        """
        Извлекает conditioning latents из аудио.
        Это "дистилляция" знаний учителя о голосе.
        """
        # В реальном XTTS:
        # gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=...)
        # return speaker_embedding

        # MOCK для запускабельности кода:
        batch_size = len(audio_list)
        return torch.randn(batch_size, Config.encoder_dim).to(self.device)


# --- 5. TRAINING LOOP ---

def train():
    device = Config.device
    print(f"Training on {device}")

    # Init Components
    # Путь к папке токенизатора (обычно внутри папки модели XTTS)
    tokenizer = VoiceBpeTokenizer(vocab_file="./xtts_vocab/vocab.json")

    dataset = PodcastDistillDataset(
        root_dir="C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped",
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)

    student = StudentModel(vocab_size=256000).to(device)  # vocab размер как у XTTS
    teacher = TeacherInterface(device)

    optimizer = optim.AdamW(student.parameters(), lr=Config.lr)
    criterion = nn.L1Loss()  # MAE for Mel Spectrograms

    student.train()
    global_step = 0
    save_every_steps = 1000  # Сохраняем каждые 1000 батчей
    try:
        for epoch in range(Config.epochs):
            total_loss = 0
            for batch in dataloader:
                text = batch["text"].to(device)
                mels = batch["mel"].to(device)
                ref_audios = batch["ref_audios"]  # Сырые данные для учителя

                optimizer.zero_grad()

                # 1. Получаем "знание" от Учителя (Speaker Embedding)
                # Мы замораживаем учителя, нам нужны только вектора голоса
                with torch.no_grad():
                    speaker_emb = teacher.get_speaker_embedding(ref_audios)

                # 2. Студент пытается синтезировать мел, используя текст и стиль учителя
                mel_pred = student(text, speaker_emb, mel_target=mels)

                # 3. Лосс
                # (Опционально: можно добавить Distillation Loss, сравнивая mel_pred с mel_pred_by_teacher,
                # но это требует прогона полного XTTS, что очень долго.
                # Обучение на реальных данных (Ground Truth) с использованием Teacher Embedding -
                # это самый эффективный способ клонирования).

                # Приводим размерности, если не совпадают из-за паддинга
                if mel_pred.shape[1] != mels.shape[1]:
                    min_len = min(mel_pred.shape[1], mels.shape[1])
                    mel_pred = mel_pred[:, :min_len, :]
                    mels = mels[:, :min_len, :]

                loss = criterion(mel_pred, mels)

                loss.backward()
                optimizer.step()
                global_step += 1

                if global_step % 10 == 0:
                    print(f"Step {global_step} | Loss: {loss.item():.4f}")

                # СОХРАНЕНИЕ ПРОМЕЖУТОЧНОГО РЕЗУЛЬТАТА
                if global_step % save_every_steps == 0:
                    checkpoint_path = f"student_step_{global_step}.pth"
                    torch.save(student.state_dict(), checkpoint_path)
                    print(f"--- Saved checkpoint: {checkpoint_path} ---")
                total_loss += loss.item()
    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем. Сохраняю текущую модель...")
        torch.save(student.state_dict(), "durationtest/student_interrupted_model.pth")
        print("Готово. Можно тестировать!")
        print(f"Epoch {epoch + 1}/{Config.epochs}, Loss: {total_loss / len(dataloader):.4f}")
    #
    #     # Сохранение
    # if (epoch + 1) % 10 == 0:
    #     torch.save(student.state_dict(), f"student_model_{epoch + 1}.pth")


if __name__ == "__main__":
    # Создайте фиктивные папки для теста или укажите реальные пути
    if not os.path.exists("C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped"):
        os.makedirs("./podcast_data")
        print("Created ./podcast_data. Please put your folders/jsons there.")
    else:
        train()