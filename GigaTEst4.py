import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional

# --- 1. CONFIGURATION & IMPORTS ---
# Предполагаем, что вы уже скачали файлы токенизатора или используете библиотеку TTS
# from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
# Для примера сделаем заглушку, если импорт не сработает,
# но в реальном коде раскомментируйте импорт выше.

class Config:
    sample_rate = 24000
    n_mels = 100  # СТРОГО 100 для Vocos 24kHz
    n_fft = 1024  # Vocos обычно учится на 1024
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

        # 2. Обработка аудио
        waveform = self.load_audio(sample["audio_path"])
        if waveform.abs().max() < 1e-5:
            # Если файл пустой, берем случайный другой (рекурсивно)
            return self.__getitem__(np.random.randint(0, len(self.samples)))

        mel_spec = self.mel_proc(waveform)  # [Frames, 80]

        if mel_spec.size(0) > 2500:
            mel_spec = mel_spec[:2500, :]
        # 3. Speaker Conditioning (Zero-Shot component)
        # В реальной дистилляции мы бы прогнали аудио через XTTS Encoder,
        # чтобы получить вектор стиля. Здесь мы либо загружаем его (если прекэширован),
        # либо будем вычислять в collate_fn. Для примера вернем сырое аудио референса.

        token_ids = self.tokenizer.encode(sample["text"], lang="ru")

        return {
            "text_ids": torch.tensor(token_ids, dtype=torch.long),
            "mel_target": mel_spec,
            "audio_path": sample["audio_path"]  # Передаем строку-путь!
        }


def collate_fn(batch):
    text_ids = [b['text_ids'] for b in batch]
    mels = [b['mel_target'] for b in batch]
    audio_paths = [b['audio_path'] for b in batch]

    # Считаем длины до того, как сделаем паддинг
    mel_lengths = torch.tensor([m.size(0) for m in mels], dtype=torch.long)
    text_lengths = torch.tensor([t.size(0) for t in text_ids], dtype=torch.long)

    # Паддинг
    text_padded = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=0)
    mels_padded = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=-11.51)

    return {
        "text": text_padded,
        "mel": mels_padded,
        "mel_lengths": mel_lengths,  # Передаем длины мела
        "text_lengths": text_lengths,
        "audio_paths": audio_paths
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
        # x: [B, T, 256]
        # speaker_emb должен быть [B, 512]

        spk_proj = self.projection(speaker_emb)  # Получаем [B, 256]
        spk_proj = spk_proj.unsqueeze(1)  # Делаем [B, 1, 256] для сложения с текстом
        return x + spk_proj


import math


class StudentModel(nn.Module):
    def __init__(self, vocab_size, speaker_dim=Config.encoder_dim):
        super().__init__()
        self.d_model = Config.d_model

        # 1. Text Encoder (Тот же самый)
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.pos_encoding = self._generate_positional_encoding(8192, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=Config.n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)  # 3 слоя хватит для начала

        # 2. Speaker Integrator
        self.speaker_integrator = SpeakerIntegrator(self.d_model, speaker_dim)

        # 3. DECODER (Самое важное изменение!)
        # Декодер принимает на вход Мел-спектрограмму (сдвинутую) и Текст (из энкодера)
        # И учится предсказывать следующий кадр.

        # Prenet (сжимает мел перед подачей в трансформер, критично для сходимости)
        self.mel_prenet = nn.Sequential(
            nn.Linear(Config.n_mels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, Config.d_model),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        dec_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=Config.n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=3)

        # 4. Mel Projection
        self.mel_proj = nn.Linear(self.d_model, Config.n_mels)

        # Stop Token (предсказывает конец речи)
        self.stop_proj = nn.Linear(self.d_model, 1)

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, text, speaker_emb, mel_target=None, ss_prob=0.0):
        # --- ENCODER ---
        x = self.embedding(text)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        memory = self.encoder(x)
        memory = self.speaker_integrator(memory, speaker_emb)

        # --- DECODER TRAINING ---
        if mel_target is not None:
            sos_frame = torch.zeros(mel_target.size(0), 1, Config.n_mels, device=mel_target.device)
            outputs = [sos_frame]

            for t in range(mel_target.size(1) - 1):
                if ss_prob > 0 and torch.rand(1).item() < ss_prob:
                    # Scheduled Sampling: берём своё последнее предсказание
                    dec_so_far = torch.cat(outputs, dim=1)
                    dec_input_ss = self.mel_prenet(dec_so_far)
                    dec_len_ss = dec_input_ss.size(1)
                    dec_input_ss = dec_input_ss + self.pos_encoding[:, :dec_len_ss, :].to(dec_input_ss.device)
                    tgt_mask_ss = self.generate_square_subsequent_mask(dec_len_ss).to(dec_input_ss.device)
                    out_ss = self.decoder(tgt=dec_input_ss, memory=memory, tgt_mask=tgt_mask_ss)
                    prev_frame = self.mel_proj(out_ss)[:, -1:, :]  # только последний кадр
                else:
                    # Teacher Forcing: берём реальный кадр
                    prev_frame = mel_target[:, t:t + 1, :]

                outputs.append(prev_frame)

            dec_input = torch.cat(outputs, dim=1)  # [B, T, n_mels]
            dec_input = self.mel_prenet(dec_input)
            dec_len = dec_input.size(1)
            dec_input = dec_input + self.pos_encoding[:, :dec_len, :].to(dec_input.device)
            tgt_mask = self.generate_square_subsequent_mask(dec_len).to(dec_input.device)

            output = self.decoder(tgt=dec_input, memory=memory, tgt_mask=tgt_mask)
            mel_out = self.mel_proj(output)
            stop_logits = self.stop_proj(output).squeeze(-1)

            return mel_out, stop_logits

        return None

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @torch.no_grad()
    def inference(self, text, speaker_emb, max_len=500,threshold = 0.5):
        # 1. Encode
        self.mel_prenet.eval()
        x = self.embedding(text)
        x = x + self.pos_encoding[:, :x.size(1), :].to(x.device)
        memory = self.encoder(x)
        memory = self.speaker_integrator(memory, speaker_emb)

        # Начинаем с кадра тишины (в логарифмах это ~ -11.51)
        curr_mel = torch.full((1, 1, Config.n_mels), -11.51).to(x.device)

        generated_frames = []
        for i in range(max_len):
            # 1. Прогоняем всё, что накопили, через декодер
            dec_input = self.mel_prenet(curr_mel)
            dec_input = dec_input + self.pos_encoding[:, :dec_input.size(1), :].to(x.device)
            tgt_mask = self.generate_square_subsequent_mask(dec_input.size(1)).to(x.device)

            output = self.decoder(tgt=dec_input, memory=memory, tgt_mask=tgt_mask)

            # 2. Берем только самый последний предсказанный кадр
            last_hidden_state = output[:, -1:, :]
            mel_frame = self.mel_proj(last_hidden_state)  # [1, 1, 80 или 100]
            stop_logit = self.stop_proj(last_hidden_state)  # [1, 1, 1]

            # 3. Добавляем предсказанный кадр в список результатов
            generated_frames.append(mel_frame)

            # 4. Обновляем curr_mel, чтобы на следующем шаге подать его на вход
            curr_mel = torch.cat([curr_mel, mel_frame], dim=1)

            # 5. Проверка стоп-токена
            if torch.sigmoid(stop_logit).item() > threshold:
                print(f"Модель решила остановиться на шаге {i}")
                break

        final_mel = torch.cat(generated_frames, dim=1)

        return final_mel


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

    def get_speaker_embedding(self, audio_paths):
        try:
            with torch.no_grad():
                gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=audio_paths)

                # speaker_embedding обычно приходит как [B, 1, 512]
                # Нам нужно строго [B, 512]
                res = speaker_embedding.view(speaker_embedding.size(0), -1)
                return res.clone()  # Клонируем, чтобы избежать ошибки backward из прошлого шага
        except Exception as e:
            print(f"Skipping batch due to error: {e}")
            # Возвращаем тензор из нулей того же размера как "заплатку"
            return torch.zeros(len(audio_paths), 512).to(self.device)


# --- 5. TRAINING LOOP ---

def train():
    writer = SummaryWriter(log_dir="runs/voice_clone_experiment")
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
    criterion_mel = nn.L1Loss()
    criterion_stop = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))

    student.train()
    global_step = 0
    save_every_steps = 500  # Сохраняем каждые 1000 батчей
    try:
        for epoch in range(Config.epochs):
            total_loss = 0
            for batch in dataloader:
                text = batch["text"].to(device)
                mels = batch["mel"].to(device)


                optimizer.zero_grad()

                # 1. Получаем "знание" от Учителя (Speaker Embedding)
                # Мы замораживаем учителя, нам нужны только вектора голоса
                with torch.no_grad():
                    # Получаем тензор и сразу клонируем его, чтобы он стал "обычным"
                    speaker_emb = teacher.get_speaker_embedding(batch["audio_paths"]).clone()
                    # На всякий случай убедимся, что для него не нужно считать градиенты
                    speaker_emb.requires_grad = False

                # 2. Студент пытается синтезировать мел, используя текст и стиль учителя
                ss_prob = min(0.5, global_step / 20000)  # плавно растёт с 0 до 0.5
                mel_pred, stop_pred = student(text, speaker_emb, mel_target=mels, ss_prob=ss_prob)

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

                mel_lengths = batch["mel_lengths"].to(device)

                # Создаем пустую маску [Batch, Max_Time]
                stop_targets = torch.zeros_like(stop_pred)
                for i, length in enumerate(mel_lengths):
                    # Только один последний кадр = стоп, не весь паддинг
                    actual_len = min(length.item(), stop_pred.size(1))
                    stop_targets[i, actual_len - 1] = 1.0

                # Увеличьте pos_weight, данные сильно несбалансированы
                criterion_stop = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]).to(device))

                loss_mel = criterion_mel(mel_pred, mels)
                loss_stop = criterion_stop(stop_pred, stop_targets)

                loss = loss_mel + loss_stop  # Складываем

                # 2. Лосс за длительность (Duration Loss)
                # Мы хотим, чтобы сумма предсказанных длительностей была близка к реальной длине аудио


                # Итоговый лосс


                loss.backward()
                optimizer.step()
                global_step += 1

                if global_step % 10 == 0:
                    writer.add_scalar("Loss/Total", loss.item(), global_step)
                    writer.add_scalar("Loss/stop", loss_stop.item(), global_step)
                    writer.add_scalar("Loss/mel", loss_mel.item(), global_step)
                    print(f"Step {global_step} | Loss: {loss.item():.4f}")

                if global_step % 500 == 0:
                    # Берем одну спектрограмму из батча (индекс 0)
                    img_mel = mel_pred[0].detach().cpu().numpy()
                    writer.add_image("Spectrogram/Train", img_mel, global_step, dataformats='HW')

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
    writer.close()
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