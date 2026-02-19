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
# -------------------------------------------------
# Positional Encoding
# -------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# -------------------------------------------------
# Duration Predictor
# -------------------------------------------------

class DurationPredictor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.linear = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        return self.linear(x).squeeze(-1)  # [B, T]


# -------------------------------------------------
# Length Regulator
# -------------------------------------------------

def length_regulator(x, durations):
    output = []
    for batch, dur in zip(x, durations):
        expanded = [
            batch[i].unsqueeze(0).repeat(int(d.item()), 1)
            for i, d in enumerate(dur)
            if int(d.item()) > 0
        ]
        if len(expanded) == 0:
            expanded = [batch[0].unsqueeze(0)]
        expanded = torch.cat(expanded, dim=0)
        output.append(expanded)
    return nn.utils.rnn.pad_sequence(output, batch_first=True)


# -------------------------------------------------
# PostNet
# -------------------------------------------------

class PostNet(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, 512, 5, padding=2),
            nn.Tanh(),
            nn.Conv1d(512, n_mels, 5, padding=2)
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


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
        audio_path = Path(sample["audio_path"])
        dur_path = audio_path.parent / f"{audio_path.stem}_durations.pt"

        # 1. Загружаем препроцессированные данные
        if dur_path.exists():
            data = torch.load(dur_path)
            token_ids = data["token_ids"]
            durations = data["durations"]
        else:
            # Fallback, если файл пропущен
            return self.__getitem__(np.random.randint(0, len(self.samples)))

        # 2. Обработка аудио
        waveform = self.load_audio(str(audio_path))
        mel_spec = self.mel_proc(waveform)

        return {
            "text_ids": torch.tensor(token_ids, dtype=torch.long),
            "mel_target": mel_spec,
            "durations": durations,  # <--- ДОБАВИЛИ ДЛИТЕЛЬНОСТИ
            "audio_path": sample["audio_path"]
        }


def collate_fn(batch):
    text_ids = [b['text_ids'] for b in batch]
    mels = [b['mel_target'] for b in batch]
    durations = [b['durations'] for b in batch] # <--- Извлекаем
    audio_paths = [b['audio_path'] for b in batch]

    text_padded = torch.nn.utils.rnn.pad_sequence(text_ids, batch_first=True, padding_value=0)
    mels_padded = torch.nn.utils.rnn.pad_sequence(mels, batch_first=True, padding_value=-11.51)
    durations_padded = torch.nn.utils.rnn.pad_sequence(durations, batch_first=True, padding_value=0) # <--- Паддинг 0

    return {
        "text": text_padded,
        "mel": mels_padded,
        "durations": durations_padded, # <--- Отдаем в батч
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


# -------------------------------------------------
# STUDENT MODEL (FastSpeech-style)
# -------------------------------------------------

class StudentModel(nn.Module):
    def __init__(self, vocab_size, speaker_dim=512, d_model=256, n_mels=100):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.spk_proj = nn.Linear(speaker_dim, d_model)

        self.duration_predictor = DurationPredictor(d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=4)

        self.mel_proj = nn.Linear(d_model, n_mels)
        self.postnet = PostNet(n_mels)

        # ДОБАВЬ target_durations В АРГУМЕНТЫ
    def forward(self, text, speaker_emb, mel_target=None, target_durations=None):

        # ----- TEXT ENCODER -----
        x = self.embedding(text)
        x = self.pos_enc(x)
        x = self.encoder(x)

        # ----- ADD SPEAKER -----
        spk = self.spk_proj(speaker_emb).unsqueeze(1)
        x = x + spk

        # ----- DURATION -----
        dur_pred = self.duration_predictor(x.detach())  # Отвязываем градиенты текста от предиктора!

        if mel_target is not None and target_durations is not None:
            # ИСПОЛЬЗУЕМ РЕАЛЬНЫЕ ДЛИТЕЛЬНОСТИ
            durations = target_durations

            # Считаем лосс предиктора
            log_dur_target = torch.log(durations.float() + 1.0)

            # Считаем лосс только по реальным токенам (игнорируя паддинг)
            mask = (text != 0).float()  # Предполагая, что 0 - это pad_token
            dur_loss = F.mse_loss(dur_pred * mask, log_dur_target * mask)

        else:
            # ИНФЕРЕНС
            durations = torch.clamp(
                torch.exp(dur_pred) - 1.0,
                min=1.0,
                max=20.0
            ).round().long()  # ОБЯЗАТЕЛЬНО .long()
            dur_loss = None

        # ----- LENGTH REGULATION -----
        x = length_regulator(x, durations)

        # ----- DECODER -----
        x = self.pos_enc(x)
        x = self.decoder(x)

        mel = self.mel_proj(x)
        mel = mel + self.postnet(mel)

        return mel, dur_loss

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    @torch.no_grad()
    def inference(student, text, speaker_emb):
        mel, _ = student(text, speaker_emb, mel_target=None)
        return mel


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


                optimizer.zero_grad()

                # 1. Получаем "знание" от Учителя (Speaker Embedding)
                # Мы замораживаем учителя, нам нужны только вектора голоса
                with torch.no_grad():
                    # Получаем тензор и сразу клонируем его, чтобы он стал "обычным"
                    speaker_emb = teacher.get_speaker_embedding(batch["audio_paths"]).clone()
                    # На всякий случай убедимся, что для него не нужно считать градиенты
                    speaker_emb.requires_grad = False

                # 2. Студент пытается синтезировать мел, используя текст и стиль учителя

                # 3. Лосс
                # (Опционально: можно добавить Distillation Loss, сравнивая mel_pred с mel_pred_by_teacher,
                # но это требует прогона полного XTTS, что очень долго.
                # Обучение на реальных данных (Ground Truth) с использованием Teacher Embedding -
                # это самый эффективный способ клонирования).
                durations = batch["durations"].to(device)
                mel_pred, dur_loss = student(text, speaker_emb, mel_target=mels, target_durations=durations)

                # Приводим размерности, если не совпадают из-за паддинга
                if mel_pred.shape[1] != mels.shape[1]:
                    min_len = min(mel_pred.shape[1], mels.shape[1])
                    mel_pred = mel_pred[:, :min_len, :]
                    mels = mels[:, :min_len, :]
                # подгон длины
                min_len = min(mel_pred.shape[1], mels.shape[1])
                mel_pred = mel_pred[:, :min_len]
                mels = mels[:, :min_len]

                mel_loss = F.l1_loss(mel_pred, mels)
                if dur_loss is not None:
                    loss = mel_loss + 0.1 * dur_loss
                else:
                    loss = mel_loss

                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)

                writer.add_scalar("Loss/Total", loss.item(), global_step)
                writer.add_scalar("Loss/Mel", mel_loss.item(), global_step)
                writer.add_scalar("Loss/Duration", dur_loss.item(), global_step)


                # 2. Лосс за длительность (Duration Loss)
                # Мы хотим, чтобы сумма предсказанных длительностей была близка к реальной длине аудио


                # Итоговый лосс


                loss.backward()
                optimizer.step()
                global_step += 1

                if global_step % 10 == 0:
                    writer.add_scalar("Loss/Total", loss.item(), global_step)
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