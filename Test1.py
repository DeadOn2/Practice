import os
import json
import torch
import torchaudio
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.vocoders import HIFIGAN


# ==========================================
# 1. DATASET: ЗАГРУЗКА ИЗ ВАШЕЙ ДИРЕКТОРИИ
# ==========================================

class PodcastDistillDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []

        print(f"Scanning directory: {root_dir}...")

        # Обходим все папки dir1, dir2...
        folders = [f for f in self.root_dir.iterdir() if f.is_dir()]

        for folder in folders:
            # Ищем json файл в папке
            json_files = list(folder.glob("*.json"))
            if not json_files:
                continue

            with open(json_files[0], 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)

            # Сопоставляем индекс в JSON с файлом mp3 (dirX_index.mp3)
            for i, entry in enumerate(metadata_list):
                # Формируем имя файла: имя_папки + _ + индекс + .mp3
                audio_filename = f"{folder.name}_{i}.mp3"
                audio_path = folder / audio_filename

                if audio_path.exists():
                    self.samples.append({
                        "text": entry["text"],
                        "audio_path": str(audio_path),
                        "speaker_id": entry.get("speaker", "unknown")
                    })

        print(f"Dataset loaded: {len(self.samples)} samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ==========================================
# 2. МОДЕЛЬ УЧЕНИКА (Lightweight Synthesizer)
# ==========================================

# class StudentSynthesizer(nn.Module):
#     """
#     Легкий синтезатор: Текст + Speaker Emb -> Mel.
#     """
#
#     def __init__(self, vocab_size=5000, speaker_dim=192, mel_dim=80):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, 256)
#         self.spk_proj = nn.Linear(speaker_dim, 256)
#
#         # Упрощенный трансформер
#         encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
#
#         self.mel_out = nn.Linear(256, mel_dim)
#
#     def forward(self, text_tokens, spk_emb, target_len):
#         # text_tokens: [B, T], spk_emb: [B, 192]
#         x = self.embedding(text_tokens)
#         s = self.spk_proj(spk_emb).unsqueeze(1)
#
#         # Добавляем стиль к тексту
#         x = x + s
#
#         # Прогон через слои
#         latent = self.transformer(x)
#
#         # Для простоты дистилляции растягиваем латент до длины таргета
#         # В продакшене тут должен быть Duration Predictor
#         latent_resized = F.interpolate(latent.transpose(1, 2), size=target_len).transpose(1, 2)
#
#         mel = self.mel_out(latent_resized)
#         return mel.transpose(1, 2)  # [B, 80, T]
#
#
# # ==========================================
# # 3. FRAMEWORK ДИСТИЛЛЯЦИИ
# # ==========================================
#
# class DistillationTrainer:
#     def __init__(self, data_path):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # 1. Загрузка инструментов
#         self.spk_encoder = EncoderClassifier.from_hparams(
#             source="speechbrain/spkrec-ecapa-voxceleb",
#             run_opts={"device": self.device}
#         )
#         self.vocoder = HIFIGAN.from_hparams(
#             source="speechbrain/tts-hifigan-ljspeech",
#             run_opts={"device": self.device}
#         )
#
#         # Трансформация в Мел (параметры под HiFi-GAN)
#         self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#             sample_rate=22050, n_mels=80, hop_length=256
#         ).to(self.device)
#
#         # 2. Загрузка Учителя (XTTS v2)
#         from TTS.api import TTS
#         print("Loading XTTS Teacher...")
#         self.teacher = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
#
#         # 3. Инициализация Ученика
#         self.student = StudentSynthesizer().to(self.device)
#         self.optimizer = torch.optim.Adam(self.student.parameters(), lr=2e-4)
#
#         # 4. Данные
#         self.dataset = PodcastDistillDataset(data_path)
#         self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
#
#     def train(self, epochs=10):
#         self.student.train()
#
#         for epoch in range(epochs):
#             for i, batch in enumerate(self.dataloader):
#                 text = batch['text'][0]
#                 audio_path = batch['audio_path'][0]
#
#                 # --- ШАГ 1: Получаем эталон от Учителя ---
#                 with torch.no_grad():
#                     # Генерируем аудио учителем
#                     wav_teacher = self.teacher.tts(text=text, speaker_wav=audio_path, language="ru")
#                     wav_teacher = torch.tensor(wav_teacher).unsqueeze(0).to(self.device)
#
#                     # Ресемплинг в 22050 для вокодера ученика
#                     resampler = torchaudio.transforms.Resample(24000, 22050).to(self.device)
#                     wav_target = resampler(wav_teacher)
#
#                     # Делаем лог-мел спектрограмму (Target)
#                     target_mel = torch.log(self.mel_spectrogram(wav_target) + 1e-6)
#
#                     # Получаем эмбеддинг спикера из оригинального файла (Zero-Shot)
#                     ref_wav, sr = torchaudio.load(audio_path)
#                     if sr != 16000:
#                         ref_wav = torchaudio.transforms.Resample(sr, 16000)(ref_wav)
#                     spk_emb = self.spk_encoder.encode_batch(ref_wav.to(self.device)).squeeze(1)
#
#                 # --- ШАГ 2: Проход ученика ---
#                 self.optimizer.zero_grad()
#
#                 # Токенизация (простой маппинг символов для примера)
#                 tokens = torch.tensor([[ord(c) % 5000 for c in text]]).to(self.device)
#
#                 # Предсказание мелов
#                 pred_mel = self.student(tokens, spk_emb, target_len=target_mel.shape[2])
#
#                 # --- ШАГ 3: Loss и оптимизация ---
#                 loss = F.l1_loss(pred_mel, target_mel)
#                 loss.backward()
#                 self.optimizer.step()
#
#                 if i % 10 == 0:
#                     print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")
#

# ==========================================
# ЗАПУСК
# ==========================================
if __name__ == "__main__":
    DATA_PATH = r"C:\Users\light\Downloads\podcasts_1_stripped_archive"
    # trainer = DistillationTrainer(DATA_PATH)
    # trainer.train()