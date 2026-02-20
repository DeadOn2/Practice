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
from vocos import Vocos
vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to("cuda")

from typing import List, Optional

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

tokenizer = VoiceBpeTokenizer(vocab_file="./xtts_vocab/vocab.json")

dataset = PodcastDistillDataset(
        root_dir="C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped",
        tokenizer=tokenizer
    )
print(dataset.samples.__getitem__(0))
# 1. Достаем спектрограмму [Frames, 80]
# 1. Достаем спектрограмму из датасета [Frames, 80]
# 1. Достаем спектрограмму [Frames, 80]
# 1. Достаем спектрограмму [201, 80]
mel = dataset[0]["mel_target"] # [201, 100]
# Переставляем оси, чтобы было [100, 201] и добавляем батч-размерность [1, 100, 201]
mel_for_vocoder = mel.transpose(0, 1).unsqueeze(0).to("cuda")
audio_res = vocoder.decode(mel_for_vocoder)

# 6. СОХРАНЕНИЕ
audio_final = audio_res.squeeze().cpu()


torchaudio.save("vocoder_sanity_check_final.wav", audio_final.unsqueeze(0), 24000)
print("Победа! Файл vocoder_sanity_check_final.wav создан.")