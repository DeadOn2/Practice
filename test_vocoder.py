# Добавь это в отдельный файл test_vocoder.py
import torch
import torchaudio
from GigaTEst3 import Config, MelSpectrogramProcessor
from vocos import Vocos

device = "cuda"
vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
mel_proc = MelSpectrogramProcessor().to(device)

# 1. Загружаем реальный голос из датасета
audio, sr = torchaudio.load("audio_2026-02-16_01-29-54.wav")
if sr != 24000:
    audio = torchaudio.transforms.Resample(sr, 24000)(audio)

# 2. Делаем "идеальную" спектрограмму
with torch.no_grad():
    mel = mel_proc(audio.to(device)).unsqueeze(0).transpose(1, 2)  # [1, 80, T]
    print(mel.shape)
    # Твой "костыль" для Vocos (80 -> 100)
    mel_100 = torch.nn.functional.interpolate(mel.transpose(1, 2), size=100).transpose(1, 2)

    # 3. Просим вокодер восстановить звук
    audio_res = vocoder.decode(mel_100)

torchaudio.save("vocoder_sanity_check.wav", audio_res.cpu(), 24000)
print("Если vocoder_sanity_check.wav звучит чисто — вокодер исправен.")