import torch
import torchaudio
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt  # Для сохранения attention
from vocos import Vocos
# from speechbrain.inference.speaker import EncoderClassifier

import GigaTestLSTM
from GigaTestLSTM import Config, TextProcessor, StudentTTS, save_mel_image

vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to("cpu")
vocoder.eval()
with torch.no_grad():
    # 1. Загружаем тензор из файла
    with torch.no_grad():
        # Загружаем
        features = torch.load(
            "C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped/100132300/100132300_44_teacher.pt")
        features = features.to("cpu")

        # Если тензор пришел без размерности батча [436, 100] -> [1, 436, 100]
        if features.ndim == 2:
            features = features.unsqueeze(0)

        # ВАЖНО: меняем местами оси времени и частот
        # Было [1, 436, 100] -> Станет [1, 100, 436]
        features = features.transpose(1, 2)

        # Теперь скармливаем вокодеру
        wav = vocoder.decode(features)
        wav = wav.squeeze().cpu().numpy()
sf.write("test_vocos.wav", wav, 24000)