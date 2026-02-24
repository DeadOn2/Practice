import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from GigaTestLSTM import Config, TextProcessor, StudentTTS

# Предполагаем, что твои классы Config, TextProcessor и StudentTTS в этом же файле
# Если в другом — импортируй их: from my_model_file import Config, StudentTTS, TextProcessor

import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path


def manual_griffin_lim(S, n_iter=64, hop_length=256, win_length=1024):
    """Ручная реализация Griffin-Lim без использования проблемных функций librosa"""
    # Создаем случайную фазу вручную через экспоненту (обходим librosa.util.phasor)
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape).astype(np.float32))
    S_complex = S.astype(np.complex64)

    # Первое обратное преобразование
    y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)

    for i in range(n_iter):
        # Прямое STFT
        stft = librosa.stft(y, n_fft=win_length, hop_length=hop_length, win_length=win_length)
        # Извлекаем фазу
        angles = np.exp(1j * np.angle(stft))
        # Обратное STFT с оригинальной амплитудой S
        y = librosa.istft(S * angles, hop_length=hop_length, win_length=win_length)
    return y


def inference_griffin_lim(text, checkpoint_path, output_filename="output_lim_1000.wav"):
    cfg = Config()
    cfg.device = torch.device("cpu")
    tp = TextProcessor(cfg.RUS_ALPHABET)

    print(f"Загрузка модели: {checkpoint_path}...")
    model = StudentTTS(cfg).to(cfg.device)
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokens = torch.tensor([tp.encode(text)], dtype=torch.long).to(cfg.device)
    lens = torch.tensor([tokens.size(1)]).to(cfg.device)

    print("Генерация спектрограммы...")
    with torch.no_grad():
        mel_output = model(tokens, lens, mels=None)

    mel = mel_output.squeeze(0).cpu().numpy().T.astype(np.float32)

    # Денормализация (модель училась в [0, 1])
    mel = np.clip(mel, 0, 1)
    mel = (mel * 80.0) - 80.0
    mel_power = librosa.db_to_power(mel)

    # Вместо mel_to_audio делаем цепочку вручную
    print("Восстановление звука (Manual Griffin-Lim)...")

    # 1. Сначала восстанавливаем линейную спектрограмму из Мел-шкалы
    # Это нужно, так как Griffin-Lim работает с STFT
    stft_mag = librosa.feature.inverse.mel_to_stft(
        mel_power,
        sr=cfg.sample_rate,
        n_fft=1024,
        power=1.0
    )

    # 2. Запускаем наш ручной алгоритм
    audio = manual_griffin_lim(
        stft_mag,
        n_iter=64,
        hop_length=cfg.hop_length,
        win_length=1024
    )

    # Сохранение
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    sf.write(output_filename, audio, cfg.sample_rate)
    print(f"✅ Готово! Файл: {output_filename}")

if __name__ == "__main__":
    # Укажи путь к своему чекпоинту
    CHECKPOINT = "checkpoints/student_step_1000.pth"
    TEST_TEXT = "Привет! Я использую алгоритм Гриффин Лим для синтеза речи."

    inference_griffin_lim(TEST_TEXT, CHECKPOINT)