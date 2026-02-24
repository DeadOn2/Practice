import torch
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from pathlib import Path
import random


def verify_precomputed_data(root_dir, num_samples=3):
    root = Path(root_dir)
    # Ищем все сохраненные тензоры учителем
    mel_files = list(root.glob("**/*_teacher.pt"))

    if not mel_files:
        print("❌ Файлы .pt не найдены. Проверьте путь или запустите препроцессинг.")
        return

    print(f"🔍 Найдено файлов: {len(mel_files)}")
    samples = random.sample(mel_files, min(num_samples, len(mel_files)))

    plt.figure(figsize=(15, 5 * len(samples)))

    for i, mel_path in enumerate(samples):
        # 1. Загрузка
        mel = torch.load(mel_path)  # Ожидаем (Time, n_mels)

        # 2. Технический аудит
        shape = mel.shape
        v_min, v_max = mel.min().item(), mel.max().item()
        v_mean = mel.mean().item()

        print(f"--- Файл {i + 1}: {mel_path.name} ---")
        print(f"   Размерность: {shape}")  # Должно быть (~frames, 100)
        print(f"   Диапазон: [{v_min:.2f}, {v_max:.2f}]")
        print(f"   Среднее: {v_mean:.2f}")

        # Проверка на NaN или бесконечность
        if torch.isnan(mel).any() or torch.isinf(mel).any():
            print("   ⚠️ ВНИМАНИЕ: Обнаружены NaN или Inf!")

        # 3. Визуализация
        plt.subplot(len(samples), 1, i + 1)
        # Транспонируем обратно в (n_mels, Time) для отображения
        mel_to_show = mel.T.numpy()

        librosa.display.specshow(
            mel_to_show,
            x_axis='time',
            y_axis='mel',
            sr=22050,
            hop_length=256,
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel Spectrogram: {mel_path.name} (Shape: {shape})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Укажите ваш путь к папке с данными
    DATA_PATH = "C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped/test2"
    verify_precomputed_data(DATA_PATH)