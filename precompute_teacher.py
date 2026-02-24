import torch
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from TTS.api import TTS
from vocos import Vocos

# Настройки
device = "cuda"
output_sample_rate = 24000  # Vocos 24k требует именно 24000


def precompute_teacher_mels_correctly(root_dir):
    # 1. Загружаем Учителя (XTTS)
    print("Загрузка XTTS v2...")
    teacher = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # 2. Загружаем Vocos (чтобы делать правильные спектрограммы)
    print("Загрузка Vocos Feature Extractor...")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    vocos.eval()  # Переводим в режим оценки

    root = Path(root_dir)
    folders = [f for f in root.iterdir() if f.is_dir()]

    for folder in folders:
        json_files = list(folder.glob("*.json"))
        if not json_files: continue

        with open(json_files[0], 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        print(f"Обработка папки: {folder.name}")

        for i, entry in enumerate(tqdm(metadata)):
            audio_path = folder / f"{folder.name}_{i}.mp3"
            mel_path = folder / f"{folder.name}_{i}_teacher.pt"

            # Пропускаем длинные тексты, чтобы не забить память
            if len(entry["text"]) > 200: continue

            # Если файл уже есть - пропускаем (удалите старые файлы вручную перед запуском!)
            if mel_path.exists(): continue

            try:
                # ШАГ 1: Учитель генерирует Аудио
                # XTTS v2 нативно выдает 24000Hz, что идеально для Vocos
                wav = teacher.tts(text=entry["text"], speaker_wav=str(audio_path), language="ru")

                # ШАГ 2: Подготовка аудио для Vocos
                # Превращаем список в тензор [1, Time]
                wav_tensor = torch.tensor(wav).float().unsqueeze(0).to(device)

                # ШАГ 3: Vocos превращает Аудио в Спектрограмму
                with torch.no_grad():
                    # feature_extractor сам сделает всё правильно (без всякой либрозы)
                    mel = vocos.feature_extractor(wav_tensor)

                    # mel сейчас имеет размер [1, 100, Time] (например [1, 100, 436])

                # ШАГ 4: Сохранение
                # Обычно модели учат предсказывать [Time, Channels] или [Channels, Time]
                # Сохраним как [Time, 100] для удобства DataLoader'а (транспонируем)
                # Если ваш Student ждет [100, Time], уберите .transpose(1, 2)
                mel_to_save = mel.transpose(1, 2).squeeze(0)

                torch.save(mel_to_save.cpu(), mel_path)

            except Exception as e:
                print(f"Ошибка на фразе '{entry['text'][:10]}...': {e}")


if __name__ == "__main__":
    # УКАЖИТЕ ПУТЬ К ПАПКЕ С ДАТАСЕТОМ
    dataset_path = "C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped"

    # ВАЖНО: Удалите старые кривые .pt файлы перед запуском!
    # Можно вручную в проводнике через поиск *.pt -> удалить
    precompute_teacher_mels_correctly(dataset_path)