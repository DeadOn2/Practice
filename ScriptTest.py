import torch
from TTS.api import TTS

# НАСТРОЙКИ

TEXT = "Это тест клонирования голоса."
SPEAKER_WAV = "audio_2026-02-16_01-29-54.wav"   # путь к референсному аудио
OUTPUT_PATH = "output.wav"

# ЗАГРУЗКА МОДЕЛИ

print("Загружаем модель XTTS-v2...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tts = TTS(
    model_name="tts_models/multilingual/multi-dataset/xtts_v2",
    progress_bar=True
).to(device)

print("Модель загружена.")

# ГЕНЕРАЦИЯ РЕЧИ

tts.tts_to_file(
    text=TEXT,
    speaker_wav=SPEAKER_WAV,
    language="ru",
    file_path=OUTPUT_PATH
)

print(f"Файл сохранён как {OUTPUT_PATH}")
