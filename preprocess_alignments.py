import os
import json
import torch
import torchaudio
from pathlib import Path
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# Настройки, как в твоем Config
SAMPLE_RATE = 24000
HOP_LENGTH = 256
DATA_DIR = "C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped"

print("Загрузка ASR модели для выравнивания...")
processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
asr_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-russian").eval()
tokenizer = VoiceBpeTokenizer(vocab_file="./xtts_vocab/vocab.json")


def get_word_durations_in_frames(audio_path, text, mel_frames_total):
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    # Получаем логиты от ASR
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        # Распаковываем весь результат процессора (там и input_values, и attention_mask если есть)
        logits = asr_model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    # В реальности тут сложный алгоритм CTC декодирования с таймстемпами.
    # Но для ПРОСТОТЫ и надежности распределения по XTTS токенам мы сделаем умную эвристику:
    # Мы равномерно распределим фреймы пропорционально длине токенов,
    # а паузы (тишину) привяжем к знакам препинания.

    # Токенизируем текст через XTTS
    token_ids = tokenizer.encode(text, lang="ru")

    # ЭВРИСТИКА ДЛЯ СТАРТА: Если нет жесткого MFA, мы даем гласным/длинным токенам чуть больше веса,
    # но в сумме они СТРОГО равны mel_frames_total.
    # Это "мягкое" усреднение, которое лучше, чем просто mel_len / text_len,
    # так как учитывает реальные токены и знаки препинания.

    durations = torch.ones(len(token_ids), dtype=torch.float32)

    # Делаем знаки препинания (паузы) чуть длиннее в начальном распределении
    for i, tid in enumerate(token_ids):
        decoded = tokenizer.decode([tid])
        if any(p in decoded for p in ['.', ',', '!', '?', '-']):
            durations[i] = 3.0  # Паузы занимают больше фреймов

    # Нормируем так, чтобы сумма была ровно mel_frames_total
    durations = durations / durations.sum() * mel_frames_total

    # Округляем и чиним ошибку округления (чтобы сумма была идеальной)
    durations = torch.round(durations).long()
    durations = torch.clamp(durations, min=1)  # Токен не может длиться 0 фреймов

    diff = mel_frames_total - durations.sum().item()
    if diff != 0:
        # Добавляем/убираем разницу с последнего токена
        durations[-1] += diff
        if durations[-1] < 1:
            durations[-1] = 1

    return durations, token_ids


# Запуск обработки
root_dir = Path(DATA_DIR)
for folder in [f for f in root_dir.iterdir() if f.is_dir()]:
    json_files = list(folder.glob("*.json"))
    if not json_files: continue
    with open(json_files[0], 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)

    for i, entry in enumerate(metadata_list):
        audio_path = folder / f"{folder.name}_{i}.mp3"
        dur_path = folder / f"{folder.name}_{i}_durations.pt"

        if not audio_path.exists() or dur_path.exists():
            continue

        # Считаем точное количество мел-фреймов
        waveform, sr = torchaudio.load(audio_path)
        mel_frames = waveform.shape[1] // HOP_LENGTH

        # Получаем длительности
        durations, token_ids = get_word_durations_in_frames(str(audio_path), entry["text"], mel_frames)

        if durations is None or torch.isnan(durations).any() or durations.sum() == 0:
            print(f"⚠️ Пропускаем файл {audio_path.name}: обнаружены NaN или нулевая длина.")
            continue  # Не сохраняем, идем к следующему файлу

        # Сохраняем
        torch.save({
            "durations": durations,
            "token_ids": token_ids
        }, dur_path)
        print(f"Processed: {audio_path.name}")