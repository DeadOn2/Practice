import torch
import torchaudio
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
from vocos import Vocos
from speechbrain.inference.speaker import EncoderClassifier

# Импортируем твои классы из файла обучения
# Убедись, что файл называется GigaTestLSTM.py или измени импорт
from GigaTestLSTM import Config, TextProcessor, StudentTTS


# ==========================================
# 1. Загрузка вспомогательных моделей
# ==========================================
def load_models(cfg, device="cpu"):
    print(f"⏳ Загрузка Vocos на {device}...")
    # Загружаем ту же модель, что использовали для подготовки данных
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    vocoder.eval()

    print(f"⏳ Загрузка Speaker Encoder (ECAPA-TDNN) на {device}...")
    spk_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    return vocoder, spk_encoder


# ==========================================
# 2. Извлечение эмбеддинга (SpeechBrain)
# ==========================================
def extract_speaker_embedding(audio_path, encoder, device):
    # SpeechBrain ECAPA-TDNN требует 16000 Hz
    signal, fs = torchaudio.load(audio_path)

    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)

    # Если стерео -> моно
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    # Нормализация громкости перед энкодером (опционально, но полезно)
    signal = signal / torch.max(torch.abs(signal))

    with torch.no_grad():
        # signal должен быть на том же устройстве, что и энкодер
        emb = encoder.encode_batch(signal.to(device))
        return emb.squeeze(1)  # [1, 192]


# ==========================================
# 3. Визуализация Attention
# ==========================================
def save_attention_image(attn, path="debug_attention.png"):
    plt.figure(figsize=(10, 6))
    plt.imshow(attn.cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
    plt.xlabel("Encoder steps (Text)")
    plt.ylabel("Decoder steps (Audio)")
    plt.title("Attention Map")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"🔍 Карта внимания сохранена в {path}")


# ==========================================
# 4. Основная функция генерации
# ==========================================
def generate_zero_shot(
        student_model,
        vocoder,
        spk_encoder,
        text,
        ref_audio_path,
        cfg,
        processor,
        output_path="zero_shot_result.wav",
        device="cpu"
):
    student_model.eval()
    student_model.to(device)

    # 1. Подготовка текста
    tokens = torch.tensor([processor.encode(text)], dtype=torch.long).to(device)
    lens = torch.tensor([tokens.size(1)]).to(device)
    print(tokens)
    # 2. Подготовка голоса
    print(f"🎤 Читаем голос из: {ref_audio_path}")
    spk_emb = extract_speaker_embedding(ref_audio_path, spk_encoder, device)

    print("🤖 Генерация спектрограммы...")
    with torch.no_grad():
        # Получаем выход модели
        # mel_output: [1, Time, 100]
        # stop_output: [1, Time, 1]
        mel_output, stop_output, attentions = student_model(tokens, lens, speaker_embs=spk_emb)

    # 3. Визуализация Attention
    save_attention_image(attentions[0], "inference_attention.png")

    # 4. Логика Stop Token
    stop_probs = torch.sigmoid(stop_output[0]).cpu().numpy()  # [Time, 1]

    stop_threshold = 0.5
    min_frames = 50  # Не останавливаться раньше ~0.5 сек

    # Ищем, где вероятность остановки превысила порог
    stop_indices = np.where(stop_probs[min_frames:] > stop_threshold)[0]

    if len(stop_indices) > 0:
        cut_idx = stop_indices[0] + min_frames
        print(f"✂️ Обрезка по Stop Token на кадре {cut_idx}")
        mel_output = mel_output[:, :cut_idx, :]
    else:
        print("⚠️ Stop Token не сработал, генерируем полную длину.")

    # 5. Синтез звука через Vocos
    print("🔊 Синтез аудио (Vocos)...")

    # ВАЖНО:
    # Модель выдала [1, Time, 100]
    # Vocos ожидает [1, 100, Time]
    features = mel_output.transpose(1, 2)

    # ВНИМАНИЕ: Мы НЕ делаем денормализацию ((x*80)-80),
    # так как модель училась предсказывать чистые признаки Vocos.

    with torch.no_grad():
        wav = vocoder.decode(features)
        wav = wav.squeeze().cpu().numpy()

    # Сохраняем (Vocos 24khz)
    sf.write(output_path, wav, 24000)
    print(f"✅ Готово! Аудио сохранено в: {output_path}")


if __name__ == "__main__":
    # Выбираем устройство (для инференса лучше CPU, если аудио короткое, или GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")

    cfg = Config()
    cfg.speaker_embedding_dim = 192  # Убеждаемся, что совпадает с SpeechBrain

    tp = TextProcessor(cfg.RUS_ALPHABET)

    # 1. Загрузка Студента
    student = StudentTTS(cfg).to(device)

    # Укажи путь к НОВОМУ чекпоинту (обученному на Vocos данных)
    # Старые чекпоинты (обученные на librosa) работать НЕ БУДУТ
    ckpt_path = "checkpoints/student_step_9500.pth"  # <--- ПОМЕНЯЙ НА СВОЙ

    if os.path.exists(ckpt_path):
        print(f"📂 Загрузка весов из {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ckpt['model_state_dict'])
    else:
        print(f"⚠️ Чекпоинт {ckpt_path} не найден! Будет шум.")

    # 2. Загрузка Вокодера и Энкодера
    vocoder, spk_encoder = load_models(cfg, device=device)

    # 3. Данные для теста
    test_text = "Привет! Это тест нового метода генерации звука."

    # Путь к файлу с голосом (любой wav/mp3)
    ref_audio = "samples/audio_2026-02-16_01-29-54.wav"

    # Если файла нет, создадим шум для теста (чтобы код не упал)
    if not os.path.exists(ref_audio):
        print("Создаю временный файл голоса для теста...")
        sf.write(ref_audio, np.random.uniform(-0.5, 0.5, 16000 * 3), 16000)
    # 4. Запуск
    generate_zero_shot(
        student,
        vocoder,
        spk_encoder,
        test_text,
        ref_audio,
        cfg,
        tp,
        output_path="result_vocos.wav",
        device=device
    )