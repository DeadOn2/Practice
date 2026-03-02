import torch
import torchaudio
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
from vocos import Vocos
from speechbrain.inference.speaker import EncoderClassifier
import time
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

    # 2. Подготовка голоса
    spk_emb = extract_speaker_embedding(ref_audio_path, spk_encoder, device)

    print("🤖 Генерация спектрограммы (с учетом Post-Net)...")
    print(f"Spk Emb Shape: {spk_emb.shape}")
    with torch.no_grad():
        # Просто передаем нужные пороги при вызове модели
        mel_raw, mel_post, stop_output, attentions = student_model(
            tokens,
            lens,
            speaker_embs=spk_emb,
            stop_threshold=0.5,  # Выставляй тут любую чувствительность
            min_stop_frames=50  # Минимум полсекунды звука
        )

    # 3. Визуализация Attention (берем из того же места)
    save_attention_image(attentions[0], f"{output_path[:len(output_path)-4]}_inference_attention.png")

    # 4. Логика Stop Token
    stop_probs = torch.sigmoid(stop_output[0]).cpu().numpy()  # [Time, 1]

    # Посмотрим, насколько модель вообще уверена
    print(f"📊 Максимальная вероятность стоп-токена за весь файл: {stop_probs.max():.6f}")

    # 5. Синтез звука через Vocos
    print("🔊 Синтез аудио (Vocos) из Post-Net выхода...")

    # Сразу переходим к Vocos, так как нормализации больше нет
    features = mel_post.transpose(1, 2)

    with torch.no_grad():
        wav = vocoder.decode(features)
        wav = wav.squeeze().cpu().numpy()

    sf.write(output_path, wav, 24000)
    print(f"✅ Готово! Аудио сохранено: {output_path}")


if __name__ == "__main__":
    # Выбираем устройство (для инференса лучше CPU, если аудио короткое, или GPU)
    device = "cpu"
    print(f"Используем устройство: {device}")

    cfg = Config()
    cfg.speaker_embedding_dim = 192  # Убеждаемся, что совпадает с SpeechBrain

    tp = TextProcessor(cfg.RUS_ALPHABET)

    # 1. Загрузка Студента
    student = StudentTTS(cfg).to(device)

    # Укажи путь к НОВОМУ чекпоинту (обученному на Vocos данных без нормализации)
    ckpt_path = "checkpoints/student_step_8750.pth"  # <--- ПОМЕНЯЙ НА СВОЙ

    if os.path.exists(ckpt_path):
        print(f"📂 Загрузка весов из {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ckpt['model_state_dict'])
    else:
        print(f"⚠️ Чекпоинт {ckpt_path} не найден! Будет шум.")

    # 2. Загрузка Вокодера и Энкодера
    vocoder, spk_encoder = load_models(cfg, device=device)

    # 3. Данные для теста
    # test_text = "А если я напишу текст, которого не было в тестовой выборке."
    test_text = "В этой серии мы говорим о том, как делать ремонт правильно, как добиваться хорошего результата и избежать основных ошибок при его проведении."
    # Путь к файлу с голосом (любой wav/mp3)
    ref_audio_1 = "samples/audio_2026-02-16_01-29-54.wav"
    ref_audio_2 = r"C:\Users\light\Downloads\podcasts_1_stripped_archive\podcasts_1_stripped\test\100605980\100605980_1.mp3"
    # Если файла нет, создадим шум для теста (чтобы код не упал)
    # if not os.path.exists(ref_audio):
    #     print("Создаю временный файл голоса для теста.")
    #     sf.write(ref_audio, np.random.uniform(-0.5, 0.5, 16000 * 3), 16000)

    # 4. Запуск
    generate_zero_shot(
        student,
        vocoder,
        spk_encoder,
        test_text,
        ref_audio_1,
        cfg,
        tp,
        output_path="result_vocos_my_sample.wav",
        device=device
    )
    time.sleep(10)

    generate_zero_shot(
        student,
        vocoder,
        spk_encoder,
        test_text,
        ref_audio_2,
        cfg,
        tp,
        output_path="result_vocos_dataset_sample.wav",
        device=device
    )