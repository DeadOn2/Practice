import torch
import torchaudio
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt  # Для сохранения attention
from vocos import Vocos
from speechbrain.inference.speaker import EncoderClassifier

import GigaTestLSTM
from GigaTestLSTM import Config, TextProcessor, StudentTTS, save_mel_image


# ==========================================
# 1. Загрузка вспомогательных моделей
# ==========================================
def load_models(cfg):
    device = "cpu"
    print("⏳ Загрузка Vocos...")
    # Обрати внимание: Vocos mel-24khz ожидает 100 бинов мела, как у тебя в конфиге
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    vocoder.eval()

    print("⏳ Загрузка Speaker Encoder (ECAPA-TDNN)...")
    spk_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    return vocoder, spk_encoder


# ==========================================
# 2. Извлечение эмбеддинга
# ==========================================
def extract_speaker_embedding(audio_path, encoder, device):
    signal, fs = torchaudio.load(audio_path)
    if fs != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
        signal = resampler(signal)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    with torch.no_grad():
        emb = encoder.encode_batch(signal.to(device))
        return emb.squeeze(1)


# ==========================================
# 3. Визуализация Attention (НОВОЕ)
# ==========================================
def save_attention_image(attn, path="debug_attention.png"):
    # attn: (T_decoder, T_encoder)
    plt.figure(figsize=(8, 6))
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
        output_path="zero_shot_result.wav"
):
    student_model.eval()
    device = "cpu"

    tokens = torch.tensor([processor.encode(text)], dtype=torch.long).to(device)
    lens = torch.tensor([tokens.size(1)]).to(device)
    spk_emb = extract_speaker_embedding(ref_audio_path, spk_encoder, device)

    with torch.no_grad():
        # ТЕПЕРЬ МОДЕЛЬ ВОЗВРАЩАЕТ 3 ЗНАЧЕНИЯ
        mel_output, stop_output, attentions = student_model(tokens, lens, speaker_embs=spk_emb)

    # 1. Визуализируем Attention (берем первый элемент батча)
    save_attention_image(attentions[0], "inference_attention.png")

    # 2. Логика обрезки по Stop Token (НОВОЕ)
    # Ищем первый кадр, где вероятность остановки > 0.5
    # --- БЕЗОПАСНАЯ ЛОГИКА ОБРЕЗКИ ---
    stop_probs = torch.sigmoid(stop_output[0]).cpu().numpy()

    # Устанавливаем порог и минимальное количество кадров (например, 50 кадров ~ 0.5 сек)
    stop_threshold = 0.1
    min_stop_frame = 40

    # Ищем кадры только ПОСЛЕ min_stop_frame
    stop_idx = np.where(stop_probs[min_stop_frame:] > stop_threshold)[0]

    if len(stop_idx) > 0:
        # Прибавляем min_stop_frame, так как поиск был со смещением
        end_frame = stop_idx[0] + min_stop_frame
        print(f"✂️ Stop Token сработал на кадре: {end_frame}")
        mel_output = mel_output[:, :end_frame, :]
    else:
        print("⚠️ Stop Token не сработал или сработал слишком рано, берем максимум.")
        # Если модель выдала 0 кадров по какой-то причине, принудительно берем всё
        if mel_output.shape[1] <= 1:
            print("📢 Force: игнорируем стоп-токен, модель выдала пустоту.")

    if mel_output.shape[1] == 0:
        print("❌ Ошибка: Пустая спектрограмма.")
        return

    # 3. Подготовка для Vocos
    mel = mel_output.transpose(1, 2)  # [1, 100, T]
    save_mel_image(mel, "melt_test.png")
    # Денормализация (ВАЖНО: проверь, не летят ли тут NaN)
    mel_db = (mel * 80.0) - 80.0

    # На всякий случай зануляем слишком маленькие значения
    mel_db = torch.clamp(mel_db, min=-100, max=0)

    # 4. Синтез
    with torch.no_grad():
        wav = vocoder.decode(mel_db)
        wav = wav.squeeze().cpu().numpy()

    sf.write(output_path, wav, 24000)
    print(f"✅ Готово! Аудио: {output_path}")
if __name__ == "__main__":
    cfg = Config()
    # Убедись, что размер эмбеддинга в конфиге совпадает со SpeechBrain (192)
    cfg.speaker_embedding_dim = 192

    tp = TextProcessor(cfg.RUS_ALPHABET)

    # Инициализация модели
    student = StudentTTS(cfg).to("cpu")

    # Загрузка чекпоинта (того, что обучался с X-векторами!)
    ckpt_path = "checp_old/student_step_9500.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        student.load_state_dict(ckpt['model_state_dict'])
        print(f"🚀 Студент загружен (Step: {ckpt.get('global_step', '?')})")
    else:
        print("⚠️ Чекпоинт не найден, инференс на случайных весах.")

    # Загрузка вокодера и энкодера спикера
    vocoder, spk_encoder = load_models(cfg)

    # --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---
    my_text = "Этот голос скибиди доп доп ес"

    # Путь к любому файлу, чей голос хотим украсть
    reference_wav = "samples/audio_2026-02-16_01-29-54.wav"

    generate_zero_shot(
        student,
        vocoder,
        spk_encoder,
        my_text,
        reference_wav,
        cfg,
        tp,
        "cloned_voice.wav"
    )