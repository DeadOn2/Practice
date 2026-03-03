import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import os
from vocos import Vocos
from speechbrain.inference.speaker import EncoderClassifier

# Импорт твоих классов
from GigaTestLSTM import Config, TextProcessor, StudentTTS

# ================= НАСТРОЙКИ =================
CHECKPOINT_PATH = "checkpoints/student_step_17250.pth"  # Твой последний чекпоинт
TEST_AUDIO_PATH = r"C:\Users\light\Downloads\podcasts_1_stripped_archive\podcasts_1_stripped\test\100605980\100605980_1.mp3"  # УКАЖИ ПУТЬ К ЛЮБОМУ ФАЙЛУ ИЗ ДАТАСЕТА
TEST_TEXT = "В этой серии мы говорим о том, как делать ремонт правильно, как добиваться хорошего результата и избежать основных ошибок при его проведении."  # Текст этого файла
DEVICE = "cpu"


# =============================================

def get_mel_from_audio(audio_path, cfg, vocoder, device="cpu"):
    """Делаем 'идеальный' мел из оригинального звука через экстрактор Vocos"""
    wav, sr = torchaudio.load(audio_path)

    # 1. Ресемплинг до 24кГц (требование Vocos)
    if sr != cfg.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=cfg.sample_rate)
        wav = resampler(wav)

    # 2. Если аудио стерео -> в моно
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    wav = wav.to(device)

    # 3. Извлекаем фичи точно так же, как при подготовке датасета
    with torch.no_grad():
        # vocoder.feature_extractor сам делает STFT, применяет свои Mel-фильтры и берет логарифм
        mel = vocoder.feature_extractor(wav)

    return mel.squeeze(0)  # Возвращаем [Mels, Time]


def plot_comparison(target_mel, pred_mel, attention):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # 1. Оригинал (Ground Truth)
    im1 = axes[0].imshow(target_mel.cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
    axes[0].set_title("Оригинал (Ground Truth) - Идеальные полоски")
    fig.colorbar(im1, ax=axes[0])

    # 2. Предсказание (Prediction)
    im2 = axes[1].imshow(pred_mel.detach().cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
    axes[1].set_title("Предсказание модели (Prediction) - Есть ли детали?")
    fig.colorbar(im2, ax=axes[1])

    # 3. Внимание (Alignment)
    im3 = axes[2].imshow(attention.detach().cpu().numpy(), aspect='auto', origin='lower', interpolation='none')
    axes[2].set_title("Внимание (Alignment) - Должна быть диагональ")

    plt.tight_layout()
    plt.savefig("comparison_debug.png")
    print("✅ Картинка сохранена в comparison_debug.png")
    plt.show()

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

def main():
    cfg = Config()
    tp = TextProcessor(cfg.RUS_ALPHABET)

    # 1. Загрузка модели
    model = StudentTTS(cfg).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("🤖 Модель загружена")

    # 2. Подготовка данных
    # Текст -> Токены
    tokens = torch.tensor([tp.encode(TEST_TEXT)], dtype=torch.long).to(DEVICE)
    lens = torch.tensor([tokens.size(1)]).to(DEVICE)
    device = "cpu"
    vocoder, spk_encoder = load_models(cfg, device=device)

    # Аудио -> Эмбеддинг (заглушка или реальный, если есть энкодер под рукой)
    # Для теста просто возьмем нули или случайный, если нет энкодера в скрипте.
    # НО лучше использовать реальный. Если сложно подключить, используй нули [1, 192]:
    spk_emb = extract_speaker_embedding(TEST_AUDIO_PATH, spk_encoder, device)
    # 3. Инференс
    with torch.no_grad():
        _, mel_post, _, attentions = model(tokens, lens, speaker_embs=spk_emb)

    # Денормализация предсказания (чтобы сравнить с сырым оригиналом)
    # Model output: [1, Time, Mels] -> (x * std) + mean
    pred_mel = mel_post
    pred_mel = pred_mel.squeeze(0).transpose(0, 1)  # [Mels, Time]

    # 4. Получение оригинала
    target_mel = get_mel_from_audio(TEST_AUDIO_PATH, cfg, vocoder).to(DEVICE)

    # Обрезаем длинный хвост, чтобы удобно смотреть
    min_len = min(target_mel.shape[1], pred_mel.shape[1])
    target_mel = target_mel[:, :min_len]
    print(f"Min: {target_mel.min()}, Max: {target_mel.max()}")
    pred_mel = pred_mel[:, :min_len]
    print(f"Min: {pred_mel.min()}, Max: {pred_mel.max()}")
    # 5. Рисуем
    plot_comparison(target_mel, pred_mel, attentions[0])


if __name__ == "__main__":
    main()