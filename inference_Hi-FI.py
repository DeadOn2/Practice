import torch
import torchaudio
import numpy as np
import soundfile as sf
import os
from bigvgan import BigVGAN
from speechbrain.inference.speaker import EncoderClassifier
from GigaTestLSTM import Config, TextProcessor, StudentTTS


# ==========================================
# 1. Ручная загрузка вокодера (Исправленный путь)
# ==========================================
def load_vocoder(device, n_mels=100):
    print(f"⏳ Загрузка BigVGAN (n_mels={n_mels})...")

    # У NVIDIA v2 100-полосная модель существует только для 24khz.
    # Это не страшно, вокодер просто восстановит звук в 24000 Гц.
    if n_mels == 100:
        repo_id = "nvidia/bigvgan_v2_24khz_100band_256x"
    else:
        repo_id = "nvidia/bigvgan_v2_22khz_80band_256x"

    try:
        model = BigVGAN.from_pretrained(repo_id, use_cuda_kernel=False)
        model.remove_weight_norm()
        model.eval().to(device)
        return model
    except Exception as e:
        print(f"❌ Ошибка при загрузке вокодера: {e}")
        return None


# ==========================================
# 2. Загрузка Speaker Encoder для Zero-Shot
# ==========================================
def load_speaker_encoder(device):
    print("⏳ Загрузка Speaker Encoder (SpeechBrain)...")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )


def get_spk_emb(audio_path, encoder, device):
    signal, fs = torchaudio.load(audio_path)
    if fs != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(signal)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    with torch.no_grad():
        emb = encoder.encode_batch(signal.to(device))
        return emb.squeeze(1)  # [1, 192]


# ==========================================
# 3. Функция генерации
# ==========================================
def generate_audio_bigvgan(student_model, vocoder, spk_encoder, text, ref_audio, cfg, processor,
                           output_path="output_bigvgan.wav"):
    student_model.eval()
    device = "cpu"

    # --- ЭТАП 1: Подготовка входа ---
    tokens = torch.tensor([processor.encode(text)], dtype=torch.long).to(device)
    lens = torch.tensor([tokens.size(1)]).to(device)

    # Клонируем голос (Zero-Shot)
    spk_emb = get_spk_emb(ref_audio, spk_encoder, device)

    # --- ЭТАП 2: Генерация мела Студентом ---
    print(f"🎤 Студент генерирует мел-спектрограмму...")
    with torch.no_grad():
        # ВАЖНО: передаем speaker_embs, так как мы в режиме Zero-Shot
        mel_output, stop_tokens, _ = student_model(tokens, lens, speaker_embs=spk_emb)

    if mel_output.shape[1] == 0:
        print("❌ Ошибка: Ранняя остановка (модель выдала пустой мел).")
        return

    mel = mel_output.squeeze(0).cpu()

    # --- ЭТАП 3: Денормализация (под BigVGAN) ---
    # Перевод из 0..1 обратно в децибелы
    mel_db = (mel * 80) - 80

    # Перевод dB в логарифмическую амплитуду (ln), которую ждет BigVGAN
    mel_log = mel_db * 0.11512925

    # Приводим к виду [Batch, Channels, Time]
    mel_input = mel_log.T.unsqueeze(0).to(device)

    # --- ЭТАП 4: Синтез звука ---
    print(f"🔊 BigVGAN синтезирует волну из {mel_input.shape[2]} кадров...")
    with torch.no_grad():
        wav = vocoder(mel_input)
        wav = wav.squeeze().cpu().numpy()

    # --- ЭТАП 5: Сохранение ---
    if np.max(np.abs(wav)) > 0:
        wav = wav / np.max(np.abs(wav))

    # ВАЖНО: Модель 100band от NVIDIA работает на 24000 Гц
    target_sr = 24000 if cfg.n_mels == 100 else 22050
    sf.write(output_path, wav, target_sr)
    print(f"✨ Победа! Файл сохранен: {output_path}")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    cfg = Config()
    cfg.n_mels = 100  # Убедись, что тут 100
    tp = TextProcessor(cfg.RUS_ALPHABET)

    student = StudentTTS(cfg).to("cpu")
    ckpt_path = "checp_old/student_step_16750.pth"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        student.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ Студент загружен (Step: 17250)")

        vocoder = load_vocoder("cpu", n_mels=cfg.n_mels)
        spk_encoder = load_speaker_encoder("cpu")

        if vocoder is not None:
            text = "Привет! Я использую вокодер Биг Ви Ган версии два для чистого звучания."
            # Замени на путь к своему реальному файлу для клонирования!
            reference = "samples/audio_2026-02-16_01-29-54.wav"

            if os.path.exists(reference):
                generate_audio_bigvgan(student, vocoder, spk_encoder, text, reference, cfg, tp)
            else:
                print(f"⚠️ Файл референса {reference} не найден.")
    else:
        print(f"❌ Чекпоинт {ckpt_path} не найден.")