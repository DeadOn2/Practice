import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.vocoders import HIFIGAN


# ==========================================
# 1. ПРЕДОБУЧЕННЫЕ КОМПОНЕНТЫ (Wrappers)
# ==========================================
class PretrainedSpeakerEncoder:
    """
    Используем ECAPA-TDNN от SpeechBrain.
    Обеспечивает Zero-Shot клонирование.
    """

    def __init__(self, device='cuda'):
        print("Loading Speaker Encoder (ECAPA-TDNN)...")
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_model/spk_encoder",
            run_opts={"device": device}
        )
        self.device = device

    def embed_audio(self, wav_tensor):
        # wav_tensor: [Batch, Time]
        # ECAPA ожидает [Batch, Time] и возвращает [Batch, 1, 192]
        with torch.no_grad():
            emb = self.encoder.encode_batch(wav_tensor)
        return emb.squeeze(1)  # [Batch, 192]


class PretrainedVocoder:
    """
    Используем HiFi-GAN от SpeechBrain.
    Нужен не только для инференса, но и для получения параметров Mel-спектрограммы.
    """

    def __init__(self, device='cuda'):
        print("Loading Vocoder (HiFi-GAN)...")
        self.hifi_gan = HIFIGAN.from_hparams(
            source="speechbrain/tts-hifigan-ljspeech",
            savedir="tmp_model/vocoder",
            run_opts={"device": device}
        )
        self.device = device

        # Вытаскиваем параметры Mel из конфига HiFi-GAN для трансформации
        # Обычно это log-mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,  # HiFiGAN speechbrain обычно 22050Hz
            n_mels=80,
            n_fft=1024,
            win_length=1024,
            hop_length=256
        ).to(device)

    def wav_to_mel(self, wav):
        """Превращает аудио (Teacher output) в таргет для ученика"""
        # wav: [Batch, Time]
        # Ресемплинг если нужно (XTTS обычно 24k, HiFiGan 22k)
        # Здесь предполагаем, что вход уже ресемплирован
        mels = self.mel_transform(wav)
        mels = torch.log(mels + 1e-6)  # Log-Mel
        return mels

    def mel_to_wav(self, mel):
        """Для проверки результата ученика"""
        with torch.no_grad():
            wav = self.hifi_gan.decode_batch(mel)
        return wav


# ==========================================
# 2. МОДЕЛЬ УЧЕНИКА (Student Synthesizer)
# ==========================================

class StudentSynthesizer(nn.Module):
    """
    Легкая модель: Текст + Вектор Спикера -> Мел-спектрограмма.
    Архитектура: Non-Autoregressive Transformer (упрощенный).
    """

    def __init__(self, vocab_size, speaker_dim=192, mel_dim=80, d_model=256):
        super().__init__()

        # 1. Text Encoder
        self.text_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 1000, d_model))  # Simple learnable pos enc

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.text_encoder = nn.TransformerEncoder(enc_layer, num_layers=4)

        # 2. Conditioning (Speaker Injection)
        self.speaker_proj = nn.Linear(speaker_dim, d_model)

        # 3. Decoder (Mel generator)
        # В реальном FastSpeech нужен Length Regulator.
        # Здесь используем простой декодер с Cross-Attention на текст.
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=4, dim_feedforward=512, batch_first=True)
        self.mel_decoder = nn.TransformerDecoder(dec_layer, num_layers=4)

        self.mel_proj = nn.Linear(d_model, mel_dim)

    def forward(self, text_idx, speaker_emb, target_len=None):
        # text_idx: [B, T_text]
        # speaker_emb: [B, 192]

        # Encode Text
        x = self.text_emb(text_idx) + self.pos_enc[:, :text_idx.size(1), :]
        memory = self.text_encoder(x)

        # Inject Speaker Style (Global Conditioning)
        style = self.speaker_proj(speaker_emb).unsqueeze(1)  # [B, 1, d_model]
        memory = memory + style  # Добавляем стиль к выходу энкодера

        # Decode to Mel
        # Для обучения нам нужна "query" последовательность нужной длины.
        # В инференсе эту длину предсказывает duration predictor (здесь опущен для краткости).
        # При обучении берем длину из Teacher Target.
        if target_len is None:
            target_len = 200  # Dummy length for inference example

        # Создаем Positional Queries для декодера
        tgt_query = torch.zeros(text_idx.size(0), target_len, 256).to(text_idx.device)
        tgt_query = tgt_query + self.pos_enc[:, :target_len, :]

        # Генерируем мелы
        out = self.mel_decoder(tgt_query, memory)
        mels = self.mel_proj(out)  # [B, T_mel, 80]

        return mels.transpose(1, 2)  # [B, 80, T_mel] для совместимости с Vocoder


# ==========================================
# 3. ПАЙПЛАЙН ДИСТИЛЛЯЦИИ
# ==========================================

class DistillationFramework:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Загрузка готовых моделей
        self.spk_encoder = PretrainedSpeakerEncoder(self.device)
        self.vocoder_tools = PretrainedVocoder(self.device)

        # Загрузка учителя XTTS (через официальный API)
        print("Loading Teacher (XTTS)...")
        from TTS.api import TTS
        self.teacher = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

        # Инициализация ученика
        self.student = StudentSynthesizer(vocab_size=5000).to(self.device)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=1e-4)
        self.loss_fn = nn.L1Loss()

    def train_step(self, text, ref_audio_path):
        """
        Один шаг дистилляции.
        text: Строка (из спец. лексики)
        ref_audio_path: Путь к файлу голоса (для клонирования)
        """
        # 1. ПОДГОТОВКА ДАННЫХ (TEACHER SIDE)
        # Генерируем аудио учителем. Важно: output должен быть WAV.
        # XTTS может быть медленным, поэтому лучше кешировать это.
        with torch.no_grad():
            # Генерируем wav (возвращает numpy list)
            # language="en" или "ru" в зависимости от вашей спец лексики
            wav_teacher_list = self.teacher.tts(text=text, speaker_wav=ref_audio_path, language="en")

            # Конвертируем в тензор
            wav_tensor = torch.tensor(wav_teacher_list).unsqueeze(0).to(self.device)

            # Ресемплинг: XTTS (24k) -> HiFiGan (22050)
            resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=22050).to(self.device)
            wav_target = resampler(wav_tensor)

            # Создаем Target Mel, который идеально подходит к нашему Vocoder
            target_mel = self.vocoder_tools.wav_to_mel(wav_target)  # [1, 80, Time]

            # Получаем эмбеддинг спикера из референса (Zero-Shot компонент)
            # Загружаем ref audio для энкодера
            ref_wav, sr = torchaudio.load(ref_audio_path)
            # Ресемплинг для ECAPA (обычно 16k)
            if sr != 16000:
                resampler_enc = torchaudio.transforms.Resample(sr, 16000)
                ref_wav = resampler_enc(ref_wav)
            ref_wav = ref_wav.to(self.device)

            speaker_emb = self.spk_encoder.embed_audio(ref_wav)  # [1, 192]

        # 2. ОБУЧЕНИЕ УЧЕНИКА (STUDENT SIDE)
        self.student.train()
        self.optimizer.zero_grad()

        # Токенизация текста (упрощенная)
        # В реальности нужен нормальный TextCleaner + Tokenizer
        text_tokens = torch.tensor([[ord(c) for c in text]]).to(self.device)  # Заглушка
        text_tokens[text_tokens > 4999] = 0  # Clip for vocab size

        # Ученик предсказывает мел
        # Передаем target_len, так как модель без duration predictor'а пока не знает длину
        pred_mel = self.student(text_tokens, speaker_emb, target_len=target_mel.shape[2])

        # 3. РАСЧЕТ ОШИБКИ
        # Сравниваем мел ученика с мелом, полученным из аудио учителя
        loss = self.loss_fn(pred_mel, target_mel)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_distillation(self, dataset_list):
        print("Starting distillation loop...")
        for epoch in range(5):
            total_loss = 0
            for i, (text, ref_path) in enumerate(dataset_list):
                loss = self.train_step(text, ref_path)
                total_loss += loss
                if i % 10 == 0:
                    print(f"Epoch {epoch}, Step {i}, Loss: {loss:.4f}")
            print(f"Epoch {epoch} finished. Avg Loss: {total_loss / len(dataset_list)}")


# ==========================================
# 4. ЗАПУСК
# ==========================================

if __name__ == "__main__":
    # Пример данных
    # Ваша спец. лексика + пара путей к файлам голосов
    data = [
        ("System initialization complete. Loading module alpha.", "voice_sample_1.wav"),
        ("Warning: Temperature critical in sector 7.", "voice_sample_1.wav"),
        ("Protocol override accepted.", "voice_sample_2.wav")
    ]

    # Создаем файлы-заглушки, чтобы код не упал при запуске без файлов
    import os

    if not os.path.exists("voice_sample_1.wav"):
        torchaudio.save("voice_sample_1.wav", torch.randn(1, 16000), 16000)
        torchaudio.save("voice_sample_2.wav", torch.randn(1, 16000), 16000)

    framework = DistillationFramework()
    framework.run_distillation(data)