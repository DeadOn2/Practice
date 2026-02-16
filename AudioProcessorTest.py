import torch
import librosa
import numpy as np
from scipy.io.wavfile import write

import matplotlib.pyplot as plt
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

device = "cuda" if torch.cuda.is_available() else "cpu"

ap = AudioProcessor(    sample_rate=24000,
    fft_size=1024,
    hop_length=256,
    win_length=1024,
    num_mels=80)
cfg = XttsConfig()

cfg.load_json("C:/Users/light/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
tts = Xtts.init_from_config(cfg).to(device)

tts.load_checkpoint(cfg, "C:/Users/light/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2")

tts.eval()

output_wav, gpt_latents, speaker_embedding = tts.synthesize("Тестирую тестовый текст для записи",config=cfg, speaker_wav="audio_2026-02-16_01-29-54.wav", language="ru")
print(output_wav)
wav_int16 = np.int16(output_wav * 32767)
write("output2.wav", 24000, wav_int16)

wav, sr = librosa.load("audio_2026-02-16_01-29-54.wav", sr = 24000)
wav = wav.astype(np.float32)
mel = ap.melspectrogram(wav)
print(mel.shape)

# plt.figure(figsize=(10, 4))
# plt.imshow(mel, aspect='auto', origin='lower')
# plt.colorbar()
# plt.title("Mel Spectrogram")
# plt.xlabel("Time")
# plt.ylabel("Mel bins")
# plt.tight_layout()
# plt.show()
