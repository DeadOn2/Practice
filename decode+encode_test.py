import torch
import torchaudio
from GigaTEst3 import StudentModel, TeacherInterface, Config, VoiceBpeTokenizer
from vocos import Vocos  # Новый вокодер

# 1. Настройки
device = "cuda"
checkpoint = "student_step_26000.pth"
vocab_path = "./xtts_vocab/vocab.json"
test_text = "Привет! Я твоя новая модель, и теперь я использую современный вокодер."
ref_audio = "audio_2026-02-16_01-29-54.wav"  # Оставляем путь строкой!

# 2. Загрузка компонентов
tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
student = StudentModel(vocab_size=256000).to(device)
student.load_state_dict(torch.load(checkpoint, map_location=device))
teacher = TeacherInterface(device)
student.eval()
with torch.no_grad():
    tokens_hi = torch.tensor(tokenizer.encode("Привет", lang="ru")).unsqueeze(0).to(device)
    tokens_bye = torch.tensor(tokenizer.encode("Очень длинная фраза для проверки", lang="ru")).unsqueeze(0).to(device)

    spk = teacher.get_speaker_embedding(["audio_2026-02-16_01-29-54.wav"])

    mel_hi = student.inference(tokens_hi, spk, max_len=100)
    mel_bye = student.inference(tokens_bye, spk, max_len=100)

    diff = (mel_hi - mel_bye).abs().sum().item()
    print(f"Разница между двумя фразами: {diff}")
    if diff < 1e-5:
        print("КРИТИЧЕСКАЯ ОШИБКА: Модель выдает одинаковый результат на разный текст!")
