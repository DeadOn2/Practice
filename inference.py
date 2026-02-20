import torch
import torchaudio
from GigaTEst3_1 import StudentModel, TeacherInterface, Config, VoiceBpeTokenizer
from vocos import Vocos  # Новый вокодер

# 1. Настройки
device = "cpu"
checkpoint = "student_step_4000.pth"
vocab_path = "./xtts_vocab/vocab.json"
test_text = "Привет! Я твоя новая модель, и теперь я использую современный вокодер."
ref_audio = "audio_2026-02-16_01-29-54.wav"  # Оставляем путь строкой!

# 2. Загрузка компонентов
tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)
student = StudentModel(vocab_size=256000).to(device)
student.load_state_dict(torch.load(checkpoint, map_location=device))
student.eval()

teacher = TeacherInterface(device)

# Загружаем Vocos (он сам скачает веса один раз)
vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
print("Все модели загружены. Начинаю генерацию...")

# 3. Генерация
with torch.no_grad():
    print("start1")
    tokens = torch.tensor(
        tokenizer.encode(test_text, lang="ru")
    ).unsqueeze(0).to(device)

    spk_emb = teacher.get_speaker_embedding([ref_audio]).to(device)

    mel_pred, _ = student(tokens, spk_emb, mel_target=None)

    mel_input = mel_pred.transpose(1, 2)

    audio_signal = vocoder.decode(mel_input)
    print("start2")
    audio_for_save = audio_signal.cpu()

    if audio_for_save.abs().max() > 1.0:
        audio_for_save = audio_for_save / audio_for_save.abs().max() * 0.95

    torchaudio.save(
        "test_step_4000_vocos.wav",
        audio_for_save,
        Config.sample_rate
    )
    print("start3")

print(mel_pred.shape, mel_input.shape)
print("Готово! Проверяй файл test_step_4000_vocos.wav")