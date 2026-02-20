import torch
import torchaudio
from GigaTEst3 import StudentModel, TeacherInterface, Config, VoiceBpeTokenizer
from vocos import Vocos  # Новый вокодер

# 1. Настройки
device = "cpu"
checkpoint = "student_step_26000.pth"
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
    # Текст -> Токены
    tokens = torch.tensor(tokenizer.encode(test_text, lang="ru")).unsqueeze(0).to(device)

    # РЕШЕНИЕ ОШИБКИ XTTS: Передаем список ПУТЕЙ, а не тензор
    spk_emb = teacher.get_speaker_embedding([ref_audio])

    # Студент -> Мел-спектрограмма (Авторегрессионный инференс)
    # Убедись, что в StudentModel есть метод .inference()
    mel_pred = student.inference(tokens, spk_emb, max_len=500)

    # Vocos ожидает [B, 80, T]. Делаем transpose
    # Если mel_pred вернул [B, T, 80], меняем размерности
    mel_input = mel_pred.transpose(1, 2)

    # Вокодер -> Аудио
    if mel_input.shape[1] == 80:
        # Используем линейную интерполяцию для изменения размера
        mel_input = torch.nn.functional.interpolate(
            mel_input,
            size=100,
            mode='linear',
            align_corners=False
        ).transpose(1, 2)  # interpolate работает по последней оси, поэтому крутим туда-сюда

        # После интерполяции возвращаем форму [1, 100, Frames]
        mel_input = mel_input.transpose(1, 2)
    audio_signal = vocoder.decode(mel_input)

    # 4. Сохранение с нормализацией
    audio_for_save = audio_signal.detach().cpu()

    if audio_for_save.abs().max() > 1.0:
        audio_for_save /= audio_for_save.abs().max()
        audio_for_save *= 0.95

    torchaudio.save("test_step_26000_vocos.wav", audio_for_save, Config.sample_rate)

print("Готово! Проверяй файл test_step_26000_vocos.wav")