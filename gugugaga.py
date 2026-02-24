import torch
from pathlib import Path
from tqdm import tqdm


def verify_and_clean_mels(root_dir, target_mels=100, delete_invalid=False):
    root = Path(root_dir)
    # Ищем все файлы с расширением _teacher.pt рекурсивно
    mel_files = list(root.rglob("*_teacher.pt"))

    invalid_files = []
    print(f"Найдено файлов для проверки: {len(mel_files)}")

    for mel_path in tqdm(mel_files):
        try:
            # map_location='cpu' чтобы не забивать VRAM при проверке
            mel_tensor = torch.load(mel_path, map_location='cpu')

            # В вашем коде формат (Time, n_mels), проверяем вторую размерность
            current_mels = mel_tensor.shape[1]

            if current_mels != target_mels:
                invalid_files.append((mel_path, current_mels))

                if delete_invalid:
                    mel_path.unlink()  # Удаляем файл

        except Exception as e:
            print(f"Ошибка при чтении {mel_path}: {e}")

    print("\n--- Итоги проверки ---")
    if not invalid_files:
        print(f"Все файлы имеют корректную размерность n_mels = {target_mels}!")
    else:
        print(f"Найдено поврежденных/неверных файлов: {len(invalid_files)}")
        for path, size in invalid_files[:10]:  # Покажем первые 10
            print(f"Файл: {path.name} | n_mels: {size}")

        if len(invalid_files) > 10:
            print(f"... и еще {len(invalid_files) - 10} файлов.")

        if delete_invalid:
            print("Все неверные файлы были УДАЛЕНЫ. Теперь запустите основной скрипт для их перегенерации.")
        else:
            print("Файлы НЕ удалены. Установите delete_invalid=True для автоматической очистки.")


if __name__ == "__main__":
    PATH = "C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped/test2"
    # Сначала запустите с False, чтобы просто посмотреть масштаб беды
    verify_and_clean_mels(PATH, target_mels=100, delete_invalid=False)