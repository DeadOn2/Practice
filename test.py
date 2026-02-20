import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Вставь сюда свой класс PodcastDistillDataset
class PodcastDistillDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.samples = []

        print(f"Scanning directory: {root_dir}...")

        # Обходим все папки dir1, dir2...
        folders = [f for f in self.root_dir.iterdir() if f.is_dir()]

        for folder in folders:
            # Ищем json файл в папке
            json_files = list(folder.glob("*.json"))
            if not json_files:
                continue

            with open(json_files[0], 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)

            # Сопоставляем индекс в JSON с файлом mp3 (dirX_index.mp3)
            for i, entry in enumerate(metadata_list):
                # Формируем имя файла: имя_папки + _ + индекс + .mp3
                audio_filename = f"{folder.name}_{i}.mp3"
                audio_path = folder / audio_filename

                if audio_path.exists():
                    self.samples.append({
                        "text": entry["text"],
                        "audio_path": str(audio_path),
                        "speaker_id": entry.get("speaker", "unknown")
                    })

        print(f"Dataset loaded: {len(self.samples)} samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
dataset = PodcastDistillDataset("C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped")

print(len(dataset))          # сколько сэмплов
print(dataset[0])            # первый элемент
