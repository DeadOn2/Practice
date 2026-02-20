import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio

class TeacherPrecomputeDataset(Dataset):
    def __init__(
        self,
        raw_dataset,
        teacher_model,
        speaker_encoder,
        tokenizer,
        audio_processor,
        save_dir,
        device="cuda"
    ):
        self.raw_dataset = raw_dataset
        self.teacher = teacher_model.to(device).eval()
        self.speaker_encoder = speaker_encoder.to(device).eval()
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def __len__(self):
        return len(self.raw_dataset)

    @torch.no_grad()
    def process_sample(self, idx):
        sample = self.raw_dataset[idx]

        text = sample["text"]
        audio_path = sample["audio_path"]

        # ---------- 1. Tokenize ----------
        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor(tokens["input_ids"]).unsqueeze(0).to(self.device)

        # ---------- 2. Load audio ----------
        wav, sr = torchaudio.load(audio_path)

        if sr != self.audio_processor.sample_rate:
            wav = torchaudio.functional.resample(
                wav, sr, self.audio_processor.sample_rate
            )

        wav = wav.to(self.device)

        # ---------- 3. Speaker embedding ----------
        speaker_emb = self.speaker_encoder(wav)

        # ---------- 4. Teacher forward ----------
        teacher_outputs = self.teacher(
            input_ids=input_ids,
            speaker_embeddings=speaker_emb
        )

        target_mel = teacher_outputs["mel"]  # убедись что ключ правильный

        # ---------- 5. Save ----------
        save_path = self.save_dir / f"sample_{idx}.pt"

        torch.save({
            "input_ids": input_ids.squeeze(0).cpu(),
            "speaker_emb": speaker_emb.squeeze(0).cpu(),
            "target_mel": target_mel.squeeze(0).cpu()
        }, save_path)

        return save_path

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
def build_teacher_dataset(dataset):
    for i in range(len(dataset)):
        path = dataset.process_sample(i)
        print(f"Saved: {path}")

raw_dataset = PodcastDistillDataset("path/to/data")

teacher_dataset = TeacherPrecomputeDataset(
    raw_dataset=raw_dataset,
    teacher_model=teacher,
    speaker_encoder=speaker_encoder,
    tokenizer=tokenizer,
    audio_processor=audio_processor,
    save_dir="teacher_precomputed",
    device="cuda"
)

build_teacher_dataset(teacher_dataset)
