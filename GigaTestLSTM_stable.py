import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from vocos import Vocos  # Используем Vocos для инференса
from speechbrain.inference.speaker import EncoderClassifier


# ==========================================
# Speaker Encoder (глобальный)
# ==========================================

print("Загрузка Speaker Encoder (ECAPA-TDNN)...")

spk_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    },
)


# ==========================================
# 1. Dataset
# ==========================================

class PodcastDistillDataset(Dataset):
    def __init__(self, root_dir, text_processor, cfg):
        self.root_dir = Path(root_dir)
        self.tp = text_processor
        self.cfg = cfg
        self.samples = []

        # Резервная загрузка Vocos (если нет .pt файлов)
        self.vocos_feature_extractor = Vocos.from_pretrained(
            "charactr/vocos-mel-24khz"
        ).to("cpu")

        print(f"Сканирование директории: {root_dir}...")
        folders = [f for f in self.root_dir.iterdir() if f.is_dir()]

        for folder in folders:
            json_files = list(folder.glob("*.json"))
            if not json_files:
                continue

            with open(json_files[0], "r", encoding="utf-8") as f:
                metadata_list = json.load(f)

            for i, entry in enumerate(metadata_list):
                audio_filename = f"{folder.name}_{i}.mp3"
                audio_path = folder / audio_filename

                teacher_mel_path = folder / f"{folder.name}_{i}_teacher.pt"
                spk_emb_path = folder / f"{folder.name}_{i}_spk.pt"

                if len(entry["text"]) > 182:
                    continue

                if audio_path.exists():
                    self.samples.append(
                        {
                            "text": entry["text"],
                            "audio_path": str(audio_path),
                            "teacher_mel_path": str(teacher_mel_path),
                            "spk_emb_path": str(spk_emb_path),
                        }
                    )

        print(f"Датасет загружен: {len(self.samples)} примеров.")

    def _get_mel_from_audio(self, audio_path):
        """
        Резервный метод: если .pt файла нет,
        генерируем Mel через Vocos на лету.
        """
        wav, sr = torchaudio.load(audio_path)

        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(
                wav,
                sr,
                self.cfg.sample_rate,
            )

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        with torch.no_grad():
            mel = self.vocos_feature_extractor.feature_extractor(wav)

        return mel.squeeze(0).transpose(0, 1)  # [Time, 100]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        text_tokens = torch.tensor(
            self.tp.encode(sample["text"]),
            dtype=torch.long,
        )

        # --- Target Mel ---
        if os.path.exists(sample["teacher_mel_path"]):
            target_mel = torch.load(sample["teacher_mel_path"])
        else:
            target_mel = self._get_mel_from_audio(sample["audio_path"])

        # --- Speaker Embedding ---
        if os.path.exists(sample["spk_emb_path"]):
            spk_emb = torch.load(sample["spk_emb_path"])
        else:
            signal, fs = torchaudio.load(sample["audio_path"])

            if fs != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=fs,
                    new_freq=16000,
                )
                signal = resampler(signal)

            with torch.no_grad():
                spk_emb = spk_classifier.encode_batch(signal)
                spk_emb = spk_emb.squeeze(0).squeeze(0)

            torch.save(spk_emb, sample["spk_emb_path"])

        return (
            text_tokens,
            target_mel,
            sample["text"],
            sample["audio_path"],
            spk_emb,
        )


# ==========================================
# Визуализация Mel
# ==========================================

def save_mel_image(mel, path="mel_spectrogram.png"):
    if torch.is_tensor(mel):
        mel = mel.detach().cpu().numpy()

    if len(mel.shape) == 3:
        mel = mel[0]

    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect="auto", origin="lower")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Generated Mel-Spectrogram")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"🖼 Спектрограмма сохранена как {path}")


# ==========================================
# Config
# ==========================================

class Config:
    RUS_ALPHABET = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя.,!?-–"
    vocab_size = len(RUS_ALPHABET) + 1

    speaker_embedding_dim = 192

    embedding_dim = 256
    encoder_hidden = 256
    decoder_hidden = 256
    attention_dim = 256

    # --- Vocos ---
    n_mels = 100
    sample_rate = 24000
    hop_length = 256

    alpha = 0.7
    beta = 0.3

    lr = 2e-4
    batch_size = 16
    epochs = 100

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ==========================================
# Text Processor
# ==========================================

class TextProcessor:
    def __init__(self, alphabet):
        self.char_to_id = {
            char: i + 1 for i, char in enumerate(alphabet)
        }
        self.id_to_char = {
            i + 1: char for i, char in enumerate(alphabet)
        }
        self.pad_id = 0

    def encode(self, text):
        text = text.lower()
        return [
            self.char_to_id[c]
            for c in text
            if c in self.char_to_id
        ]

    def decode(self, ids):
        return "".join(
            self.id_to_char[i]
            for i in ids
            if i in self.id_to_char
        )


# ==========================================
# Location Sensitive Attention
# ==========================================

class LocationSensitiveAttention(nn.Module):
    def __init__(
        self,
        encoder_dim,
        decoder_dim,
        attention_dim,
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
    ):
        super().__init__()

        self.W1 = nn.Linear(
            encoder_dim,
            attention_dim,
            bias=False,
        )
        self.W2 = nn.Linear(
            decoder_dim,
            attention_dim,
            bias=False,
        )
        self.V = nn.Linear(
            attention_dim,
            1,
            bias=False,
        )

        padding = attention_location_kernel_size // 2

        self.location_conv = nn.Conv1d(
            in_channels=1,
            out_channels=attention_location_n_filters,
            kernel_size=attention_location_kernel_size,
            padding=padding,
            bias=False,
        )

        self.location_dense = nn.Linear(
            attention_location_n_filters,
            attention_dim,
            bias=False,
        )

    def forward(self, query, keys, prev_weights, mask=None):
        """
        query: (B, 1, dec_dim)
        keys: (B, T, enc_dim)
        prev_weights: (B, T)
        """

        proj_key = self.W1(keys)
        proj_query = self.W2(query)

        loc_feat = self.location_conv(
            prev_weights.unsqueeze(1)
        )
        loc_feat = loc_feat.transpose(1, 2)
        proj_loc = self.location_dense(loc_feat)

        scores = self.V(
            torch.tanh(proj_key + proj_query + proj_loc)
        ).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(
            weights.unsqueeze(1),
            keys,
        )

        return context, weights
# ==========================================
# 4. Encoder Module
# ==========================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x, lengths):
        x = self.embedding(x)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_outputs, _ = self.lstm(packed_x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True
        )

        return outputs  # (B, T, hidden_dim * 2)


# ==========================================
# 5. Decoder Module
# ==========================================

def guided_attention_loss(attentions, text_lens, mel_lens, g=0.2):
    """
    attentions: (B, T_dec, T_enc)
    text_lens: (B)
    mel_lens: (B)
    """

    B, T_dec, T_enc = attentions.size()
    device = attentions.device
    loss = 0.0

    for i in range(B):
        N = text_lens[i].item()
        M = mel_lens[i].item()

        if N == 0 or M == 0:
            continue

        grid_n, grid_m = torch.meshgrid(
            torch.arange(N, device=device),
            torch.arange(M, device=device),
            indexing="ij"
        )

        W = 1.0 - torch.exp(
            -((grid_n.float() / N - grid_m.float() / M) ** 2) / (2 * g ** 2)
        )

        W = W.T  # (M, N)

        attn_slice = attentions[i, :M, :N]
        loss += torch.mean(attn_slice * W)

    return loss / B


# ==========================================
# PreNet
# ==========================================

class PreNet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 256]):
        super().__init__()

        self.layer1 = nn.Linear(in_dim, sizes[0])
        self.layer2 = nn.Linear(sizes[0], sizes[1])

    def forward(self, x):
        # Dropout ВСЕГДА включен (Tacotron-style)
        x = F.dropout(F.relu(self.layer1(x)), p=0.5, training=True)
        x = F.dropout(F.relu(self.layer2(x)), p=0.5, training=True)
        return x


# ==========================================
# PostNet
# ==========================================


# ==========================================
# Decoder
# ==========================================

class Decoder(nn.Module):
    def __init__(
        self,
        n_mels,
        decoder_hidden,
        encoder_total_dim,
        attention_dim,
        speaker_dim
    ):
        super().__init__()

        self.n_mels = n_mels
        self.decoder_hidden = decoder_hidden

        self.prenet = PreNet(n_mels, [256, 256])

        self.lstm_input_size = 256 + encoder_total_dim + speaker_dim
        self.lstm = nn.LSTMCell(self.lstm_input_size, decoder_hidden)

        self.attention = LocationSensitiveAttention(
            encoder_total_dim,
            decoder_hidden,
            attention_dim
        )

        self.linear_input_size = decoder_hidden + encoder_total_dim + speaker_dim
        self.linear = nn.Linear(self.linear_input_size, n_mels)
        self.stop_linear = nn.Linear(self.linear_input_size, 1)

    def forward(
        self,
        encoder_outputs,
        encoder_mask,
        spk_emb,
        teacher_mels=None,
        max_len=1000
    ):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        prev_weights = torch.zeros(
            batch_size,
            encoder_outputs.size(1),
            device=device
        )

        h = torch.zeros(batch_size, self.decoder_hidden, device=device)
        c = torch.zeros(batch_size, self.decoder_hidden, device=device)

        mel_input = torch.zeros(batch_size, self.n_mels, device=device)

        outputs = []
        stop_tokens = []
        attentions = []

        steps = (
            teacher_mels.size(1)
            if teacher_mels is not None
            else max_len
        )

        for t in range(steps):

            # 1. Attention
            context, attn_weights = self.attention(
                h.unsqueeze(1),
                encoder_outputs,
                prev_weights,
                encoder_mask
            )

            context = context.squeeze(1)
            prev_weights = attn_weights.squeeze(1)

            attentions.append(prev_weights)

            # 2. PreNet
            prenet_out = self.prenet(mel_input)

            # 3. LSTM
            rnn_input = torch.cat([prenet_out, context, spk_emb], dim=-1)
            h, c = self.lstm(rnn_input, (h, c))

            # 4. Output layers
            concat_out = torch.cat([h, context, spk_emb], dim=-1)

            mel_out = self.linear(concat_out)
            stop_out = self.stop_linear(concat_out)

            outputs.append(mel_out)
            stop_tokens.append(stop_out)

            # Inference stop condition
            if teacher_mels is None:
                stop_prob = torch.sigmoid(stop_out).mean().item()

                if t > 30 and stop_prob > 0.5:
                    print(f"DEBUG: Stop at step {t}")
                    break

                mel_input = mel_out
                continue

            # Teacher forcing + scheduled sampling
            if t < teacher_mels.size(1) - 1:
                if (
                    self.training
                    and t > 0
                    and torch.rand(1).item() < 0.25
                ):
                    mel_input = mel_out.detach()
                else:
                    mel_input = teacher_mels[:, t, :]
            else:
                mel_input = mel_out

        return (
            torch.stack(outputs, dim=1),
            torch.stack(stop_tokens, dim=1),
            torch.stack(attentions, dim=1),
        )


# ==========================================
# 6. Student TTS Model
# ==========================================

class StudentTTS(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = Encoder(
            cfg.vocab_size,
            cfg.embedding_dim,
            cfg.encoder_hidden
        )

        self.decoder = Decoder(
            cfg.n_mels,
            cfg.decoder_hidden,
            cfg.encoder_hidden * 2,
            cfg.attention_dim,
            speaker_dim=cfg.speaker_embedding_dim
        )

    def forward(self, text, text_lengths, speaker_embs, mels=None):
        device = text.device

        mask = (
            torch.arange(text.size(1), device=device)
            .expand(len(text_lengths), text.size(1))
            < text_lengths.unsqueeze(1)
        )

        encoder_outputs = self.encoder(text, text_lengths)

        mel_outputs, stop_outputs, attentions = self.decoder(
            encoder_outputs,
            mask,
            speaker_embs,
            teacher_mels=mels
        )

        return mel_outputs, stop_outputs, attentions


# ==========================================
# 8. Dataset & Collate
# ==========================================

class RussianTTSDataset(Dataset):
    def __init__(self, texts, gt_mels, teacher_mels, processor):
        self.texts = texts
        self.gt_mels = gt_mels
        self.teacher_mels = teacher_mels
        self.processor = processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = torch.tensor(
            self.processor.encode(self.texts[idx]),
            dtype=torch.long
        )

        return tokens, self.gt_mels[idx], self.teacher_mels[idx]


def save_checkpoint(model, optimizer, epoch, global_step, loss, path):
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, path)
    print(f"--- Чекпоинт сохранен: {path} ---")

# ==========================================
# 3. Обновленный Collate и Training Loop
# ==========================================

def collate_fn_podcast(batch):
    # Сортировка по убыванию длины текста
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    tokens, mels, raw_texts, audio_paths, spk_embs = zip(*batch)

    token_lens = torch.tensor([len(x) for x in tokens])
    tokens_padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True)

    mel_lens = torch.tensor([x.size(0) for x in mels])
    mels_padded = nn.utils.rnn.pad_sequence(mels, batch_first=True)

    spk_embs_tensor = torch.stack(spk_embs)

    return (
        tokens_padded,
        token_lens,
        mels_padded,
        mel_lens,
        raw_texts,
        audio_paths,
        spk_embs_tensor,
    )


# ==========================================
# Training
# ==========================================

def train_with_distillation(root_dir):

    # --- Masked losses ---
    def masked_mse(preds, targets, mask):
        diff = (preds - targets) ** 2
        return (diff * mask).sum() / (mask.sum() * cfg.n_mels + 1e-8)

    def masked_l1(preds, targets, mask):
        diff = torch.abs(preds - targets)
        return (diff * mask).sum() / (mask.sum() * cfg.n_mels + 1e-8)

    # --- Config ---
    cfg = Config()
    tp = TextProcessor(cfg.RUS_ALPHABET)

    dataset = PodcastDistillDataset(root_dir, tp, cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn_podcast,
        shuffle=True
    )

    student = StudentTTS(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr)

    writer = SummaryWriter(log_dir="runs/fast_distill_v2")

    # Stop-token loss
    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([15.0]).to(cfg.device),
        reduction="none"
    )

    global_step = 0
    start_epoch = 0

    # --- Load checkpoint ---
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith(".pth") and "step" in f
    ]

    if checkpoints:
        checkpoints.sort(
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        last_checkpoint = os.path.join(
            checkpoint_dir,
            checkpoints[-1]
        )

        print(f"Загрузка чекпоинта: {last_checkpoint}")

        try:
            ckpt = torch.load(last_checkpoint)
            student.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            global_step = ckpt.get("global_step", 0)
            start_epoch = ckpt.get("epoch", 0)

            print(
                f"Восстановлено: Эпоха {start_epoch}, "
                f"Шаг {global_step}"
            )
        except Exception as e:
            print(f"Ошибка загрузки чекпоинта: {e}")

    print("Начало обучения с Stop Token...")
    student.train()

    try:
        for epoch in range(start_epoch, cfg.epochs):
            for batch in dataloader:

                (
                    tokens,
                    token_lens,
                    gts,
                    gt_lens,
                    raw_texts,
                    audio_paths,
                    speaker_ids,
                ) = batch

                tokens = tokens.to(cfg.device)
                token_lens = token_lens.to(cfg.device)
                gts = gts.to(cfg.device)
                gt_lens = gt_lens.to(cfg.device)
                speaker_ids = speaker_ids.to(cfg.device)

                target_mels = gts

                optimizer.zero_grad()

                # Forward
                pred_mels, pred_stops, attentions = student(
                    tokens,
                    token_lens,
                    speaker_embs=speaker_ids,
                    mels=target_mels
                )

                # Trim to minimum length
                min_len = min(
                    pred_mels.size(1),
                    target_mels.size(1)
                )

                p_mel = pred_mels[:, :min_len, :]
                t_mel = target_mels[:, :min_len, :]
                p_stop = pred_stops[:, :min_len, :]

                # Mask
                mask = (
                    torch.arange(min_len, device=cfg.device)
                    .expand(len(gt_lens), min_len)
                    < gt_lens.unsqueeze(1)
                )

                mask_expanded = mask.unsqueeze(-1).float()

                # Mel losses
                loss_mse = (
                    ((p_mel - t_mel) ** 2 * mask_expanded).sum()
                    / (mask.sum() * cfg.n_mels + 1e-8)
                )

                loss_l1 = (
                    (torch.abs(p_mel - t_mel) * mask_expanded).sum()
                    / (mask.sum() * cfg.n_mels + 1e-8)
                )

                # Stop targets
                stop_targets = torch.zeros_like(p_stop)

                for i, length in enumerate(gt_lens):
                    if length < min_len:
                        stop_targets[i, length:, 0] = 1.0

                loss_stop = bce_loss(p_stop, stop_targets).mean()

                # Guided attention
                loss_guide = guided_attention_loss(
                    attentions,
                    token_lens,
                    gt_lens
                )

                # Total loss
                loss = (
                    (cfg.alpha * loss_mse)
                    + (cfg.beta * loss_l1)
                    + loss_stop
                    + (10 * loss_guide)
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    1.0
                )

                optimizer.step()
                global_step += 1

                # Logging
                if global_step % 10 == 0:
                    writer.add_scalar("Loss/Total", loss.item(), global_step)
                    writer.add_scalar("Loss/Guide", loss_guide.item(), global_step)
                    writer.add_scalar("Loss/Mel_MSE", loss_mse.item(), global_step)
                    writer.add_scalar("Loss/L1", loss_l1.item(), global_step)
                    writer.add_scalar("Loss/Stop_BCE", loss_stop.item(), global_step)

                    attn_matrix = attentions[0].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(6, 4))
                    im = ax.imshow(
                        attn_matrix,
                        aspect="auto",
                        origin="lower",
                        interpolation="none"
                    )
                    fig.colorbar(im, ax=ax)

                    plt.title(f"Attention (Step {global_step})")
                    plt.xlabel("Encoder Steps")
                    plt.ylabel("Decoder Steps")
                    plt.tight_layout()

                    writer.add_figure(
                        "Attention_Alignment",
                        fig,
                        global_step
                    )

                    print(
                        f"Epoch {epoch}/{cfg.epochs} | "
                        f"Step {global_step} | "
                        f"Total: {loss.item():.6f} | "
                        f"Mel: {loss_mse.item():.6f} | "
                        f"L1: {loss_l1.item():.6f} | "
                        f"Stop: {loss_stop.item():.6f} | "
                        f"Guide: {loss_guide.item():.6f}"
                    )

                # Save checkpoint
                if global_step % 250 == 0:
                    save_path = os.path.join(
                        checkpoint_dir,
                        f"student_step_{global_step}.pth"
                    )

                    torch.save(
                        {
                            "global_step": global_step,
                            "epoch": epoch,
                            "model_state_dict": student.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss.item(),
                        },
                        save_path
                    )

                    print(f"💾 Чекпоинт сохранен: {save_path}")

    except KeyboardInterrupt:
        print("\nОстановка обучения пользователем...")

        save_path = os.path.join(
            checkpoint_dir,
            "interrupted.pth"
        )

        torch.save(
            {
                "global_step": global_step,
                "epoch": epoch,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            },
            save_path
        )

        print("💾 Аварийный чекпоинт сохранен.")

    writer.close()
    print("Обучение завершено.")


# ==========================================
# 10. Inference & Main
# ==========================================

def inference(model, text, ref_audio_path, cfg, processor):
    model.eval()

    tokens = torch.tensor(
        [processor.encode(text)],
        dtype=torch.long
    ).to(cfg.device)

    lens = torch.tensor([tokens.size(1)]).to(cfg.device)

    signal, fs = torchaudio.load(ref_audio_path)

    if fs != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=fs,
            new_freq=16000
        )
        signal = resampler(signal)

    with torch.no_grad():
        signal = signal.to(cfg.device)
        spk_emb = (
            spk_classifier.encode_batch(signal)
            .squeeze(0)
            .squeeze(0)
        )

        mel, _, _ = model(
            tokens,
            lens,
            speaker_embs=spk_emb.unsqueeze(0)
        )

    return mel


def save_audio_vocos(
    mel_output,
    filename="output_vocos.wav",
    device="cuda"
):
    print("Инициализация Vocos...")

    vocoder = Vocos.from_pretrained(
        "charactr/vocos-mel-24khz"
    ).to(device)

    features = mel_output.transpose(1, 2)

    with torch.no_grad():
        wav = vocoder.decode(features)

    import soundfile as sf

    sf.write(
        filename,
        wav.squeeze().cpu().numpy(),
        24000
    )

    print(f"🎧 Аудио сохранено: {filename}")


if __name__ == "__main__":
    cfg = Config()
    tp = TextProcessor(cfg.RUS_ALPHABET)

    print("Starting training on Russian dataset...")

    train_with_distillation(
        "C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped/test"
    )