import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from vocos import Vocos  # <--- ВАЖНО: Используем Vocos для инференса
from speechbrain.inference.speaker import EncoderClassifier

# Инициализируем экстрактор голоса (скачается автоматически при первом запуске)
# Используем ECAPA-TDNN, он выдает вектор размерностью 192
print("Загрузка Speaker Encoder (ECAPA-TDNN)...")
spk_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
class PodcastDistillDataset(Dataset):
    def __init__(self, root_dir, text_processor, cfg):
        self.root_dir = Path(root_dir)
        self.tp = text_processor
        self.cfg = cfg
        self.samples = []

        # Для подстраховки (если нет .pt), загружаем Vocos для генерации на лету
        # Но лучше, чтобы все файлы были прекомпилированы твоим скриптом!
        self.vocos_feature_extractor = Vocos.from_pretrained("charactr/vocos-mel-24khz").to("cpu")

        print(f"Сканирование директории: {root_dir}...")
        folders = [f for f in self.root_dir.iterdir() if f.is_dir()]

        for folder in folders:
            json_files = list(folder.glob("*.json"))
            if not json_files: continue

            with open(json_files[0], 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)

            for i, entry in enumerate(metadata_list):
                audio_filename = f"{folder.name}_{i}.mp3"
                audio_path = folder / audio_filename

                # Мы ищем именно тот файл _teacher.pt, который ты создал скриптом с Vocos
                teacher_mel_path = folder / f"{folder.name}_{i}_teacher.pt"
                spk_emb_path = folder / f"{folder.name}_{i}_spk.pt"

                if len(entry["text"]) > 182: continue

                # Добавляем в список только если есть аудио
                if audio_path.exists():
                    self.samples.append({
                        "text": entry["text"],
                        "audio_path": str(audio_path),
                        "teacher_mel_path": str(teacher_mel_path),
                        "spk_emb_path": str(spk_emb_path),
                    })

        print(f"Датасет загружен: {len(self.samples)} примеров.")

    def _get_mel_from_audio(self, audio_path):
        # Резервный метод: если .pt файла нет, генерируем через Vocos на лету
        # ВАЖНО: Никакой нормализации / 80. Vocos сам знает, что делать.
        wav, sr = torchaudio.load(audio_path)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)

        # Vocos ожидает [1, Time], если стерео - усредняем
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        with torch.no_grad():
            mel = self.vocos_feature_extractor.feature_extractor(wav)  # [1, 100, Time]

        return mel.squeeze(0).transpose(0, 1)  # [Time, 100]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text_tokens = torch.tensor(self.tp.encode(sample["text"]), dtype=torch.long)

        # 1. Загрузка Mel-спектрограммы (Target)
        if os.path.exists(sample["teacher_mel_path"]):
            target_mel = torch.load(sample["teacher_mel_path"])
        else:
            target_mel = self._get_mel_from_audio(sample["audio_path"])

        # 2. Извлечение вектора голоса (Speaker Embedding)
        if os.path.exists(sample["spk_emb_path"]):
            spk_emb = torch.load(sample["spk_emb_path"])
        else:
            signal, fs = torchaudio.load(sample["audio_path"])
            # SpeechBrain требует 16000Hz! Ресемплим с 24000 (или исходного)
            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                signal = resampler(signal)

            with torch.no_grad():
                spk_emb = spk_classifier.encode_batch(signal)
                spk_emb = spk_emb.squeeze(0).squeeze(0)  # (192,)

            torch.save(spk_emb, sample["spk_emb_path"])

        return text_tokens, target_mel, sample["text"], sample["audio_path"], spk_emb

import matplotlib.pyplot as plt

# class AudioNormalizer:
#     def __init__(self):
#         self.mean = -2.1932
#         self.std = 4.7414
#
#     def normalize(self, mel):
#         return (mel - self.mean) / self.std
#
#     def denormalize(self, mel):
#         return (mel * self.std) + self.mean
#
# normalizer = AudioNormalizer()

def save_mel_image(mel, path="mel_spectrogram.png"):
    # Если это тензор PyTorch, переносим на CPU и превращаем в numpy
    if torch.is_tensor(mel):
        mel = mel.detach().cpu().numpy()

    # Если пришел батч [1, n_mels, Time], убираем лишнюю размерность
    if len(mel.shape) == 3:
        mel = mel[0]

    plt.figure(figsize=(10, 4))
    # Важно:imshow ожидает (высота, ширина), т.е. (n_mels, Time)
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Generated Mel-Spectrogram")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"🖼 Спектрограмма сохранена как {path}")
# Вызывай это после денормализации:
# save_mel_image(mel, "debug_mel.png")

class Config:
    # Алфавит (оставляем как есть)
    RUS_ALPHABET = " абвгдеёжзийклмнопрстуфхцчшщъыьэюя.,!?-–"
    vocab_size = len(RUS_ALPHABET) + 1

    speaker_embedding_dim = 192

    embedding_dim = 256
    encoder_hidden = 256
    decoder_hidden = 256
    attention_dim = 256

    # --- ИЗМЕНЕНИЯ ПОД VOCOS ---
    n_mels = 100  # Vocos использует 100 полос
    sample_rate = 24000  # Vocos работает на 24кГц
    hop_length = 256  # Стандарт для Vocos 24khz
    # ---------------------------

    alpha = 1  # Вес MSE
    beta = 0.1  # Вес L1

    lr = 1e-4
    batch_size = 24
    epochs = 50
    device = torch.device("cuda")


# ==========================================
# 2. Text Preprocessing Utility
# ==========================================
class TextProcessor:
    def __init__(self, alphabet):
        self.char_to_id = {char: i + 1 for i, char in enumerate(alphabet)}
        self.id_to_char = {i + 1: char for i, char in enumerate(alphabet)}
        self.pad_id = 0

    def encode(self, text):
        text = text.lower()
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids):
        return "".join([self.id_to_char[i] for i in ids if i in self.id_to_char])
class LocationSensitiveAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim, attention_location_n_filters=32,
                 attention_location_kernel_size=31):
        super().__init__()
        self.W1 = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W2 = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.V = nn.Linear(attention_dim, 1, bias=False)

        # Свертка смотрит на то, куда модель смотрела на прошлом шаге
        padding = attention_location_kernel_size // 2
        self.location_conv = nn.Conv1d(
            in_channels=1,
            out_channels=attention_location_n_filters,
            kernel_size=attention_location_kernel_size,
            padding=padding,
            bias=False
        )
        self.location_dense = nn.Linear(attention_location_n_filters, attention_dim, bias=False)

    def forward(self, query, keys, prev_weights, mask=None):
        # query: (B, 1, dec_dim), keys: (B, T, enc_dim), prev_weights: (B, T)
        proj_key = self.W1(keys)
        proj_query = self.W2(query)  # Транслируется автоматически благодаря бродкастингу

        # Извлекаем признаки локации из предыдущих весов внимания
        loc_feat = self.location_conv(prev_weights.unsqueeze(1))  # (B, filters, T)
        loc_feat = loc_feat.transpose(1, 2)  # (B, T, filters)
        proj_loc = self.location_dense(loc_feat)  # (B, T, attention_dim)

        # Складываем контент + запрос + позицию
        scores = self.V(torch.tanh(proj_key + proj_query + proj_loc)).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), keys)
        return context, weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
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
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, _ = self.lstm(packed_x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs  # (B, T, hidden_dim * 2)

def guided_attention_loss(attentions, text_lens, mel_lens, g=0.2):

    B, T_dec, T_enc = attentions.size()
    device = attentions.device
    loss = 0.0

    for i in range(B):
        N = text_lens[i].item()  # Реальная длина текста (без паддинга)
        M = mel_lens[i].item()  # Реальная длина аудио (без паддинга)

        if N == 0 or M == 0: continue

        # Создаем сетку координат
        grid_n, grid_m = torch.meshgrid(
            torch.arange(N, device=device),
            torch.arange(M, device=device),
            indexing='ij'
        )

        # Матрица штрафов: 0 на диагонали, близко к 1 по краям
        W = 1.0 - torch.exp(-((grid_n.float() / N - grid_m.float() / M) ** 2) / (2 * g ** 2))
        W = W.T  # Транспонируем, чтобы получить форму (M, N)

        # Умножаем предсказанное внимание на матрицу штрафов
        # Берем только реальные длины, игнорируя паддинги
        attn_slice = attentions[i, :M, :N]
        loss += torch.mean(attn_slice * W)

    return loss / B

class PreNet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 256]):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, sizes[0])
        self.layer2 = nn.Linear(sizes[0], sizes[1])

    def forward(self, x):
        # ВАЖНО: training=True стоит ЖЕСТКО. Dropout должен работать всегда!
        x = F.dropout(F.relu(self.layer1(x)), p=0.5, training=True)
        x = F.dropout(F.relu(self.layer2(x)), p=0.5, training=True)
        return x


class PostNet(nn.Module):
    def __init__(self, n_mels=100, postnet_embedding_dim=512, kernel_size=5, dropout=0.1):
        super().__init__()

        self.convolutions = nn.ModuleList()

        # Первый слой (in: n_mels, out: 512)
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(n_mels, postnet_embedding_dim, kernel_size, stride=1, padding=int((kernel_size - 1) / 2)),
                nn.BatchNorm1d(postnet_embedding_dim)
            )
        )

        # Средние 3 слоя (in: 512, out: 512)
        for _ in range(3):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(postnet_embedding_dim, postnet_embedding_dim, kernel_size, stride=1,
                              padding=int((kernel_size - 1) / 2)),
                    nn.BatchNorm1d(postnet_embedding_dim)
                )
            )

        # Последний слой (in: 512, out: n_mels) - БЕЗ активации в конце
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(postnet_embedding_dim, n_mels, kernel_size, stride=1, padding=int((kernel_size - 1) / 2)),
                nn.BatchNorm1d(n_mels)
            )
        )

        self.dropout = dropout

    def forward(self, x):

        x = x.transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), p=self.dropout, training=self.training)

        # Последний слой без Tanh
        x = F.dropout(self.convolutions[-1](x), p=self.dropout, training=self.training)

        # Возвращаем к размерности [Batch, Time, Mels]
        x = x.transpose(1, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, n_mels, decoder_hidden, encoder_total_dim, attention_dim, speaker_dim):
        super().__init__()
        self.n_mels = n_mels
        self.decoder_hidden = decoder_hidden

        # НОВОЕ: Инициализируем Pre-Net
        self.prenet = PreNet(n_mels, [256, 256])

        # Обновляем размер входа: выход Pre-Net (256) + контекст + спикер
        self.lstm_input_size = 256 + encoder_total_dim + speaker_dim
        self.lstm = nn.LSTMCell(self.lstm_input_size, decoder_hidden)

        self.attention = LocationSensitiveAttention(encoder_total_dim, decoder_hidden, attention_dim)

        self.linear_input_size = decoder_hidden + encoder_total_dim + speaker_dim
        self.linear = nn.Linear(self.linear_input_size, n_mels)
        self.stop_linear = nn.Linear(self.linear_input_size, 1)

    def forward(self, encoder_outputs, encoder_mask, spk_emb, teacher_mels=None, teacher_forcing_ratio=1.0, max_len=1500, stop_threshold=0.5, min_stop_frames=30):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        prev_weights = torch.zeros(batch_size, encoder_outputs.size(1)).to(device)
        h = torch.zeros(batch_size, self.decoder_hidden).to(device)
        c = torch.zeros(batch_size, self.decoder_hidden).to(device)
        mel_input = torch.zeros(batch_size, self.n_mels).to(device)

        outputs = []
        stop_tokens = []
        attentions = []

        steps = teacher_mels.size(1) if teacher_mels is not None else max_len

        for t in range(steps):
            # 1. Attention
            context, attn_weights = self.attention(h.unsqueeze(1), encoder_outputs, prev_weights, encoder_mask)
            context = context.squeeze(1)

            attentions.append(attn_weights.squeeze(1))

            prev_weights = attn_weights.squeeze(1)  # Запоминаем для следующего шага

            # НОВОЕ: Пропускаем mel_input через Pre-Net
            prenet_out = self.prenet(mel_input)

            # 2. Шаг LSTM (используем prenet_out вместо сырого mel_input)
            rnn_input = torch.cat([prenet_out, context, spk_emb], dim=-1)
            h, c = self.lstm(rnn_input, (h, c))

            # 3. Формируем выход
            concat_out = torch.cat([h, context, spk_emb], dim=-1)

            mel_out = self.linear(concat_out)
            stop_out = self.stop_linear(concat_out)

            outputs.append(mel_out)
            stop_tokens.append(stop_out)

            # Teacher forcing c вероятностью (Scheduled Sampling)
            if teacher_mels is None:
                stop_prob = torch.sigmoid(stop_out)[0].item()
                if t > min_stop_frames and stop_prob > stop_threshold:
                    print(f"🛑 Модель остановилась на шаге {t} (prob: {stop_prob:.3f})")
                    break

            if teacher_mels is not None:
                # Если мы не в конце последовательности
                if t < teacher_mels.size(1) - 1:
                    # Решаем, что подать на вход на СЛЕДУЮЩЕМ шаге
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        # Берем ПРАВИЛЬНЫЙ кадр (Ground Truth)
                        mel_input = teacher_mels[:, t, :]
                    else:
                        # Берем ТО, ЧТО ТОЛЬКО ЧТО ПРЕДСКАЗАЛИ (detach обязательно!)
                        mel_input = mel_out.detach()
                else:
                    mel_input = mel_out
            else:
                # Инференс (всегда берем свой выход)
                mel_input = mel_out

        return torch.stack(outputs, dim=1), torch.stack(stop_tokens, dim=1), torch.stack(attentions, dim=1)

class StudentTTS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg.vocab_size, cfg.embedding_dim, cfg.encoder_hidden)
        self.decoder = Decoder(
            cfg.n_mels,
            cfg.decoder_hidden,
            cfg.encoder_hidden * 2,
            cfg.attention_dim,
            # УДАЛИЛИ: num_speakers
            speaker_dim=cfg.speaker_embedding_dim
        )
        self.postnet = PostNet(n_mels=cfg.n_mels)

    def forward(self, text, text_lengths, speaker_embs, mels=None, teacher_forcing_ratio=1.0, stop_threshold=0.5, min_stop_frames=30):
        device = text.device
        mask = torch.arange(text.size(1), device=device).expand(len(text_lengths),
                                                                text.size(1)) < text_lengths.unsqueeze(1)

        encoder_outputs = self.encoder(text, text_lengths)
        mel_outputs, stop_outputs, attentions = self.decoder(
            encoder_outputs, mask, speaker_embs,
            teacher_mels=mels,
            teacher_forcing_ratio=teacher_forcing_ratio,
            stop_threshold=stop_threshold,  # <--- Передали в декодер
            min_stop_frames=min_stop_frames  # <--- Передали в декодер
        )
        # --- НОВОЕ: Пропускаем через Post-Net и прибавляем к сырому выходу ---
        mel_outputs_post = mel_outputs + self.postnet(mel_outputs)

        # Возвращаем 4 значения (сырой мел и улучшенный мел)
        return mel_outputs, mel_outputs_post, stop_outputs, attentions

class RussianTTSDataset(Dataset):
    def __init__(self, texts, gt_mels, teacher_mels, processor):
        self.texts = texts
        self.gt_mels = gt_mels
        self.teacher_mels = teacher_mels
        self.processor = processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.processor.encode(self.texts[idx]), dtype=torch.long)
        return tokens, self.gt_mels[idx], self.teacher_mels[idx]

def save_checkpoint(model, optimizer, epoch, global_step, loss, path):
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"--- Чекпоинт сохранен: {path} ---")

def collate_fn_podcast(batch):
    # Сортировка для падинга (по убыванию длины текста)
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    tokens, mels, raw_texts, audio_paths, spk_embs = zip(*batch)

    token_lens = torch.tensor([len(x) for x in tokens])
    tokens_padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True)

    mel_lens = torch.tensor([x.size(0) for x in mels])
    mels_padded = nn.utils.rnn.pad_sequence(mels, batch_first=True) # Паддинг нулями - ок для Vocos

    spk_embs_tensor = torch.stack(spk_embs)

    return tokens_padded, token_lens, mels_padded, mel_lens, raw_texts, audio_paths, spk_embs_tensor


import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def train_with_distillation(root_dir):
    # --- Вспомогательные функции Loss ---
    def masked_mse(preds, targets, mask):
        diff = (preds - targets) ** 2
        return (diff * mask).sum() / (mask.sum() * cfg.n_mels + 1e-8)

    def masked_l1(preds, targets, mask):
        diff = torch.abs(preds - targets)
        return (diff * mask).sum() / (mask.sum() * cfg.n_mels + 1e-8)

    # Конфигурация
    cfg = Config()
    tp = TextProcessor(cfg.RUS_ALPHABET)

    dataset = PodcastDistillDataset(root_dir, tp, cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collate_fn_podcast, shuffle=True)

    student = StudentTTS(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=1e-6)
    # 2. Определяем функцию шедулера (Noam-style, но проще)
    def get_scheduler(optimizer, warmup_steps, total_steps):
        def lr_lambda(current_step):
            # Если шаг меньше warmup - растем линейно
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # После warmup - убываем (Cosine Annealing)
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    import math

    # Настройка:
    warmup_steps = 1500  # Примерно 1-2 эпохи
    total_steps = cfg.epochs * len(dataloader)  # Общее число шагов

    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

    writer = SummaryWriter(log_dir="runs/fast_distill_v2")

    # bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).to(cfg.device), reduction='none')

    global_step = 0
    start_epoch = 0
    # --- Загрузка чекпоинта ---
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and 'step' in f]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        last_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"Загрузка чекпоинта: {last_checkpoint}")
        try:
            ckpt = torch.load(last_checkpoint)

            student.load_state_dict(ckpt['model_state_dict'], strict=False)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            # --- НОВОЕ: Принудительное обновление LR УДАЛЕНО ---
            # Теперь мы доверяем шедулеру и загруженному состоянию оптимизатора.
            current_lr = optimizer.param_groups[0]['lr']
            print(f"📉 Восстановлен Learning Rate: {current_lr:.6e}")

            global_step = ckpt.get('global_step', 0)
            start_epoch = ckpt.get('epoch', 0)

            print(f"✅ Успех! Модель подхватила старые веса. Начинаем с шага {global_step}")
        except Exception as e:
            print(f"Ошибка загрузки: {e}")

    print("Начало обучения с Stop Token...")
    student.train()

    try:
        tf_start = 1.0  # Начинаем с полной помощи
        tf_end = 0.0  # В конце полностью самостоятельная (можно оставить 0.1 для стабильности)
        tf_decay_steps = 50000  # За сколько шагов спуститься до минимума
        for epoch in range(start_epoch, cfg.epochs):

            # --- НОВОЕ: Переменные для подсчета среднего лосса за эпоху ---
            epoch_loss_sum = 0.0
            epoch_loss_mse_raw_sum = 0.0
            epoch_loss_mse_post_sum = 0.0
            epoch_loss_l1_sum = 0.0
            epoch_loss_guide_sum = 0.0

            num_batches = 0

            for batch in dataloader:
                tokens, token_lens, gts, gt_lens, raw_texts, audio_paths, speaker_ids = batch

                tokens = tokens.to(cfg.device)
                token_lens = token_lens.to(cfg.device)
                gts = gts.to(cfg.device)
                gt_lens = gt_lens.to(cfg.device)
                speaker_ids = speaker_ids.to(cfg.device)

                target_mels = gts

                current_tf_ratio = max(tf_end, tf_start - (tf_start - tf_end) * (global_step / tf_decay_steps))

                optimizer.zero_grad()

                # Передаем в модель
                pred_mels_raw, pred_mels_post, pred_stops, attentions = student(
                    tokens, token_lens,
                    speaker_embs=speaker_ids,
                    mels=target_mels,
                    teacher_forcing_ratio=current_tf_ratio  # <--- НОВЫЙ АРГУМЕНТ
                )

                min_len = min(pred_mels_raw.size(1), target_mels.size(1))

                p_mel_raw = pred_mels_raw[:, :min_len, :]
                p_mel_post = pred_mels_post[:, :min_len, :]
                t_mel = target_mels[:, :min_len, :]
                p_stop = pred_stops[:, :min_len, :]

                mask = torch.arange(min_len, device=cfg.device).expand(len(gt_lens), min_len) < gt_lens.unsqueeze(1)
                mask_expanded = mask.unsqueeze(-1).float()

                loss_mse_raw = ((p_mel_raw - t_mel) ** 2 * mask_expanded).sum() / (mask.sum() * cfg.n_mels + 1e-8)
                loss_mse_post = ((p_mel_post - t_mel) ** 2 * mask_expanded).sum() / (mask.sum() * cfg.n_mels + 1e-8)
                loss_l1 = (torch.abs(p_mel_post - t_mel) * mask_expanded).sum() / (mask.sum() * cfg.n_mels + 1e-8)

                stop_targets = torch.zeros_like(p_stop)
                for i, length in enumerate(gt_lens):
                    # length.item() даст число. Берем max(0, length - 1), чтобы избежать индекса -1
                    last_valid_frame = max(0, length.item() - 1)
                    # Ставим 1 на последний кадр звука и на весь последующий паддинг
                    stop_targets[i, last_valid_frame:, 0] = 1.0

                # Рассчитываем динамический вес позитивного класса для текущего батча
                num_positives = stop_targets.sum()
                num_negatives = stop_targets.numel() - num_positives

                if num_positives > 0:
                    # Ограничим максимальный вес, чтобы градиенты не взорвались
                    dynamic_weight = torch.clamp(num_negatives / num_positives, max=100.0)
                else:
                    dynamic_weight = torch.tensor(1.0).to(cfg.device)

                # Логируем в Tensorboard, чтобы видеть прогресс
                if global_step % 100 == 0:
                    writer.add_scalar('Training/Teacher_Forcing_Ratio', current_tf_ratio, global_step)

                # Обновляем функцию потерь с новым весом (придется создавать ее заново на каждом шаге,
                # либо использовать F.binary_cross_entropy_with_logits напрямую)
                loss_stop = F.binary_cross_entropy_with_logits(
                    p_stop,
                    stop_targets,
                    pos_weight=dynamic_weight,
                    reduction='mean'
                )

                loss_guide = guided_attention_loss(attentions, token_lens, gt_lens)



                if global_step < 10000:
                    w_guide = 100
                    mse_coeff = cfg.alpha
                    l1_coeff = cfg.beta
                elif global_step < 21000:
                    w_guide = 50
                    mse_coeff = cfg.alpha
                    l1_coeff = 0.5
                elif global_step < 22000:
                    w_guide = 25
                elif global_step < 23000:
                    w_guide = 12
                elif global_step < 24000:
                    w_guide = 6
                elif global_step < 25000:
                    w_guide = 3
                elif global_step < 27500:
                    w_guide = 1


                loss = (mse_coeff * loss_mse_raw) + (mse_coeff * loss_mse_post) + (l1_coeff * loss_l1) + loss_stop + (
                            w_guide * loss_guide)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()  # <--- Сдвигаем LR каждый шаг

                # --- НОВОЕ: Накапливаем лосс ---
                epoch_loss_sum += loss.item()
                epoch_loss_mse_raw_sum += loss_mse_raw.item()
                epoch_loss_mse_post_sum += loss_mse_post.item()
                epoch_loss_l1_sum += loss_l1.item()
                epoch_loss_guide_sum += loss_guide.item()

                num_batches += 1
                global_step += 1

                if global_step % 10 == 0:
                    writer.add_scalar('Loss/Total', loss.item(), global_step)
                    writer.add_scalar('Loss/Guide', loss_guide.item(), global_step)
                    writer.add_scalar('Loss/Mel_MSE_RAW', loss_mse_raw.item(), global_step)
                    writer.add_scalar('Loss/Mel_MSE_POST', loss_mse_post.item(), global_step)
                    writer.add_scalar('Loss/L1', loss_l1.item(), global_step)
                    writer.add_scalar('Loss/Stop_BCE', loss_stop.item(), global_step)
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('Training/Learning_Rate', current_lr, global_step)

                    attn_matrix = attentions[0].detach().cpu().numpy()

                    fig, ax = plt.subplots(figsize=(6, 4))
                    im = ax.imshow(attn_matrix, aspect='auto', origin='lower', interpolation='none')
                    fig.colorbar(im, ax=ax)
                    plt.title(f"Attention (Step {global_step})")
                    plt.xlabel("Encoder Steps (Text)")
                    plt.ylabel("Decoder Steps (Audio)")
                    plt.tight_layout()

                    writer.add_figure('Attention_Alignment', fig, global_step)
                    plt.close(fig)
                    print(
                        f"Epoch {epoch}/{cfg.epochs} | Step {global_step} | Total: {loss.item():.6f} | MSE_raw(multiplied by mse_coeff {mse_coeff}): {loss_mse_raw.item()*mse_coeff:.6f} | MSE_post(multiplied by mse_coeff {mse_coeff}): {loss_mse_post.item()*mse_coeff:.6f} | L1(multiplied by l1_coeff {l1_coeff}): {loss_l1.item()*l1_coeff:.6f}| Stop: {loss_stop.item():.6f} | Guide: {loss_guide.item():.6f} | LR: {current_lr:.6e} | Current_guide_weight: {w_guide:.6f}")

                    # Сохранение чекпоинта
                    if global_step % 250 == 0:
                        save_path = os.path.join(checkpoint_dir, f"student_step_{global_step}.pth")

                        # Получаем текущий LR для вывода и сохранения
                        current_lr = optimizer.param_groups[0]['lr']

                        torch.save({
                            'global_step': global_step,
                            'epoch': epoch,
                            'model_state_dict': student.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': loss.item(),
                            'lr': current_lr  # Сохраним и в словаре для истории
                        }, save_path)

                        print(
                            f"💾 Шаг {global_step}: Чекпоинт сохранен. Текущий LR: {current_lr:.6e} | Путь: {save_path}")

            # --- НОВОЕ: Шаг шедулера в конце эпохи ---
            # Вычисляем средний лосс за всю эпоху
            avg_epoch_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0
            avg_mse_raw_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0
            avg_mse_post_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0
            avg_l1_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0
            avg_guide_loss = epoch_loss_sum / num_batches if num_batches > 0 else 0

            print(
                f"🔻 Эпоха {epoch} завершена. Средние лоссы за эпоху: Total_avg {avg_epoch_loss:.6f} | avg_mse_raw_loss {avg_mse_raw_loss:.6f} | avg_mse_post_loss {avg_mse_post_loss:.6f} | avg_l1_loss {avg_l1_loss:.6f} | avg_guide_los {avg_guide_loss:.6f} Текущий LR: {optimizer.param_groups[0]['lr']:.6e}")

    except KeyboardInterrupt:
        print("\nОстановка обучения пользователем...")
        save_path = os.path.join(checkpoint_dir, "interrupted.pth")
        torch.save({
            'global_step': global_step,
            'epoch': epoch,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss.item(),
        }, save_path)
        print("💾 Аварийный чекпоинт сохранен.")

    writer.close()
    try:
        plt.close(fig)
    except NameError:
        pass
    print("Обучение завершено.")

import torchaudio

def save_audio_vocos(mel_output, filename="output_vocos.wav", device="cuda"):
    """
    Превращает Mel-спектрограмму в звук с помощью Vocos.
    """
    # Загружаем Vocos (если еще не загружен где-то глобально)
    print("Инициализация Vocos для рендеринга...")
    vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)

    # mel_output приходит из модели в формате [1, Time, 100]
    # Vocos ждет [1, 100, Time], поэтому транспонируем
    features = mel_output.transpose(1, 2)

    with torch.no_grad():
        wav = vocoder.decode(features)

    # Сохраняем
    import soundfile as sf
    sf.write(filename, wav.squeeze().cpu().numpy(), 24000)
    print(f"🎧 Аудио сохранено: {filename}")

if __name__ == "__main__":
    cfg = Config()
    tp = TextProcessor(cfg.RUS_ALPHABET)

    print("Starting training on Russian dataset...")
    train_with_distillation("C:/Users/light/Downloads/podcasts_1_stripped_archive/podcasts_1_stripped/test")
