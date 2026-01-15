"""CharLM: упрощённое трёхстадийное обучение для OCR-коррекции.

Stage A: Lexicon MLM (простой span-masked MLM на 200-300k словах)
Stage B: Context MLM (окна 1-3 слова, OCR-шум только в контексте)
Stage C: Contrastive Learning (InfoNCE на OCR-парах)
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

from .config import DEFAULT_CONFIG
from .model import CharTransformerMLM
from .utils import (
    Logger,
    build_vocab,
    encode_str,
    choose_spans,
    masked_accuracy,
    load_charset,
    tokenize_by_charset,
    add_ocr_noise,
    filter_words,
    is_valid_word,
    clean_word,
    load_allowed_chars,
)


# ============================================================
# DATASETS
# ============================================================


class LexiconMLMDataset(Dataset):
    """Stage A: простой span-masked MLM на словах из лексикона."""

    def __init__(self, words: list[str], c2i: dict, cfg: dict):
        self.words = words
        self.c2i = c2i
        self.cfg = cfg
        self.mask_id = c2i["<MASK>"]

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        w = self.words[idx]
        ids = encode_str(w, self.c2i, self.cfg["max_len"])
        L = min(len(w), self.cfg["max_len"])

        # Span masking
        mask_pos = choose_spans(
            L,
            self.cfg["span_min"],
            self.cfg["span_max"],
            self.cfg["num_spans_min"],
            self.cfg["num_spans_max"],
        )

        x = ids.copy()
        y = [-100] * self.cfg["max_len"]

        for p in mask_pos:
            y[p] = ids[p]
            if random.random() < self.cfg["mask_prob"]:
                x[p] = self.mask_id

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class ContextMLMDataset(IterableDataset):
    """Stage B: контекстное MLM с окнами 1-3 слова, OCR-шум только в контексте."""

    def __init__(self, tokens: list[str], c2i: dict, cfg: dict):
        super().__init__()
        self.tokens = tokens
        self.c2i = c2i
        self.cfg = cfg
        self.N = len(tokens)
        self.mask_id = c2i["<MASK>"]

    def _sample_window(self):
        """Выбрать окно 1-3 слова."""
        r = random.random()
        p1 = self.cfg.get("p_win_1", 0.3)
        p2 = self.cfg.get("p_win_2", 0.3)

        if r < p1:  # 1 слово
            idx = random.randrange(self.N)
            return [self.tokens[idx]], 0
        elif r < p1 + p2:  # 2 слова
            idx = random.randrange(max(1, self.N - 1))
            return [self.tokens[idx], self.tokens[idx + 1]], random.randint(0, 1)
        else:  # 3 слова
            idx = random.randrange(max(1, self.N - 2))
            return [self.tokens[idx], self.tokens[idx + 1], self.tokens[idx + 2]], 1

    def __iter__(self):
        while True:
            words, center_idx = self._sample_window()
            center = words[center_idx]

            if len(center) < self.cfg["min_word_len"]:
                continue

            # OCR-шум ТОЛЬКО в контексте (не в центре)
            noisy_words = []
            for i, w in enumerate(words):
                if i != center_idx:
                    noisy_words.append(add_ocr_noise(w, self.cfg))
                else:
                    noisy_words.append(w)

            clean_seq = " ".join(words)
            noisy_seq = " ".join(noisy_words)

            # Позиции центрального слова
            prefix_len = sum(len(words[j]) + 1 for j in range(center_idx))
            center_start = min(prefix_len, self.cfg["max_len"])
            center_end = min(prefix_len + len(center), self.cfg["max_len"])

            if center_end - center_start < self.cfg["min_word_len"]:
                continue

            clean_ids = encode_str(clean_seq, self.c2i, self.cfg["max_len"])
            noisy_ids = encode_str(noisy_seq, self.c2i, self.cfg["max_len"])

            x = noisy_ids.copy()
            y = [-100] * self.cfg["max_len"]

            # Маскируем позиции в центральном слове
            center_len = center_end - center_start
            rel_positions = choose_spans(
                center_len,
                self.cfg["span_min"],
                self.cfg["span_max"],
                self.cfg["num_spans_min"],
                self.cfg["num_spans_max"],
            )

            for rp in rel_positions:
                p = center_start + rp
                if 0 <= p < self.cfg["max_len"]:
                    y[p] = clean_ids[p]
                    if random.random() < self.cfg["mask_prob"]:
                        x[p] = self.mask_id

            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class ContrastiveDataset(Dataset):
    """Stage C: Contrastive Learning на OCR-парах (InfoNCE).

    Anchor = incorrect, Positive = correct, Negatives = in-batch + random words.
    """

    def __init__(
        self, pairs_df: pd.DataFrame, c2i: dict, cfg: dict, all_words: list[str]
    ):
        self.incorrect = pairs_df["incorrect"].tolist()
        self.correct = pairs_df["correct"].tolist()
        self.c2i = c2i
        self.cfg = cfg
        self.all_words = all_words

    def __len__(self):
        return len(self.incorrect)

    def __getitem__(self, idx):
        inc = self.incorrect[idx]
        cor = self.correct[idx]

        inc_ids = encode_str(inc, self.c2i, self.cfg["max_len"])
        cor_ids = encode_str(cor, self.c2i, self.cfg["max_len"])

        # Случайные негативы
        n_neg = self.cfg.get("n_random_negatives", 3)
        neg_ids_list = []
        for _ in range(n_neg):
            neg_word = random.choice(self.all_words)
            neg_ids_list.append(encode_str(neg_word, self.c2i, self.cfg["max_len"]))

        return {
            "anchor": torch.tensor(inc_ids, dtype=torch.long),
            "positive": torch.tensor(cor_ids, dtype=torch.long),
            "negatives": torch.tensor(neg_ids_list, dtype=torch.long),
        }


# ============================================================
# TRAINING FUNCTIONS
# ============================================================


def train_epoch_mlm(
    model,
    loader,
    optimizer,
    device,
    vocab_size,
    steps_limit=None,
    scheduler=None,
    grad_clip=None,
):
    """Эпоха MLM обучения (Stage A, B)."""
    model.train()
    total_loss = total_acc = 0.0
    n_steps = 0

    iterator = iter(loader)
    max_steps = steps_limit or len(loader)
    pbar = tqdm(range(max_steps), leave=False)

    for _ in pbar:
        try:
            x, y = next(iterator)
        except StopIteration:
            break

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), y.view(-1), ignore_index=-100
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler:
            scheduler.step()

        acc = masked_accuracy(logits, y)
        total_loss += loss.item()
        total_acc += acc
        n_steps += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc*100:.1f}%")

    return total_loss / max(1, n_steps), total_acc / max(1, n_steps)


def get_embeddings(model, x):
    """Получить эмбеддинги (mean pooling по не-pad позициям)."""
    B, T = x.shape
    pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    h = model.emb(x) + model.pos(pos_ids)
    h = model.encoder(h, src_key_padding_mask=(x == model.pad_idx))

    mask = (x != model.pad_idx).unsqueeze(-1).float()
    h = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return h


def info_nce_loss(anchor_emb, positive_emb, negative_embs, temperature=0.07):
    """InfoNCE loss: anchor → positive, with in-batch + random negatives."""
    anchor_emb = F.normalize(anchor_emb, dim=-1)
    positive_emb = F.normalize(positive_emb, dim=-1)
    negative_embs = F.normalize(negative_embs, dim=-1)

    # Positive similarity
    pos_sim = (anchor_emb * positive_emb).sum(dim=-1, keepdim=True) / temperature

    # Random negatives similarity
    neg_sim = (
        torch.bmm(negative_embs, anchor_emb.unsqueeze(-1)).squeeze(-1) / temperature
    )

    # In-batch negatives (all positives except own)
    in_batch_sim = torch.mm(anchor_emb, positive_emb.t()) / temperature
    B = anchor_emb.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=anchor_emb.device)
    in_batch_neg = in_batch_sim[mask].view(B, B - 1)

    # Combine: [pos, random_neg, in_batch_neg]
    logits = torch.cat([pos_sim, neg_sim, in_batch_neg], dim=-1)
    labels = torch.zeros(B, dtype=torch.long, device=anchor_emb.device)

    return F.cross_entropy(logits, labels)


def train_epoch_contrastive(
    model, loader, optimizer, device, scheduler=None, grad_clip=None, temperature=0.07
):
    """Эпоха Contrastive обучения (Stage C)."""
    model.train()
    total_loss = 0.0
    n_steps = 0

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        anchor = batch["anchor"].to(device)
        positive = batch["positive"].to(device)
        negatives = batch["negatives"].to(device)

        B, N, T = negatives.shape

        anchor_emb = get_embeddings(model, anchor)
        positive_emb = get_embeddings(model, positive)

        neg_flat = negatives.view(B * N, T)
        neg_emb = get_embeddings(model, neg_flat).view(B, N, -1)

        loss = info_nce_loss(anchor_emb, positive_emb, neg_emb, temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        n_steps += 1
        pbar.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / max(1, n_steps)


# ============================================================
# MAIN TRAINING
# ============================================================


def train(config: dict = None):
    """Основная функция обучения CharLM (3 стадии)."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Setup
    exp_dir = cfg["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg["device"] != "auto":
        device = cfg["device"]

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # Paths
    model_a_path = os.path.join(exp_dir, "model_a.pt")
    model_b_path = os.path.join(exp_dir, "model_b.pt")
    model_path = os.path.join(exp_dir, "model.pt")
    vocab_path = os.path.join(exp_dir, "vocab.json")
    config_path = os.path.join(exp_dir, "config.json")
    model_path = os.path.join(exp_dir, "model.pt")
    vocab_path = os.path.join(exp_dir, "vocab.json")
    config_path = os.path.join(exp_dir, "config.json")

    logger = Logger(os.path.join(exp_dir, "train.log"))
    logger.log(f"Device: {device}")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # ============================================================
    # LOAD DATA
    # ============================================================

    # Загружаем разрешённые символы из charset (только буквы)
    charset_path = cfg.get("charset_path", "charset.txt")
    allowed_chars = load_allowed_chars(charset_path)
    logger.log(f"Allowed chars (letters from charset): {len(allowed_chars)}")

    logger.log("Loading lexicon...")
    raw_words = []
    with open(cfg["lexicon_path"], encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                raw_words.append(w)
    logger.log(f"Raw words: {len(raw_words):,}")

    # Фильтрация: только буквы из charset
    all_words = filter_words(
        raw_words, min_len=cfg["min_word_len"], allowed_chars=allowed_chars
    )
    logger.log(f"After filtering (letters only): {len(all_words):,}")

    # Stage A: sample 200-300k слов
    max_words_a = cfg.get("max_words_a", 250000)
    if len(all_words) > max_words_a:
        words_a = random.sample(all_words, max_words_a)
    else:
        words_a = all_words
    logger.log(f"Stage A words: {len(words_a):,}")

    # Vocab
    c2i, i2c, chars = build_vocab(all_words, include_space=True)
    vocab_size = len(chars)
    logger.log(f"Vocab size: {vocab_size}")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)

    # Context tokens (Stage B) - фильтруем
    charset = load_charset(charset_path) if charset_path else None
    tokens = []
    if cfg.get("text_path") and os.path.exists(cfg["text_path"]):
        with open(cfg["text_path"], encoding="utf-8") as f:
            text = f.read()
        raw_tokens = tokenize_by_charset(text, charset)
        tokens = [
            clean_word(t, allowed_chars)
            for t in raw_tokens
            if is_valid_word(t, allowed_chars) and len(t) >= cfg["min_word_len"]
        ]
        logger.log(f"Context tokens (filtered): {len(tokens):,}")

    # ============================================================
    # MODEL
    # ============================================================

    model = CharTransformerMLM(
        vocab_size=vocab_size,
        emb_size=cfg["emb_size"],
        max_len=cfg["max_len"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        ffn_size=cfg["ffn_size"],
        dropout=cfg["dropout"],
        pad_idx=c2i["<PAD>"],
    ).to(device)

    logger.log(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    grad_clip = cfg.get("grad_clip", 1.0)

    # ============================================================
    # STAGE A: Lexicon MLM (5-8 epochs)
    # ============================================================

    logger.log("\n=== STAGE A: Lexicon MLM ===")

    loader_a = DataLoader(
        LexiconMLMDataset(words_a, c2i, cfg),
        batch_size=cfg["batch_a"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    opt_a = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr_a"], weight_decay=cfg.get("weight_decay", 0.01)
    )

    total_steps_a = cfg["epochs_a"] * len(loader_a)
    scheduler_a = torch.optim.lr_scheduler.OneCycleLR(
        opt_a,
        max_lr=cfg["lr_a"],
        total_steps=total_steps_a,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    best_loss_a = float("inf")
    for ep in range(1, cfg["epochs_a"] + 1):
        loss, acc = train_epoch_mlm(
            model,
            loader_a,
            opt_a,
            device,
            vocab_size,
            scheduler=scheduler_a,
            grad_clip=grad_clip,
        )
        logger.log(f"[A] Epoch {ep:02d} | loss={loss:.4f} | acc={acc*100:.1f}%")
        if loss < best_loss_a:
            best_loss_a = loss
            torch.save(model.state_dict(), model_a_path)

    model.load_state_dict(
        torch.load(model_a_path, map_location=device, weights_only=True)
    )
    logger.log(f"Stage A done. Best loss: {best_loss_a:.4f}")

    # ============================================================
    # STAGE B: Context MLM
    # ============================================================

    if tokens:
        logger.log("\n=== STAGE B: Context MLM ===")

        loader_b = DataLoader(
            ContextMLMDataset(tokens, c2i, cfg),
            batch_size=cfg["batch_b"],
            num_workers=0,
        )

        opt_b = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr_b"],
            weight_decay=cfg.get("weight_decay", 0.01),
        )

        steps_per_epoch = cfg.get("steps_per_epoch_b", 10000)
        total_steps_b = cfg["epochs_b"] * steps_per_epoch
        scheduler_b = torch.optim.lr_scheduler.OneCycleLR(
            opt_b,
            max_lr=cfg["lr_b"],
            total_steps=total_steps_b,
            pct_start=0.05,
            anneal_strategy="cos",
        )

        best_loss_b = float("inf")
        for ep in range(1, cfg["epochs_b"] + 1):
            loss, acc = train_epoch_mlm(
                model,
                loader_b,
                opt_b,
                device,
                vocab_size,
                steps_limit=steps_per_epoch,
                scheduler=scheduler_b,
                grad_clip=grad_clip,
            )
            logger.log(f"[B] Epoch {ep:02d} | loss={loss:.4f} | acc={acc*100:.1f}%")
            if loss < best_loss_b:
                best_loss_b = loss
                torch.save(model.state_dict(), model_b_path)

        model.load_state_dict(
            torch.load(model_b_path, map_location=device, weights_only=True)
        )
        logger.log(f"Stage B done. Best loss: {best_loss_b:.4f}")

    # ============================================================
    # STAGE C: Contrastive Learning (InfoNCE)
    # ============================================================

    pairs_path = cfg.get("pairs_path")
    if pairs_path and os.path.exists(pairs_path):
        logger.log("\n=== STAGE C: Contrastive Learning ===")

        pairs_df = pd.read_csv(pairs_path)
        logger.log(f"Raw pairs: {len(pairs_df):,}")

        # Фильтрация пар: только валидные слова (буквы из charset)
        pairs_df = pairs_df[
            pairs_df["incorrect"].apply(lambda x: is_valid_word(str(x), allowed_chars))
            & pairs_df["correct"].apply(lambda x: is_valid_word(str(x), allowed_chars))
        ].copy()
        pairs_df["incorrect"] = pairs_df["incorrect"].apply(
            lambda x: clean_word(str(x), allowed_chars)
        )
        pairs_df["correct"] = pairs_df["correct"].apply(
            lambda x: clean_word(str(x), allowed_chars)
        )
        logger.log(f"Filtered pairs: {len(pairs_df):,}")

        loader_c = DataLoader(
            ContrastiveDataset(pairs_df, c2i, cfg, all_words),
            batch_size=cfg.get("batch_c", 64),
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        opt_c = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("lr_c", 1e-5),
            weight_decay=cfg.get("weight_decay", 0.01),
        )

        total_steps_c = cfg.get("epochs_c", 5) * len(loader_c)
        scheduler_c = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_c, T_max=total_steps_c
        )

        temperature = cfg.get("contrastive_temperature", 0.07)

        best_loss_c = float("inf")
        for ep in range(1, cfg.get("epochs_c", 5) + 1):
            loss = train_epoch_contrastive(
                model,
                loader_c,
                opt_c,
                device,
                scheduler=scheduler_c,
                grad_clip=grad_clip,
                temperature=temperature,
            )
            logger.log(f"[C] Epoch {ep:02d} | loss={loss:.4f}")
            if loss < best_loss_c:
                best_loss_c = loss
                torch.save(model.state_dict(), model_path)

        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        logger.log(f"Stage C done. Best loss: {best_loss_c:.4f}")
    else:
        torch.save(model.state_dict(), model_path)

    logger.log(f"\n=== TRAINING COMPLETE ===")
    logger.log(f"Final model: {model_path}")

    return model, (c2i, i2c, chars), exp_dir


if __name__ == "__main__":
    train()
