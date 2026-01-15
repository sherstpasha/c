"""CharLM обучение: трёхстадийное MLM для OCR-коррекции."""

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
    Logger, build_vocab, encode_str, choose_spans, 
    insert_split, masked_accuracy, masked_topk_accuracy,
    load_charset, tokenize_by_charset, add_ocr_noise, find_diff_positions
)


class LexiconMLMDataset(Dataset):
    """Stage A: MLM на словах из лексикона."""
    
    def __init__(self, words: list[str], c2i: dict, cfg: dict):
        self.words = words
        self.c2i = c2i
        self.cfg = cfg
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        w = self.words[idx]
        if random.random() < self.cfg["split_prob_a"]:
            w = insert_split(w)
        
        ids = encode_str(w, self.c2i, self.cfg["max_len"])
        L = min(len(w), self.cfg["max_len"])
        mask_pos = choose_spans(L, self.cfg["span_min"], self.cfg["span_max"],
                                self.cfg["num_spans_min"], self.cfg["num_spans_max"])
        
        x = ids.copy()
        y = [-100] * self.cfg["max_len"]
        mask_id = self.c2i["<MASK>"]
        
        for p in mask_pos:
            y[p] = ids[p]
            if random.random() < self.cfg["mask_prob"]:
                x[p] = mask_id
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class ContextMLMDataset(IterableDataset):
    """Stage B: контекстное MLM с OCR-шумом в контексте (не в центре)."""
    
    def __init__(self, tokens: list[str], c2i: dict, cfg: dict):
        super().__init__()
        self.tokens = tokens
        self.c2i = c2i
        self.cfg = cfg
        self.N = len(tokens)
    
    def _sample_window(self):
        r = random.random()
        p1, p2 = self.cfg["p_win_1"], self.cfg["p_win_2"]
        
        if r < p1:
            w = [self.tokens[random.randrange(self.N)]]
            return w, 0
        elif r < p1 + p2:
            i = random.randrange(1, self.N - 1)
            if random.random() < 0.5:
                return [self.tokens[i-1], self.tokens[i]], 1
            return [self.tokens[i], self.tokens[i+1]], 0
        else:
            i = random.randrange(1, self.N - 1)
            return [self.tokens[i-1], self.tokens[i], self.tokens[i+1]], 1
    
    def __iter__(self):
        while True:
            words, c_idx = self._sample_window()
            center = words[c_idx]
            
            if len(center) < self.cfg["min_word_len"]:
                continue
            
            # OCR-шум только в контекст (не в центр)
            noisy_words = []
            for i, w in enumerate(words):
                if i != c_idx:
                    noisy_words.append(add_ocr_noise(w, self.cfg))
                else:
                    noisy_words.append(w)
            
            clean_seq = " ".join(words)
            noisy_seq = " ".join(noisy_words)
            
            prefix_chars = sum(len(words[j]) + 1 for j in range(c_idx))
            cs = min(prefix_chars, self.cfg["max_len"])
            ce = min(prefix_chars + len(center), self.cfg["max_len"])
            
            if ce - cs < self.cfg["min_word_len"]:
                continue
            
            clean_ids = encode_str(clean_seq, self.c2i, self.cfg["max_len"])
            noisy_ids = encode_str(noisy_seq, self.c2i, self.cfg["max_len"])
            
            x = noisy_ids.copy()
            y = [-100] * self.cfg["max_len"]
            
            center_len = ce - cs
            rel_positions = choose_spans(center_len, self.cfg["span_min"], 
                                        self.cfg["span_max"], self.cfg["num_spans_min"],
                                        self.cfg["num_spans_max"])
            mask_id = self.c2i["<MASK>"]
            
            for rp in rel_positions:
                p = cs + rp
                if 0 <= p < self.cfg["max_len"]:
                    y[p] = clean_ids[p]
                    if random.random() < self.cfg["mask_prob"]:
                        x[p] = mask_id
            
            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class PairsMLMDataset(Dataset):
    """Stage C: обучение на парах (incorrect → correct).
    
    Маскируем ТОЛЬКО позиции, где incorrect отличается от correct.
    """
    
    def __init__(self, pairs_df: pd.DataFrame, c2i: dict, cfg: dict):
        super().__init__()
        self.c2i = c2i
        self.cfg = cfg
        self.mask_id = c2i["<MASK>"]
        
        # Подготавливаем пары
        self.incorrect_list = pairs_df["incorrect"].tolist()
        self.correct_list = pairs_df["correct"].tolist()
        self.N = len(self.incorrect_list)
        
        # Вероятности для окон
        self.p1 = cfg.get("p_win_1_c", 0.2)  # одна пара
        self.p2 = cfg.get("p_win_2_c", 0.3)  # две пары
        # 1 - p1 - p2 = три пары
    
    def __len__(self):
        return self.N
    
    def _sample_window(self, idx):
        """Формируем окно из 1/2/3 пар, центральная пара - idx."""
        r = random.random()
        
        if r < self.p1:
            # Одна пара
            return [self.incorrect_list[idx]], [self.correct_list[idx]], 0
        elif r < self.p1 + self.p2:
            # Две пары
            if idx == 0 or (idx < self.N - 1 and random.random() < 0.5):
                # idx и следующая
                if idx < self.N - 1:
                    inc = [self.incorrect_list[idx], self.incorrect_list[idx+1]]
                    cor = [self.correct_list[idx], self.correct_list[idx+1]]
                    return inc, cor, 0
            # Предыдущая и idx
            if idx > 0:
                inc = [self.incorrect_list[idx-1], self.incorrect_list[idx]]
                cor = [self.correct_list[idx-1], self.correct_list[idx]]
                return inc, cor, 1
            # Fallback
            return [self.incorrect_list[idx]], [self.correct_list[idx]], 0
        else:
            # Три пары
            if idx > 0 and idx < self.N - 1:
                inc = [self.incorrect_list[idx-1], self.incorrect_list[idx], self.incorrect_list[idx+1]]
                cor = [self.correct_list[idx-1], self.correct_list[idx], self.correct_list[idx+1]]
                return inc, cor, 1
            elif idx == 0 and self.N >= 3:
                inc = [self.incorrect_list[0], self.incorrect_list[1], self.incorrect_list[2]]
                cor = [self.correct_list[0], self.correct_list[1], self.correct_list[2]]
                return inc, cor, 0
            elif idx == self.N - 1 and self.N >= 3:
                inc = [self.incorrect_list[-3], self.incorrect_list[-2], self.incorrect_list[-1]]
                cor = [self.correct_list[-3], self.correct_list[-2], self.correct_list[-1]]
                return inc, cor, 2
            else:
                return [self.incorrect_list[idx]], [self.correct_list[idx]], 0
    
    def __getitem__(self, idx):
        inc_words, cor_words, center_idx = self._sample_window(idx)
        
        inc_seq = " ".join(inc_words)
        cor_seq = " ".join(cor_words)
        
        # Encode
        inc_ids = encode_str(inc_seq, self.c2i, self.cfg["max_len"])
        cor_ids = encode_str(cor_seq, self.c2i, self.cfg["max_len"])
        
        # Находим позиции центрального слова в последовательности
        prefix_len = sum(len(inc_words[j]) + 1 for j in range(center_idx))
        center_word_inc = inc_words[center_idx]
        center_word_cor = cor_words[center_idx]
        
        # Находим позиции различий в центральном слове
        diff_positions = find_diff_positions(center_word_inc, center_word_cor)
        
        x = inc_ids.copy()
        y = [-100] * self.cfg["max_len"]
        
        # Маскируем только позиции с различиями в центральном слове
        for rel_pos in diff_positions:
            abs_pos = prefix_len + rel_pos
            if 0 <= abs_pos < self.cfg["max_len"]:
                y[abs_pos] = cor_ids[abs_pos]
                if random.random() < self.cfg["mask_prob"]:
                    x[abs_pos] = self.mask_id
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ============================================================
# TRAIN FUNCTIONS
# ============================================================

def train_epoch(model, loader, optimizer, device, vocab_size, steps_limit=None, 
                scheduler=None, grad_clip=None):
    """Один эпох обучения."""
    model.train()
    tot_loss = tot_acc1 = tot_acc5 = 0.0
    n_steps = 0
    
    it = iter(loader)
    pbar = tqdm(range(steps_limit) if steps_limit else loader, leave=False)
    
    for _ in pbar:
        try:
            x, y = next(it) if steps_limit else _
        except StopIteration:
            break
        
        if not steps_limit:
            x, y = _
        
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # LR scheduler step
        if scheduler:
            scheduler.step()
        
        acc1 = masked_accuracy(logits, y)
        acc5 = masked_topk_accuracy(logits, y, k=5)
        
        tot_loss += loss.item()
        tot_acc1 += acc1
        tot_acc5 += acc5
        n_steps += 1
        
        pbar.set_postfix(loss=f"{loss.item():.3f}", acc1=f"{acc1*100:.1f}%")
    
    return tot_loss / max(1, n_steps), tot_acc1 / max(1, n_steps), tot_acc5 / max(1, n_steps)


def train(config: dict = None):
    """
    Основная функция обучения CharLM.
    
    Args:
        config: словарь конфигурации (переопределяет DEFAULT_CONFIG)
    
    Returns:
        model: обученная модель
        vocab: (c2i, i2c, chars)
        exp_dir: путь к папке эксперимента
    """
    # Объединяем конфиги
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    
    # Создаём папку эксперимента
    exp_dir = cfg["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    
    # Пути к артефактам в папке эксперимента (раздельно по стадиям)
    checkpoint_a_path = os.path.join(exp_dir, "model_a.pt")
    checkpoint_b_path = os.path.join(exp_dir, "model_b.pt")
    checkpoint_c_path = os.path.join(exp_dir, "model_c.pt")
    checkpoint_final_path = os.path.join(exp_dir, "model.pt")  # финальная модель
    vocab_path = os.path.join(exp_dir, "vocab.json")
    config_path = os.path.join(exp_dir, "config.json")
    log_path = os.path.join(exp_dir, "train.log")
    
    # Device
    if cfg["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["device"]
    
    # Seed
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    
    # Logger
    logger = Logger(log_path)
    logger.log(f"Experiment dir: {exp_dir}")
    logger.log(f"Config: {json.dumps(cfg, indent=2, ensure_ascii=False)}")
    logger.log(f"Device: {device}")
    
    # Сохраняем конфиг
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    
    # ============================================================
    # LOAD DATA
    # ============================================================
    
    logger.log("Loading lexicon...")
    lex_words = []
    try:
        with open(cfg["lexicon_path"], encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if len(w) >= cfg["min_word_len"]:
                    lex_words.append(w)
        logger.log(f"Lexicon words: {len(lex_words):,}")
    except Exception as e:
        logger.log(f"ERROR loading lexicon: {e}")
        raise
    
    # Vocab
    c2i, i2c, chars = build_vocab(lex_words, include_space=True)
    vocab_size = len(chars)
    logger.log(f"Vocab size: {vocab_size}")
    
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)
    
    # Load charset для токенизации
    charset = None
    if cfg.get("charset_path"):
        try:
            charset = load_charset(cfg["charset_path"])
            logger.log(f"Loaded charset: {len(charset)} chars")
        except Exception as e:
            logger.log(f"ERROR loading charset: {e}")
            raise
    
    # Context tokens
    tokens = []
    try:
        with open(cfg["text_path"], encoding="utf-8") as f:
            text = f.read()
        tokens = tokenize_by_charset(text, charset)
        logger.log(f"Context tokens: {len(tokens):,}")
    except Exception as e:
        logger.log(f"ERROR loading text: {e}")
        raise
    
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
        pad_idx=c2i["<PAD>"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model params: {total_params:,}")
    
    # ============================================================
    # STAGE A: PRETRAIN ON LEXICON
    # ============================================================
    
    logger.log("\n=== STAGE A: PRETRAIN ON LEXICON ===")
    
    loader_a = DataLoader(
        LexiconMLMDataset(lex_words, c2i, cfg),
        batch_size=cfg["batch_a"],
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    
    opt_a = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg["lr_a"],
        weight_decay=cfg.get("weight_decay", 0.01)
    )
    
    # Cosine annealing scheduler with warmup
    scheduler_a = None
    if cfg.get("use_scheduler", True):
        total_steps_a = cfg["epochs_a"] * len(loader_a)
        warmup_steps = cfg.get("warmup_steps", 500)
        scheduler_a = torch.optim.lr_scheduler.OneCycleLR(
            opt_a,
            max_lr=cfg["lr_a"],
            total_steps=total_steps_a,
            pct_start=warmup_steps / total_steps_a,
            anneal_strategy='cos'
        )
    
    best_a = float("inf")
    grad_clip = cfg.get("grad_clip", 1.0)
    
    for ep in range(1, cfg["epochs_a"] + 1):
        loss, acc1, acc5 = train_epoch(
            model, loader_a, opt_a, device, vocab_size,
            scheduler=scheduler_a, grad_clip=grad_clip
        )
        msg = f"[A] Epoch {ep:02d} | loss={loss:.4f} | acc@1={acc1*100:.2f}% | acc@5={acc5*100:.2f}%"
        logger.log(msg)
        
        if loss < best_a:
            best_a = loss
            torch.save(model.state_dict(), checkpoint_a_path)
            logger.log("  saved best (Stage A)")
    
    # Load best from Stage A
    model.load_state_dict(torch.load(checkpoint_a_path, map_location=device))
    logger.log(f"Stage A done. Best model: {checkpoint_a_path}")
    
    # ============================================================
    # STAGE B: FINETUNE ON CONTEXT
    # ============================================================
    
    logger.log("\n=== STAGE B: FINETUNE ON CONTEXT ===")
    
    loader_b = DataLoader(
        ContextMLMDataset(tokens, c2i, cfg),
        batch_size=cfg["batch_b"],
        shuffle=False,
        drop_last=True,
        num_workers=0
    )
    
    opt_b = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg["lr_b"],
        weight_decay=cfg.get("weight_decay", 0.01)
    )
    
    # Scheduler for Stage B
    scheduler_b = None
    if cfg.get("use_scheduler", True):
        total_steps_b = cfg["epochs_b"] * cfg["steps_per_epoch_b"]
        warmup_steps = min(cfg.get("warmup_steps", 500), total_steps_b // 10)
        scheduler_b = torch.optim.lr_scheduler.OneCycleLR(
            opt_b,
            max_lr=cfg["lr_b"],
            total_steps=total_steps_b,
            pct_start=warmup_steps / total_steps_b,
            anneal_strategy='cos'
        )
    
    best_b = float("inf")
    
    for ep in range(1, cfg["epochs_b"] + 1):
        loss, acc1, acc5 = train_epoch(
            model, loader_b, opt_b, device, vocab_size, 
            steps_limit=cfg["steps_per_epoch_b"],
            scheduler=scheduler_b, grad_clip=grad_clip
        )
        msg = f"[B] Epoch {ep:02d} | loss={loss:.4f} | acc@1={acc1*100:.2f}% | acc@5={acc5*100:.2f}%"
        logger.log(msg)
        
        if loss < best_b:
            best_b = loss
            torch.save(model.state_dict(), checkpoint_b_path)
            logger.log("  ✅ saved best (Stage B)")
    
    # Load best from Stage B
    model.load_state_dict(torch.load(checkpoint_b_path, map_location=device))
    logger.log(f"Stage B done. Best model: {checkpoint_b_path}")
    
    # ============================================================
    # STAGE C: FINETUNE ON PAIRS (incorrect → correct)
    # ============================================================
    
    pairs_path = cfg.get("pairs_path")
    if pairs_path and os.path.exists(pairs_path):
        logger.log("\n=== STAGE C: FINETUNE ON PAIRS ===")
        
        # Load pairs
        pairs_df = pd.read_csv(pairs_path)
        logger.log(f"Loaded {len(pairs_df):,} pairs from {pairs_path}")
        
        loader_c = DataLoader(
            PairsMLMDataset(pairs_df, c2i, cfg),
            batch_size=cfg.get("batch_c", 128),
            shuffle=True,
            drop_last=True,
            num_workers=0
        )
        
        opt_c = torch.optim.AdamW(
            model.parameters(), 
            lr=cfg.get("lr_c", 1e-6),
            weight_decay=cfg.get("weight_decay", 0.01)
        )
        
        # Scheduler for Stage C (очень консервативный)
        scheduler_c = None
        if cfg.get("use_scheduler", True):
            total_steps_c = cfg.get("epochs_c", 5) * len(loader_c)
            # Без warmup для Stage C, просто линейное снижение
            scheduler_c = torch.optim.lr_scheduler.LinearLR(
                opt_c,
                start_factor=1.0,
                end_factor=0.5,
                total_iters=total_steps_c
            )
        
        best_c = float("inf")
        
        for ep in range(1, cfg.get("epochs_c", 5) + 1):
            loss, acc1, acc5 = train_epoch(
                model, loader_c, opt_c, device, vocab_size,
                scheduler=scheduler_c, grad_clip=grad_clip
            )
            msg = f"[C] Epoch {ep:02d} | loss={loss:.4f} | acc@1={acc1*100:.2f}% | acc@5={acc5*100:.2f}%"
            logger.log(msg)
            
            if loss < best_c:
                best_c = loss
                torch.save(model.state_dict(), checkpoint_c_path)
                logger.log("  ✅ saved best (Stage C)")
        
        # Load best from Stage C
        model.load_state_dict(torch.load(checkpoint_c_path, map_location=device))
        logger.log(f"Stage C done. Best model: {checkpoint_c_path}")
    else:
        logger.log("\n[INFO] Skipping Stage C (no pairs_path or file not found)")
    
    # Save final model
    torch.save(model.state_dict(), checkpoint_final_path)
    
    logger.log(f"\n=== TRAINING COMPLETE ===")
    logger.log(f"Experiment dir: {exp_dir}")
    logger.log(f"  Model A (lexicon): {checkpoint_a_path}")
    logger.log(f"  Model B (context): {checkpoint_b_path}")
    if pairs_path and os.path.exists(pairs_path):
        logger.log(f"  Model C (pairs): {checkpoint_c_path}")
    logger.log(f"  Final model: {checkpoint_final_path}")
    logger.log(f"  Vocab: {vocab_path}")
    logger.log(f"  Config: {config_path}")
    
    return model, (c2i, i2c, chars), exp_dir


if __name__ == "__main__":
    train()
