"""
CharLM обучение: трёхстадийное metric learning для OCR-коррекции.

Stage A: Lexicon embedding pretraining (in-batch negatives, fast)
Stage B: Context-aware embedding
Stage C: Supervised OCR pairs (hard negative mining)
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
from .model import CharLM
from .utils import (
    Logger,
    build_vocab,
    encode_str,
    encode_batch,
    load_charset,
    tokenize_by_charset,
    apply_ocr_noise,
    create_corrupted_version,
    apply_mlm_mask,
    infonce_loss_inbatch,
    mine_hard_negatives,
    masked_accuracy,
    compute_recall_at_k,
)


# ============================================================
# DATASETS
# ============================================================


class LexiconDatasetFast(Dataset):
    """
    Stage A: Fast lexicon dataset with in-batch negatives.

    Returns only anchor and positive - negatives are other samples in batch.
    """

    def __init__(self, words: list[str], c2i: dict, cfg: dict):
        self.words = words
        self.c2i = c2i
        self.cfg = cfg
        self.vocab_size = len(c2i)
        self.n_words = len(words)

    def __len__(self):
        return self.n_words

    def __getitem__(self, idx):
        word = self.words[idx]

        # Anchor: with probability noise_prob, corrupt the word
        if random.random() < self.cfg.get("noise_prob_a", 0.5):
            anchor_word = create_corrupted_version(word, self.cfg)
        else:
            anchor_word = word

        # Encode
        anchor_ids = encode_str(anchor_word, self.c2i, self.cfg["max_len"])
        positive_ids = encode_str(word, self.c2i, self.cfg["max_len"])

        # MLM masking on positive (optional, low weight)
        mlm_weight = self.cfg.get("mlm_weight_a", 0.1)
        if mlm_weight > 0:
            masked_ids, mlm_labels = apply_mlm_mask(
                positive_ids,
                self.c2i,
                mask_prob=self.cfg.get("mask_prob", 0.15),
                vocab_size=self.vocab_size,
            )
        else:
            masked_ids = positive_ids
            mlm_labels = [-100] * len(positive_ids)

        return {
            "anchor": torch.tensor(anchor_ids, dtype=torch.long),
            "positive": torch.tensor(positive_ids, dtype=torch.long),
            "masked": torch.tensor(masked_ids, dtype=torch.long),
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long),
        }


class ContextDataset(IterableDataset):
    """
    Stage B: Context-aware embedding dataset.

    Windows of 1-3 words, center word is anchor.
    Context words may have OCR noise.
    """

    def __init__(self, tokens: list[str], c2i: dict, cfg: dict):
        super().__init__()
        self.tokens = tokens
        self.c2i = c2i
        self.cfg = cfg
        self.N = len(tokens)
        self.vocab_size = len(c2i)

    def _sample_window(self):
        """Sample a window of 1-3 words, return words and center index."""
        r = random.random()
        p1 = self.cfg.get("p_win_1", 0.4)
        p2 = self.cfg.get("p_win_2", 0.3)

        if r < p1:
            w = self.tokens[random.randrange(self.N)]
            return [w], 0
        elif r < p1 + p2:
            i = random.randrange(1, self.N)
            if random.random() < 0.5:
                return [self.tokens[i - 1], self.tokens[i]], random.choice([0, 1])
            return [self.tokens[i - 1], self.tokens[i]], 0
        else:
            i = random.randrange(1, self.N - 1)
            return [self.tokens[i - 1], self.tokens[i], self.tokens[i + 1]], 1

    def __iter__(self):
        while True:
            words, center_idx = self._sample_window()
            center_word = words[center_idx]

            if len(center_word) < self.cfg.get("min_word_len", 3):
                continue

            # Apply noise to non-center words
            noisy_words = []
            for i, w in enumerate(words):
                if i != center_idx and random.random() < self.cfg.get(
                    "context_noise_prob", 0.3
                ):
                    noisy_words.append(apply_ocr_noise(w, self.cfg))
                else:
                    noisy_words.append(w)

            noisy_seq = " ".join(noisy_words)
            anchor_ids = encode_str(noisy_seq, self.c2i, self.cfg["max_len"])
            positive_ids = encode_str(center_word, self.c2i, self.cfg["max_len"])

            # MLM
            clean_seq = " ".join(words)
            clean_ids = encode_str(clean_seq, self.c2i, self.cfg["max_len"])
            masked_ids, mlm_labels = apply_mlm_mask(
                clean_ids,
                self.c2i,
                mask_prob=self.cfg.get("mask_prob", 0.15),
                vocab_size=self.vocab_size,
            )

            yield {
                "anchor": torch.tensor(anchor_ids, dtype=torch.long),
                "positive": torch.tensor(positive_ids, dtype=torch.long),
                "masked": torch.tensor(masked_ids, dtype=torch.long),
                "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long),
            }


class OCRPairsDataset(Dataset):
    """
    Stage C: Supervised OCR pairs dataset.
    """

    def __init__(self, pairs_df: pd.DataFrame, c2i: dict, cfg: dict):
        self.incorrect = pairs_df["incorrect"].tolist()
        self.correct = pairs_df["correct"].tolist()
        self.c2i = c2i
        self.cfg = cfg
        self.vocab_size = len(c2i)
        self.N = len(self.incorrect)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        inc = self.incorrect[idx]
        cor = self.correct[idx]

        anchor_ids = encode_str(inc, self.c2i, self.cfg["max_len"])
        positive_ids = encode_str(cor, self.c2i, self.cfg["max_len"])

        masked_ids, mlm_labels = apply_mlm_mask(
            positive_ids,
            self.c2i,
            mask_prob=self.cfg.get("mask_prob", 0.15),
            vocab_size=self.vocab_size,
        )

        return {
            "anchor": torch.tensor(anchor_ids, dtype=torch.long),
            "positive": torch.tensor(positive_ids, dtype=torch.long),
            "masked": torch.tensor(masked_ids, dtype=torch.long),
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long),
            "pair_idx": idx,
        }


# ============================================================
# TRAINING FUNCTIONS
# ============================================================


def freeze_encoder(model: CharLM):
    """Freeze encoder parameters."""
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder(model: CharLM):
    """Unfreeze encoder parameters."""
    for p in model.encoder.parameters():
        p.requires_grad = True


def train_stage_a(
    model: CharLM,
    words: list[str],
    c2i: dict,
    cfg: dict,
    device: str,
    logger: Logger,
    checkpoint_path: str,
):
    """
    Stage A: Fast lexicon embedding pretraining with in-batch negatives.

    - No explicit negative sampling
    - In-batch negatives only (O(B²) similarity matrix)
    - Optional encoder freezing for first N epochs
    """
    logger.log("\n=== STAGE A: LEXICON EMBEDDING PRETRAINING ===")
    logger.log(
        f"Words: {len(words):,}, Batch: {cfg['batch_a']}, Epochs: {cfg['epochs_a']}"
    )

    dataset = LexiconDatasetFast(words, c2i, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_a"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=True,
    )

    freeze_epochs = cfg.get("freeze_encoder_a_epochs", 0)
    if freeze_epochs > 0:
        logger.log(f"Freezing encoder for first {freeze_epochs} epochs")
        freeze_encoder(model)

    # Optimizer - will be recreated after unfreeze
    def make_optimizer(encoder_lr):
        return torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": encoder_lr},
                {"params": model.embed_head.parameters(), "lr": cfg["lr_embed_head_a"]},
                {"params": model.mlm_head.parameters(), "lr": cfg["lr_embed_head_a"]},
            ],
            weight_decay=cfg.get("weight_decay", 0.01),
        )

    encoder_lr = 0.0 if freeze_epochs > 0 else cfg["lr_encoder_a"]
    optimizer = make_optimizer(encoder_lr)

    # Scheduler
    total_steps = cfg["epochs_a"] * len(loader)
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[encoder_lr or 1e-8, cfg["lr_embed_head_a"], cfg["lr_embed_head_a"]],
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    vocab_size = len(c2i)
    metric_weight = cfg.get("metric_weight_a", 1.0)
    mlm_weight = cfg.get("mlm_weight_a", 0.1)
    temperature = cfg.get("temperature", 0.07)
    grad_clip = cfg.get("grad_clip", 1.0)

    best_loss = float("inf")

    for epoch in range(1, cfg["epochs_a"] + 1):
        # Unfreeze encoder after freeze_epochs
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            logger.log(f"Unfreezing encoder at epoch {epoch}")
            unfreeze_encoder(model)
            # Recreate optimizer with proper encoder LR
            optimizer = make_optimizer(cfg["lr_encoder_a"])
            remaining_steps = (cfg["epochs_a"] - epoch + 1) * len(loader)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[
                    cfg["lr_encoder_a"],
                    cfg["lr_embed_head_a"],
                    cfg["lr_embed_head_a"],
                ],
                total_steps=remaining_steps,
                pct_start=0.1,
                anneal_strategy="cos",
            )

        model.train()
        total_loss = total_metric = total_mlm = 0.0
        n_steps = 0

        pbar = tqdm(loader, desc=f"[A] Epoch {epoch}", leave=False)
        for batch in pbar:
            anchor = batch["anchor"].to(device, non_blocking=True)
            positive = batch["positive"].to(device, non_blocking=True)
            masked = batch["masked"].to(device, non_blocking=True)
            mlm_labels = batch["mlm_labels"].to(device, non_blocking=True)

            # Forward - only 2 encoder passes per batch
            anchor_emb = model.encode_words(anchor)  # [B, D]
            positive_emb = model.encode_words(positive)  # [B, D]

            # In-batch InfoNCE loss
            metric_loss = infonce_loss_inbatch(anchor_emb, positive_emb, temperature)

            # MLM loss (optional)
            if mlm_weight > 0:
                mlm_logits = model.forward_mlm(masked)
                mlm_loss = F.cross_entropy(
                    mlm_logits.view(-1, vocab_size),
                    mlm_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                mlm_loss = torch.tensor(0.0, device=device)

            loss = metric_weight * metric_loss + mlm_weight * mlm_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_metric += metric_loss.item()
            total_mlm += mlm_loss.item() if mlm_weight > 0 else 0
            n_steps += 1

            pbar.set_postfix(
                loss=f"{loss.item():.3f}", metric=f"{metric_loss.item():.3f}"
            )

        avg_loss = total_loss / n_steps
        avg_metric = total_metric / n_steps
        avg_mlm = total_mlm / n_steps

        frozen_str = " [frozen]" if epoch <= freeze_epochs else ""
        logger.log(
            f"[A] Epoch {epoch:02d}{frozen_str} | loss={avg_loss:.4f} | "
            f"metric={avg_metric:.4f} | mlm={avg_mlm:.4f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.log("  ✅ saved best")

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    logger.log(f"Stage A done. Best loss: {best_loss:.4f}")

    return best_loss


def train_stage_b(
    model: CharLM,
    tokens: list[str],
    c2i: dict,
    cfg: dict,
    device: str,
    logger: Logger,
    checkpoint_path: str,
):
    """
    Stage B: Context-aware embedding training with in-batch negatives.
    """
    logger.log("\n=== STAGE B: CONTEXT-AWARE EMBEDDING ===")

    dataset = ContextDataset(tokens, c2i, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_b"],
        num_workers=cfg.get("num_workers", 0),
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": cfg["lr_encoder_b"]},
            {"params": model.embed_head.parameters(), "lr": cfg["lr_embed_head_b"]},
            {"params": model.mlm_head.parameters(), "lr": cfg["lr_embed_head_b"]},
        ],
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    steps_per_epoch = cfg.get("steps_per_epoch_b", 8000)
    total_steps = cfg["epochs_b"] * steps_per_epoch
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg["lr_encoder_b"], cfg["lr_embed_head_b"], cfg["lr_embed_head_b"]],
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    vocab_size = len(c2i)
    metric_weight = cfg.get("metric_weight_b", 1.0)
    mlm_weight = cfg.get("mlm_weight_b", 0.1)
    temperature = cfg.get("temperature", 0.07)
    grad_clip = cfg.get("grad_clip", 1.0)

    best_loss = float("inf")

    for epoch in range(1, cfg["epochs_b"] + 1):
        model.train()
        total_loss = total_metric = total_mlm = 0.0
        n_steps = 0

        data_iter = iter(loader)
        pbar = tqdm(range(steps_per_epoch), desc=f"[B] Epoch {epoch}", leave=False)

        for _ in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            anchor = batch["anchor"].to(device, non_blocking=True)
            positive = batch["positive"].to(device, non_blocking=True)
            masked = batch["masked"].to(device, non_blocking=True)
            mlm_labels = batch["mlm_labels"].to(device, non_blocking=True)

            anchor_emb = model.encode_words(anchor)
            positive_emb = model.encode_words(positive)

            # In-batch negatives
            metric_loss = infonce_loss_inbatch(anchor_emb, positive_emb, temperature)

            # MLM loss
            if mlm_weight > 0:
                mlm_logits = model.forward_mlm(masked)
                mlm_loss = F.cross_entropy(
                    mlm_logits.view(-1, vocab_size),
                    mlm_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                mlm_loss = torch.tensor(0.0, device=device)

            loss = metric_weight * metric_loss + mlm_weight * mlm_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_metric += metric_loss.item()
            total_mlm += mlm_loss.item() if mlm_weight > 0 else 0
            n_steps += 1

            pbar.set_postfix(
                loss=f"{loss.item():.3f}", metric=f"{metric_loss.item():.3f}"
            )

        avg_loss = total_loss / n_steps
        avg_metric = total_metric / n_steps
        avg_mlm = total_mlm / n_steps

        logger.log(
            f"[B] Epoch {epoch:02d} | loss={avg_loss:.4f} | "
            f"metric={avg_metric:.4f} | mlm={avg_mlm:.4f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.log("  ✅ saved best")

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    logger.log(f"Stage B done. Best loss: {best_loss:.4f}")

    return best_loss


def encode_lexicon_batched(
    model: CharLM, lexicon_ids: torch.Tensor, batch_size: int = 512
):
    """Encode lexicon in batches to avoid OOM."""
    model.eval()
    emb_list = []
    with torch.no_grad():
        for i in range(0, len(lexicon_ids), batch_size):
            batch_ids = lexicon_ids[i : i + batch_size]
            batch_emb = model.encode_words(batch_ids)
            emb_list.append(batch_emb)
    return torch.cat(emb_list, dim=0)


def train_stage_c(
    model: CharLM,
    pairs_df: pd.DataFrame,
    lexicon_words: list[str],
    c2i: dict,
    cfg: dict,
    device: str,
    logger: Logger,
    checkpoint_path: str,
):
    """
    Stage C: Supervised OCR pairs training with hard negative mining.

    Lexicon embeddings are pre-computed and updated every N epochs.
    """
    logger.log("\n=== STAGE C: SUPERVISED OCR PAIRS ===")
    logger.log(f"Pairs: {len(pairs_df):,}, Lexicon: {len(lexicon_words):,}")

    dataset = OCRPairsDataset(pairs_df, c2i, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_c"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": cfg["lr_encoder_c"]},
            {"params": model.embed_head.parameters(), "lr": cfg["lr_embed_head_c"]},
            {"params": model.mlm_head.parameters(), "lr": cfg["lr_embed_head_c"]},
        ],
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    total_steps = cfg["epochs_c"] * len(loader)
    warmup_steps = int(total_steps * cfg.get("warmup_ratio", 0.1))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[cfg["lr_encoder_c"], cfg["lr_embed_head_c"], cfg["lr_embed_head_c"]],
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    vocab_size = len(c2i)
    metric_weight = cfg.get("metric_weight_c", 1.0)
    mlm_weight = cfg.get("mlm_weight_c", 0.05)
    temperature = cfg.get("temperature", 0.07)
    grad_clip = cfg.get("grad_clip", 1.0)
    hard_neg_k = cfg.get("hard_neg_k", 10)
    update_every = cfg.get("update_lexicon_emb_every", 3)

    # Pre-encode lexicon
    logger.log("Pre-encoding lexicon for hard negative mining...")
    lexicon_ids = encode_batch(lexicon_words, c2i, cfg["max_len"]).to(device)
    lexicon_emb = encode_lexicon_batched(model, lexicon_ids)
    logger.log(f"Lexicon embeddings: {lexicon_emb.shape}")

    # Build correct word -> lexicon index mapping
    correct_words = pairs_df["correct"].tolist()
    correct_to_idx = {}
    for i, w in enumerate(lexicon_words):
        if w not in correct_to_idx:
            correct_to_idx[w] = i

    best_loss = float("inf")

    for epoch in range(1, cfg["epochs_c"] + 1):
        # Update lexicon embeddings periodically
        if epoch > 1 and (epoch - 1) % update_every == 0:
            logger.log(f"  Updating lexicon embeddings...")
            lexicon_emb = encode_lexicon_batched(model, lexicon_ids)

        model.train()
        total_loss = total_metric = total_mlm = 0.0
        n_steps = 0

        pbar = tqdm(loader, desc=f"[C] Epoch {epoch}", leave=False)
        for batch in pbar:
            anchor = batch["anchor"].to(device, non_blocking=True)
            positive = batch["positive"].to(device, non_blocking=True)
            masked = batch["masked"].to(device, non_blocking=True)
            mlm_labels = batch["mlm_labels"].to(device, non_blocking=True)
            pair_idx = batch["pair_idx"]

            anchor_emb = model.encode_words(anchor)
            positive_emb = model.encode_words(positive)

            # Get positive indices in lexicon
            pos_indices = torch.tensor(
                [correct_to_idx.get(correct_words[i], 0) for i in pair_idx],
                device=device,
            )

            # Mine hard negatives from lexicon (detached)
            hard_neg_emb = mine_hard_negatives(
                anchor_emb.detach(), lexicon_emb, pos_indices, k=hard_neg_k
            )

            # InfoNCE with hard negatives
            pos_sim = (anchor_emb * positive_emb).sum(dim=-1) / temperature
            neg_sim = (
                torch.bmm(hard_neg_emb, anchor_emb.unsqueeze(-1)).squeeze(-1)
                / temperature
            )
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            targets = torch.zeros(anchor_emb.size(0), dtype=torch.long, device=device)
            metric_loss = F.cross_entropy(logits, targets)

            # MLM loss
            if mlm_weight > 0:
                mlm_logits = model.forward_mlm(masked)
                mlm_loss = F.cross_entropy(
                    mlm_logits.view(-1, vocab_size),
                    mlm_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                mlm_loss = torch.tensor(0.0, device=device)

            loss = metric_weight * metric_loss + mlm_weight * mlm_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_metric += metric_loss.item()
            total_mlm += mlm_loss.item() if mlm_weight > 0 else 0
            n_steps += 1

            pbar.set_postfix(
                loss=f"{loss.item():.3f}", metric=f"{metric_loss.item():.3f}"
            )

        avg_loss = total_loss / n_steps
        avg_metric = total_metric / n_steps
        avg_mlm = total_mlm / n_steps

        # Evaluate recall
        model.eval()
        with torch.no_grad():
            # Update embeddings for eval
            lexicon_emb_eval = encode_lexicon_batched(model, lexicon_ids)

            sample_size = min(1000, len(dataset))
            sample_indices = random.sample(range(len(dataset)), sample_size)

            query_ids = torch.stack([dataset[i]["anchor"] for i in sample_indices]).to(
                device
            )
            query_emb = model.encode_words(query_ids)
            labels = torch.tensor(
                [correct_to_idx.get(correct_words[i], 0) for i in sample_indices],
                device=device,
            )

            recall_1 = compute_recall_at_k(query_emb, lexicon_emb_eval, labels, k=1)
            recall_10 = compute_recall_at_k(query_emb, lexicon_emb_eval, labels, k=10)

        logger.log(
            f"[C] Epoch {epoch:02d} | loss={avg_loss:.4f} | "
            f"metric={avg_metric:.4f} | mlm={avg_mlm:.4f} | "
            f"R@1={recall_1*100:.1f}% | R@10={recall_10*100:.1f}%"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)
            logger.log("  ✅ saved best")

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    logger.log(f"Stage C done. Best loss: {best_loss:.4f}")

    return best_loss


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================


def train(config: dict = None):
    """
    Main training function for CharLM.

    Three-stage training:
    - Stage A: Lexicon embedding pretraining
    - Stage B: Context-aware embedding
    - Stage C: Supervised OCR pairs

    Args:
        config: configuration dict (overrides DEFAULT_CONFIG)

    Returns:
        model: trained model
        vocab: (c2i, i2c, chars)
        exp_dir: experiment directory path
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    # Create experiment directory
    exp_dir = cfg["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)

    # Paths
    checkpoint_a = os.path.join(exp_dir, "model_a.pt")
    checkpoint_b = os.path.join(exp_dir, "model_b.pt")
    checkpoint_c = os.path.join(exp_dir, "model_c.pt")
    checkpoint_final = os.path.join(exp_dir, "model.pt")
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
    logger.log(f"Device: {device}")
    logger.log(f"Config:\n{json.dumps(cfg, indent=2, ensure_ascii=False, default=str)}")

    # Save config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False, default=str)

    # ============================================================
    # LOAD DATA
    # ============================================================

    logger.log("\n=== LOADING DATA ===")

    # Lexicon
    lex_words = []
    with open(cfg["lexicon_path"], encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if len(w) >= cfg.get("min_word_len", 3):
                lex_words.append(w)

    # Apply max_lexicon_words limit for Stage A efficiency
    max_lex = cfg.get("max_lexicon_words")
    if max_lex and len(lex_words) > max_lex:
        logger.log(f"Lexicon original: {len(lex_words):,} words")
        random.shuffle(lex_words)
        lex_words = lex_words[:max_lex]
        logger.log(
            f"Lexicon limited to: {len(lex_words):,} words (max_lexicon_words={max_lex:,})"
        )
    else:
        logger.log(f"Lexicon: {len(lex_words):,} words")

    # Build vocab
    c2i, i2c, chars = build_vocab(lex_words, include_space=True)
    vocab_size = len(chars)
    logger.log(f"Vocab: {vocab_size} chars")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)

    # Charset for tokenization
    charset = None
    if cfg.get("charset_path"):
        charset = load_charset(cfg["charset_path"])
        logger.log(f"Charset: {len(charset)} chars")

    # Context tokens
    tokens = []
    if cfg.get("text_path") and os.path.exists(cfg["text_path"]):
        with open(cfg["text_path"], encoding="utf-8") as f:
            text = f.read()
        tokens = tokenize_by_charset(text, charset)
        logger.log(f"Context tokens: {len(tokens):,}")

    # OCR pairs
    pairs_df = None
    if cfg.get("pairs_path") and os.path.exists(cfg["pairs_path"]):
        pairs_df = pd.read_csv(cfg["pairs_path"])
        logger.log(f"OCR pairs: {len(pairs_df):,}")

    # ============================================================
    # MODEL
    # ============================================================

    model = CharLM(
        vocab_size=vocab_size,
        emb_size=cfg["emb_size"],
        embed_dim=cfg.get("embed_dim", cfg["emb_size"]),
        max_len=cfg["max_len"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        ffn_size=cfg["ffn_size"],
        dropout=cfg["dropout"],
        pad_idx=c2i["<PAD>"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model params: {total_params:,}")

    # ============================================================
    # TRAINING
    # ============================================================

    # Stage A
    train_stage_a(model, lex_words, c2i, cfg, device, logger, checkpoint_a)

    # Stage B
    if tokens:
        train_stage_b(model, tokens, c2i, cfg, device, logger, checkpoint_b)
    else:
        logger.log("\n[INFO] Skipping Stage B (no context tokens)")

    # Stage C
    if pairs_df is not None:
        train_stage_c(
            model, pairs_df, lex_words, c2i, cfg, device, logger, checkpoint_c
        )
    else:
        logger.log("\n[INFO] Skipping Stage C (no pairs)")

    # Save final model
    torch.save(model.state_dict(), checkpoint_final)

    logger.log("\n=== TRAINING COMPLETE ===")
    logger.log(f"Final model: {checkpoint_final}")

    return model, (c2i, i2c, chars), exp_dir


if __name__ == "__main__":
    train()
