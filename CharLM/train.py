# train.py
import os
import json
import random
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import DEFAULT_CONFIG
from .model import CharTransformerMLM
from .dataset import LexiconMLMDataset, OCRMLMDataset, MixedMLMDataset
from .utils import (
    Logger,
    build_vocab,
    evaluate_ocr_confidence,
    evaluate_ocr_with_cer,
    masked_accuracy,
    filter_words,
    load_allowed_chars,
    log_random_examples,
)


# ============================================================
# OCR evaluation
# ============================================================


def load_eval_pairs(path, allowed_chars, eval_ratio=0.15, seed=42):
    pairs = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inc = row["incorrect"].strip().lower()
            cor = row["correct"].strip().lower()
            if len(inc) < 4:
                continue
            if len(inc) != len(cor):
                continue
            if inc == cor:
                continue
            if not all(ch in allowed_chars for ch in inc):
                continue
            if not all(ch in allowed_chars for ch in cor):
                continue

            pairs.append((inc, cor))

    random.seed(seed)
    random.shuffle(pairs)

    n_eval = int(len(pairs) * eval_ratio)
    return pairs[:n_eval], pairs[n_eval:]


# ============================================================
# training
# ============================================================


def train(config: dict = None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    os.makedirs(cfg["exp_dir"], exist_ok=True)
    ckpt_dir = os.path.join(cfg["exp_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(os.path.join(cfg["exp_dir"], "train.log"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"Device: {device}")

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # -------------------- DATA --------------------
    allowed_chars = load_allowed_chars(cfg["charset_path"])

    with open(cfg["lexicon_path"], encoding="utf-8") as f:
        raw_words = [w.strip() for w in f if w.strip()]

    words = filter_words(
        raw_words,
        min_len=cfg["min_word_len"],
        allowed_chars=allowed_chars,
    )

    lexicon = set(words)

    if cfg.get("max_words_a") and len(words) > cfg["max_words_a"]:
        words = random.sample(words, cfg["max_words_a"])

    logger.log(f"Training words: {len(words):,}")

    c2i, i2c, chars = build_vocab(words)
    vocab_size = len(chars)

    with open(os.path.join(cfg["exp_dir"], "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)

    # -------------------- OCR eval data --------------------
    eval_pairs = []
    holdout = []

    train_pairs = []

    if cfg.get("pairs_path") and os.path.exists(cfg["pairs_path"]):
        eval_pairs, train_pairs = load_eval_pairs(
            cfg["pairs_path"],
            allowed_chars,
            eval_ratio=cfg.get("eval_ratio", 0.15),
            seed=cfg["seed"],
        )
        logger.log(f"OCR train pairs: {len(train_pairs)}")

    lex_dataset = LexiconMLMDataset(words, c2i, cfg)

    datasets = [lex_dataset]
    sampling_probs = [cfg.get("lexicon_prob", 0.7)]

    if train_pairs:

        ocr_dataset = OCRMLMDataset(
            train_pairs,
            c2i=c2i,
            max_len=cfg["max_len"],
            mask_id=c2i["<MASK>"],
        )
        datasets.append(ocr_dataset)
        sampling_probs.append(cfg.get("ocr_prob", 0.3))

    dataset = MixedMLMDataset(datasets, sampling_probs)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_a"],
        shuffle=True,
        drop_last=True,
    )

    # -----------------------------------------
    # Save OCR eval pairs for inspection
    # -----------------------------------------
    eval_dump_path = os.path.join(cfg["exp_dir"], "ocr_eval_pairs.tsv")

    with open(eval_dump_path, "w", encoding="utf-8") as f:
        f.write("incorrect\tcorrect\n")
        for inc, cor in eval_pairs:
            f.write(f"{inc}\t{cor}\n")

    logger.log(f"OCR eval pairs saved to: {eval_dump_path}")

    holdout_dump_path = os.path.join(cfg["exp_dir"], "ocr_holdout_pairs.tsv")

    with open(holdout_dump_path, "w", encoding="utf-8") as f:
        f.write("incorrect\tcorrect\n")
        for inc, cor in holdout:
            f.write(f"{inc}\t{cor}\n")

    logger.log(f"OCR holdout pairs saved to: {holdout_dump_path}")

    # -------------------- MODEL --------------------
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr_a"],
        weight_decay=cfg.get("weight_decay", 0.01),
    )
    total_steps = cfg["epochs_a"] * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps),
    )

    # -------------------- TRAIN LOOP --------------------
    for epoch in range(1, cfg["epochs_a"] + 1):
        model.train()
        total_loss = total_acc = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()

            # ---- grad clipping ----
            grad_clip = cfg.get("grad_clip", 0.0)
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            # ---- scheduler step (per batch) ----
            scheduler.step()

            acc = masked_accuracy(logits, y)

            total_loss += loss.item()
            total_acc += acc
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc*100:.1f}%")

        mean_loss = total_loss / len(loader)
        mean_acc = total_acc / len(loader)

        logger.log(
            f"[Epoch {epoch}] MLM loss={mean_loss:.4f} | MLM masked_acc={mean_acc*100:.2f}%"
        )

        # log MLM examples
        log_random_examples(model, (x, y), logits, c2i, i2c, logger)

        if eval_pairs:
            model.eval()
            if epoch == 1 or epoch % cfg.get("conf_eval_every", 5) == 0:
                corr_stats, incorr_stats = evaluate_ocr_confidence(
                    model,
                    eval_pairs,
                    c2i,
                    device,
                    cfg["max_len"],
                )

                logger.log("==== OCR CONFIDENCE STATS ====")
                logger.log(f"Correct symbols  : {corr_stats}")
                logger.log(f"Incorrect symbols: {incorr_stats}")
                logger.log("==============================")

        csv_path = os.path.join(cfg["exp_dir"], f"ocr_epoch_{epoch}.csv")

        stats = evaluate_ocr_with_cer(
            model,
            eval_pairs,
            c2i,
            i2c,
            device,
            cfg["max_len"],
            mask_threshold=0.2,
            apply_threshold=0.95,
            max_edits=3,
            csv_path=csv_path,
            lexicon=lexicon,
        )

        logger.log("==== OCR CER STATS ====")
        for k, v in stats.items():
            logger.log(f"{k:15s}: {v:.4f}")
        logger.log("======================")

        if epoch % 1 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(ckpt_dir, f"charlm_epoch_{epoch}.pt"),
            )
    return model, (c2i, i2c, chars)
