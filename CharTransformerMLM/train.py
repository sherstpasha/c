import os
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from CharTransformerMLM.model import CharTransformerMLM
from CharTransformerMLM.dataset import CharOCRDenoiseDataset
from CharTransformerMLM.vocab import CharVocab
from CharTransformerMLM.utils.collate import collate_denoise


# ============================================================
# LOGGER
# ============================================================


class TxtLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # data
    "charset_path": "data/charset.txt",
    "text_path": "data/extracted_texts_cleaned.txt",
    "pairs_csv": "data/pairs_with_errors.csv",
    "words_path": "data/all_words.txt",
    "max_words": 3,
    "p_real_start": 0.1,
    "p_real_end": 0.3,
    "p_ending_swap": 0.03,
    "p_extra_punct": 0.02,
    # model
    "emb_size": 160,
    "n_layers": 4,
    "n_heads": 5,
    "ffn_size": 640,
    "dropout": 0.1,
    # training
    "epochs": 10,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    # logging
    "eval_every_epochs": 1,
    "save_dir": "checkpoints",
    "resume": None,
}


# ============================================================
# UTILS
# ============================================================


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mlm_accuracy(logits, targets):
    mask = targets != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    preds = logits.argmax(dim=-1)
    return (preds[mask] == targets[mask]).float().mean()


@torch.no_grad()
def inspect_denoise_predictions(model, vocab, dataset, device, n_samples=3):
    model.eval()
    print("\n=== DENOISING PREDICTIONS CHECK ===")

    def pretty_full(ids):
        s = vocab.decode(ids)
        return s.replace("<EOW>", " | ")

    for _ in range(n_samples):
        x, y = dataset[random.randrange(len(dataset))]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        logits, _ = model(x, y)
        preds = logits.argmax(dim=-1)

        x_ids = x[0].tolist()
        y_ids = y[0].tolist()
        p_ids = preds[0].tolist()

        noisy_full = pretty_full(x_ids)
        pred_full = pretty_full(p_ids)

        tgt_full = []
        for xi, yi in zip(x_ids, y_ids):
            if yi != -100:
                tgt_full.append(vocab.id_to_token[yi])
            else:
                tgt_full.append(vocab.id_to_token[xi])
        tgt_full = "".join(tgt_full).replace("<EOW>", " | ")

        fix_noisy = []
        fix_tgt = []
        fix_pred = []

        for xi, yi, pi in zip(x_ids, y_ids, p_ids):
            if yi != -100:
                fix_noisy.append(vocab.id_to_token[xi])
                fix_tgt.append(vocab.id_to_token[yi])
                fix_pred.append(vocab.id_to_token[pi])

        print("\n--- SAMPLE ---")
        print("NOISY FULL:  ", noisy_full)
        print("TARGET FULL: ", tgt_full)
        print("PRED FULL:   ", pred_full)

        if fix_noisy:
            print("\nFIX ONLY:")
            print("NOISY:  ", "".join(fix_noisy))
            print("TARGET: ", "".join(fix_tgt))
            print("PRED:   ", "".join(fix_pred))


@torch.no_grad()
def inspect_word_embeddings(model, vocab, dataset, device, n_words=3, top_k=5):
    """Показать слова и их ближайших соседей по эмбеддингам (на чистых словах)"""
    model.eval()
    print("\n=== WORD EMBEDDINGS CHECK ===")

    # Собираем чистые слова из текстового файла
    words = []
    embeddings = []

    # Берём случайные строки из dataset.lines (чистые тексты)
    sample_lines = random.sample(dataset.lines, min(100, len(dataset.lines)))

    for line in sample_lines:
        # Разбиваем на слова
        line_words = line.split()
        for word in line_words:
            # Фильтруем: только буквенные слова 3-15 символов
            if len(word) < 3 or len(word) > 15:
                continue
            if not all(ch in vocab.token_to_id for ch in word):
                continue
            if not any(ch.isalpha() for ch in word):
                continue

            # Кодируем слово + EOW
            ids = vocab.encode(word) + [vocab.eow]
            x = torch.tensor([ids], device=device)

            _, hidden = model(x)
            word_embs = model.extract_word_embeddings(hidden, x)

            if len(word_embs[0]) > 0:
                words.append(word)
                embeddings.append(word_embs[0][0])

        if len(words) >= 150:
            break

    if len(words) < n_words + top_k:
        print("Недостаточно слов для анализа")
        return

    # Стекаем эмбеддинги
    emb_matrix = torch.stack(embeddings)  # [N, D]
    emb_matrix = emb_matrix / emb_matrix.norm(dim=-1, keepdim=True)  # Нормализация

    # Выбираем n_words случайных слов
    indices = random.sample(range(len(words)), n_words)

    for idx in indices:
        query_word = words[idx]
        query_emb = emb_matrix[idx : idx + 1]  # [1, D]

        # Косинусное сходство
        similarities = (emb_matrix @ query_emb.T).squeeze()  # [N]

        # Топ-k+1 (включая само слово)
        top_indices = similarities.argsort(descending=True)[: top_k + 1]

        neighbors = []
        for ti in top_indices:
            if ti != idx:
                neighbors.append(f"{words[ti]} ({similarities[ti]:.3f})")

        print(f"\n'{query_word}' → {', '.join(neighbors[:top_k])}")


# ============================================================
# TRAIN
# ============================================================


def train(config):
    set_seed(config["seed"])
    device = config["device"]

    os.makedirs(config["save_dir"], exist_ok=True)
    logger = TxtLogger(Path(config["save_dir"]) / "train.log")

    vocab = CharVocab(config["charset_path"])

    dataset = CharOCRDenoiseDataset(
        text_path=config["text_path"],
        pairs_csv_path=config["pairs_csv"],
        vocab=vocab,
        words_path=config.get("words_path"),
        max_words=config["max_words"],
        p_ending_swap=config.get("p_ending_swap", 0.03),
        p_extra_punct=config.get("p_extra_punct", 0.02),
    )

    # Сохранить топ окончаний в JSON для проверки
    if dataset.top_endings:
        endings_path = Path(config["save_dir"]) / "top_endings.json"
        with open(endings_path, "w", encoding="utf-8") as f:
            json.dump(dataset.top_endings, f, ensure_ascii=False, indent=2)
        logger.log(f"Сохранено {len(dataset.top_endings)} окончаний в {endings_path}")

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_denoise(b, vocab.pad),
    )

    model = CharTransformerMLM(
        vocab_size=len(vocab.token_to_id),
        emb_size=config["emb_size"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        ffn_size=config["ffn_size"],
        dropout=config["dropout"],
        pad_idx=vocab.pad,
        eow_idx=vocab.eow,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["lr"] * 0.01,
    )

    ce_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    best_acc = 0.0
    start_epoch = 0

    if config["resume"]:
        ckpt = torch.load(config["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        best_acc = ckpt["best_acc"]
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, config["epochs"]):
        # Линейно увеличиваем p_real от start до end
        progress = epoch / max(1, config["epochs"] - 1)
        dataset.p_real = config["p_real_start"] + progress * (
            config["p_real_end"] - config["p_real_start"]
        )

        model.train()

        run_loss = 0.0
        run_acc = 0.0

        pbar = tqdm(loader, desc=f"epoch {epoch}", ncols=100)

        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            logits, _ = model(x, y)

            eow = vocab.eow
            mask = (x != eow) & (y == -100)
            logits[..., eow] -= mask * 1e9

            loss = ce_loss(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            acc = mlm_accuracy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            run_loss += loss.item()
            run_acc += acc.item()

            pbar.set_postfix(
                loss=f"{run_loss/(pbar.n+1):.3f}",
                acc=f"{run_acc/(pbar.n+1):.3f}",
            )

        ep_loss = run_loss / len(loader)
        ep_acc = run_acc / len(loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.log(
            f"epoch {epoch} | loss={ep_loss:.4f} acc={ep_acc:.3f} lr={current_lr:.2e} p_real={dataset.p_real:.2f}"
        )

        if (epoch + 1) % config["eval_every_epochs"] == 0:
            inspect_denoise_predictions(model, vocab, dataset, device)
            inspect_word_embeddings(model, vocab, dataset, device)

        if ep_acc > best_acc:
            best_acc = ep_acc

            save_dir = Path(config["save_dir"])

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config,
                },
                save_dir / "best.pt",
            )

            torch.save(
                model.state_dict(),
                save_dir / "best_weights.pt",
            )

            # Сохранить конфиг как JSON
            with open(save_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            logger.log(f"Saved BEST checkpoint and weights (acc={best_acc:.3f})")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": config,
            },
            save_dir / "last.pt",
        )


if __name__ == "__main__":
    train(CONFIG)
