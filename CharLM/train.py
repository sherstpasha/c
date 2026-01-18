import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import faiss

from vocab import CharVocab
from dataset import (
    WordMLMDataset,
    CollateMLMStageA,
    ContextRerankerDataset,
    CollateReranker,
    OCRPairsRerankerDataset,
)
from model import CharTransformerMLM
from corrector import FaissRerankCorrector


# ============================================================
# Utils
# ============================================================

import csv


def load_pairs_csv(path: str):
    pairs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                {
                    "incorrect": row["incorrect"],
                    "correct": row["correct"],
                }
            )
    return pairs


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def cycle_iter(loader: DataLoader):
    """Бесконечный итератор по loader."""
    while True:
        for b in loader:
            yield b


# ============================================================
# Logging helpers
# ============================================================


@torch.no_grad()
def decode_example(
    model: CharTransformerMLM,
    vocab: CharVocab,
    x: torch.Tensor,  # [T]
    y: torch.Tensor,  # [T]
) -> tuple[str, str, str]:
    """
    Возвращает (input_masked, prediction, gold)
    """
    model.eval()
    logits = model(x.unsqueeze(0))  # [1, T, V]
    preds = logits.argmax(dim=-1)[0]  # [T]

    input_ids = x.tolist()
    gold_ids = y.tolist()

    pred_ids = []
    for xi, yi, pi in zip(input_ids, gold_ids, preds.tolist()):
        if yi != -100:
            pred_ids.append(pi)  # предсказываем только mask
        else:
            pred_ids.append(xi)  # копируем вход

    input_str = vocab.decode(input_ids)
    pred_str = vocab.decode(pred_ids)
    gold_str = vocab.decode(
        [yi if yi != -100 else xi for xi, yi in zip(input_ids, gold_ids)]
    )

    return input_str, pred_str, gold_str


def pick_better_example_idx(
    bx: torch.Tensor, by: torch.Tensor, min_visible: int = 2
) -> int:
    """
    Хочется видеть пример не из одних <MASK>.
    y == -100 значит "не маска" (видимый символ).
    """
    # visible_count: сколько символов не замаскировано
    visible_count = (by == -100).sum(dim=1)  # [B]
    candidates = (visible_count >= min_visible).nonzero(as_tuple=True)[0]
    if candidates.numel() > 0:
        return int(candidates[random.randrange(candidates.numel())].item())
    return random.randrange(bx.size(0))


# ============================================================
# FAISS helpers (lexicon inspection)
# ============================================================


@torch.no_grad()
def build_word_embeddings(
    model: CharTransformerMLM,
    vocab: CharVocab,
    words: List[str],
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """
    Возвращает tensor [N, D] на CPU
    ВАЖНО: режем до model.max_len, чтобы совпадало с train-time.
    """
    model.eval()
    all_embs = []

    for i in tqdm(
        range(0, len(words), batch_size),
        desc="  [FAISS] encoding",
        leave=False,
    ):
        batch_words = words[i : i + batch_size]
        encoded = [vocab.encode(w)[: model.max_len] for w in batch_words if w]
        if not encoded:
            continue

        max_len = max(len(x) for x in encoded)
        x = torch.full(
            (len(encoded), max_len),
            vocab.pad,
            dtype=torch.long,
            device=device,
        )
        for j, ids in enumerate(encoded):
            x[j, : len(ids)] = torch.tensor(ids, device=device)

        h = model.encode(x)
        emb = h.mean(dim=1)  # [B, D]
        all_embs.append(emb.cpu())

    return (
        torch.cat(all_embs, dim=0)
        if all_embs
        else torch.empty((0, model.mlm_head.in_features))
    )


def build_faiss_index(embs: torch.Tensor) -> faiss.Index:
    x = embs.numpy().astype("float32")
    faiss.normalize_L2(x)
    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)
    return index


def faiss_neighbors(
    index: faiss.Index,
    embs: torch.Tensor,
    words: List[str],
    query_idx: int,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    q = embs[query_idx : query_idx + 1].numpy().astype("float32")
    faiss.normalize_L2(q)

    sims, idxs = index.search(q, top_k + 1)

    result = []
    for i, s in zip(idxs[0], sims[0]):
        if i == query_idx:
            continue
        result.append((words[i], float(s)))
        if len(result) == top_k:
            break
    return result


# ============================================================
# Losses
# ============================================================


def rerank_margin_loss(
    pos_score: torch.Tensor, neg_score: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    # хотим pos >= neg + margin  =>  margin - (pos-neg) <= 0
    return torch.relu(margin - (pos_score - neg_score)).mean()


@torch.no_grad()
def log_rerank_example(
    model: CharTransformerMLM,
    vocab: CharVocab,
    rerank_dataset,
    device: torch.device,
):
    model.eval()

    idx = random.randrange(len(rerank_dataset))
    raw = rerank_dataset.get_raw(idx)

    pos = raw["pos_ids"].unsqueeze(0).to(device)
    neg = raw["neg_ids"].unsqueeze(0).to(device)

    ps = model.score(pos).item()
    ns = model.score(neg).item()

    print("\n[RERANK EXAMPLE]")
    print(f"CTX : {raw['left']} ___ {raw['right']}")
    print(f"POS : {raw['pos_word']:<20} score = {ps:+.3f}")
    print(f"NEG : {raw['neg_word']:<20} score = {ns:+.3f}")
    print(f"Δ   : {(ps - ns):+.3f}")


# ============================================================
# Training
# ============================================================
@torch.no_grad()
def log_ocr_rerank_example(model, vocab, ocr_dataset, device):
    model.eval()
    idx = random.randrange(len(ocr_dataset))

    # OCRPairsRerankerDataset хранит так:
    pos = ocr_dataset.pos_tensors[idx].unsqueeze(0).to(device)  # [1, T]
    neg = ocr_dataset.neg_tensors[idx].unsqueeze(0).to(device)  # [1, T]
    meta = ocr_dataset.meta[idx]

    # быстрее: один прогон
    both = torch.cat([pos, neg], dim=0)  # [2, T]
    scores = model.score(both)  # [2]
    pos_score, neg_score = scores[0].item(), scores[1].item()

    ctx = f"{meta['left']} ___ {meta['right']}"

    print("\n[RERANK EXAMPLE – OCR]")
    print(f"CTX : {ctx}")
    print(f"POS : {meta['pos']:<20} score = {pos_score:+.3f}")
    print(f"NEG : {meta['neg']:<20} score = {neg_score:+.3f}")
    print(f"Δ   : {pos_score - neg_score:+.3f}")


def train(config: Dict):
    # ----------------------------
    # CONFIG
    # ----------------------------
    lexicon_path = config["lexicon_path"]
    charset_path = config["charset_path"]
    text_path = config["text_path"]  # data/extracted_texts_cleaned.txt

    exp_dir = Path(config.get("exp_dir", "exp_multitask"))
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config.get("device", "auto"))
    set_seed(config.get("seed", 42))

    # multitask
    p_rerank = float(config.get("p_rerank", 0.3))  # доля шагов reranker
    rerank_margin = float(config.get("rerank_margin", 1.0))
    rerank_window = int(config.get("rerank_window", 5))

    # ----------------------------
    # VOCAB
    # ----------------------------
    vocab = CharVocab(charset_path)

    eval_pairs = load_pairs_csv(config["eval_pairs_path"])
    print(f"[EVAL] loaded correction pairs = {len(eval_pairs)}")

    # ----------------------------
    # DATA: MLM
    # ----------------------------
    mlm_dataset = WordMLMDataset(
        words_path=lexicon_path,
        vocab=vocab,
        max_len=config.get("max_len", 32),
        mask_prob=config.get("mask_prob", 0.15),
        min_word_len=config.get("min_word_len", 1),
    )
    mlm_loader = DataLoader(
        mlm_dataset,
        batch_size=config.get("batch_size", 256),
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=CollateMLMStageA(vocab.pad),
    )
    mlm_iter = cycle_iter(mlm_loader)

    # ----------------------------
    # DATA: RERANKER
    # ----------------------------
    rerank_dataset = ContextRerankerDataset(
        text_path=text_path,
        vocab=vocab,
        max_len=config.get("max_len", 32),
        window=rerank_window,
    )
    rerank_loader = DataLoader(
        rerank_dataset,
        batch_size=config.get("batch_size", 256),
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=CollateReranker(vocab.pad),
    )
    rerank_iter = cycle_iter(rerank_loader)

    ocr_dataset = OCRPairsRerankerDataset(
        pairs_csv=config["ocr_pairs_path"],
        text_path=text_path,
        vocab=vocab,
        max_len=config.get("max_len", 32),
        window=rerank_window,
    )
    ocr_loader = DataLoader(
        ocr_dataset,
        batch_size=config.get("batch_size", 256),
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        pin_memory=(device.type == "cuda"),
    )
    ocr_iter = cycle_iter(ocr_loader)
    print("[OCR CHECK]")
    print("dataset size:", len(ocr_dataset))
    print("example:", ocr_dataset[0]["meta"])
    # ----------------------------
    # MODEL
    # ----------------------------
    model = CharTransformerMLM(
        vocab_size=len(vocab),
        emb_size=config.get("emb_size", 256),
        max_len=config.get("max_len", 32),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 8),
        ffn_size=config.get("ffn_size", 1024),
        dropout=config.get("dropout", 0.1),
        pad_idx=vocab.pad,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 0.01),
    )

    mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    epochs = int(config.get("epochs", 5))
    steps_per_epoch = int(
        config.get("steps_per_epoch", len(mlm_loader))
    )  # можно фиксировать
    grad_clip = config.get("grad_clip", 1.0)

    # FAISS params (lexicon inspection)
    build_every = int(config.get("build_faiss_every", 3))
    faiss_top_k = int(config.get("faiss_top_k", 5))
    faiss_bs = int(config.get("faiss_batch_size", 4096))
    faiss_max_words = config.get("faiss_max_words")  # None или число

    # ----------------------------
    # TRAIN LOOP
    # ----------------------------
    best_correction_score = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mlm = 0.0
        total_rer = 0.0
        total_delta = 0.0
        total_margin_ok = 0.0
        n_mlm = 0
        n_rer = 0

        # для красивого примера MLM
        example_batch = None

        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}/{epochs}", ncols=110)
        for step in pbar:
            do_rerank = random.random() < p_rerank

            optimizer.zero_grad()
            src = "mlm"
            if do_rerank:
                if random.random() < config.get("p_ocr", 0.5):
                    batch = next(ocr_iter)
                    src = "ocr"
                else:
                    batch = next(rerank_iter)
                    src = "synt"
                pos = batch["pos"].to(device)
                neg = batch["neg"].to(device)

                # ОБЪЕДИНЯЕМ В ОДИН БАТЧ
                both = torch.cat([pos, neg], dim=0)  # [2B, T]

                scores = model.score(both)  # [2B]
                pos_score, neg_score = scores.chunk(2)  # два [B]

                loss = rerank_margin_loss(pos_score, neg_score, margin=rerank_margin)

                with torch.no_grad():
                    delta = (pos_score - neg_score).mean().item()
                    margin_ok = (
                        (pos_score > neg_score + rerank_margin).float().mean().item()
                    )

                total_delta += delta
                total_margin_ok += margin_ok

                total_rer += float(loss.item())
                n_rer += 1
            else:
                batch = next(mlm_iter)
                if example_batch is None:
                    example_batch = batch

                x = batch["x"].to(device)
                y = batch["y"].to(device)

                logits = model(x)
                loss = mlm_loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

                total_mlm += float(loss.item())
                n_mlm += 1

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(
                L=f"{total_loss/(step+1):.3f}",
                M=f"{(total_mlm/max(n_mlm,1)):.2f}",
                R=f"{(total_rer/max(n_rer,1)):.2f}",
                Δ=f"{(total_delta/max(n_rer,1)):.3f}",
                ok=f"{(total_margin_ok/max(n_rer,1)):.2f}",
                src=src,
            )

        print(
            f"\n[epoch {epoch}] "
            f"loss={total_loss/steps_per_epoch:.4f} | "
            f"mlm={total_mlm/max(n_mlm,1):.4f} (n={n_mlm}) | "
            f"rer={total_rer/max(n_rer,1):.4f} (n={n_rer}) | "
            f"Δ={total_delta/max(n_rer,1):.3f} | "
            f"ok={total_margin_ok/max(n_rer,1):.2f}"
        )
        print(
            f"[RERANK DATASET] unique (pos, neg) pairs = {len(rerank_dataset.neg_pairs)}"
        )

        # ----------------------------
        # EXAMPLE (INPUT / PRED / GOLD) - из MLM
        # ----------------------------
        if example_batch is not None:
            model.eval()
            bx = example_batch["x"]
            by = example_batch["y"]
            i = pick_better_example_idx(bx, by, min_visible=2)

            x_ex = bx[i].to(device)
            y_ex = by[i].to(device)

            inp, pred, gold = decode_example(model, vocab, x_ex, y_ex)

            print("\n[EXAMPLE MLM]")
            print(f"  IN : {inp}")
            print(f"  PRD: {pred}")
            print(f"  GT : {gold}")

        # ----------------------------
        # RERANK EXAMPLE
        # ----------------------------
        if n_rer > 0:
            # берём свежий батч reranker
            print("\n[RERANK EXAMPLE – synthetic]")
            log_rerank_example(model, vocab, rerank_dataset, device)

            print("\n[RERANK EXAMPLE – OCR]")
            log_ocr_rerank_example(model, vocab, ocr_dataset, device)

        # ----------------------------
        # FAISS INSPECTION (lexicon)
        # ----------------------------
        if epoch % build_every == 0:
            print("\n[FAISS] building index")

            words = mlm_dataset.words
            if faiss_max_words is not None:
                words = words[: int(faiss_max_words)]

            embs = build_word_embeddings(
                model, vocab, words, device, batch_size=faiss_bs
            )
            if embs.numel() > 0:
                index = build_faiss_index(embs)

                probes = random.sample(range(len(words)), 3)
                print("[FAISS] nearest neighbors")
                for idx in probes:
                    w = words[idx]
                    neigh = faiss_neighbors(index, embs, words, idx, top_k=faiss_top_k)
                    print(f"  '{w}'")
                    for nw, sim in neigh:
                        print(f"    {nw:<20} {sim:.4f}")

            # ----------------------------
            # CORRECTOR EVAL
            # ----------------------------
            corrector = FaissRerankCorrector(
                model=model,
                vocab=vocab,
                lexicon_words=words,
                device=device,
                max_len=config.get("max_len", 32),
                top_k=config.get("corrector_top_k", 10),
                sim_threshold=config.get("corrector_threshold", 0.1),
            )

            metrics = corrector.evaluate_on_pairs_csv("data/pairs.csv", max_items=1000)

            print(
                f"[CORRECTOR] "
                f"changed={metrics['changed']} | "
                f"improved={metrics['improved']} | "
                f"eff={metrics['efficiency']:.3f}"
            )

            if metrics["efficiency"] > best_correction_score:
                best_correction_score = metrics["efficiency"]

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                    exp_dir / "best_corrector.pt",
                )

                print(f"[CORRECTOR] new best = {best_correction_score:.3f}")

        # ----------------------------
        # SAVE
        # ----------------------------
        torch.save(
            {"model_state": model.state_dict(), "config": config},
            exp_dir / f"epoch_{epoch}.pt",
        )

    return model, vocab, exp_dir


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    cfg = {
        # paths
        "lexicon_path": "data/all_words.txt",
        "text_path": "data/extracted_texts_cleaned.txt",
        "charset_path": "data/charset.txt",
        # exp
        "exp_dir": "exp_multitask2",
        "device": "auto",
        "seed": 42,
        # model
        "max_len": 16,
        "emb_size": 192,
        "n_layers": 4,
        "n_heads": 6,
        "ffn_size": 768,
        "dropout": 0.1,
        # optimization
        "batch_size": 256,
        "epochs": 15,
        "steps_per_epoch": 3000,  # можно = len(mlm_loader) для “одной эпохи по лексикону”
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "num_workers": 2,
        # MLM
        "mask_prob": 0.9,
        "min_word_len": 4,
        # multitask
        "p_rerank": 0.3,
        "rerank_margin": 1.0,
        "rerank_window": 1,
        # FAISS
        "build_faiss_every": 1,
        "faiss_top_k": 5,
        "faiss_batch_size": 4096,
        "faiss_max_words": 1500000,
        "ocr_pairs_path": "data/pairs_with_errors.csv",
        "p_ocr": 0.5,
        "eval_pairs_path": "data/pairs.csv",
        "corrector_top_k": 10,
        "corrector_threshold": 0.1,
    }

    train(cfg)
