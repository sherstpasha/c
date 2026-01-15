"""
CharLM утилиты: кодирование, аугментация, метрики, логирование.
"""

import random
import torch
from datetime import datetime


class Logger:
    """Простой логгер в файл и консоль."""

    def __init__(self, path: str = None):
        self.path = path
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"=== Training started: {datetime.now()} ===\n")

    def log(self, msg: str):
        print(msg)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


def load_charset(path: str) -> set[str]:
    """Загрузить набор разрешённых символов из файла."""
    charset = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if line and not line.startswith("<"):
                charset.add(line)
    return charset


def tokenize_by_charset(text: str, charset: set[str]) -> list[str]:
    """Токенизировать текст по разрешённым символам."""
    tokens = []
    current = []
    for ch in text:
        if ch in charset:
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


def build_vocab(
    words: list[str], include_space: bool = True
) -> tuple[dict, dict, list]:
    """Построить словарь символов из списка слов."""
    chars = set()
    for w in words:
        chars.update(w)
    if include_space:
        chars.add(" ")

    chars = ["<PAD>", "<MASK>", "<UNK>"] + sorted(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c, chars


def encode_str(s: str, c2i: dict, max_len: int) -> list[int]:
    """Закодировать строку в список индексов с паддингом."""
    unk = c2i["<UNK>"]
    pad = c2i["<PAD>"]
    ids = [c2i.get(ch, unk) for ch in s[:max_len]]
    return ids + [pad] * (max_len - len(ids))


def encode_batch(strings: list[str], c2i: dict, max_len: int) -> torch.Tensor:
    """Закодировать батч строк в тензор."""
    batch = [encode_str(s, c2i, max_len) for s in strings]
    return torch.tensor(batch, dtype=torch.long)


# ============================================================
# OCR AUGMENTATION
# ============================================================


def apply_ocr_noise(word: str, cfg: dict) -> str:
    """
    Применить OCR-подобный шум к слову.

    Аугментации:
    - insert: вставить случайный символ
    - delete: удалить случайный символ
    - substitute: заменить на визуально похожий
    - swap: поменять соседние символы местами
    - space_insert: вставить пробел
    - duplicate: продублировать символ
    """
    if len(word) < 2:
        return word

    result = list(word)
    confusables = cfg.get("confusables", {})

    # Character substitution (visually similar)
    if random.random() < cfg.get("p_char_substitute", 0.12):
        candidates = [(i, c) for i, c in enumerate(result) if c in confusables]
        if candidates:
            i, c = random.choice(candidates)
            result[i] = random.choice(confusables[c])

    # Character deletion
    if random.random() < cfg.get("p_char_delete", 0.08) and len(result) > 2:
        i = random.randint(0, len(result) - 1)
        result.pop(i)

    # Character insertion
    if random.random() < cfg.get("p_char_insert", 0.08):
        i = random.randint(0, len(result))
        # Insert a random character from the word
        c = random.choice(word)
        result.insert(i, c)

    # Adjacent swap
    if random.random() < cfg.get("p_char_swap", 0.05) and len(result) > 1:
        i = random.randint(0, len(result) - 2)
        result[i], result[i + 1] = result[i + 1], result[i]

    # Space insertion
    if random.random() < cfg.get("p_space_insert", 0.05) and len(result) > 2:
        i = random.randint(1, len(result) - 1)
        result.insert(i, " ")

    # Character duplication
    if random.random() < cfg.get("p_duplicate", 0.03):
        i = random.randint(0, len(result) - 1)
        result.insert(i, result[i])

    return "".join(result)


def create_corrupted_version(
    word: str, cfg: dict, min_edits: int = 1, max_edits: int = 3
) -> str:
    """Создать зашумлённую версию слова с гарантированными изменениями."""
    result = word
    n_edits = random.randint(min_edits, max_edits)

    for _ in range(n_edits):
        result = apply_ocr_noise(result, cfg)

    # Ensure at least some change
    if result == word and len(word) > 2:
        # Force at least one edit
        i = random.randint(0, len(word) - 1)
        chars = list(result)
        confusables = cfg.get("confusables", {})
        if word[i] in confusables:
            chars[i] = random.choice(confusables[word[i]])
        else:
            chars.pop(i) if len(chars) > 2 else None
        result = "".join(chars)

    return result


# ============================================================
# MLM MASKING
# ============================================================


def apply_mlm_mask(
    ids: list[int],
    c2i: dict,
    mask_prob: float = 0.15,
    vocab_size: int = None,
) -> tuple[list[int], list[int]]:
    """
    Применить MLM маскирование.

    Returns:
        masked_ids: входные id с маской
        labels: целевые id (-100 для немаскированных)
    """
    mask_id = c2i["<MASK>"]
    pad_id = c2i["<PAD>"]

    masked = ids.copy()
    labels = [-100] * len(ids)

    for i, tok in enumerate(ids):
        if tok == pad_id:
            continue
        if random.random() < mask_prob:
            labels[i] = tok
            r = random.random()
            if r < 0.8:
                masked[i] = mask_id
            elif r < 0.9:
                # Random token
                masked[i] = random.randint(3, vocab_size - 1) if vocab_size else tok
            # else keep original (10%)

    return masked, labels


# ============================================================
# METRIC LEARNING LOSSES
# ============================================================


def infonce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss.

    Args:
        anchor: [B, D] anchor embeddings
        positive: [B, D] positive embeddings
        negatives: [B, K, D] or [K, D] negative embeddings
        temperature: softmax temperature

    Returns:
        loss: scalar
    """
    # Positive similarity
    pos_sim = (anchor * positive).sum(dim=-1) / temperature  # [B]

    # Negative similarity
    if negatives.dim() == 2:
        # Shared negatives [K, D]
        neg_sim = torch.mm(anchor, negatives.T) / temperature  # [B, K]
    else:
        # Per-sample negatives [B, K, D]
        neg_sim = (
            torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / temperature
        )  # [B, K]

    # Concat positive and negatives
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [B, 1+K]

    # Target is always 0 (positive is first)
    targets = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)

    return torch.nn.functional.cross_entropy(logits, targets)


def infonce_loss_inbatch(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss with in-batch negatives.

    Uses all other positives in the batch as negatives.
    Much faster than explicit negative sampling (only 2 encoder forwards per batch).

    Args:
        anchor: [B, D] anchor embeddings (L2 normalized)
        positive: [B, D] positive embeddings (L2 normalized)
        temperature: softmax temperature

    Returns:
        loss: scalar
    """
    # Similarity matrix [B, B] - row i, col j = sim(anchor_i, positive_j)
    # Diagonal entries are positive pairs, off-diagonal are negatives
    sim = torch.mm(anchor, positive.T) / temperature  # [B, B]

    # Labels: for each row, the positive is on the diagonal
    labels = torch.arange(anchor.size(0), device=anchor.device)

    return torch.nn.functional.cross_entropy(sim, labels)


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Triplet margin loss.

    Args:
        anchor: [B, D] anchor embeddings
        positive: [B, D] positive embeddings
        negative: [B, D] negative embeddings (hardest)
        margin: triplet margin

    Returns:
        loss: scalar
    """
    pos_dist = (anchor - positive).pow(2).sum(dim=-1)  # [B]
    neg_dist = (anchor - negative).pow(2).sum(dim=-1)  # [B]

    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()


def compute_metric_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    loss_type: str = "infonce",
    temperature: float = 0.07,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Compute metric learning loss.

    Args:
        anchor: [B, D] anchor embeddings
        positive: [B, D] positive embeddings
        negatives: [B, K, D] or [K, D] negative embeddings
        loss_type: "infonce" or "triplet"
        temperature: for InfoNCE
        margin: for triplet

    Returns:
        loss: scalar
    """
    if loss_type == "infonce":
        return infonce_loss(anchor, positive, negatives, temperature)
    elif loss_type == "triplet":
        # For triplet, use hardest negative
        if negatives.dim() == 3:
            # [B, K, D] -> find hardest per sample
            neg_dist = (anchor.unsqueeze(1) - negatives).pow(2).sum(dim=-1)  # [B, K]
            hardest_idx = neg_dist.argmin(dim=-1)  # [B]
            hardest_neg = negatives[torch.arange(anchor.size(0)), hardest_idx]  # [B, D]
        else:
            # Shared negatives - find hardest for each anchor
            neg_dist = torch.cdist(anchor, negatives)  # [B, K]
            hardest_idx = neg_dist.argmin(dim=-1)  # [B]
            hardest_neg = negatives[hardest_idx]  # [B, D]
        return triplet_loss(anchor, positive, hardest_neg, margin)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def mine_hard_negatives(
    anchor_emb: torch.Tensor,
    all_emb: torch.Tensor,
    positive_idx: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """
    Mine hard negatives from a pool of embeddings.

    Args:
        anchor_emb: [B, D] anchor embeddings
        all_emb: [N, D] pool of all embeddings
        positive_idx: [B] indices of positives in all_emb
        k: number of hard negatives per anchor

    Returns:
        hard_negatives: [B, K, D] hard negative embeddings
    """
    B, D = anchor_emb.shape
    N = all_emb.size(0)
    device = anchor_emb.device

    # Compute similarities
    sim = torch.mm(anchor_emb, all_emb.T)  # [B, N]

    # Mask out positives
    mask = torch.zeros(B, N, device=device, dtype=torch.bool)
    mask[torch.arange(B), positive_idx] = True
    sim[mask] = -float("inf")

    # Get top-k most similar (hardest negatives)
    _, hard_idx = sim.topk(k, dim=-1)  # [B, K]

    # Gather embeddings
    hard_negatives = all_emb[hard_idx]  # [B, K, D]

    return hard_negatives


# ============================================================
# METRICS
# ============================================================


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy on masked positions."""
    with torch.no_grad():
        mask = targets != -100
        if mask.sum().item() == 0:
            return 0.0
        preds = logits.argmax(dim=-1)
        return (preds[mask] == targets[mask]).float().mean().item()


def masked_topk_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 5
) -> float:
    """Compute top-k accuracy on masked positions."""
    with torch.no_grad():
        mask = targets != -100
        if mask.sum().item() == 0:
            return 0.0
        topk = logits.topk(k, dim=-1).indices
        hit = (topk == targets.unsqueeze(-1)).any(dim=-1)
        return hit[mask].float().mean().item()


def compute_recall_at_k(
    query_emb: torch.Tensor,
    key_emb: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Compute Recall@K for retrieval.

    Args:
        query_emb: [B, D] query embeddings
        key_emb: [N, D] key embeddings
        labels: [B] indices of correct keys for each query
        k: top-k

    Returns:
        recall: float
    """
    with torch.no_grad():
        sim = torch.mm(query_emb, key_emb.T)  # [B, N]
        _, topk_idx = sim.topk(k, dim=-1)  # [B, K]
        hits = (topk_idx == labels.unsqueeze(-1)).any(dim=-1)  # [B]
        return hits.float().mean().item()
