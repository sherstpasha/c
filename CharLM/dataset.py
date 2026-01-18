import random
import re
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


# ============================================================
# Utils
# ============================================================

WORD_RE = re.compile(r"[А-Яа-яѣѢёЁ]+")  # под твой корпус


def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text)


def corrupt_word(word: str) -> str:
    """
    Простая OCR-подобная порча слова (негатив).
    """
    if len(word) < 3:
        return word

    i = random.randrange(len(word))
    return word[:i] + random.choice(word) + word[i + 1 :]


def extract_suffixes(words, min_len=2, max_len=5, min_freq=50):
    from collections import Counter

    cnt = Counter()
    for w in words:
        for l in range(min_len, max_len + 1):
            if len(w) > l:
                cnt[w[-l:]] += 1
    return [s for s, c in cnt.items() if c >= min_freq]


# ============================================================
# MLM DATASET (как раньше)
# ============================================================


class WordMLMDataset(Dataset):
    def __init__(
        self,
        words_path: str,
        vocab,
        max_len: int = 32,
        mask_prob: float = 0.15,
        min_word_len: int = 1,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.mask_prob = mask_prob

        with open(words_path, encoding="utf-8") as f:
            words = [w.strip() for w in f if w.strip()]

        if min_word_len > 1:
            words = [w for w in words if len(w) >= min_word_len]

        self.words = words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        ids = self.vocab.encode(word)[: self.max_len]

        x = ids.copy()
        y = [-100] * len(ids)

        masked = False
        for i in range(len(x)):
            if random.random() < self.mask_prob:
                y[i] = x[i]
                x[i] = self.vocab.mask
                masked = True

        # гарантируем якорь
        if not masked:
            i = random.randrange(len(x))
            y[i] = x[i]
            x[i] = self.vocab.mask

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


import csv
from collections import defaultdict


def load_ocr_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inc = row["incorrect"].strip()
            cor = row["correct"].strip()
            if inc and cor and inc != cor:
                pairs.append((inc, cor))
    return pairs


class OCRPairsRerankerDataset(Dataset):
    """
    OCR-based reranker dataset.
    POS = correct word in real context
    NEG = OCR error word in same context
    """

    def __init__(
        self,
        pairs_csv: str,
        text_path: str,
        vocab,
        max_len: int = 64,
        window: int = 5,
        max_ctx_per_pair: int = 5,
    ):
        self.vocab = vocab
        self.max_len = max_len
        pad = vocab.pad

        # ----------------------------
        # 1. Load OCR pairs
        # ----------------------------
        self.pairs = load_ocr_pairs(pairs_csv)
        print(f"[OCR] loaded pairs = {len(self.pairs)}")

        # ----------------------------
        # 2. Load text and tokenize
        # ----------------------------

        with open(text_path, encoding="utf-8") as f:
            text = f.read()

        words = tokenize_words(text)

        # ----------------------------
        # 3. Build index: word -> positions
        # ----------------------------
        positions = defaultdict(list)
        for i, w in enumerate(words):
            positions[w].append(i)

        # ----------------------------
        # 4. Collect context samples
        # ----------------------------
        samples = []
        for inc, cor in self.pairs:
            if cor not in positions:
                continue
            for pos_idx in positions[cor][:max_ctx_per_pair]:
                samples.append((pos_idx, inc, cor))

        print(f"[OCR] context samples = {len(samples)}")

        # ----------------------------
        # 5. Precompute & pad tensors
        # ----------------------------
        self.pos_tensors = []
        self.neg_tensors = []
        self.meta = []  # для логов / примеров

        def encode(words_list):
            ids = vocab.encode(" ".join(words_list))[:max_len]
            out = torch.full((max_len,), pad, dtype=torch.long)
            out[: len(ids)] = torch.tensor(ids, dtype=torch.long)
            return out

        for pos_idx, inc, cor in samples:
            left = words[max(0, pos_idx - window) : pos_idx]
            right = words[pos_idx + 1 : pos_idx + 1 + window]

            self.pos_tensors.append(encode(left + [cor] + right))
            self.neg_tensors.append(encode(left + [inc] + right))

            self.meta.append(
                {
                    "left": " ".join(left),
                    "right": " ".join(right),
                    "pos": cor,
                    "neg": inc,
                }
            )

        print(f"[OCR] cached samples = {len(self.pos_tensors)}")

        assert len(self.pos_tensors) == len(self.neg_tensors)

    def __len__(self):
        return len(self.pos_tensors)

    def __getitem__(self, idx):
        return {
            "pos": self.pos_tensors[idx],
            "neg": self.neg_tensors[idx],
            "meta": self.meta[idx],  # можно не использовать в train
        }


class CollateMLMStageA:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        xs, ys = zip(*batch)
        max_len = max(x.size(0) for x in xs)

        bx = torch.full((len(xs), max_len), self.pad_idx, dtype=torch.long)
        by = torch.full((len(xs), max_len), -100, dtype=torch.long)

        for i, (x, y) in enumerate(zip(xs, ys)):
            bx[i, : x.size(0)] = x
            by[i, : y.size(0)] = y

        return {"x": bx, "y": by}


# ============================================================
# RERANKER DATASET
# ============================================================

MIN_LEN_FOR_RERANK = 3
MAX_NEG_TRIES = 5


class ContextRerankerDataset(Dataset):
    """
    Dataset для обучения reranker'а:
    (left context, candidate, right context) -> score
    """

    def __init__(
        self,
        text_path: str,
        vocab,
        max_len: int = 64,
        window: int = 5,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.window = window

        with open(text_path, encoding="utf-8") as f:
            text = f.read()

        words = tokenize_words(text)
        self.words = words
        self.suffixes = extract_suffixes(self.words)
        self.neg_pairs = set()

    def __len__(self):
        return len(self.words)

    def encode_sequence(self, words: List[str]) -> List[int]:
        text = " ".join(words)
        return self.vocab.encode(text)[: self.max_len]

    def get_raw(self, idx: int) -> dict:
        target = self.words[idx]

        if len(target) < MIN_LEN_FOR_RERANK:
            return self.get_raw((idx + 1) % len(self.words))

        left = self.words[max(0, idx - self.window) : idx]
        right = self.words[idx + 1 : idx + 1 + self.window]

        neg_word = None
        for _ in range(MAX_NEG_TRIES):
            neg_word = self.make_hard_negative(target)
            if neg_word is not None and neg_word != target:
                break

        if neg_word is None or neg_word == target:
            return self.get_raw((idx + 1) % len(self.words))

        pos_ids = self.encode_sequence(left + [target] + right)
        neg_ids = self.encode_sequence(left + [neg_word] + right)

        return {
            "left": " ".join(left),
            "right": " ".join(right),
            "pos_word": target,
            "neg_word": neg_word,
            "pos_ids": torch.tensor(pos_ids, dtype=torch.long),
            "neg_ids": torch.tensor(neg_ids, dtype=torch.long),
        }

    def make_hard_negative(self, word: str) -> str | None:
        if len(word) < MIN_LEN_FOR_RERANK:
            return None

        # 1) suffix-aware negatives (основные)
        for suf in sorted(self.suffixes, key=len, reverse=True):
            if word.endswith(suf) and len(word) > len(suf) + 1:
                base = word[: -len(suf)]
                for _ in range(3):
                    alt = random.choice(self.suffixes)
                    if alt != suf:
                        cand = base + alt
                        if cand != word:
                            return cand

        # 2) fallback — редкий OCR-шум
        cand = corrupt_word(word)
        if cand != word:
            return cand

        return None

    def __getitem__(self, idx):
        target = self.words[idx]

        # фильтр коротких
        if len(target) < MIN_LEN_FOR_RERANK:
            return self.__getitem__((idx + 1) % len(self.words))

        left = self.words[max(0, idx - self.window) : idx]
        right = self.words[idx + 1 : idx + 1 + self.window]

        # позитив
        pos_ids = self.encode_sequence(left + [target] + right)

        # негатив — ТОЛЬКО hard
        neg_word = None
        for _ in range(5):
            neg_word = self.make_hard_negative(target)
            if neg_word is not None and neg_word != target:
                break

        if neg_word is None or neg_word == target:
            return self.__getitem__((idx + 1) % len(self.words))

        self.neg_pairs.add((target, neg_word))
        neg_ids = self.encode_sequence(left + [neg_word] + right)

        return {
            "pos": torch.tensor(pos_ids, dtype=torch.long),
            "neg": torch.tensor(neg_ids, dtype=torch.long),
            "pos_word": target,
            "neg_word": neg_word,
            "left": " ".join(left),
            "right": " ".join(right),
        }


class CollateReranker:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        pos = [b["pos"] for b in batch]
        neg = [b["neg"] for b in batch]

        max_len = max(max(x.size(0), y.size(0)) for x, y in zip(pos, neg))

        def pad(xs):
            out = torch.full(
                (len(xs), max_len),
                self.pad_idx,
                dtype=torch.long,
            )
            for i, x in enumerate(xs):
                out[i, : x.size(0)] = x
            return out

        return {
            "pos": pad(pos),
            "neg": pad(neg),
        }
