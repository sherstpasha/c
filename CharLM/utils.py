"""Утилиты CharLM: кодирование, маскирование, метрики."""

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
    """Загрузить набор символов из файла."""
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
    """Построить словарь символов."""
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


def choose_spans(
    L: int, span_min: int, span_max: int, num_spans_min: int, num_spans_max: int
) -> list[int]:
    """Выбрать позиции для span masking (избегая краёв)."""
    if L <= 3:
        return []
    n_spans = random.randint(num_spans_min, num_spans_max)
    positions = set()
    for _ in range(n_spans):
        span_len = random.randint(span_min, span_max)
        start_max = min(L - 2, L - 1 - span_len)
        if start_max < 1:
            continue
        start = random.randint(1, start_max)
        for p in range(start, start + span_len):
            if 1 <= p <= L - 2:
                positions.add(p)
    return sorted(positions)


# OCR confusables
CONFUSABLES = {
    "о": "а",
    "а": "о",
    "е": "ё",
    "ё": "е",
    "и": "й",
    "й": "и",
    "п": "г",
    "г": "п",
    "н": "м",
    "м": "н",
    "ь": "ъ",
    "ъ": "ь",
    "ѣ": "е",
    "і": "и",
}


def add_ocr_noise(text: str, cfg: dict) -> str:
    """Добавить OCR-шум (1 операция на вызов)."""
    if len(text) < 2:
        return text

    r = random.random()
    cumulative = 0

    cumulative += cfg.get("p_swap", 0.05)
    if r < cumulative:
        candidates = [
            (i, CONFUSABLES[c]) for i, c in enumerate(text) if c in CONFUSABLES
        ]
        if candidates:
            i, new_c = random.choice(candidates)
            return text[:i] + new_c + text[i + 1 :]
        return text

    cumulative += cfg.get("p_delete", 0.03)
    if r < cumulative:
        i = random.randint(0, len(text) - 1)
        return text[:i] + text[i + 1 :]

    cumulative += cfg.get("p_insert_space", 0.03)
    if r < cumulative:
        i = random.randint(1, len(text) - 1)
        return text[:i] + " " + text[i:]

    cumulative += cfg.get("p_duplicate", 0.01)
    if r < cumulative:
        i = random.randint(0, len(text) - 1)
        return text[:i] + text[i] + text[i:]

    return text


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Accuracy по маскированным позициям."""
    with torch.no_grad():
        mask = targets != -100
        if mask.sum().item() == 0:
            return 0.0
        preds = logits.argmax(dim=-1)
        return (preds[mask] == targets[mask]).float().mean().item()


def load_allowed_chars(charset_path: str) -> set[str]:
    """Загрузить разрешённые символы из charset.txt (только буквы)."""
    allowed = set()
    with open(charset_path, encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n\r")
            if len(ch) == 1 and ch.isalpha():
                allowed.add(ch)
    return allowed


# Глобальная переменная - загружается при первом использовании
_ALLOWED_CHARS = None


def get_allowed_chars(charset_path: str = "charset.txt") -> set[str]:
    """Получить разрешённые символы (с кэшированием)."""
    global _ALLOWED_CHARS
    if _ALLOWED_CHARS is None:
        _ALLOWED_CHARS = load_allowed_chars(charset_path)
    return _ALLOWED_CHARS


def is_valid_word(word: str, allowed_chars: set[str] = None) -> bool:
    """Проверить, что слово состоит только из разрешённых букв."""
    if allowed_chars is None:
        allowed_chars = get_allowed_chars()
    return all(ch in allowed_chars for ch in word)


def clean_word(word: str, allowed_chars: set[str] = None) -> str:
    """Оставить только разрешённые символы."""
    if allowed_chars is None:
        allowed_chars = get_allowed_chars()
    return "".join(ch for ch in word if ch in allowed_chars)


def filter_words(
    words: list[str], min_len: int = 1, allowed_chars: set[str] = None
) -> list[str]:
    """Фильтрация слов: только валидные слова из букв."""
    if allowed_chars is None:
        allowed_chars = get_allowed_chars()
    result = []
    for w in words:
        cleaned = "".join(ch for ch in w if ch in allowed_chars)
        if len(cleaned) >= min_len:
            result.append(cleaned)
    return result
