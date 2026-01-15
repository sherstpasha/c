"""
CharLM - Character-level Masked Language Model для OCR-коррекции.

Трёхстадийное обучение:
- Stage A: Lexicon MLM (span-masked MLM на 200-300k словах)
- Stage B: Context MLM (окна 1-3 слова, OCR-шум только в контексте)
- Stage C: Contrastive Learning (InfoNCE на OCR-парах)

Использование:
    from CharLM import train, DEFAULT_CONFIG, CharTransformerMLM

    # С дефолтной конфигурацией
    model, vocab, exp_dir = train()

    # С кастомной конфигурацией
    config = {"exp_dir": "exp1", "epochs_a": 8}
    model, vocab, exp_dir = train(config)
"""

from .config import DEFAULT_CONFIG
from .model import CharTransformerMLM
from .train import train, LexiconMLMDataset, ContextMLMDataset, ContrastiveDataset
from .utils import (
    build_vocab,
    encode_str,
    Logger,
    load_charset,
    tokenize_by_charset,
    add_ocr_noise,
    masked_accuracy,
    filter_words,
    is_valid_word,
    clean_word,
    load_allowed_chars,
    get_allowed_chars,
)

__all__ = [
    "DEFAULT_CONFIG",
    "CharTransformerMLM",
    "train",
    "LexiconMLMDataset",
    "ContextMLMDataset",
    "ContrastiveDataset",
    "build_vocab",
    "encode_str",
    "Logger",
    "load_charset",
    "tokenize_by_charset",
    "add_ocr_noise",
    "masked_accuracy",
    "filter_words",
    "is_valid_word",
    "clean_word",
    "load_allowed_chars",
    "get_allowed_chars",
]
