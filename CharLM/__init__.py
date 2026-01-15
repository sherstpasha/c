"""
CharLM - Character-level Masked Language Model для OCR-коррекции.

Трёхстадийное обучение:
- Stage A: pretrain на лексиконе (отдельные слова)
- Stage B: finetune на контексте (окна 1/2/3 слова) с OCR-шумом
- Stage C: finetune на парах (incorrect → correct), маска только на diff

Использование:
    from CharLM import train, DEFAULT_CONFIG, CharTransformerMLM
    
    # С дефолтной конфигурацией
    model, vocab, exp_dir = train()
    
    # С кастомной конфигурацией
    config = {"exp_dir": "exp1", "epochs_a": 10}
    model, vocab, exp_dir = train(config)
"""

from .config import DEFAULT_CONFIG
from .model import CharTransformerMLM
from .train import train, LexiconMLMDataset, ContextMLMDataset, PairsMLMDataset
from .utils import (
    build_vocab, encode_str, Logger, load_charset, 
    tokenize_by_charset, add_ocr_noise, find_diff_positions
)

__all__ = [
    "DEFAULT_CONFIG",
    "CharTransformerMLM", 
    "train",
    "LexiconMLMDataset",
    "ContextMLMDataset",
    "PairsMLMDataset",
    "build_vocab",
    "encode_str",
    "Logger",
    "load_charset",
    "tokenize_by_charset",
    "add_ocr_noise",
    "find_diff_positions",
]
