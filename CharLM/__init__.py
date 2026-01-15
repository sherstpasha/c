"""
CharLM - Character-level Embedding Model for OCR Correction.

Architecture:
- Shared Transformer Encoder (Pre-LN, GELU, learnable positions)
- Embedding Head (main): mean pooling → Linear → LayerNorm → L2 normalize
- MLM Head (aux): Linear projection for regularization

Three-stage training:
- Stage A: Lexicon embedding pretraining (metric + MLM)
- Stage B: Context-aware embedding (windows 1-3 words)
- Stage C: Supervised OCR pairs with hard negative mining

Usage:
    from CharLM import train, CharLM, DEFAULT_CONFIG

    # Train with default config
    model, vocab, exp_dir = train()

    # Train with custom config
    config = {"exp_dir": "exp1", "epochs_a": 20}
    model, vocab, exp_dir = train(config)

    # Use model for encoding
    embeddings = model.encode_words(input_ids)  # [B, D] L2-normalized
"""

from .config import DEFAULT_CONFIG
from .model import CharLM, CharLMEncoder, EmbeddingHead, MLMHead
from .train import train, LexiconDatasetFast, ContextDataset, OCRPairsDataset
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
    compute_metric_loss,
    infonce_loss,
    infonce_loss_inbatch,
    triplet_loss,
    mine_hard_negatives,
    masked_accuracy,
    masked_topk_accuracy,
    compute_recall_at_k,
)

__all__ = [
    # Config
    "DEFAULT_CONFIG",
    # Model
    "CharLM",
    "CharLMEncoder",
    "EmbeddingHead",
    "MLMHead",
    # Training
    "train",
    "LexiconDatasetFast",
    "ContextDataset",
    "OCRPairsDataset",
    # Utils
    "Logger",
    "build_vocab",
    "encode_str",
    "encode_batch",
    "load_charset",
    "tokenize_by_charset",
    "apply_ocr_noise",
    "create_corrupted_version",
    "apply_mlm_mask",
    "compute_metric_loss",
    "infonce_loss",
    "infonce_loss_inbatch",
    "triplet_loss",
    "mine_hard_negatives",
    "masked_accuracy",
    "masked_topk_accuracy",
    "compute_recall_at_k",
]
