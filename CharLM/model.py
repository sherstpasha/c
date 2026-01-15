"""
CharLM модель: Transformer с двумя головами для OCR-коррекции.

Архитектура:
- Shared Transformer Encoder (Pre-LN, GELU, learnable positional embeddings)
- Embedding Head (main): mean pooling → Linear → LayerNorm → L2 normalize
- MLM Head (aux): Linear projection to vocab

API:
- forward_mlm(x) -> logits [B, T, V]
- encode_words(x, mask) -> embeddings [B, D] (L2-normalized)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharLMEncoder(nn.Module):
    """
    Character-level Transformer Encoder.

    Pre-LN architecture with GELU activation and learnable positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_size: int = 256,
        max_len: int = 64,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_size: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb_size = emb_size

        self.char_emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, emb_size)
        self.emb_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] input character indices
        Returns:
            hidden: [B, T, D] encoder hidden states
        """
        B, T = x.shape
        device = x.device

        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.char_emb(x) + self.pos_emb(pos_ids)
        h = self.emb_dropout(h)

        pad_mask = x == self.pad_idx
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.final_norm(h)

        return h

    def get_pad_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Return boolean mask where True = PAD position."""
        return x == self.pad_idx


class EmbeddingHead(nn.Module):
    """
    Embedding Head for metric learning.

    Mean pooling over non-PAD positions → Linear → LayerNorm → L2 normalize.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, hidden: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, T, D] encoder hidden states
            pad_mask: [B, T] boolean mask where True = PAD
        Returns:
            embeddings: [B, output_dim] L2-normalized embeddings
        """
        # Mean pooling over non-PAD positions
        valid_mask = ~pad_mask  # [B, T]
        valid_mask = valid_mask.unsqueeze(-1).float()  # [B, T, 1]

        # Sum and normalize
        sum_hidden = (hidden * valid_mask).sum(dim=1)  # [B, D]
        count = valid_mask.sum(dim=1).clamp(min=1)  # [B, 1]
        pooled = sum_hidden / count  # [B, D]

        # Project and normalize
        out = self.proj(pooled)
        out = self.norm(out)
        out = F.normalize(out, p=2, dim=-1)

        return out


class MLMHead(nn.Module):
    """
    MLM Head for auxiliary masked language modeling.

    Simple linear projection to vocabulary.
    """

    def __init__(self, input_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: [B, T, D] encoder hidden states
        Returns:
            logits: [B, T, V] vocabulary logits
        """
        return self.proj(hidden)


class CharLM(nn.Module):
    """
    Character-level Language Model for OCR correction with metric learning.

    Two-head architecture:
    1. Embedding Head (main): produces L2-normalized word embeddings
    2. MLM Head (aux): masked language modeling for regularization
    """

    def __init__(
        self,
        vocab_size: int,
        emb_size: int = 256,
        embed_dim: int = 256,
        max_len: int = 64,
        n_layers: int = 6,
        n_heads: int = 8,
        ffn_size: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx

        # Shared encoder
        self.encoder = CharLMEncoder(
            vocab_size=vocab_size,
            emb_size=emb_size,
            max_len=max_len,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_size=ffn_size,
            dropout=dropout,
            pad_idx=pad_idx,
        )

        # Two heads
        self.embed_head = EmbeddingHead(emb_size, embed_dim)
        self.mlm_head = MLMHead(emb_size, vocab_size)

    def forward_mlm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLM.

        Args:
            x: [B, T] input character indices
        Returns:
            logits: [B, T, V] vocabulary logits
        """
        hidden = self.encoder(x)
        return self.mlm_head(hidden)

    def encode_words(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode words to embeddings.

        Args:
            x: [B, T] input character indices
        Returns:
            embeddings: [B, embed_dim] L2-normalized embeddings
        """
        hidden = self.encoder(x)
        pad_mask = self.encoder.get_pad_mask(x)
        return self.embed_head(hidden, pad_mask)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass returning both outputs.

        Args:
            x: [B, T] input character indices
        Returns:
            embeddings: [B, embed_dim] L2-normalized embeddings
            logits: [B, T, V] vocabulary logits
        """
        hidden = self.encoder(x)
        pad_mask = self.encoder.get_pad_mask(x)

        embeddings = self.embed_head(hidden, pad_mask)
        logits = self.mlm_head(hidden)

        return embeddings, logits

    def get_encoder_params(self):
        """Get encoder parameters (for separate LR)."""
        return self.encoder.parameters()

    def get_head_params(self):
        """Get head parameters (for separate LR)."""
        return list(self.embed_head.parameters()) + list(self.mlm_head.parameters())

    def get_embed_head_params(self):
        """Get embedding head parameters only."""
        return self.embed_head.parameters()

    def get_mlm_head_params(self):
        """Get MLM head parameters only."""
        return self.mlm_head.parameters()
