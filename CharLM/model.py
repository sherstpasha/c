import torch
import torch.nn as nn


class CharTransformerMLM(nn.Module):
    """
    Char-level Transformer Encoder с двумя головами:
    - MLM head (token-level)
    - Reranker head (sequence-level score)

    Encoder общий для обеих задач.
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
        self.max_len = max_len

        # ------------------------
        # Embeddings
        # ------------------------
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.pos = nn.Embedding(max_len, emb_size)
        self.emb_dropout = nn.Dropout(dropout)

        # ------------------------
        # Encoder
        # ------------------------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ------------------------
        # Heads
        # ------------------------
        # MLM head
        self.mlm_head = nn.Linear(emb_size, vocab_size)

        # Reranker head (sequence-level)
        self.rerank_head = nn.Linear(emb_size, 1)

    # =========================================================
    # Core encoder
    # =========================================================

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T] — индексы символов
        return: [B, T, D] — скрытые представления
        """
        B, T = x.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len={self.max_len}")

        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        h = self.emb(x) + self.pos(pos_ids)
        h = self.emb_dropout(h)

        h = self.encoder(
            h,
            src_key_padding_mask=(x == self.pad_idx),
        )
        return h

    # =========================================================
    # MLM
    # =========================================================

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLM forward

        return: [B, T, vocab_size]
        """
        h = self.encode(x)
        return self.mlm_head(h)

    # =========================================================
    # Reranker
    # =========================================================

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T]
        return: [B]
        """
        h = self.encode(x)  # [B, T, D]

        # берём embedding ПЕРВОГО НЕ pad токена слова
        mask = (x != self.pad_idx).float()  # [B, T]
        lengths = mask.sum(dim=1).long() - 1
        lengths = lengths.clamp(min=0)

        word_emb = h[torch.arange(h.size(0)), lengths]  # [B, D]
        return self.rerank_head(word_emb).squeeze(-1)
