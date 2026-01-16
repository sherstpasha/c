import torch
import torch.nn as nn


class CharTransformerMLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_size: int = 160,
        max_len: int = 128,
        n_layers: int = 4,
        n_heads: int = 5,
        ffn_size: int = 640,
        dropout: float = 0.1,
        pad_idx: int = 0,
        eow_idx: int = 1,
        copy_strength: float = 6.0,
    ):
        super().__init__()

        self.pad_idx = pad_idx
        self.eow_idx = eow_idx
        self.emb_size = emb_size
        self.copy_strength = copy_strength

        # --- embeddings ---
        self.char_emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, emb_size)
        self.emb_drop = nn.Dropout(dropout)

        # --- transformer encoder ---
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=n_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # --- correction head (Δ) ---
        self.char_head = nn.Linear(emb_size, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.char_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        # Обнулить эмбеддинг для pad
        with torch.no_grad():
            self.char_emb.weight[self.pad_idx].zero_()

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        h = self.char_emb(x) + self.pos_emb(pos_ids)
        h = self.emb_drop(h)

        h = self.encoder(
            h,
            src_key_padding_mask=(x == self.pad_idx),
        )

        logits = self.char_head(h)

        if y is not None:
            copy_mask = y == -100  # [B, T]
            copy_bias = torch.zeros_like(logits)

            copy_bias.scatter_(
                -1,
                x.unsqueeze(-1),
                1.0,
            )

            logits = logits + self.copy_strength * copy_bias * copy_mask.unsqueeze(-1)

        return logits, h

    @torch.no_grad()
    def extract_word_embeddings(self, hidden: torch.Tensor, x: torch.Tensor):
        batch_embs = []

        for i in range(x.size(0)):
            eow_pos = (x[i] == self.eow_idx).nonzero(as_tuple=False)
            if eow_pos.numel() == 0:
                batch_embs.append(torch.empty(0, hidden.size(-1), device=hidden.device))
                continue

            batch_embs.append(hidden[i, eow_pos.squeeze(1)])

        return batch_embs
