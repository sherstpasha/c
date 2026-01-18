import torch
import torch.nn as nn


class EditVocab:
    """
    Edit operations vocabulary:
    COPY
    DELETE
    INSERT_x
    REPLACE_x
    """

    def __init__(self, char_vocab):
        self.ops = []
        self.op_to_id = {}
        self.id_to_op = {}

        self.COPY = self._add("COPY")
        self.DELETE = self._add("DELETE")
        self.char_vocab = char_vocab

        for ch, cid in char_vocab.token_to_id.items():
            if ch.startswith("<"):  # skip special tokens
                continue
            self._add(f"INSERT_{ch}")
            self._add(f"REPLACE_{ch}")

        self.size = len(self.ops)

    def _add(self, op):
        idx = len(self.ops)
        self.ops.append(op)
        self.op_to_id[op] = idx
        self.id_to_op[idx] = op
        return idx


class CharTransformerEdit(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        edit_vocab_size: int,
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
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # --- edit head ---
        self.edit_head = nn.Linear(emb_size, edit_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.char_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.char_emb.weight[self.pad_idx].zero_()

    def forward(self, x: torch.Tensor, edit_targets=None):
        """
        x: [B, T] input character ids
        edit_targets: [B, T] edit op ids (optional)
        """
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        h = self.char_emb(x) + self.pos_emb(pos_ids)
        h = self.emb_drop(h)

        h = self.encoder(
            h,
            src_key_padding_mask=(x == self.pad_idx),
        )

        logits = self.edit_head(h)  # [B, T, edit_vocab_size]

        # ---- COPY bias ----
        if edit_targets is not None:
            # Encourage COPY when no edit is needed
            copy_mask = edit_targets == -100  # same semantics as before

            # bias only COPY logit
            copy_bias = torch.zeros_like(logits[..., 0])
            copy_bias[copy_mask] = self.copy_strength

            logits[..., 0] += copy_bias  # COPY is always id=0

        return logits, h
