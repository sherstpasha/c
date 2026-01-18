import torch

from CharTransformerMLM.dataset import CharOCREditDataset
from CharTransformerMLM.vocab import CharVocab
from CharTransformerMLM.model import EditVocab

from CharTransformerMLM.utils.collate import collate_edit


# -------------------------
# Decode helpers
# -------------------------


def decode_chars(vocab, ids):
    return "".join(vocab.id_to_token[i] for i in ids if i in vocab.id_to_token)


def decode_ops(edit_vocab, op_ids):
    out = []
    for oi in op_ids:
        if oi == -100:
            out.append("·")
        else:
            out.append(edit_vocab.id_to_op[oi])
    return " | ".join(out)


def apply_edit_ops(vocab, edit_vocab, x_ids, op_ids):
    """
    Apply edit operations to noisy input to reconstruct target string.
    """
    out = []

    for xi, oi in zip(x_ids, op_ids):
        ch = vocab.id_to_token.get(xi, "")

        if oi == -100 or edit_vocab.id_to_op[oi] == "COPY":
            out.append(ch)

        elif edit_vocab.id_to_op[oi] == "DELETE":
            continue

        elif edit_vocab.id_to_op[oi].startswith("REPLACE_"):
            out.append(edit_vocab.id_to_op[oi].replace("REPLACE_", ""))

        elif edit_vocab.id_to_op[oi].startswith("INSERT_"):
            ins = edit_vocab.id_to_op[oi].replace("INSERT_", "")
            out.append(ins)
            out.append(ch)

    return "".join(out)


# -------------------------
# Main check
# -------------------------


def main():
    vocab = CharVocab("data/charset.txt")
    edit_vocab = EditVocab(vocab)

    print("=== VOCAB INFO ===")
    print("char vocab size:", len(vocab.token_to_id))
    print("edit vocab size:", edit_vocab.size)
    print("PAD:", vocab.pad, "EOW:", vocab.eow)
    print()

    ds = CharOCREditDataset(
        text_path="data/extracted_texts_cleaned.txt",
        pairs_csv_path="data/pairs_with_errors.csv",
        vocab=vocab,
        edit_vocab=edit_vocab,
        max_len=128,
        max_words=5,
        noise_prob=0.15,
        p_real=0.4,
    )

    print("=== SINGLE SAMPLES ===")

    for i in range(3):
        x, y = ds[i]

        x_ids = x.tolist()
        y_ids = y.tolist()

        print(f"\n--- Sample {i} ---")
        print("Noisy x:    ", decode_chars(vocab, x_ids))
        print("Edit ops:   ", decode_ops(edit_vocab, y_ids))
        print("Reconstructed:", apply_edit_ops(vocab, edit_vocab, x_ids, y_ids))

        print("<EOW> count:", (x == vocab.eow).sum().item())
        assert (x == vocab.eow).sum() >= 1, "EOW not found!"

    print("\n=== COLLATE CHECK ===")

    batch = [ds[i] for i in range(4)]
    out = collate_edit(batch, vocab.pad)

    x = out["x"]
    y = out["y"]
    mask = out["attn_mask"]

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("attn_mask shape:", mask.shape)

    print("\nPadded batch decoded:")

    for i in range(x.size(0)):
        print(f"[{i}]", decode_chars(vocab, x[i].tolist()))

    assert x.shape == y.shape
    assert mask.shape == x.shape
    assert (x == vocab.pad).sum() >= 0
    assert (y == -100).sum() >= 0

    print("\n✓✓✓ EDIT DATASET CHECK PASSED ✓✓✓")


if __name__ == "__main__":
    main()
