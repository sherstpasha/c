import torch

from CharTransformerMLM.dataset import CharOCRDenoiseDataset
from CharTransformerMLM.vocab import CharVocab


from CharTransformerMLM.utils.collate import collate_denoise


def decode(vocab, ids):
    return "".join(vocab.id_to_token[i] for i in ids)


def decode_target(vocab, x, y):

    out = []
    for xi, yi in zip(x, y):
        if yi == -100:
            out.append("·")
        else:
            out.append(vocab.id_to_token[yi])
    return "".join(out)


def main():
    vocab = CharVocab("data/charset.txt")

    print("=== VOCAB INFO ===")
    print("vocab size:", len(vocab.token_to_id))
    print("PAD:", vocab.pad, "EOW:", vocab.eow)
    print()

    ds = CharOCRDenoiseDataset(
        text_path="data/extracted_texts_cleaned.txt",
        pairs_csv_path="data/pairs_with_errors.csv",
        vocab=vocab,
        max_len=128,
        max_words=5,
        noise_prob=0.15,
        p_real=0.4,
    )

    print("=== SINGLE SAMPLES ===")

    for i in range(3):
        x, y = ds[i]

        print(f"\n--- Sample {i} ---")
        print("Noisy x:   ", decode(vocab, x.tolist()))
        print("Target y:  ", decode_target(vocab, x.tolist(), y.tolist()))

        print("<EOW> count:", (x == vocab.eow).sum().item())
        assert (x == vocab.eow).sum() >= 1, "EOW not found!"

    print("\n=== COLLATE CHECK ===")

    batch = [ds[i] for i in range(4)]
    out = collate_denoise(batch, vocab.pad)

    x = out["x"]
    y = out["y"]
    mask = out["attn_mask"]

    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("attn_mask shape:", mask.shape)

    print("\nPadded batch decoded:")

    for i in range(x.size(0)):
        print(f"[{i}]", decode(vocab, x[i].tolist()))

    assert x.shape == y.shape
    assert mask.shape == x.shape
    assert (x == vocab.pad).sum() >= 0
    assert (y == -100).sum() >= 0

    print("\n✓✓✓ DATASET CHECK PASSED ✓✓✓")


if __name__ == "__main__":
    main()
