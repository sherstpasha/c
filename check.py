import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CharLM.dataset import (
    WordMLMDataset,
    CollateMLMStageA,
    ContextRerankerDataset,
    CollateReranker,
)
from CharLM.model import CharTransformerMLM
from CharLM.vocab import CharVocab


def main():
    # =========================
    # PATHS
    # =========================
    WORDS_PATH = "data/all_words.txt"
    TEXT_PATH = "data/extracted_texts_cleaned.txt"
    CHARSET_PATH = "data/charset.txt"

    # =========================
    # PARAMS
    # =========================
    BATCH_SIZE = 8
    MAX_LEN = 32
    MASK_PROB = 0.3
    MIN_WORD_LEN = 4

    EMB_SIZE = 128
    N_LAYERS = 2
    N_HEADS = 4
    FFN_SIZE = 256

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[check] device = {DEVICE}")

    # =========================
    # VOCAB
    # =========================
    vocab = CharVocab(CHARSET_PATH)
    print(f"[check] vocab size = {len(vocab)}")

    # =========================================================
    # MLM CHECK
    # =========================================================
    print("\n[check] MLM pipeline")

    mlm_dataset = WordMLMDataset(
        words_path=WORDS_PATH,
        vocab=vocab,
        max_len=MAX_LEN,
        mask_prob=MASK_PROB,
        min_word_len=MIN_WORD_LEN,
    )

    mlm_loader = DataLoader(
        mlm_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=CollateMLMStageA(vocab.pad),
    )

    mlm_batch = next(iter(mlm_loader))
    x = mlm_batch["x"].to(DEVICE)
    y = mlm_batch["y"].to(DEVICE)

    print(f"[check] MLM batch x shape = {x.shape}")
    print(f"[check] MLM batch y shape = {y.shape}")

    model = CharTransformerMLM(
        vocab_size=len(vocab),
        emb_size=EMB_SIZE,
        max_len=MAX_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        ffn_size=FFN_SIZE,
        dropout=0.1,
        pad_idx=vocab.pad,
    ).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(x)

    assert logits.shape[:2] == x.shape
    assert logits.shape[2] == len(vocab)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

    print(f"[check] MLM loss = {loss.item():.4f}")
    assert torch.isfinite(loss)

    num_masked = (y != -100).sum().item()
    print(f"[check] masked tokens = {num_masked}")
    assert num_masked > 0

    # =========================================================
    # RERANKER CHECK
    # =========================================================
    print("\n[check] Reranker pipeline")

    rerank_dataset = ContextRerankerDataset(
        text_path=TEXT_PATH,
        vocab=vocab,
        max_len=MAX_LEN,
        window=5,
        num_workers=0,
    )

    rerank_loader = DataLoader(
        rerank_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=CollateReranker(vocab.pad),
        num_workers=0,
    )

    rerank_batch = next(iter(rerank_loader))
    pos = rerank_batch["pos"].to(DEVICE)
    neg = rerank_batch["neg"].to(DEVICE)

    print(f"[check] pos shape = {pos.shape}")
    print(f"[check] neg shape = {neg.shape}")

    with torch.no_grad():
        pos_score = model.score(pos)
        neg_score = model.score(neg)

    print(f"[check] pos score mean = {pos_score.mean().item():.4f}")
    print(f"[check] neg score mean = {neg_score.mean().item():.4f}")

    assert pos_score.shape == neg_score.shape == (BATCH_SIZE,)
    assert torch.isfinite(pos_score).all()
    assert torch.isfinite(neg_score).all()

    print("\n✅ CHECK PASSED: MLM + Reranker pipelines are consistent")


if __name__ == "__main__":
    main()
