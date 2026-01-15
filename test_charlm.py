"""Пример использования CharLM с metric learning."""

from CharLM import train

config = {
    "exp_dir": "exp1",
    # === Данные ===
    "lexicon_path": "all_words.txt",
    "text_path": "extracted_texts_cleaned.txt",
    "charset_path": "charset.txt",
    "pairs_path": "pairs.csv",
    # === Модель ===
    "max_len": 64,
    "emb_size": 256,
    "embed_dim": 256,  # размерность выходного эмбеддинга
    "n_layers": 6,
    "n_heads": 8,
    "ffn_size": 1024,
    "dropout": 0.1,
    # === Stage A: Lexicon Embedding Pretraining ===
    "batch_a": 128,
    "epochs_a": 5,
    "lr_encoder_a": 1e-3,
    "lr_embed_head_a": 3e-3,
    "mlm_weight_a": 0.3,
    "metric_weight_a": 1.0,
    "n_negatives_a": 15,
    "noise_prob_a": 0.5,
    # === Stage B: Context-Aware Embedding ===
    "batch_b": 128,
    "epochs_b": 30,
    "steps_per_epoch_b": 10000,
    "lr_encoder_b": 1e-5,
    "lr_embed_head_b": 3e-4,
    "mlm_weight_b": 0.1,
    "metric_weight_b": 1.0,
    "context_noise_prob": 0.3,
    # === Stage C: Supervised OCR Pairs ===
    "batch_c": 64,
    "epochs_c": 20,
    "lr_encoder_c": 5e-6,
    "lr_embed_head_c": 1e-4,
    "mlm_weight_c": 0.05,
    "metric_weight_c": 1.0,
    "hard_neg_k": 10,
    # === Metric Learning ===
    "metric_loss_type": "infonce",  # "triplet" или "infonce"
    "temperature": 0.07,
    "triplet_margin": 0.3,
    # === OCR Augmentation ===
    "p_char_insert": 0.08,
    "p_char_delete": 0.08,
    "p_char_substitute": 0.12,
    "p_char_swap": 0.05,
    "p_space_insert": 0.05,
    "p_duplicate": 0.03,
    # === MLM Masking ===
    "min_word_len": 3,
    "mask_prob": 0.15,
    # === Context Windows ===
    "p_win_1": 0.4,
    "p_win_2": 0.3,
    "p_win_3": 0.3,
    # === Optimization ===
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "device": "cuda",
    "seed": 42,
}

if __name__ == "__main__":
    model, (c2i, i2c, chars), exp_dir = train(config)
    print(f"\nVocab size: {len(chars)}")
    print(f"Embedding dim: {model.embed_dim}")
    print(f"Experiment saved to: {exp_dir}")

    # Quick test
    import torch
    from CharLM import encode_batch

    test_words = ["слово", "текст", "пример"]
    device = next(model.parameters()).device

    ids = encode_batch(test_words, c2i, config["max_len"]).to(device)
    with torch.no_grad():
        emb = model.encode_words(ids)

    print(f"\nTest embeddings shape: {emb.shape}")
    print(f"L2 norms: {emb.norm(dim=-1).tolist()}")

    # Cosine similarity matrix
    sim = torch.mm(emb, emb.T)
    print(f"\nCosine similarity matrix:")
    for i, w in enumerate(test_words):
        print(f"  {w}: {[f'{s:.3f}' for s in sim[i].tolist()]}")
