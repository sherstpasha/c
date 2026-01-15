"""CharLM конфигурация для обучения эмбеддингов с metric learning."""

DEFAULT_CONFIG = {
    "exp_dir": "exp",
    # === Данные ===
    "lexicon_path": "all_words.txt",
    "text_path": "extracted_texts_cleaned.txt",
    "charset_path": "charset.txt",
    "pairs_path": "pairs.csv",
    "max_lexicon_words": 300000,  # ограничение лексикона для Stage A
    # === Модель ===
    "max_len": 64,
    "emb_size": 256,
    "n_layers": 6,
    "n_heads": 8,
    "ffn_size": 1024,
    "dropout": 0.1,
    "embed_dim": 256,  # размерность выходного эмбеддинга
    # === Stage A: Lexicon Embedding Pretraining ===
    "batch_a": 512,  # большой batch для in-batch negatives
    "epochs_a": 8,
    "lr_encoder_a": 1e-3,
    "lr_embed_head_a": 3e-3,
    "mlm_weight_a": 0.1,  # слабый MLM
    "metric_weight_a": 1.0,
    "noise_prob_a": 0.5,
    "freeze_encoder_a_epochs": 2,  # заморозить encoder на первых N эпохах
    # === Stage B: Context-Aware Embedding ===
    "batch_b": 256,
    "epochs_b": 20,
    "steps_per_epoch_b": 8000,
    "lr_encoder_b": 1e-5,
    "lr_embed_head_b": 3e-4,
    "mlm_weight_b": 0.1,
    "metric_weight_b": 1.0,
    "context_noise_prob": 0.3,
    # === Stage C: Supervised OCR Pairs ===
    "batch_c": 128,
    "epochs_c": 15,
    "lr_encoder_c": 5e-6,
    "lr_embed_head_c": 1e-4,
    "mlm_weight_c": 0.05,
    "metric_weight_c": 1.0,
    "hard_neg_k": 10,
    "update_lexicon_emb_every": 3,  # обновлять lexicon embeddings каждые N эпох
    # === Metric Learning ===
    "temperature": 0.07,
    # === OCR Augmentation ===
    "p_char_insert": 0.08,
    "p_char_delete": 0.08,
    "p_char_substitute": 0.12,
    "p_char_swap": 0.05,
    "p_space_insert": 0.05,
    "p_duplicate": 0.03,
    # === Confusable Characters ===
    "confusables": {
        "о": ["а", "0"],
        "а": ["о"],
        "е": ["ё", "с"],
        "ё": ["е"],
        "и": ["й", "п"],
        "й": ["и"],
        "п": ["г", "н"],
        "г": ["п", "т"],
        "н": ["м", "и"],
        "м": ["н", "ш"],
        "ь": ["ъ"],
        "ъ": ["ь"],
        "ш": ["щ"],
        "щ": ["ш"],
        "ц": ["у"],
        "у": ["ц"],
        "б": ["в"],
        "в": ["б"],
        "ѣ": ["е"],
        "і": ["и", "1"],
    },
    # === Masking (for MLM) ===
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
    "device": "auto",
    "seed": 42,
    "num_workers": 0,
}
