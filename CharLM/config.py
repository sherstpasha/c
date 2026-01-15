"""Дефолтная конфигурация CharLM (упрощённая)."""

DEFAULT_CONFIG = {
    "exp_dir": "exp",
    # Данные
    "lexicon_path": "all_words.txt",
    "text_path": "extracted_texts_cleaned.txt",
    "charset_path": "charset.txt",
    "pairs_path": "pairs.csv",
    # Модель
    "max_len": 64,
    "emb_size": 256,
    "n_layers": 6,
    "n_heads": 8,
    "ffn_size": 1024,
    "dropout": 0.1,
    # Stage A (Lexicon MLM)
    "batch_a": 256,
    "epochs_a": 6,  # 5-8 эпох достаточно
    "lr_a": 1e-3,
    "max_words_a": 250000,  # random sample 200-300k слов
    # Stage B (Context MLM)
    "batch_b": 256,
    "epochs_b": 10,
    "lr_b": 5e-5,
    "steps_per_epoch_b": 10000,
    # Stage C (Contrastive Learning)
    "batch_c": 64,
    "epochs_c": 5,
    "lr_c": 1e-5,
    "n_random_negatives": 3,
    "contrastive_temperature": 0.07,
    # Маскирование
    "min_word_len": 4,
    "mask_prob": 0.9,
    "span_min": 1,
    "span_max": 3,
    "num_spans_min": 1,
    "num_spans_max": 2,
    # Контекстные окна (1-3 слова)
    "p_win_1": 0.3,
    "p_win_2": 0.3,
    # p_win_3 = 1 - p_win_1 - p_win_2 = 0.4
    # OCR-шум (только для контекста в Stage B)
    "p_swap": 0.05,
    "p_delete": 0.03,
    "p_insert_space": 0.03,
    "p_duplicate": 0.01,
    # Оптимизация
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "device": "auto",
    "seed": 42,
}
