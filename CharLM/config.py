"""Дефолтная конфигурация CharLM."""

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
    
    # Stage A (лексикон)
    "batch_a": 256,
    "epochs_a": 30,
    "lr_a": 1e-3,
    "split_prob_a": 0.03,
    
    # Stage B (контекст)
    "batch_b": 256,
    "epochs_b": 36,
    "lr_b": 1e-5,
    "steps_per_epoch_b": 12000,
    
    # Stage C (пары error->correct) — легкий finetune
    "batch_c": 64,
    "epochs_c": 5,
    "lr_c": 1e-6,
    "p_win_1_c": 0.4,
    "p_win_2_c": 0.3,
    "p_win_3_c": 0.3,
    
    # Маскирование
    "min_word_len": 4,
    "mask_prob": 0.95,
    "span_min": 1,
    "span_max": 3,
    "num_spans_min": 1,
    "num_spans_max": 2,
    
    # Контекстные окна
    "p_win_1": 0.35,
    "p_win_2": 0.25,
    "p_win_3": 0.40,
    
    # OCR-шум (Stage B)
    "p_swap": 0.05,
    "p_delete": 0.03,
    "p_insert_space": 0.03,
    "p_duplicate": 0.01,
    
    # Оптимизация
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "use_scheduler": True,
    
    "device": "auto",
    "seed": 42,
}
