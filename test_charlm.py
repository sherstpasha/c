"""Пример использования CharLM."""

from CharLM import train

config = {
    "exp_dir": "exp1",
    # Данные
    "lexicon_path": "all_words.txt",
    "text_path": "extracted_texts_cleaned.txt",
    "charset_path": "charset.txt",
    # Модель
    "max_len": 32,
    "emb_size": 192,
    "n_layers": 6,
    "n_heads": 6,
    "ffn_size": 768,
    "dropout": 0.1,
    # Stage A: pretrain на лексиконе
    "batch_a": 256,
    "epochs_a": 30,
    "lr_a": 1e-3,
    "split_prob_a": 0.03,
    # Stage B: finetune на контексте с OCR-шумом
    "batch_b": 256,
    "epochs_b": 30,
    "lr_b": 1e-5,
    "steps_per_epoch_b": 12000,
    # Stage C: finetune на парах (легкий)
    "pairs_path": "pairs.csv",
    "batch_c": 64,
    "epochs_c": 10,
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
    # OCR-шум (только в контексте, не в центре)
    "p_swap": 0.01,
    "p_delete": 0.01,
    "p_insert_space": 0.01,
    "p_duplicate": 0.01,
    # Оптимизация
    "grad_clip": 1.0,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "use_scheduler": True,
    "device": "cuda",
    "seed": 42,
}

if __name__ == "__main__":
    model, (c2i, i2c, chars), exp_dir = train(config)
    print(f"Vocab size: {len(chars)}")
    print(f"Experiment saved to: {exp_dir}")
