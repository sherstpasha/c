from CharTransformerMLM.train import train

CONFIG = {
    # reproducibility
    "seed": 42,
    "device": "cuda",  # или "cpu"
    # data
    "charset_path": "data/charset.txt",
    "text_path": "data/extracted_texts_cleaned.txt",
    "pairs_csv": "data/pairs_with_errors_combined.csv",
    "words_path": "data/all_words.txt",
    "max_words": 7,
    "p_real_start": 0.01,
    "p_real_end": 0.1,
    # augmentation probabilities
    "noise_prob": 0.15,
    "p_ending_swap": 0.05,
    "p_extra_punct": 0.02,
    "p_hyphen_comma": 0.015,  # дефис-запятая в конце слова
    "p_comma_prefix": 0.01,  # запятая в начале слова
    "p_repeat_ending": 0.01,  # повторы окончаний
    "p_single_hyphen": 0.02,  # одиночный дефис-разрыв
    # model
    "emb_size": 192,
    "n_layers": 6,
    "n_heads": 6,
    "ffn_size": 768,
    "dropout": 0.1,
    # training
    "epochs": 100,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    # logging / checkpoints
    "eval_every_epochs": 1,
    "save_dir": "checkpoints2",
    "resume": None,  # "checkpoints/last.pt"
}

if __name__ == "__main__":
    train(CONFIG)
