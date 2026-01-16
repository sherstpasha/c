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
    "max_words": 4,
    "p_real_start": 0.01,
    "p_real_end": 0.1,
    "p_ending_swap": 0.05,
    "p_extra_punct": 0.01,
    # model
    "emb_size": 128,
    "n_layers": 4,
    "n_heads": 4,
    "ffn_size": 512,
    "dropout": 0.1,
    # training
    "epochs": 100,
    "batch_size": 256,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    # logging / checkpoints
    "eval_every_epochs": 1,
    "save_dir": "checkpoints",
    "resume": None,  # "checkpoints/last.pt"
}

if __name__ == "__main__":
    train(CONFIG)
