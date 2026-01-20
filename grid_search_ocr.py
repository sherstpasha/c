# grid_search_ocr.py
import torch
import csv
from itertools import product
from CharLM.utils import evaluate_ocr_with_cer
from CharLM.config import DEFAULT_CONFIG
from CharLM.model import CharTransformerMLM
from CharLM.utils import build_vocab, load_allowed_chars, filter_words
import json
import os

# =======================
# CONFIG
# =======================
EXP_DIR = "exp_stage_a1"  # <-- путь к обученной модели
MODEL_PATH = os.path.join(EXP_DIR, "ocr_epoch_30.csv")
PAIRS_PATH = os.path.join(EXP_DIR, "ocr_holdout_pairs.tsv")
RESULTS_CSV = os.path.join(EXP_DIR, "grid_results.csv")

MASK_THRESHOLDS = [0.05, 0.1, 0.2, 0.3]
APPLY_THRESHOLDS = [0.7, 0.8, 0.9, 0.95]
MAX_EDITS = [1, 2, 3]

device = "cuda" if torch.cuda.is_available() else "cpu"

# =======================
# LOAD CONFIG + VOCAB
# =======================
cfg = DEFAULT_CONFIG

with open(os.path.join(EXP_DIR, "vocab.json"), encoding="utf-8") as f:
    chars = json.load(f)

c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}

# =======================
# LOAD MODEL
# =======================
model = CharTransformerMLM(
    vocab_size=len(chars),
    emb_size=cfg["emb_size"],
    max_len=cfg["max_len"],
    n_layers=cfg["n_layers"],
    n_heads=cfg["n_heads"],
    ffn_size=cfg["ffn_size"],
    dropout=0.0,
    pad_idx=c2i["<PAD>"],
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =======================
# LOAD PAIRS
# =======================
pairs = []
with open(PAIRS_PATH, encoding="utf-8") as f:
    next(f)
    for line in f:
        inc, cor = line.strip().split("\t")
        pairs.append((inc, cor))

print(f"Loaded {len(pairs)} OCR pairs")

# =======================
# GRID SEARCH
# =======================
rows = []

for mask_t, apply_t, max_e in product(MASK_THRESHOLDS, APPLY_THRESHOLDS, MAX_EDITS):
    stats = evaluate_ocr_with_cer(
        model,
        pairs,
        c2i,
        i2c,
        device,
        cfg["max_len"],
        mask_threshold=mask_t,
        apply_threshold=apply_t,
        max_edits=max_e,
    )

    row = {
        "mask_threshold": mask_t,
        "apply_threshold": apply_t,
        "max_edits": max_e,
        **stats,
        "net_gain": stats["improved_pct"] - stats["worsened_pct"],
    }

    rows.append(row)

    print(
        f"mask={mask_t:.2f} apply={apply_t:.2f} edits={max_e} | "
        f"+{stats['improved_pct']:.2f} / -{stats['worsened_pct']:.2f} "
        f"net={row['net_gain']:.2f}"
    )

# =======================
# SAVE
# =======================
rows.sort(key=lambda r: (-r["net_gain"], r["worsened_pct"]))

with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved grid results to {RESULTS_CSV}")
