# ======================================================
# Masked Character Transformer for OCR correction
# - Trains on clean lexicon only
# - Drop-in Transformer Encoder (instead of GRU)
# - Span masking (BERT-style)
# - Predict ONLY masked positions (more correct MLM objective)
# ======================================================

import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import Levenshtein

# ======================================================
# CONFIG
# ======================================================

LEXICON_PATH = "all_words_with_gt.txt"

MAX_LEN = 32
EMB_SIZE = 256

# Transformer config
N_LAYERS = 6
N_HEADS = 8
FFN_SIZE = 1024
DROPOUT = 0.1

BATCH_SIZE = 256          # was 1024 for GRU; Transformer uses more memory
EPOCHS = 40
LR = 1e-4

MIN_WORD_LEN = 4

# Masking config
MASK_PROB = 0.95          # probability to actually mask chosen span positions
SPAN_MIN = 1              # span length range
SPAN_MAX = 3
NUM_SPANS_MIN = 1         # number of spans per word
NUM_SPANS_MAX = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

CKPT = "char_masked_transformer.pt"
VOCAB_PATH = "char_lm_vocab.json"

# ======================================================
# SEED
# ======================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ======================================================
# LOAD LEXICON
# ======================================================

words = []
with open(LEXICON_PATH, encoding="utf-8") as f:
    for line in f:
        w = line.strip()
        if len(w) >= MIN_WORD_LEN:
            words.append(w)

print(f"Lexicon words: {len(words)}")

# ======================================================
# BUILD CHAR VOCAB
# ======================================================

chars = set()
for w in words:
    chars.update(w)

chars = ["<PAD>", "<MASK>", "<UNK>"] + sorted(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}

PAD = c2i["<PAD>"]
MASK = c2i["<MASK>"]
UNK = c2i["<UNK>"]
VOCAB = len(chars)

with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(chars, f, ensure_ascii=False)

print(f"Vocab size: {VOCAB}")

# ======================================================
# DATASET
# ======================================================

def encode(word: str):
    ids = [c2i.get(c, UNK) for c in word[:MAX_LEN]]
    return ids + [PAD] * (MAX_LEN - len(ids))

def choose_spans(L: int):
    """
    Choose 1..2 spans, each length 1..3, avoiding edges.
    Returns sorted unique positions to supervise.
    """
    if L <= 3:
        return []

    eligible_start_min = 1
    eligible_start_max = max(1, L - 2)  # keep room for edge-avoid & span
    n_spans = random.randint(NUM_SPANS_MIN, NUM_SPANS_MAX)

    positions = set()
    for _ in range(n_spans):
        span_len = random.randint(SPAN_MIN, SPAN_MAX)
        # ensure span stays inside [1, L-2]
        start_max = min(L - 2, L - 1 - span_len)
        start_min = 1
        if start_max < start_min:
            continue
        start = random.randint(start_min, start_max)
        for p in range(start, start + span_len):
            if 1 <= p <= L - 2:
                positions.add(p)

    return sorted(list(positions))

class SpanMaskedCharDataset(Dataset):
    def __init__(self, words):
        self.words = words

    def __len__(self):
        return len(self.words)

    def __getitem__(self, i):
        w = self.words[i]
        ids = encode(w)

        L = min(len(w), MAX_LEN)
        mask_pos = choose_spans(L)

        x = ids.copy()
        y = [-100] * MAX_LEN  # ignore index for CE

        for p in mask_pos:
            y[p] = ids[p]
            if random.random() < MASK_PROB:
                x[p] = MASK
            # else: keep original char (BERT-style "sometimes keep")

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
            w
        )

loader = DataLoader(
    SpanMaskedCharDataset(words),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

# ======================================================
# MODEL (Transformer Encoder)
# ======================================================

class CharTransformerMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, EMB_SIZE, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, EMB_SIZE)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=EMB_SIZE,
            nhead=N_HEADS,
            dim_feedforward=FFN_SIZE,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.out = nn.Linear(EMB_SIZE, VOCAB)

    def forward(self, x):
        # x: [B, T]
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)

        # key padding mask: True where PAD (to be ignored)
        pad_mask = (x == PAD)  # [B, T]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.out(h)  # [B, T, V]

model = CharTransformerMLM().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

# ======================================================
# TRAIN
# ======================================================

print("\n=== TRAINING SPAN-MASKED CHAR TRANSFORMER MLM ===")

best = 1e9

for ep in range(1, EPOCHS + 1):
    model.train()
    tot = 0.0

    for x, y, _ in tqdm(loader, leave=False):
        x = x.to(DEVICE)  # [B,T]
        y = y.to(DEVICE)  # [B,T] with -100 for non-masked

        logits = model(x)  # [B,T,V]
        loss = F.cross_entropy(
            logits.view(-1, VOCAB),
            y.view(-1),
            ignore_index=-100
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        tot += loss.item()

    avg = tot / len(loader)
    print(f"Epoch {ep:02d} | loss={avg:.4f}")

    if avg < best:
        best = avg
        torch.save(model.state_dict(), CKPT)
        print("  ✅ saved")

model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

# ======================================================
# SCORING UTILITIES
# NOTE: This stays similar to your original, but with MLM it's more correct to
# score each position by masking it and using p(char|context).
# That’s slower, but higher quality. If needed, we can cache or batch it.
# ======================================================

@torch.no_grad()
def masked_char_logprob(word: str, pos: int):
    """
    log p(word[pos] | word with pos masked)
    """
    if pos < 0 or pos >= min(len(word), MAX_LEN):
        return 0.0

    ids = encode(word)
    true_id = ids[pos]
    ids[pos] = MASK

    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    logits = model(x)[0, pos]  # [V]
    lp = F.log_softmax(logits, dim=0)[true_id].item()
    return lp

@torch.no_grad()
def word_logprob_mlm(word: str):
    """
    Sum over positions of log p(char_i | word with char_i masked).
    More expensive than your GRU word_logprob, but consistent with MLM training.
    """
    L = min(len(word), MAX_LEN)
    if L == 0:
        return 0.0

    s = 0.0
    for i in range(L):
        # you can optionally skip PAD tail; here L already ignores it
        s += masked_char_logprob(word, i)
    return s

@torch.no_grad()
def find_worst_position(word: str):
    """
    Find position with lowest conditional log-prob (i.e., most 'surprising' char).
    Avoid edges (1..L-2) like before.
    """
    L = min(len(word), MAX_LEN)
    if L <= 3:
        return None, 0.0

    worst_pos = None
    worst_drop = 0.0

    for i in range(1, L - 1):
        lp = masked_char_logprob(word, i)
        drop = -lp
        if drop > worst_drop:
            worst_drop = drop
            worst_pos = i

    return worst_pos, worst_drop

@torch.no_grad()
def suggest_fixes(word: str, topk=5):
    pos, _ = find_worst_position(word)
    if pos is None:
        return []

    ids = encode(word)
    ids[pos] = MASK
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    logits = model(x)[0, pos]
    p = F.log_softmax(logits, dim=0)

    cand = []
    for i in torch.topk(p, topk).indices:
        c = i2c[i.item()]
        if c in ("<PAD>", "<MASK>", "<UNK>"):
            continue
        w2 = word[:pos] + c + word[pos+1:]
        cand.append((w2, word_logprob_mlm(w2)))

    return sorted(cand, key=lambda x: -x[1])

# ======================================================
# DIAGNOSTIC ON PAIRS
# ======================================================

def diagnostic_pairs(pairs_csv, n=2000):
    import pandas as pd
    df = pd.read_csv(pairs_csv).dropna()
    df["ocr"] = df["incorrect"].astype(str)
    df["gt"] = df["correct"].astype(str)

    diffs = []
    for _, r in df.sample(min(n, len(df))).iterrows():
        lo = word_logprob_mlm(r["ocr"])
        lg = word_logprob_mlm(r["gt"])
        diffs.append(lg - lo)

    print("\n=== DIAGNOSTIC: MLM LM signal ===")
    print(f"mean(gt - ocr) = {np.mean(diffs):.4f}")
    print(f"p90 = {np.percentile(diffs,90):.4f}")
    print(f"p99 = {np.percentile(diffs,99):.4f}")

# ======================================================
# EXAMPLE
# ======================================================

print("\n=== EXAMPLE ===")
w = "распространепіе"
print("word:", w)
print("logP:", word_logprob_mlm(w))
print("worst pos:", find_worst_position(w))
print("suggestions:", suggest_fixes(w))

diagnostic_pairs("pairs.csv")

# ======================================================
# APPLY CONFIG
# ======================================================

DELTA_TH = 1.0        # LM confidence threshold
TOPK = 5
MIN_APPLY_LEN = 4

RESULTS_CSV = "ocr_lm_results.csv"
LEXICON_PATH = "all_words_with_gt.txt"
LEXICON = set(w.strip().lower() for w in open(LEXICON_PATH, encoding="utf-8"))

@torch.no_grad()
def correct_word_lm(word: str):
    if len(word) < MIN_APPLY_LEN:
        return word, False, 0.0

    w_norm = word.lower()
    in_lexicon = w_norm in LEXICON

    base_lp = word_logprob_mlm(word)
    pos, _ = find_worst_position(word)

    if pos is None:
        return word, False, 0.0

    # get distribution at worst pos
    ids = encode(word)
    ids[pos] = MASK
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    logits = model(x)[0, pos]
    p = F.log_softmax(logits, dim=0)

    best_word = word
    best_lp = base_lp

    for i in torch.topk(p, TOPK).indices:
        c = i2c[i.item()]
        if c in ("<PAD>", "<MASK>", "<UNK>"):
            continue

        cand = word[:pos] + c + word[pos+1:]
        cand_norm = cand.lower()

        # 🔒 CRITICAL RULE: if original is in lexicon, do not leave lexicon
        if in_lexicon and cand_norm not in LEXICON:
            continue

        lp = word_logprob_mlm(cand)
        if lp > best_lp:
            best_lp = lp
            best_word = cand

    delta = best_lp - base_lp
    if delta > DELTA_TH:
        return best_word, True, delta

    return word, False, delta

# ======================================================
# APPLY TO PAIRS + METRICS
# ======================================================

import pandas as pd
import evaluate

cer_metric = evaluate.load("cer")

df = pd.read_csv("pairs.csv").dropna()
df["ocr"] = df["incorrect"].astype(str)
df["gt"] = df["correct"].astype(str)

rows = []
ocr_all = []
cor_all = []
gt_all = []

applied = 0
improved = 0

for _, r in tqdm(df.iterrows(), total=len(df)):
    ocr = r["ocr"]
    gt = r["gt"]

    # only single alpha-words (same as your current pipeline)
    if not ocr.isalpha() or not gt.isalpha():
        cor = ocr
        applied_flag = False
        delta = 0.0
    else:
        cor, applied_flag, delta = correct_word_lm(ocr)
        if applied_flag:
            applied += 1
            if Levenshtein.distance(cor, gt) < Levenshtein.distance(ocr, gt):
                improved += 1

    cer_before = Levenshtein.distance(ocr, gt)
    cer_after = Levenshtein.distance(cor, gt)

    if cor == ocr:
        effect = "NOT_APPLIED"
    elif cer_after < cer_before:
        effect = "IMPROVED"
    elif cer_after > cer_before:
        effect = "WORSENED"
    else:
        effect = "NO_EFFECT"

    rows.append({
        "effect": effect,
        "ocr": ocr,
        "corrected": cor,
        "gt": gt,
        "cer_before": cer_before,
        "cer_after": cer_after,
        "cer_delta": cer_before - cer_after,
        "lm_delta": delta
    })

    ocr_all.append(ocr)
    cor_all.append(cor)
    gt_all.append(gt)

print("\n=== METRICS (CHAR TRANSFORMER MLM SAFE CORRECTION) ===")
print("CER ДО    :", cer_metric.compute(predictions=ocr_all, references=gt_all))
print("CER ПОСЛЕ :", cer_metric.compute(predictions=cor_all, references=gt_all))
print(f"Правок применено: {applied}")
print(f"Precision правок: {improved/applied:.3f}" if applied else "—")

res_df = pd.DataFrame(rows).sort_values(
    by=["effect", "cer_delta"],
    ascending=[True, False]
)

res_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
print(f"Saved results to: {RESULTS_CSV}")

print("\n=== TOP IMPROVED ===")
print(res_df[res_df.effect=="IMPROVED"].head(10)[["ocr","corrected","gt","lm_delta"]])

print("\n=== TOP WORSENED ===")
print(res_df[res_df.effect=="WORSENED"].head(10)[["ocr","corrected","gt","lm_delta"]])
