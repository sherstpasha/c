# ======================================================
# OCR post-correction: TRIGRAM COPY TEXT-TO-TEXT + ATTENTION
# ======================================================

import json
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import Levenshtein
import evaluate

# ======================================================
# CONFIG
# ======================================================

DATA_PATH = "pairs.csv"

MAX_WORD_LEN = 24
EMB_SIZE = 64
WORD_HID = 96
CTX_HID = 128

BATCH_SIZE = 256
EPOCHS = 60
LR = 1e-3
PATIENCE = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

CKPT = "best_trigram_copy_t2t.pt"
VOCAB_PATH = "trigram_copy_t2t_vocab.json"

# ======================================================
# SEED
# ======================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_all(SEED)

# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"incorrect": "ocr", "correct": "gt"})
df = df.dropna(subset=["ocr", "gt"])
df["ocr"] = df["ocr"].astype(str)
df["gt"] = df["gt"].astype(str)

print(f"Всего строк: {len(df)}")

# ======================================================
# TOKENIZATION
# ======================================================

WORD_RE = re.compile(r"[А-Яа-яЁёѣіІѢЪЬъь]+", re.UNICODE)

def tokenize(text):
    return [m.group() for m in WORD_RE.finditer(text)]

# ======================================================
# BUILD WORD-LEVEL SAMPLES
# ======================================================

samples = []

for ocr, gt in zip(df["ocr"], df["gt"]):
    ocr_words = tokenize(ocr)
    gt_words = tokenize(gt)

    if len(ocr_words) != len(gt_words):
        continue

    for i, (w_ocr, w_gt) in enumerate(zip(ocr_words, gt_words)):
        samples.append({
            "prev": ocr_words[i-1] if i > 0 else "<BOS>",
            "word": w_ocr,
            "next": ocr_words[i+1] if i < len(ocr_words)-1 else "<EOS>",
            "target": w_gt
        })

print(f"Word samples: {len(samples)}")

# ======================================================
# SPLIT
# ======================================================

train_s, val_s = train_test_split(samples, test_size=0.15, random_state=SEED)
print(f"Train: {len(train_s)} | Val: {len(val_s)}")

# ======================================================
# CHAR VOCAB
# ======================================================

chars = set()
for s in samples:
    for w in (s["prev"], s["word"], s["next"], s["target"]):
        chars.update(w)

chars = ["<PAD>", "<UNK>"] + sorted(chars)
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}

PAD = c2i["<PAD>"]
UNK = c2i["<UNK>"]
VOCAB = len(chars)

with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(chars, f, ensure_ascii=False)

def enc(word):
    ids = [c2i.get(c, UNK) for c in word[:MAX_WORD_LEN]]
    return ids + [PAD] * (MAX_WORD_LEN - len(ids))

# ======================================================
# DATASET
# ======================================================

class TrigramDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        s = self.data[i]
        return (
            torch.tensor(enc(s["prev"])),
            torch.tensor(enc(s["word"])),
            torch.tensor(enc(s["next"])),
            torch.tensor(enc(s["target"])),
            torch.tensor(enc(s["word"])),  # input word for copy
            s
        )

train_dl = DataLoader(TrigramDataset(train_s), BATCH_SIZE, shuffle=True)
val_dl = DataLoader(TrigramDataset(val_s), BATCH_SIZE)

# ======================================================
# MODEL
# ======================================================

class WordEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, EMB_SIZE, padding_idx=PAD)
        self.rnn = nn.GRU(EMB_SIZE, WORD_HID, bidirectional=True, batch_first=True)

    def forward(self, x):
        out, _ = self.rnn(self.emb(x))
        return out  # [B, L, 2H]

class TrigramCopyT2T(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = WordEncoder()

        self.att = nn.Linear(WORD_HID * 2, 1)
        self.ctx = nn.Linear(WORD_HID * 6, CTX_HID)

        self.copy = nn.Linear(CTX_HID, MAX_WORD_LEN)
        self.char = nn.Linear(CTX_HID, MAX_WORD_LEN * VOCAB)

    def attend(self, h):
        a = self.att(h).squeeze(-1)
        w = torch.softmax(a, dim=1)
        return (h * w.unsqueeze(-1)).sum(dim=1)

    def forward(self, p, w, n):
        hp = self.attend(self.enc(p))
        hw = self.attend(self.enc(w))
        hn = self.attend(self.enc(n))

        ctx = torch.relu(self.ctx(torch.cat([hp, hw, hn], dim=1)))

        copy_logits = self.copy(ctx)
        char_logits = self.char(ctx).view(-1, MAX_WORD_LEN, VOCAB)

        return copy_logits, char_logits

model = TrigramCopyT2T().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

# ======================================================
# LOSS
# ======================================================

def loss_fn(copy_logits, char_logits, inp, tgt):
    copy_target = (inp == tgt).float()

    loss_copy = F.binary_cross_entropy_with_logits(
        copy_logits, copy_target, reduction="mean"
    )

    loss_char = F.cross_entropy(
        char_logits.view(-1, VOCAB),
        tgt.view(-1),
        ignore_index=PAD
    )

    return loss_copy + loss_char

# ======================================================
# TRAIN
# ======================================================

best = 1e9
bad = 0

print("\n=== TRAINING ===")

for ep in range(1, EPOCHS+1):
    model.train()
    tr = 0

    for p,w,n,t,inp,_ in tqdm(train_dl, leave=False):
        p,w,n,t,inp = p.to(DEVICE), w.to(DEVICE), n.to(DEVICE), t.to(DEVICE), inp.to(DEVICE)

        copy_logits, char_logits = model(p,w,n)
        loss = loss_fn(copy_logits, char_logits, inp, t)

        opt.zero_grad()
        loss.backward()
        opt.step()
        tr += loss.item()

    model.eval()
    va = 0
    with torch.no_grad():
        for p,w,n,t,inp,_ in val_dl:
            p,w,n,t,inp = p.to(DEVICE), w.to(DEVICE), n.to(DEVICE), t.to(DEVICE), inp.to(DEVICE)
            va += loss_fn(*model(p,w,n), inp, t).item()

    print(f"Epoch {ep:02d} | train={tr/len(train_dl):.4f} | val={va/len(val_dl):.4f}")

    if va < best:
        best = va
        bad = 0
        torch.save(model.state_dict(), CKPT)
        print("  ✅ saved")
    else:
        bad += 1
        if bad >= PATIENCE:
            print("  🛑 early stopping")
            break

model.load_state_dict(torch.load(CKPT))
model.eval()

# ======================================================
# INFERENCE + DIAGNOSTICS
# ======================================================

cer_metric = evaluate.load("cer")
rows = []
refs, ocrs, cors = [], [], []
changed = improved = 0

with torch.no_grad():
    for s in val_s:
        p = torch.tensor([enc(s["prev"])]).to(DEVICE)
        w = torch.tensor([enc(s["word"])]).to(DEVICE)
        n = torch.tensor([enc(s["next"])]).to(DEVICE)

        copy_logits, char_logits = model(p,w,n)
        copy = torch.sigmoid(copy_logits)[0]
        chars = char_logits.argmax(-1)[0]

        out = []
        for i, c in enumerate(s["word"]):
            if i >= MAX_WORD_LEN:
                break
            out.append(c if copy[i] > 0.5 else i2c[chars[i].item()])

        pred_word = "".join(out)
        ocr_word = s["word"]
        gt_word = s["target"]

        cb = Levenshtein.distance(ocr_word, gt_word)
        ca = Levenshtein.distance(pred_word, gt_word)

        if pred_word == ocr_word:
            effect = "NOT_CHANGED"
        else:
            changed += 1
            if ca < cb:
                effect = "IMPROVED"
                improved += 1
            elif ca > cb:
                effect = "WORSENED"
            else:
                effect = "NO_EFFECT"

        rows.append({
            "effect": effect,
            "ocr": ocr_word,
            "predicted": pred_word,
            "gt": gt_word,
            "cer_before": cb,
            "cer_after": ca,
            "cer_delta": cb - ca
        })

        refs.append(gt_word)
        ocrs.append(ocr_word)
        cors.append(pred_word)

print("\n=== METRICS ===")
print("CER ДО   :", cer_metric.compute(predictions=ocrs, references=refs))
print("CER ПОСЛЕ:", cer_metric.compute(predictions=cors, references=refs))
print(f"Правок: {changed}")
print(f"Precision: {improved/changed:.3f}" if changed else "—")

pd.DataFrame(rows).to_csv("ocr_trigram_copy_t2t_results.csv", index=False)
print("\nSaved: ocr_trigram_copy_t2t_results.csv")
