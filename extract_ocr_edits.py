import pandas as pd
import Levenshtein
import re
import numpy as np
import evaluate

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


# ======================================================
# 0. Утилиты
# ======================================================

NO_CHANGE = "__NO_CHANGE__"

WORD_CHAR_RE = re.compile(r"^\w+$", re.UNICODE)
DIGIT_RE = re.compile(r"\d")
LATIN_RE = re.compile(r"[a-zA-Z]")


def is_valid_symbol(s: str) -> bool:
    if not s:
        return False
    if not WORD_CHAR_RE.match(s):
        return False
    if DIGIT_RE.search(s):
        return False
    if "_" in s:
        return False
    return True


def is_latin(s: str) -> bool:
    return bool(LATIN_RE.search(s))


def is_case_only_pair(a: str, b: str) -> bool:
    """
    True если отличие ТОЛЬКО в регистре
    """
    return a.lower() == b.lower() and a != b


# ======================================================
# 1. Загрузка данных
# ======================================================

GT_PATH = "YeniseiGovReports-HWR_gt_mapped.csv"
PRED_PATH = "YeniseiGovReports-HWR_trba_lite_g1_mapped.csv"

gt = pd.read_csv(GT_PATH)
pred = pd.read_csv(PRED_PATH)

gt = gt.rename(columns={"filename": "image", "text": "gt"})
pred = pred.rename(columns={"image": "image", "prediction": "ocr"})

df = gt.merge(pred, on="image", how="inner")
df = df.dropna(subset=["gt", "ocr"])

print(f"Всего пар слов: {len(df)}")


# ======================================================
# 2. POSITIVE samples (ТОЛЬКО реальные OCR-ошибки)
# ======================================================

pos_rows = []

for _, row in df.iterrows():
    ocr = str(row["ocr"])
    gt = str(row["gt"])

    if ocr == gt:
        continue

    ops = Levenshtein.editops(ocr, gt)
    for op, i, j in ops:
        if op != "replace":
            continue

        ch_ocr = ocr[i]
        ch_gt = gt[j]

        # базовая фильтрация
        if not is_valid_symbol(ch_ocr) or not is_valid_symbol(ch_gt):
            continue

        # 🚫 выкидываем регистр
        if is_case_only_pair(ch_ocr, ch_gt):
            continue

        # 🚫 выкидываем латиницу
        if is_latin(ch_ocr) or is_latin(ch_gt):
            continue

        left = ocr[max(0, i - 2) : i]
        right = ocr[i + 1 : i + 3]

        pos_rows.append(
            {
                "ocr": ch_ocr,
                "y": ch_gt,
                "left": left,
                "right": right,
            }
        )

pos_df = pd.DataFrame(pos_rows)

# фильтр по частоте
pos_df = pos_df[pos_df.groupby(["ocr", "y"])["ocr"].transform("count") >= 3]

print(f"Positive samples (clean): {len(pos_df)}")


# ======================================================
# 3. NEGATIVE samples (NO_CHANGE)
# ======================================================

neg_rows = []

for _, row in df.iterrows():
    ocr = str(row["ocr"])
    gt = str(row["gt"])

    for i in range(min(len(ocr), len(gt))):
        if ocr[i] == gt[i] and is_valid_symbol(ocr[i]):
            left = ocr[max(0, i - 2) : i]
            right = ocr[i + 1 : i + 3]

            neg_rows.append(
                {
                    "ocr": ocr[i],
                    "y": NO_CHANGE,
                    "left": left,
                    "right": right,
                }
            )

neg_df = pd.DataFrame(neg_rows)

# балансировка
neg_df = neg_df.sample(n=min(len(neg_df), len(pos_df) * 2), random_state=42)

print(f"Negative samples: {len(neg_df)}")


# ======================================================
# 4. Train set
# ======================================================

train_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42)
print(f"Всего обучающих примеров: {len(train_df)}")


# ======================================================
# 5. Признаки
# ======================================================


def make_features(row):
    feats = {f"ocr={row.ocr}": 1}

    for i, ch in enumerate(row.left[::-1]):
        if is_valid_symbol(ch):
            feats[f"L{i+1}={ch}"] = 1

    for i, ch in enumerate(row.right):
        if is_valid_symbol(ch):
            feats[f"R{i+1}={ch}"] = 1

    return feats


X = train_df.apply(make_features, axis=1)
y = train_df["y"]


# ======================================================
# 6. Обучение модели
# ======================================================

vec = DictVectorizer(sparse=True)
X_vec = vec.fit_transform(X)

clf = LogisticRegression(max_iter=300)
clf.fit(X_vec, y)

classes = list(clf.classes_)
no_change_idx = classes.index(NO_CHANGE)

print("\nМодель обучена.")
print("Классы:", classes)


# ======================================================
# 7. Применение модели (GATED, БЕЗ РЕГИСТРА)
# ======================================================


def correct_word(ocr, clf, vec, gate_k=3.0, prob_threshold=0.6):
    best = None

    for pos in range(len(ocr)):
        ch = ocr[pos]
        if not is_valid_symbol(ch):
            continue

        left = ocr[max(0, pos - 2) : pos]
        right = ocr[pos + 1 : pos + 3]

        feats = {f"ocr={ch}": 1}
        for i, c in enumerate(left[::-1]):
            if is_valid_symbol(c):
                feats[f"L{i+1}={c}"] = 1
        for i, c in enumerate(right):
            if is_valid_symbol(c):
                feats[f"R{i+1}={c}"] = 1

        X = vec.transform([feats])
        probs = clf.predict_proba(X)[0]

        p_no = probs[no_change_idx]

        for cls, p in zip(classes, probs):
            if cls == NO_CHANGE:
                continue
            if is_latin(cls):
                continue

            # gating
            if p < prob_threshold:
                continue
            if p <= p_no * gate_k:
                continue

            if best is None or p > best["prob"]:
                best = {"pos": pos, "char": cls, "prob": p}

    if best:
        return (ocr[: best["pos"]] + best["char"] + ocr[best["pos"] + 1 :], best)

    return ocr, None


# ======================================================
# 8. Оценка
# ======================================================

cer_metric = evaluate.load("cer")

refs, ocr_preds, corr_preds = [], [], []
applied = 0
improved = 0

for _, row in df.iterrows():
    ocr = str(row["ocr"])
    gt = str(row["gt"])

    corrected, info = correct_word(ocr, clf, vec)

    refs.append(gt)
    ocr_preds.append(ocr)
    corr_preds.append(corrected)

    if info is not None:
        applied += 1
        if Levenshtein.distance(corrected, gt) < Levenshtein.distance(ocr, gt):
            improved += 1

cer_before = cer_metric.compute(predictions=ocr_preds, references=refs)
cer_after = cer_metric.compute(predictions=corr_preds, references=refs)

acc_before = sum(o == g for o, g in zip(ocr_preds, refs)) / len(refs)
acc_after = sum(c == g for c, g in zip(corr_preds, refs)) / len(refs)

print("\n=== METRICS ===")
print(f"CER ДО        : {cer_before:.4f}")
print(f"CER ПОСЛЕ     : {cer_after:.4f}")
print(f"Δ CER         : {cer_before - cer_after:.4f}")

print(f"\nWord Accuracy ДО    : {acc_before:.4f}")
print(f"Word Accuracy ПОСЛЕ : {acc_after:.4f}")

print(f"\nПравок применено    : {applied}")
print(f"Precision правок    : {improved / applied:.4f}" if applied else "—")


# ======================================================
# 9. Сохранение результатов (ОТСОРТИРОВАННЫЕ)
# ======================================================

out_rows = []

for ocr, cor, gt in zip(ocr_preds, corr_preds, refs):
    cer_b = Levenshtein.distance(ocr, gt)
    cer_a = Levenshtein.distance(cor, gt)

    if cor != ocr:
        if cer_a > cer_b:
            effect = "WORSENED"
            rank = 0
        elif cer_a < cer_b:
            effect = "IMPROVED"
            rank = 1
        else:
            effect = "NO_EFFECT"
            rank = 2
    else:
        effect = "NOT_APPLIED"
        rank = 3

    out_rows.append(
        {
            "effect": effect,
            "rank": rank,
            "ocr": ocr,
            "corrected": cor,
            "gt": gt,
            "cer_before": cer_b,
            "cer_after": cer_a,
            "cer_delta": cer_b - cer_a,
        }
    )

res_df = pd.DataFrame(out_rows)
res_df = res_df.sort_values(by=["rank", "cer_delta"], ascending=[True, False])
res_df = res_df.drop(columns=["rank"])

res_df.to_csv("ocr_postcorrect_results_clean_sorted.csv", index=False)

print("\nРезультаты сохранены в:")
print(" - ocr_postcorrect_results_clean_sorted.csv")
