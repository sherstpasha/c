import pandas as pd
import re
import numpy as np
import evaluate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve

# ======================================================
# 0. Утилиты
# ======================================================

LATIN_RE = re.compile(r"[a-zA-Z]")
DIGIT_RE = re.compile(r"\d")


def normalize_word(w: str) -> str:
    """
    Минимальная нормализация:
    - приводим к строке
    - убираем лишние пробелы
    """
    return str(w).strip()


def is_valid_word(w: str) -> bool:
    """
    Оставляем только слова без цифр
    (gate для OCR-текста, не для чисел)
    """
    if DIGIT_RE.search(w):
        return False
    return True


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

df["ocr"] = df["ocr"].apply(normalize_word)
df["gt"] = df["gt"].apply(normalize_word)

df = df[df["ocr"].apply(is_valid_word)]

print(f"Всего слов: {len(df)}")

# ======================================================
# 2. Labels: need_fix
# ======================================================

df["need_fix"] = (df["ocr"] != df["gt"]).astype(int)

print(df["need_fix"].value_counts())

# ======================================================
# 3. Признаки: char n-grams (1–3) + позиции
# ======================================================


def add_position_markers(w: str) -> str:
    """
    Добавляем маркеры начала и конца слова
    """
    return f"^{w}$"


texts = df["ocr"].apply(add_position_markers)

vectorizer = CountVectorizer(
    analyzer="char",
    ngram_range=(1, 3),
    min_df=3,  # важно: режем шум
)

X = vectorizer.fit_transform(texts)
y = df["need_fix"].values

print(f"Размерность признаков: {X.shape}")

# ======================================================
# 4. Обучение gate-модели
# ======================================================

clf = LogisticRegression(
    max_iter=300,
    class_weight="balanced",
    n_jobs=-1,
)

clf.fit(X, y)

print("Gate-модель обучена")

# ======================================================
# 5. Качество классификации
# ======================================================

probs = clf.predict_proba(X)[:, 1]

roc = roc_auc_score(y, probs)
print(f"\nROC-AUC: {roc:.4f}")

precision, recall, thresholds = precision_recall_curve(y, probs)

# выберем порог с высокой точностью
TARGET_PRECISION = 0.9
best_idx = np.where(precision >= TARGET_PRECISION)[0]
if len(best_idx) > 0:
    idx = best_idx[-1]
else:
    idx = np.argmax(precision)

thr = thresholds[max(idx - 1, 0)]

print(f"\nВыбран порог: {thr:.3f}")
print(f"Precision: {precision[idx]:.3f}")
print(f"Recall   : {recall[idx]:.3f}")

# ======================================================
# 6. CER до / после (gate-only)
# ======================================================

cer_metric = evaluate.load("cer")

refs = df["gt"].tolist()
ocr_preds = df["ocr"].tolist()

# если gate говорит "не исправлять" — оставляем как есть
gate_preds = [
    ocr if p < thr else ocr  # gate пока только решает, а не исправляет
    for ocr, p in zip(ocr_preds, probs)
]

cer_before = cer_metric.compute(predictions=ocr_preds, references=refs)
cer_after = cer_metric.compute(predictions=gate_preds, references=refs)

print("\n=== GATE METRICS ===")
print(f"CER ДО    : {cer_before:.4f}")
print(f"CER ПОСЛЕ : {cer_after:.4f}")

# ======================================================
# 7. Интерпретация: ТОП n-грамм
# ======================================================

feat_names = np.array(vectorizer.get_feature_names_out())
coefs = clf.coef_[0]

top_pos = np.argsort(coefs)[-30:][::-1]
top_neg = np.argsort(coefs)[:30]

print("\n=== TOP 'need_fix' n-grams ===")
for i in top_pos:
    print(f"{feat_names[i]:8s} {coefs[i]:.3f}")

print("\n=== TOP 'no_fix' n-grams ===")
for i in top_neg:
    print(f"{feat_names[i]:8s} {coefs[i]:.3f}")

# ======================================================
# 8. Сохранение результатов
# ======================================================

out = df.copy()
out["gate_prob"] = probs
out["gate_decision"] = (probs >= thr).astype(int)

out.to_csv("ocr_gate_results.csv", index=False)

print("\nРезультаты сохранены:")
print(" - ocr_gate_results.csv")
