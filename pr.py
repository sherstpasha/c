import pandas as pd

# ======================================================
# paths
# ======================================================

GT_PATH = "YeniseiGovReports-HWR_gt_mapped.csv"
PRED_PATH = "YeniseiGovReports-HWR_trba_lite_g1_mapped.csv"

OUT_PATH = "ocr_pairs_incorrect_correct.csv"

# ======================================================
# load
# ======================================================

gt = pd.read_csv(GT_PATH)
pred = pd.read_csv(PRED_PATH)

# ожидаемые колонки:
# gt:   filename, text
# pred: image, prediction

gt = gt.rename(
    columns={
        "filename": "image",
        "text": "correct",
    }
)

pred = pred.rename(
    columns={
        "image": "image",
        "prediction": "incorrect",
    }
)

# ======================================================
# merge
# ======================================================

df = gt.merge(pred, on="image", how="inner")

df = df.dropna(subset=["incorrect", "correct"])

# ======================================================
# reorder columns (красиво)
# ======================================================

df = df[["image", "incorrect", "correct"]]

print(f"Всего строк: {len(df)}")

# ======================================================
# save
# ======================================================

df.to_csv(OUT_PATH, index=False)

print("Файл сохранён:")
print(f" - {OUT_PATH}")
