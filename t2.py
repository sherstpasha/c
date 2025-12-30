import pandas as pd
import unicodedata

# === CONFIG ===
GT_PATH = "YeniseiGovReports-PRT_gt.csv"
OUT_PATH = "YeniseiGovReports-PRT_gt_mapped.csv"

# === GLOBAL CHAR MAP ===
CHAR_MAP = {
    "i": "і",
}


# === NORMALIZATION ===
def apply_char_map(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(CHAR_MAP.get(ch, ch) for ch in text)


# === RUN ===
df = pd.read_csv(GT_PATH)

# применяем маппинг к колонке text
df["text"] = df["text"].astype(str).apply(apply_char_map)

df.to_csv(OUT_PATH, index=False, encoding="utf-8")

print(f"Saved mapped GT to: {OUT_PATH}")
