import re

# ================= CONFIG =================
GT_PATH = "YeniseiGovReports-PRT_gt_mapped.csv"
DICT_PATH = "all_words_with_gt.txt"
OUT_PATH = "all_words_with_gt.txt"
# =========================================


def load_dict(path):
    with open(path, encoding="utf-8") as f:
        return set(w.strip().lower() for w in f if w.strip())


def extract_tokens_from_gt(gt_path):
    import pandas as pd

    df = pd.read_csv(gt_path)

    token_re = re.compile(r"[А-Яа-яёЁѣѢіІ]+")
    tokens = set()

    for text in df["text"].astype(str):
        for m in token_re.finditer(text):
            tok = m.group(0).lower()
            if len(tok) >= 3:
                tokens.add(tok)

    return tokens


def main():
    words = load_dict(DICT_PATH)
    gt_tokens = extract_tokens_from_gt(GT_PATH)

    print(f"Dict size before: {len(words)}")
    print(f"GT tokens found : {len(gt_tokens)}")

    merged = words | gt_tokens

    print(f"Dict size after : {len(merged)}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for w in sorted(merged):
            f.write(w + "\n")

    print(f"\nSaved merged dictionary to: {OUT_PATH}")


if __name__ == "__main__":
    main()
