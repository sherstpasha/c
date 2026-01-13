import Levenshtein
import re
from typing import List, Dict
import pandas as pd


# ======================================================
# 0. Утилиты
# ======================================================

WORD_CHAR_RE = re.compile(r"^\w+$", re.UNICODE)
LATIN_RE = re.compile(r"[a-zA-Z]")
DIGIT_RE = re.compile(r"\d")


def is_valid_symbol(ch: str) -> bool:
    if not ch:
        return False
    if not WORD_CHAR_RE.match(ch):
        return False
    if DIGIT_RE.search(ch):
        return False
    if LATIN_RE.search(ch):
        return False
    if "_" in ch:
        return False
    return True


def is_valid_ngram(s: str) -> bool:
    return s and all(is_valid_symbol(c) for c in s)


def is_case_only(a: str, b: str) -> bool:
    return a.lower() == b.lower() and a != b


# ======================================================
# 1. RuleExtractor
# ======================================================


class RuleExtractor:
    """
    Extracts local OCR→GT rules (n=1..N) using sliding window
    with встроенной фильтрацией по частоте
    """

    def __init__(
        self,
        max_ngram: int = 3,
        max_context_dist: int = 1,
        context: int = 3,
        min_count: int = 3,
    ):
        self.max_ngram = max_ngram
        self.max_context_dist = max_context_dist
        self.context = context
        self.min_count = min_count

    # --------------------------------------------------
    # extract rules from ONE pair
    # --------------------------------------------------

    def extract_rules_from_pair(self, ocr: str, gt: str) -> List[Dict]:
        rules = []

        if ocr == gt:
            return rules

        max_len = min(len(ocr), len(gt))

        for n in range(1, self.max_ngram + 1):
            for i in range(max_len - n + 1):
                o_ng = ocr[i : i + n]
                g_ng = gt[i : i + n]

                if o_ng == g_ng:
                    continue

                if not is_valid_ngram(o_ng) or not is_valid_ngram(g_ng):
                    continue

                if is_case_only(o_ng, g_ng):
                    continue

                # запрет начала слова для одиночных замен
                if n == 1 and i == 0:
                    continue

                left_dist = Levenshtein.distance(ocr[:i], gt[:i])
                right_dist = Levenshtein.distance(ocr[i + n :], gt[i + n :])

                if left_dist > self.max_context_dist:
                    continue
                if right_dist > self.max_context_dist:
                    continue

                rules.append(
                    {
                        "ocr_ngram": o_ng,
                        "gt_ngram": g_ng,
                        "len": n,
                        "pos": i,
                        "left_ctx": ocr[max(0, i - self.context) : i],
                        "right_ctx": ocr[i + n : i + n + self.context],
                        "left_dist": left_dist,
                        "right_dist": right_dist,
                    }
                )

        return rules

    # --------------------------------------------------
    # dataset-level builder
    # --------------------------------------------------

    def build_rule_table(self, df: pd.DataFrame, save_raw: bool = True):
        all_rules = []

        for _, row in df.iterrows():
            ocr = str(row["ocr"])
            gt = str(row["gt"])
            all_rules.extend(self.extract_rules_from_pair(ocr, gt))

        raw_df = pd.DataFrame(all_rules)

        if raw_df.empty:
            return raw_df, raw_df

        # считаем частоты
        raw_df["count"] = raw_df.groupby(["ocr_ngram", "gt_ngram", "len"])[
            "ocr_ngram"
        ].transform("count")

        # фильтрация по частоте
        filtered_df = raw_df[raw_df["count"] >= self.min_count].copy()

        # сохраняем
        if save_raw:
            raw_df.to_csv("rules_extracted_raw.csv", index=False)
            filtered_df.to_csv("rules_extracted_filtered.csv", index=False)

        return raw_df, filtered_df


GT_PATH = "YeniseiGovReports-HWR_gt_mapped.csv"
PRED_PATH = "YeniseiGovReports-HWR_trba_lite_g1_mapped.csv"

gt = pd.read_csv(GT_PATH).rename(columns={"filename": "image", "text": "gt"})
pred = pd.read_csv(PRED_PATH).rename(columns={"image": "image", "prediction": "ocr"})

df = gt.merge(pred, on="image", how="inner").dropna(subset=["gt", "ocr"])

print(f"Всего пар слов: {len(df)}")

extractor = RuleExtractor(
    max_ngram=3,
    max_context_dist=1,
    context=3,
    min_count=3,
)

raw_df, rules_df = extractor.build_rule_table(df)

print("RAW правил:", len(raw_df))
print("Отфильтрованных правил:", len(rules_df))

top = (
    rules_df.groupby(["ocr_ngram", "gt_ngram", "len"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

top.to_csv("rules_extracted_top.csv", index=False)
print("TOP правила сохранены: rules_extracted_top.csv")
