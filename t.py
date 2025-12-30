# ================= GLOBAL CHAR MAPPING =================
# РЕДАКТИРУЕШЬ ТОЛЬКО ЭТОТ СЛОВАРЬ
CHAR_MAP = {
    "i": "і",
    "I": "І",
}
# ======================================================

import re
import unicodedata
from collections import defaultdict
import pandas as pd
from rapidfuzz.distance import Levenshtein
from evaluate import load
from tqdm import tqdm


class SimpleSpellCorrector:
    def __init__(
        self,
        words,
        min_token_len=4,
        max_edit=2,
        char_map=None,
        skip_hyphen_tokens=True,
        protect_prefix_len=2,
    ):
        self.min_token_len = min_token_len
        self.max_edit = max_edit
        self.char_map = char_map or {}
        self.skip_hyphen_tokens = skip_hyphen_tokens
        self.protect_prefix_len = protect_prefix_len

        self.corrections_log = []
        # self.cache = {}  # 🔥 кеш результатов

        # prefix | word | suffix
        self._punct_re = re.compile(r"^([^\w]*)([\wА-Яа-яёЁ]+)([^\w]*)$")

        # ---------- словарь ----------
        self.words = set(w.lower() for w in words)

        # нормализованные формы
        self.norm_words = {w: self.normalize_for_match(w) for w in self.words}

        # группировка по длине
        self.words_by_len = defaultdict(list)
        for w in self.words:
            self.words_by_len[len(w)].append(w)

        # группировка по первой букве
        self.words_by_first = defaultdict(list)
        for w in self.words:
            if w:
                self.words_by_first[w[0]].append(w)

    # ---------- utils ----------
    def normalize_for_match(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        return "".join(self.char_map.get(ch, ch) for ch in s)

    def restore_case(self, original: str, corrected: str) -> str:
        if original.isupper():
            return corrected.upper()
        if original and original[0].isupper():
            return corrected.capitalize()
        return corrected.lower()

    def split_token(self, token: str):
        m = self._punct_re.match(token)
        if not m:
            return None, token, None
        return m.group(1), m.group(2), m.group(3)

    def prefix_compatible(self, src: str, dst: str) -> bool:
        n = self.protect_prefix_len
        if n <= 0:
            return True
        if len(src) < n or len(dst) < n:
            return False

        for i in range(n):
            a = src[i]
            b = dst[i]
            if a == b:
                continue
            if self.char_map.get(a) == b:
                continue
            if self.char_map.get(b) == a:
                continue
            return False
        return True

    def is_correction_candidate(self, token: str) -> bool:
        if len(token) < self.min_token_len:
            return False
        if any(c.isdigit() for c in token):
            return False
        if any(c in token for c in "_/\\|@#$%^&*+=<>[]{}"):
            return False
        letters = sum(c.isalpha() for c in token)
        if letters / len(token) < 0.8:
            return False
        if len(set(token.lower())) <= 2:
            return False
        return True

    # ---------- correction ----------
    def correct_word_only(self, word: str) -> str:
        word_l = word.lower()

        # 🔥 кеш
        # if word_l in self.cache:
        #    return self.cache[word_l]

        word_n = self.normalize_for_match(word_l)

        best = word_l
        best_dist = self.max_edit + 1

        # кандидаты по длине
        for L in range(
            len(word_l) - self.max_edit,
            len(word_l) + self.max_edit + 1,
        ):
            for w in self.words_by_len.get(L, []):
                # фильтр по префиксу
                if not self.prefix_compatible(word_l, w):
                    continue

                w_n = self.norm_words[w]
                d = Levenshtein.distance(word_n, w_n)

                if d < best_dist:
                    best_dist = d
                    best = w
                    if d == 0:
                        break

        result = self.restore_case(word, best) if best_dist <= self.max_edit else word

        # self.cache[word_l] = result
        return result

    def correct_text(self, text: str, row_id: int, gt_text: str):
        tokens = text.split()
        out = []
        changed = False

        for t in tokens:
            prefix, core, suffix = self.split_token(t)
            if prefix is None:
                out.append(t)
                continue

            if self.skip_hyphen_tokens and ("-" in prefix or "-" in suffix):
                out.append(t)
                continue

            if self.is_correction_candidate(core):
                corrected_core = self.correct_word_only(core)
                corrected = f"{prefix}{corrected_core}{suffix}"

                if corrected != t:
                    self.corrections_log.append(
                        {
                            "row_id": row_id,
                            "token_before": t,
                            "token_after": corrected,
                            "before_text": text,
                            "after_text": None,
                            "gt_text": gt_text,
                        }
                    )
                    changed = True

                out.append(corrected)
            else:
                out.append(t)

        new_text = " ".join(out)

        if changed:
            for item in reversed(self.corrections_log):
                if item["row_id"] != row_id:
                    break
                if item["after_text"] is None:
                    item["after_text"] = new_text

        return new_text

    def save_log(self, path):
        if self.corrections_log:
            pd.DataFrame(self.corrections_log).to_csv(
                path, index=False, encoding="utf-8"
            )
            print(f"Saved {len(self.corrections_log)} corrections to {path}")
        else:
            print("No corrections were made.")


def evaluate_texts(preds, gts, title):
    cer = load("cer").compute(predictions=preds, references=gts)
    wer = load("wer").compute(predictions=preds, references=gts)
    print(title)
    print(f"  CER: {cer:.4f}")
    print(f"  WER: {wer:.4f}")


# ========================== RUN ==========================
if __name__ == "__main__":
    GT_PATH = "YeniseiGovReports-HWR_gt_mapped.csv"
    PRED_PATH = "YeniseiGovReports-HWR_trba_lite_g1_mapped.csv"
    DICT_PATH = "all_words_with_gt.txt"

    LIMIT = 5000
    MIN_TOKEN_LEN = 3
    MAX_EDIT = 2

    LOG_PATH = "corrections_log.csv"

    gt_df = pd.read_csv(GT_PATH)
    pred_df = pd.read_csv(PRED_PATH)

    gt_texts = gt_df["text"].astype(str).tolist()[:LIMIT]
    pred_texts = pred_df["prediction"].astype(str).tolist()[:LIMIT]

    with open(DICT_PATH, encoding="utf-8") as f:
        words = set(w.strip().lower() for w in f if w.strip())

    corrector = SimpleSpellCorrector(
        words=words,
        min_token_len=MIN_TOKEN_LEN,
        max_edit=MAX_EDIT,
        char_map=CHAR_MAP,
    )

    print("=== BEFORE CORRECTION ===")
    evaluate_texts(pred_texts, gt_texts, "Before")

    print("\nCorrecting texts...")
    corrected_texts = [
        corrector.correct_text(pred, i, gt_texts[i])
        for i, pred in tqdm(
            enumerate(pred_texts),
            total=len(pred_texts),
            desc="Correcting",
        )
    ]

    print("\n=== AFTER CORRECTION ===")
    evaluate_texts(corrected_texts, gt_texts, "After")

    corrector.save_log(LOG_PATH)
