# ================= GLOBAL CHAR MAPPING =================
CHAR_MAP = {
    "i": "і",
    "I": "І",
}
# ======================================================

import re
import unicodedata
import os
import psutil
import pandas as pd
from symspellpy import SymSpell, Verbosity
from rapidfuzz.distance import Levenshtein
from evaluate import load
from tqdm import tqdm


class OCRSpellCorrector:
    def __init__(
        self,
        words,
        max_edit=2,
        min_token_len=3,
        symspell_prefix_len=6,
        protect_prefix_len=2,
        char_map=None,
        skip_hyphen_tokens=True,
        max_candidates=10,
        oracle_mode=True,  # 🔥 ГЛАВНЫЙ ТУМБЛЕР
    ):
        self.max_edit = max_edit
        self.min_token_len = min_token_len
        self.protect_prefix_len = protect_prefix_len
        self.char_map = char_map or {}
        self.skip_hyphen_tokens = skip_hyphen_tokens
        self.max_candidates = max_candidates
        self.oracle_mode = oracle_mode

        self.words = set(w.lower() for w in words)
        self.cache = {}
        self.log = []

        self._punct_re = re.compile(r"^([^\w]*)([\wА-Яа-яёЁ]+)([^\w]*)$")

        # -------- SymSpell (ТОЛЬКО генерация кандидатов) --------
        self.symspell = SymSpell(
            max_dictionary_edit_distance=max_edit,
            prefix_length=symspell_prefix_len,
        )

        for w in self.words:
            self.symspell.create_dictionary_entry(self.normalize(w), 1)

        self.print_ram("After SymSpell build")

    # ---------------- utils ----------------
    def print_ram(self, label):
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        print(f"[RAM] {label}: {rss:.1f} MB")

    def normalize(self, s: str) -> str:
        s = unicodedata.normalize("NFKC", s)
        return "".join(self.char_map.get(c, c) for c in s)

    def restore_case(self, src: str, dst: str) -> str:
        if src.isupper():
            return dst.upper()
        if src and src[0].isupper():
            return dst.capitalize()
        return dst.lower()

    def split_token(self, token):
        m = self._punct_re.match(token)
        if not m:
            return None, token, None
        return m.group(1), m.group(2), m.group(3)

    # ---------------- фильтры ----------------
    def prefix_ok(self, a: str, b: str) -> bool:
        n = self.protect_prefix_len
        if n <= 0:
            return True
        if len(a) < n or len(b) < n:
            return False
        for i in range(n):
            x, y = a[i], b[i]
            if x == y:
                continue
            if self.char_map.get(x) == y:
                continue
            if self.char_map.get(y) == x:
                continue
            return False
        return True

    def suffix_ok(self, a: str, b: str) -> bool:
        BAD = [
            ("е", "и"),
            ("и", "е"),
            ("ть", "те"),
            ("ть", "т"),
            ("й", "я"),
            ("я", "й"),
        ]
        for x, y in BAD:
            if a.endswith(x) and b.endswith(y):
                return False
        return True

    def is_candidate(self, token):
        if len(token) < self.min_token_len:
            return False
        if any(c.isdigit() for c in token):
            return False
        if any(c in token for c in "_/\\|@#$%^&*+=<>[]{}"):
            return False
        letters = sum(c.isalpha() for c in token)
        return letters / len(token) >= 0.8

    # ---------------- correction ----------------
    def correct_word(self, word, gt_word=None):
        word_l = word.lower()

        # 🔒 если слово есть в словаре — не трогаем
        if word_l in self.words:
            return word

        if word_l in self.cache:
            return self.cache[word_l]

        norm = self.normalize(word_l)

        suggestions = self.symspell.lookup(
            norm,
            Verbosity.TOP,
            max_edit_distance=self.max_edit,
        )[: self.max_candidates]

        # ========== ORACLE MODE ==========
        if self.oracle_mode and gt_word:
            gt_norm = self.normalize(gt_word.lower())
            for s in suggestions:
                if s.term == gt_norm:
                    result = self.restore_case(word, gt_word)
                    self.cache[word_l] = result
                    return result
        # =================================

        # ---- обычный реранкер ----
        best = word_l
        best_dist = self.max_edit + 1

        for s in suggestions:
            cand = s.term
            if not self.prefix_ok(norm, cand):
                continue
            if not self.suffix_ok(word_l, cand):
                continue

            d = Levenshtein.distance(norm, cand)
            if d < best_dist:
                best_dist = d
                best = cand

        if best_dist < self.max_edit:
            result = self.restore_case(word, best)
        else:
            result = word

        self.cache[word_l] = result
        return result

    def correct_text(self, text, row_id, gt_text):
        tokens = text.split()
        gt_tokens = gt_text.split()

        out = []
        changed = False

        for i, t in enumerate(tokens):
            p, core, s = self.split_token(t)
            if p is None:
                out.append(t)
                continue

            if self.skip_hyphen_tokens and ("-" in p or "-" in s):
                out.append(t)
                continue

            gt_core = None
            if i < len(gt_tokens):
                _, gt_core, _ = self.split_token(gt_tokens[i])

            if self.is_candidate(core):
                new_core = self.correct_word(core, gt_core)
                new_tok = f"{p}{new_core}{s}"

                if new_tok != t:
                    self.log.append(
                        {
                            "row_id": row_id,
                            "token_before": t,
                            "token_after": new_tok,
                            "before_text": text,
                            "after_text": None,
                            "gt_text": gt_text,
                        }
                    )
                    changed = True

                out.append(new_tok)
            else:
                out.append(t)

        new_text = " ".join(out)

        if changed:
            for item in reversed(self.log):
                if item["row_id"] != row_id:
                    break
                if item["after_text"] is None:
                    item["after_text"] = new_text

        return new_text

    def save_log(self, path):
        if self.log:
            pd.DataFrame(self.log).to_csv(path, index=False, encoding="utf-8")
            print(f"Saved {len(self.log)} corrections → {path}")
        else:
            print("No corrections.")


# ========================== RUN ==========================

if __name__ == "__main__":
    GT_PATH = "YeniseiGovReports-HWR_gt_mapped.csv"
    PRED_PATH = "YeniseiGovReports-HWR_trba_lite_g1_mapped.csv"
    DICT_PATH = "all_words_with_gt.txt"

    LIMIT = 5000
    LOG_PATH = "corrections_log_oracle.csv"

    gt = pd.read_csv(GT_PATH)["text"].astype(str).tolist()[:LIMIT]
    pred = pd.read_csv(PRED_PATH)["prediction"].astype(str).tolist()[:LIMIT]

    with open(DICT_PATH, encoding="utf-8") as f:
        words = set(w.strip() for w in f if w.strip())

    corrector = OCRSpellCorrector(
        words=words,
        max_edit=4,
        min_token_len=3,
        symspell_prefix_len=5,
        protect_prefix_len=2,
        char_map=CHAR_MAP,
        max_candidates=100,
        oracle_mode=True,  # 🔥 ВКЛЮЧЕНО
    )

    cer_metric = load("cer")
    wer_metric = load("wer")

    print("\n=== BEFORE ===")
    print("CER:", cer_metric.compute(predictions=pred, references=gt))
    print("WER:", wer_metric.compute(predictions=pred, references=gt))

    print("\nCorrecting (ORACLE MODE)...")
    corrected = [
        corrector.correct_text(p, i, gt[i])
        for i, p in tqdm(enumerate(pred), total=len(pred))
    ]

    print("\n=== AFTER (ORACLE) ===")
    print("CER:", cer_metric.compute(predictions=corrected, references=gt))
    print("WER:", wer_metric.compute(predictions=corrected, references=gt))

    corrector.save_log(LOG_PATH)
