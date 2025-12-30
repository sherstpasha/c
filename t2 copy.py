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
        symspell_prefix_len=7,
        protect_prefix_len=2,
        char_map=None,
        skip_hyphen_tokens=True,
        max_candidates=10,  # 🔥 ограничиваем взрыв вариантов
    ):
        self.max_edit = max_edit
        self.min_token_len = min_token_len
        self.protect_prefix_len = protect_prefix_len
        self.char_map = char_map or {}
        self.skip_hyphen_tokens = skip_hyphen_tokens
        self.max_candidates = max_candidates

        self.cache = {}
        self.log = []

        self.words = set(w.lower() for w in words)

        self._punct_re = re.compile(r"^([^\w]*)([\wА-Яа-яёЁ]+)([^\w]*)$")

        # -------- SymSpell: ТОЛЬКО КАНДИДАТЫ --------
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

    # ---------- ТВОИ ФИЛЬТРЫ ----------
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
    def correct_word(self, word):
        word_l = word.lower()

        # 🔒 уже словарное — НЕ ТРОГАЕМ
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

        # 🔒 не берём граничные случаи
        if best_dist >= self.max_edit:
            result = word
        else:
            result = self.restore_case(word, best)

        self.cache[word_l] = result
        return result

    def correct_text(self, text, row_id, gt_text):
        tokens = text.split()
        out = []
        changed = False

        for t in tokens:
            p, core, s = self.split_token(t)
            if p is None:
                out.append(t)
                continue

            if self.skip_hyphen_tokens and ("-" in p or "-" in s):
                out.append(t)
                continue

            if self.is_candidate(core):
                new_core = self.correct_word(core)
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
# ================= GRID SEARCH MAIN =================

import time
import csv
import psutil
import os
import pandas as pd
from evaluate import load
from tqdm import tqdm

# ---------- CONFIG ----------
GT_PATH = "YeniseiGovReports-HWR_gt_mapped.csv"
PRED_PATH = "YeniseiGovReports-HWR_trba_lite_g1_mapped.csv"
DICT_PATH = "all_words_with_gt.txt"

LIMIT = 5000
RESULTS_CSV = "grid_results.csv"

# ---------- PARAM GRID ----------
PREFIX_LENGTHS = [5, 6, 7]
PROTECT_PREFIX_LENS = [1, 2]
MAX_CANDIDATES = [5, 10, 20]
MIN_TOKEN_LENS = [3, 4]

# ---------- LOAD DATA ----------
gt = pd.read_csv(GT_PATH)["text"].astype(str).tolist()[:LIMIT]
pred = pd.read_csv(PRED_PATH)["prediction"].astype(str).tolist()[:LIMIT]

with open(DICT_PATH, encoding="utf-8") as f:
    words = set(w.strip() for w in f if w.strip())

cer_metric = load("cer")
wer_metric = load("wer")


# ---------- RAM UTILS ----------
def ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


# ---------- RUN GRID ----------
results = []

exp_id = 0
total = (
    len(PREFIX_LENGTHS)
    * len(PROTECT_PREFIX_LENS)
    * len(MAX_CANDIDATES)
    * len(MIN_TOKEN_LENS)
)

print(f"\n🔬 Total experiments: {total}\n")

for prefix_len in PREFIX_LENGTHS:
    for protect_len in PROTECT_PREFIX_LENS:
        for max_cand in MAX_CANDIDATES:
            for min_tok in MIN_TOKEN_LENS:
                exp_id += 1
                print(
                    f"\n=== EXP {exp_id}/{total} | "
                    f"prefix={prefix_len}, protect={protect_len}, "
                    f"cands={max_cand}, min_tok={min_tok} ==="
                )

                t0 = time.time()

                corrector = OCRSpellCorrector(
                    words=words,
                    max_edit=2,
                    min_token_len=min_tok,
                    symspell_prefix_len=prefix_len,
                    protect_prefix_len=protect_len,
                    char_map=CHAR_MAP,
                )

                # 🔧 ограничение кандидатов
                corrector.max_candidates = max_cand

                build_ram = ram_mb()

                corrected = [
                    corrector.correct_text(p, i, gt[i])
                    for i, p in tqdm(
                        enumerate(pred),
                        total=len(pred),
                        desc="Correcting",
                        leave=False,
                    )
                ]

                t1 = time.time()

                cer = cer_metric.compute(
                    predictions=corrected,
                    references=gt,
                )
                wer = wer_metric.compute(
                    predictions=corrected,
                    references=gt,
                )

                elapsed = t1 - t0
                ram_used = ram_mb()

                print(
                    f"CER={cer:.5f} | WER={wer:.5f} | "
                    f"time={elapsed:.1f}s | RAM={ram_used:.0f}MB"
                )

                results.append(
                    {
                        "prefix_length": prefix_len,
                        "protect_prefix_len": protect_len,
                        "max_candidates": max_cand,
                        "min_token_len": min_tok,
                        "CER": cer,
                        "WER": wer,
                        "time_sec": elapsed,
                        "ram_mb": ram_used,
                    }
                )

                # 💾 incremental save
                pd.DataFrame(results).to_csv(
                    RESULTS_CSV,
                    index=False,
                )

print("\n✅ GRID SEARCH FINISHED")
print(f"Results saved to {RESULTS_CSV}")
