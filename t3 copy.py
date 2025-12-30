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
import time
from collections import Counter, defaultdict

import pandas as pd
from rapidfuzz.distance import Levenshtein
from evaluate import load
from tqdm import tqdm


def count_token_errors(pred_texts, gt_texts):
    """
    Считает количество несовпадающих токенов
    (простое выравнивание по split(), как и в корректоре)
    """
    errors = 0
    total = 0

    for p, g in zip(pred_texts, gt_texts):
        p_toks = p.split()
        g_toks = g.split()
        n = min(len(p_toks), len(g_toks))

        for i in range(n):
            total += 1
            if p_toks[i] != g_toks[i]:
                errors += 1

        # лишние токены тоже считаем ошибками
        extra = abs(len(p_toks) - len(g_toks))
        errors += extra
        total += extra

    return errors, total


class MorphIndexCorrector:
    """
    Морфо-индексный корректировщик:
      - строит топ приставок/окончаний из словаря
      - строит индекс (prefix, stem)->Counter(endings)
      - ищет похожие stems через trigram inverted index
      - генерит кандидатов и реранкает Levenshtein-ом
    """

    def __init__(
        self,
        words,
        max_edit=2,
        min_token_len=3,
        # статистика морфем из словаря
        max_prefix_len=4,
        max_ending_len=6,
        min_stem_len=3,
        top_prefixes=150,
        top_endings=800,
        # генерация/поиск кандидатов
        top_splits=8,
        ngram_n=3,
        max_stem_candidates=200,
        max_endings_per_key=30,
        # фильтры
        protect_prefix_len=2,
        char_map=None,
        skip_hyphen_tokens=True,
        # лёгкий “анти-лекс-подмена” барьер
        dont_touch_short_len=4,
        # лог
        enable_log=True,
    ):
        self.max_edit = max_edit
        self.min_token_len = min_token_len

        self.max_prefix_len = max_prefix_len
        self.max_ending_len = max_ending_len
        self.min_stem_len = min_stem_len
        self.top_prefixes_k = top_prefixes
        self.top_endings_k = top_endings

        self.top_splits = top_splits
        self.ngram_n = ngram_n
        self.max_stem_candidates = max_stem_candidates
        self.max_endings_per_key = max_endings_per_key

        self.protect_prefix_len = protect_prefix_len
        self.char_map = char_map or {}
        self.skip_hyphen_tokens = skip_hyphen_tokens
        self.dont_touch_short_len = dont_touch_short_len

        self.enable_log = enable_log
        self.log = []
        self.cache = {}

        # prefix | core | suffix
        self._punct_re = re.compile(r"^([^\w]*)([\wА-Яа-яёЁ]+)([^\w]*)$")

        # нормализованный словарь
        self.words_raw = set(w.strip() for w in words if w and w.strip())
        self.words_norm = set(self.normalize(w.lower()) for w in self.words_raw)

        self._print_ram("Before morph build")
        self._build_morph_stats_and_index()
        self._build_stem_trigram_index()
        self._print_ram("After morph build")

    # ---------------- utils ----------------
    def _print_ram(self, label):
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

    def split_token(self, token: str):
        m = self._punct_re.match(token)
        if not m:
            return None, token, None
        return m.group(1), m.group(2), m.group(3)

    def is_candidate_token(self, token: str) -> bool:
        if len(token) < self.min_token_len:
            return False
        if any(c.isdigit() for c in token):
            return False
        if any(c in token for c in "_/\\|@#$%^&*+=<>[]{}"):
            return False
        letters = sum(c.isalpha() for c in token)
        return letters / len(token) >= 0.8

    def prefix_ok(self, a: str, b: str) -> bool:
        """
        a,b — уже нормализованные строки
        проверяем совпадение первых protect_prefix_len символов
        с учётом CHAR_MAP (в нормализованных обычно уже не нужно,
        но оставим на случай смешанной формы).
        """
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

    # ---------------- build morph index ----------------
    def _build_morph_stats_and_index(self):
        """
        1) первый проход: считаем частоты всех префиксов/окончаний в допустимых длинах
        2) выбираем топ-K
        3) второй проход: строим индекс (prefix, stem)->Counter(ending)
        """
        pref_cnt = Counter()
        end_cnt = Counter()

        # --- pass 1 ---
        for w_norm in self.words_norm:
            if len(w_norm) < (self.min_stem_len + 1):
                continue

            # приставки (включая пустую)
            pref_cnt[""] += 1
            for lp in range(
                1, min(self.max_prefix_len, len(w_norm) - self.min_stem_len) + 1
            ):
                pref_cnt[w_norm[:lp]] += 1

            # окончания (включая пустое)
            end_cnt[""] += 1
            for le in range(
                1, min(self.max_ending_len, len(w_norm) - self.min_stem_len) + 1
            ):
                end_cnt[w_norm[-le:]] += 1

        # выбираем топы
        self.top_prefixes = [p for p, _ in pref_cnt.most_common(self.top_prefixes_k)]
        self.top_endings = [e for e, _ in end_cnt.most_common(self.top_endings_k)]

        # веса для скоринга разрезов (чтобы брать top_splits)
        # (логарифмирование можно, но и так норм)
        self.pref_weight = {p: c for p, c in pref_cnt.items()}
        self.end_weight = {e: c for e, c in end_cnt.items()}

        print(
            f"[Morph] Top prefixes: {len(self.top_prefixes)} | Top endings: {len(self.top_endings)}"
        )

        top_pref_set = set(self.top_prefixes)
        top_end_set = set(self.top_endings)

        # --- pass 2: индекс ---
        self.morph_index = defaultdict(Counter)  # (prefix, stem)->Counter(endings)
        self.stems_set = set()

        added = 0
        for w_norm in self.words_norm:
            L = len(w_norm)
            if L < (self.min_stem_len + 1):
                continue

            # перебираем только топовые приставки/окончания
            # но не все комбинации: ограничиваемся теми, что действительно совпадают по краям
            pref_cands = []
            for p in self.top_prefixes:
                if p == "" or (len(p) < L and w_norm.startswith(p)):
                    pref_cands.append(p)

            end_cands = []
            for e in self.top_endings:
                if e == "" or (len(e) < L and w_norm.endswith(e)):
                    end_cands.append(e)

            # строим разрезы
            for p in pref_cands:
                for e in end_cands:
                    stem = w_norm[len(p) : L - len(e)]
                    if len(stem) < self.min_stem_len:
                        continue
                    self.morph_index[(p, stem)][e] += 1
                    self.stems_set.add(stem)
                    added += 1

        print(f"[Morph] Index keys: {len(self.morph_index):,}")
        print(f"[Morph] Indexed triples added: {added:,}")

    def _stem_ngrams(self, s: str):
        n = self.ngram_n
        if len(s) < n:
            return [s] if s else []
        return [s[i : i + n] for i in range(len(s) - n + 1)]

    def _build_stem_trigram_index(self):
        """
        Инвертированный индекс n-грамм -> список stems
        Храним stem как строку в stems_list, postings — список индексов.
        """
        self.stems_list = list(self.stems_set)
        self.stem_id = {s: i for i, s in enumerate(self.stems_list)}

        postings = defaultdict(list)
        for i, stem in enumerate(self.stems_list):
            grams = set(self._stem_ngrams(stem))
            for g in grams:
                postings[g].append(i)

        self.postings = postings
        print(
            f"[Morph] Stem grams keys: {len(self.postings):,} | stems: {len(self.stems_list):,}"
        )

    # ---------------- candidate generation ----------------
    def _top_segmentations(self, w_norm: str):
        """
        Возвращает top_splits разрезов (p, stem, e, score),
        где p/e из топовых приставок/окончаний и совпадают по краям.
        """
        L = len(w_norm)
        segs = []

        for p in self.top_prefixes:
            if p != "" and not w_norm.startswith(p):
                continue
            lp = len(p)
            if L - lp < self.min_stem_len:
                continue

            for e in self.top_endings:
                if e != "" and not w_norm.endswith(e):
                    continue
                le = len(e)
                stem = w_norm[lp : L - le]
                if len(stem) < self.min_stem_len:
                    continue

                # скоринг разреза: частота приставки + частота окончания + бонус за длинный stem
                score = (
                    self.pref_weight.get(p, 1)
                    + self.end_weight.get(e, 1)
                    + 2 * len(stem)
                )
                segs.append((p, stem, e, score))

        if not segs:
            return []

        segs.sort(key=lambda x: x[3], reverse=True)
        return segs[: self.top_splits]

    def _find_similar_stems(self, stem: str):
        """
        Быстрый поиск похожих stems:
          - считаем пересечения по n-граммам
          - берём top по счёту пересечений
        """
        if stem in self.stem_id:
            # всегда включаем exact
            exact_id = self.stem_id[stem]
        else:
            exact_id = None

        grams = set(self._stem_ngrams(stem))
        if not grams:
            return []

        scores = defaultdict(int)
        for g in grams:
            for sid in self.postings.get(g, []):
                scores[sid] += 1

        if exact_id is not None:
            scores[exact_id] += 10_000  # гарантированно в топ

        # берём top-N
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[
            : self.max_stem_candidates
        ]
        return [self.stems_list[sid] for sid, _ in top]

    # ---------------- correction ----------------
    def correct_word(self, word: str):
        word_l = word.lower()
        w_norm = self.normalize(word_l)

        # 0) очень короткие — не трогаем (защита от "сил->сие", "суд->суа" и т.п.)
        if len(w_norm) <= self.dont_touch_short_len:
            return word

        # 1) уже словарное — не трогаем
        if w_norm in self.words_norm:
            return word

        # 2) кеш
        if w_norm in self.cache:
            return self.cache[w_norm]

        # 3) top сегментации
        segs = self._top_segmentations(w_norm)
        if not segs:
            self.cache[w_norm] = word
            return word

        best = w_norm
        best_dist = self.max_edit + 1

        # 4) перебор сегментаций -> похожие stems -> допустимые endings
        for p, stem, e_ocr, _score in segs:
            # похожие основы
            stem_cands = self._find_similar_stems(stem)
            if not stem_cands:
                continue

            # фиксируем префикс-кандидат (можно расширять позже: похожие приставки)
            p_cand = p

            for stem2 in stem_cands:
                key = (p_cand, stem2)
                if key not in self.morph_index:
                    continue

                # топ окончаний для данного (p,stem)
                endings = self.morph_index[key].most_common(self.max_endings_per_key)

                for end2, _cnt in endings:
                    cand = f"{p_cand}{stem2}{end2}"

                    # быстрый фильтр: первые protect_prefix_len должны совпасть
                    if not self.prefix_ok(w_norm, cand):
                        continue

                    d = Levenshtein.distance(w_norm, cand)
                    if d < best_dist:
                        best_dist = d
                        best = cand
                        if d == 0:
                            break

        if best_dist <= self.max_edit and best != w_norm:
            out = self.restore_case(word, best)
        else:
            out = word

        self.cache[w_norm] = out
        return out

    def correct_text(self, text: str, row_id: int, gt_text: str):
        tokens = text.split()
        out = []
        changed = False

        for t in tokens:
            p, core, s = self.split_token(t)
            if p is None:
                out.append(t)
                continue

            # тумблер дефисов/переносов
            if self.skip_hyphen_tokens and ("-" in p or "-" in s):
                out.append(t)
                continue

            if self.is_candidate_token(core):
                new_core = self.correct_word(core)
                new_tok = f"{p}{new_core}{s}"

                if new_tok != t and self.enable_log:
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

        if changed and self.enable_log:
            for item in reversed(self.log):
                if item["row_id"] != row_id:
                    break
                if item["after_text"] is None:
                    item["after_text"] = new_text

        return new_text

    def save_log(self, path):
        if not self.enable_log:
            return
        if self.log:
            pd.DataFrame(self.log).to_csv(path, index=False, encoding="utf-8")
            print(f"Saved {len(self.log)} corrections → {path}")
        else:
            print("No corrections.")


# ========================== RUN (EXPERIMENT) ==========================
if __name__ == "__main__":
    GT_PATH = "YeniseiGovReports-HWR_gt_mapped.csv"
    PRED_PATH = "YeniseiGovReports-HWR_trba_lite_g1_mapped.csv"
    DICT_PATH = "all_words_with_gt.txt"

    LIMIT = 5000
    LOG_PATH = "morph_v2_log.csv"

    # ---- load data ----
    gt = pd.read_csv(GT_PATH)["text"].astype(str).tolist()[:LIMIT]
    pred = pd.read_csv(PRED_PATH)["prediction"].astype(str).tolist()[:LIMIT]

    with open(DICT_PATH, encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]

    # ---- metrics ----
    cer_metric = load("cer")
    wer_metric = load("wer")

    print("\n=== BEFORE ===")
    print("CER:", cer_metric.compute(predictions=pred, references=gt))
    print("WER:", wer_metric.compute(predictions=pred, references=gt))

    err_before, total_tokens = count_token_errors(pred, gt)
    print(f"[Errors BEFORE] {err_before} / {total_tokens}")

    # ---- build corrector ----
    t0 = time.time()
    corrector = MorphIndexCorrector(
        words=words,
        max_edit=2,
        min_token_len=3,
        # морфо-статистика
        max_prefix_len=4,
        max_ending_len=6,
        min_stem_len=3,
        top_prefixes=150,  # можно крутить
        top_endings=800,  # можно крутить
        # поиск/генерация
        top_splits=8,  # можно крутить
        ngram_n=3,
        max_stem_candidates=200,  # можно крутить
        max_endings_per_key=30,  # можно крутить
        # фильтры
        protect_prefix_len=2,
        char_map=CHAR_MAP,
        skip_hyphen_tokens=True,
        dont_touch_short_len=4,
        enable_log=True,
    )
    t1 = time.time()
    print(f"[Time] Build: {t1 - t0:.1f}s")

    # ---- correct ----
    print("\nCorrecting (MORPH-V2)...")
    corrected = [
        corrector.correct_text(p, i, gt[i])
        for i, p in tqdm(enumerate(pred), total=len(pred))
    ]

    err_after, _ = count_token_errors(corrected, gt)
    print(f"[Errors AFTER]  {err_after} / {total_tokens}")

    if err_before > 0:
        eff = err_after / err_before
        print(f"[Efficiency] errors_after / errors_before = {eff:.4f}")
        print(f"[Reduction] {(1 - eff) * 100:.2f}%")

    print("\n=== AFTER (MORPH-V2) ===")
    print("CER:", cer_metric.compute(predictions=corrected, references=gt))
    print("WER:", wer_metric.compute(predictions=corrected, references=gt))

    corrector.save_log(LOG_PATH)
