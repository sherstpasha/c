"""
Simple OCR Corrector: только правила + LogisticRegression
Без Gate модели. Используется только логистическая регрессия.

Возможности:
1. Языковые признаки (use_ngrams, use_entropy):
   - Биграммы слева/справа от символа
   - Триграммы (символ с контекстом)
   - Энтропия символа (сколько вариантов замен)

2. Beam Search (use_beam_search):
   - Рассматривает несколько кандидатов на каждой позиции
   - Может делать несколько замен в одном слове
   - Параметры: beam_size, beam_lambda, top_k_candidates

Запуск:
  python simple_corrector.py              # обычный режим
  python simple_corrector.py optuna 50    # оптимизация Optuna
"""

import pandas as pd
import numpy as np
import re
import Levenshtein
import evaluate
import warnings

from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import Parallel, delayed


# ======================================================
# ПАРАМЕТРЫ
# ======================================================

CONFIG = {
    # === Данные ===
    "data_path": "pairs.csv",
    "ocr_column": "incorrect",
    "gt_column": "correct",
    "image_column": "image",
    # === Модель ===
    "model_type": "logistic",
    # === LogisticRegression параметры ===
    "lr_C": 1.0,
    "lr_max_iter": 300,
    # === Corrector параметры ===
    "context_size": 2,
    "min_rule_freq": 3,
    "negative_ratio": 2.0,
    "prob_threshold": 0.5,
    "gate_k": 2.0,  # p_change > p_no_change * gate_k
    # === Языковые признаки ===
    "use_ngrams": True,  # биграммы и триграммы
    "use_entropy": True,  # энтропия символа
    # === Beam Search ===
    "use_beam_search": False,  # включить beam search
    "beam_size": 5,  # размер beam
    "beam_lambda": 0.1,  # вес языковой модели
    "top_k_candidates": 3,  # топ-K кандидатов на позицию
    # === Parallelism ===
    "n_jobs": 1,
    # === Общие ===
    "random_state": 42,
}


# ======================================================
# УТИЛИТЫ
# ======================================================

WORD_CHAR_RE = re.compile(r"^\w+$", re.UNICODE)
LATIN_RE = re.compile(r"[a-zA-Z]")
DIGIT_RE = re.compile(r"\d")

NO_CHANGE = "__NO_CHANGE__"


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


def is_case_only(a: str, b: str) -> bool:
    return a.lower() == b.lower() and a != b


def is_valid_word(w: str) -> bool:
    if DIGIT_RE.search(w):
        return False
    return True


def normalize_word(w: str) -> str:
    return str(w).strip()


# ======================================================
# SIMPLE CORRECTOR
# ======================================================


class SimpleCorrector:
    """
    Простой корректор OCR без Gate модели.
    Использует правила замен + классификатор.
    """

    def __init__(self, config: Dict = None):
        self.config = config or CONFIG.copy()

        self.vec = None
        self.clf = None
        self.classes = None
        self.no_change_idx = None
        self.df = None
        
        # Статистика для энтропии символов
        self.char_entropy = {}  # ocr_char -> entropy

    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Загрузить данные"""
        data_path = data_path or self.config["data_path"]

        df = pd.read_csv(data_path)
        df = df.rename(
            columns={
                self.config["ocr_column"]: "ocr",
                self.config["gt_column"]: "gt",
                self.config["image_column"]: "image",
            }
        )

        df = df.dropna(subset=["gt", "ocr"])
        df["ocr"] = df["ocr"].apply(normalize_word)
        df["gt"] = df["gt"].apply(normalize_word)
        df = df[df["ocr"].apply(is_valid_word)]

        self.df = df
        print(f"Loaded {len(df)} word pairs")

        return df

    def _extract_positive_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечь positive samples (реальные OCR ошибки)"""
        pos_rows = []
        ctx = self.config["context_size"]

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

                if not is_valid_symbol(ch_ocr) or not is_valid_symbol(ch_gt):
                    continue

                if is_case_only(ch_ocr, ch_gt):
                    continue

                if is_latin(ch_ocr) or is_latin(ch_gt):
                    continue

                left = ocr[max(0, i - ctx) : i]
                right = ocr[i + 1 : i + 1 + ctx]

                pos_rows.append(
                    {
                        "ocr": ch_ocr,
                        "y": ch_gt,
                        "left": left,
                        "right": right,
                        "pos_in_word": i,
                        "word_len": len(ocr),
                    }
                )

        pos_df = pd.DataFrame(pos_rows)

        if pos_df.empty:
            return pos_df

        # фильтр по частоте
        min_freq = self.config["min_rule_freq"]
        pos_df = pos_df[
            pos_df.groupby(["ocr", "y"])["ocr"].transform("count") >= min_freq
        ]

        return pos_df

    def _extract_negative_samples(
        self, df: pd.DataFrame, n_positive: int
    ) -> pd.DataFrame:
        """Извлечь negative samples (не нужно исправлять)"""
        neg_rows = []
        ctx = self.config["context_size"]

        for _, row in df.iterrows():
            ocr = str(row["ocr"])
            gt = str(row["gt"])

            for i in range(min(len(ocr), len(gt))):
                if ocr[i] == gt[i] and is_valid_symbol(ocr[i]):
                    left = ocr[max(0, i - ctx) : i]
                    right = ocr[i + 1 : i + 1 + ctx]

                    neg_rows.append(
                        {
                            "ocr": ocr[i],
                            "y": NO_CHANGE,
                            "left": left,
                            "right": right,
                            "pos_in_word": i,
                            "word_len": len(ocr),
                        }
                    )

        neg_df = pd.DataFrame(neg_rows)

        # балансировка
        n_neg = int(n_positive * self.config["negative_ratio"])
        if len(neg_df) > n_neg:
            neg_df = neg_df.sample(n=n_neg, random_state=self.config["random_state"])

        return neg_df

    def _make_features(self, row) -> Dict:
        """Создать признаки для одного примера"""
        ch = row['ocr']
        left = row['left']
        right = row['right']
        
        feats = {f"ocr={ch}": 1}

        # Контекст слева
        for i, c in enumerate(left[::-1]):
            if is_valid_symbol(c):
                feats[f"L{i+1}={c}"] = 1

        # Контекст справа
        for i, c in enumerate(right):
            if is_valid_symbol(c):
                feats[f"R{i+1}={c}"] = 1

        # Позиционные признаки
        feats["pos_ratio"] = row["pos_in_word"] / max(row["word_len"], 1)
        feats["is_start"] = 1 if row["pos_in_word"] == 0 else 0
        feats["is_end"] = 1 if row["pos_in_word"] == row["word_len"] - 1 else 0

        # === ЯЗЫКОВЫЕ ПРИЗНАКИ ===
        if self.config.get("use_ngrams", True):
            # Биграммы
            if len(left) > 0 and is_valid_symbol(left[-1]):
                feats[f"bigram_L={left[-1]}{ch}"] = 1
            if len(right) > 0 and is_valid_symbol(right[0]):
                feats[f"bigram_R={ch}{right[0]}"] = 1
            
            # Триграммы
            if len(left) > 0 and len(right) > 0:
                if is_valid_symbol(left[-1]) and is_valid_symbol(right[0]):
                    feats[f"trigram={left[-1]}{ch}{right[0]}"] = 1

        # === ЭНТРОПИЯ СИМВОЛА ===
        if self.config.get("use_entropy", True):
            entropy = self.char_entropy.get(ch, 0.0)
            feats["ocr_entropy"] = entropy

        return feats

    def _create_model(self):
        """Создать модель по конфигу"""
        return LogisticRegression(
            C=self.config["lr_C"],
            max_iter=self.config["lr_max_iter"],
            random_state=self.config["random_state"],
        )

    def _compute_char_entropy(self, pos_df: pd.DataFrame):
        """Вычислить энтропию для каждого OCR символа"""
        from collections import defaultdict
        
        char_replacements = defaultdict(lambda: defaultdict(int))
        
        # Подсчитываем частоты замен
        for _, row in pos_df.iterrows():
            ocr_ch = row['ocr']
            y_ch = row['y']
            char_replacements[ocr_ch][y_ch] += 1
        
        # Вычисляем энтропию
        self.char_entropy = {}
        for ocr_ch, replacements in char_replacements.items():
            total = sum(replacements.values())
            entropy = 0.0
            for count in replacements.values():
                p = count / total
                if p > 0:
                    entropy -= p * np.log2(p)
            self.char_entropy[ocr_ch] = entropy
        
        print(f"Computed entropy for {len(self.char_entropy)} characters")

    def fit(self, df: pd.DataFrame = None) -> "SimpleCorrector":
        """Обучить модель"""
        df = df if df is not None else self.df

        # Positive samples
        pos_df = self._extract_positive_samples(df)
        print(f"Positive samples: {len(pos_df)}")
        
        # Вычисляем энтропию символов
        if self.config.get("use_entropy", True):
            self._compute_char_entropy(pos_df)

        # Negative samples
        neg_df = self._extract_negative_samples(df, len(pos_df))
        print(f"Negative samples: {len(neg_df)}")

        # Объединяем
        train_df = pd.concat([pos_df, neg_df]).sample(
            frac=1, random_state=self.config["random_state"]
        )
        print(f"Total training samples: {len(train_df)}")

        if train_df.empty:
            raise ValueError("No training samples!")

        # Features
        X = train_df.apply(self._make_features, axis=1)
        y = train_df["y"]

        # Vectorize
        self.vec = DictVectorizer(sparse=True)
        X_vec = self.vec.fit_transform(X)

        # Train
        print(f"Training {self.config['model_type']} model...")
        self.clf = self._create_model()
        self.clf.fit(X_vec, y)

        self.classes = list(self.clf.classes_)
        self.no_change_idx = self.classes.index(NO_CHANGE)

        print(f"Model trained. Classes: {len(self.classes)}")

        return self

    def _get_position_candidates(self, ocr: str, pos: int) -> List[Tuple[str, float]]:
        """Получить топ-K кандидатов для позиции"""
        ctx = self.config["context_size"]
        prob_threshold = self.config["prob_threshold"]
        gate_k = self.config["gate_k"]
        top_k = self.config.get("top_k_candidates", 3)

        ch = ocr[pos]
        if not is_valid_symbol(ch):
            return [(ch, 1.0)]  # NO_CHANGE с вероятностью 1

        left = ocr[max(0, pos - ctx) : pos]
        right = ocr[pos + 1 : pos + 1 + ctx]

        # Формируем признаки (с учётом новых фич)
        feats = {f"ocr={ch}": 1}
        for i, c in enumerate(left[::-1]):
            if is_valid_symbol(c):
                feats[f"L{i+1}={c}"] = 1
        for i, c in enumerate(right):
            if is_valid_symbol(c):
                feats[f"R{i+1}={c}"] = 1

        feats["pos_ratio"] = pos / max(len(ocr), 1)
        feats["is_start"] = 1 if pos == 0 else 0
        feats["is_end"] = 1 if pos == len(ocr) - 1 else 0

        # N-граммы
        if self.config.get("use_ngrams", True):
            if len(left) > 0 and is_valid_symbol(left[-1]):
                feats[f"bigram_L={left[-1]}{ch}"] = 1
            if len(right) > 0 and is_valid_symbol(right[0]):
                feats[f"bigram_R={ch}{right[0]}"] = 1
            if len(left) > 0 and len(right) > 0:
                if is_valid_symbol(left[-1]) and is_valid_symbol(right[0]):
                    feats[f"trigram={left[-1]}{ch}{right[0]}"] = 1

        # Энтропия
        if self.config.get("use_entropy", True):
            entropy = self.char_entropy.get(ch, 0.0)
            feats["ocr_entropy"] = entropy

        X = self.vec.transform([feats])
        probs = self.clf.predict_proba(X)[0]

        p_no = probs[self.no_change_idx]

        candidates = [(ch, p_no)]  # NO_CHANGE всегда включаем

        for cls, p in zip(self.classes, probs):
            if cls == NO_CHANGE:
                continue
            if is_latin(cls):
                continue

            # Фильтрация
            if p < prob_threshold:
                continue
            if p <= p_no * gate_k:
                continue

            candidates.append((cls, p))

        # Сортируем по вероятности и берём топ-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def _beam_search(self, ocr: str) -> Tuple[str, float]:
        """Beam Search по символам слова"""
        beam_size = self.config.get("beam_size", 5)
        beam_lambda = self.config.get("beam_lambda", 0.1)

        # Beam: [(word, log_prob)]
        beam = [(ocr, 0.0)]

        for pos in range(len(ocr)):
            new_beam = []

            for current_word, current_score in beam:
                candidates = self._get_position_candidates(current_word, pos)

                for new_char, prob in candidates:
                    # Применяем замену
                    new_word = current_word[:pos] + new_char + current_word[pos + 1:]
                    
                    # Считаем score
                    log_prob = np.log(prob + 1e-10)
                    new_score = current_score + log_prob

                    # TODO: можно добавить языковую модель
                    # new_score += beam_lambda * language_model_score(new_word)

                    new_beam.append((new_word, new_score))

            # Оставляем топ beam_size
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_size]

        # Возвращаем лучший результат
        best_word, best_score = beam[0]
        return best_word, best_score

    def correct_word(self, ocr: str) -> Tuple[str, Optional[Dict]]:
        """Исправить одно слово"""
        if self.config.get("use_beam_search", False):
            corrected, score = self._beam_search(ocr)
            if corrected != ocr:
                return corrected, {"method": "beam_search", "score": score}
            return ocr, None

        # Старый greedy подход
        best = None
        ctx = self.config["context_size"]
        prob_threshold = self.config["prob_threshold"]
        gate_k = self.config["gate_k"]

        for pos in range(len(ocr)):
            ch = ocr[pos]
            if not is_valid_symbol(ch):
                continue

            left = ocr[max(0, pos - ctx) : pos]
            right = ocr[pos + 1 : pos + 1 + ctx]

            feats = {f"ocr={ch}": 1}
            for i, c in enumerate(left[::-1]):
                if is_valid_symbol(c):
                    feats[f"L{i+1}={c}"] = 1
            for i, c in enumerate(right):
                if is_valid_symbol(c):
                    feats[f"R{i+1}={c}"] = 1

            feats["pos_ratio"] = pos / max(len(ocr), 1)
            feats["is_start"] = 1 if pos == 0 else 0
            feats["is_end"] = 1 if pos == len(ocr) - 1 else 0
            
            # N-граммы
            if self.config.get("use_ngrams", True):
                if len(left) > 0 and is_valid_symbol(left[-1]):
                    feats[f"bigram_L={left[-1]}{ch}"] = 1
                if len(right) > 0 and is_valid_symbol(right[0]):
                    feats[f"bigram_R={ch}{right[0]}"] = 1
                if len(left) > 0 and len(right) > 0:
                    if is_valid_symbol(left[-1]) and is_valid_symbol(right[0]):
                        feats[f"trigram={left[-1]}{ch}{right[0]}"] = 1

            # Энтропия
            if self.config.get("use_entropy", True):
                entropy = self.char_entropy.get(ch, 0.0)
                feats["ocr_entropy"] = entropy

            X = self.vec.transform([feats])
            probs = self.clf.predict_proba(X)[0]

            p_no = probs[self.no_change_idx]

            for cls, p in zip(self.classes, probs):
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
            corrected = ocr[: best["pos"]] + best["char"] + ocr[best["pos"] + 1 :]
            return corrected, best

        return ocr, None

    def evaluate(self, df: pd.DataFrame = None) -> Dict:
        """Оценить качество"""
        df = df if df is not None else self.df

        cer_metric = evaluate.load("cer")

        # parallelize per-row correction & scoring
        n_jobs = self.config.get("n_jobs", 1)

        def _process_row(row):
            ocr = str(row["ocr"])
            gt = str(row["gt"])
            corrected, info = self.correct_word(ocr)

            applied = 0
            improved = 0
            worsened = 0

            if info is not None:
                applied = 1
                dist_before = Levenshtein.distance(ocr, gt)
                dist_after = Levenshtein.distance(corrected, gt)
                if dist_after < dist_before:
                    improved = 1
                elif dist_after > dist_before:
                    worsened = 1

            return gt, ocr, corrected, applied, improved, worsened

        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_row)(row) for _, row in df.iterrows()
        )

        refs = [r[0] for r in results]
        ocr_preds = [r[1] for r in results]
        corr_preds = [r[2] for r in results]
        applied = sum(r[3] for r in results)
        improved = sum(r[4] for r in results)
        worsened = sum(r[5] for r in results)

        cer_before = cer_metric.compute(predictions=ocr_preds, references=refs)
        cer_after = cer_metric.compute(predictions=corr_preds, references=refs)

        acc_before = sum(o == g for o, g in zip(ocr_preds, refs)) / len(refs)
        acc_after = sum(c == g for c, g in zip(corr_preds, refs)) / len(refs)

        return {
            "cer_before": cer_before,
            "cer_after": cer_after,
            "cer_delta": cer_before - cer_after,
            "acc_before": acc_before,
            "acc_after": acc_after,
            "acc_delta": acc_after - acc_before,
            "total_words": len(df),
            "corrections_applied": applied,
            "improved": improved,
            "worsened": worsened,
            "no_effect": applied - improved - worsened,
            "precision": improved / applied if applied > 0 else 0,
        }

    def print_evaluation(self, results: Dict = None):
        """Вывести результаты"""
        if results is None:
            results = self.evaluate()

        print()
        print("=" * 50)
        print(f"MODEL: {self.config['model_type']}")
        print("=" * 50)
        print()
        print(f"Total words:           {results['total_words']}")
        print(f"Corrections applied:   {results['corrections_applied']}")
        print()
        print("--- CER ---")
        print(f"CER before:            {results['cer_before']:.4f}")
        print(f"CER after:             {results['cer_after']:.4f}")
        print(f"CER delta:             {results['cer_delta']:+.4f}")
        print()
        print("--- Word Accuracy ---")
        print(f"Accuracy before:       {results['acc_before']:.4f}")
        print(f"Accuracy after:        {results['acc_after']:.4f}")
        print(f"Accuracy delta:        {results['acc_delta']:+.4f}")
        print()
        print("--- Correction Quality ---")
        print(f"Improved:              {results['improved']}")
        print(f"Worsened:              {results['worsened']}")
        print(f"No effect:             {results['no_effect']}")
        print(f"Precision:             {results['precision']:.4f}")


# ======================================================
# OPTUNA OPTIMIZATION
# ======================================================


def optuna_optimize(config_base: Dict = None, n_trials: int = 100, n_jobs: int = 1):
    """Optuna оптимизация"""
    import optuna

    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    config_base = config_base or CONFIG.copy()

    # Загружаем данные один раз
    corr = SimpleCorrector(config_base)
    df = corr.load_data()

    def objective(trial):
        config = config_base.copy()

        # === Параметры LogisticRegression ===
        config["lr_C"] = trial.suggest_float("lr_C", 0.01, 10.0, log=True)
        config["lr_max_iter"] = trial.suggest_int("lr_max_iter", 100, 500)

        # === Общие параметры корректора ===
        config["context_size"] = trial.suggest_int("context_size", 1, 4)
        config["min_rule_freq"] = trial.suggest_int("min_rule_freq", 2, 10)
        config["negative_ratio"] = trial.suggest_float("negative_ratio", 1.0, 5.0)
        config["prob_threshold"] = trial.suggest_float("prob_threshold", 0.2, 0.8)
        config["gate_k"] = trial.suggest_float("gate_k", 0.5, 10.0)

        # === Языковые признаки ===
        config["use_ngrams"] = trial.suggest_categorical("use_ngrams", [True, False])
        config["use_entropy"] = trial.suggest_categorical("use_entropy", [True, False])

        # === Beam Search ===
        config["use_beam_search"] = trial.suggest_categorical("use_beam_search", [True, False])
        if config["use_beam_search"]:
            config["beam_size"] = trial.suggest_int("beam_size", 3, 10)
            config["beam_lambda"] = trial.suggest_float("beam_lambda", 0.0, 1.0)
            config["top_k_candidates"] = trial.suggest_int("top_k_candidates", 2, 5)

        try:
            import io, sys

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            corr = SimpleCorrector(config)
            corr.df = df
            corr.fit(df)
            result = corr.evaluate(df)

            sys.stdout = old_stdout

            applied = result["corrections_applied"]
            precision = result["precision"]

            if applied < 10:
                return 0.0

            # Score: precision с бонусом за количество
            score = precision * min(1.0, applied / 50)

            trial.set_user_attr("precision", precision)
            trial.set_user_attr("applied", applied)
            trial.set_user_attr("improved", result["improved"])
            trial.set_user_attr("worsened", result["worsened"])
            trial.set_user_attr("cer_delta", result["cer_delta"])
            trial.set_user_attr("acc_delta", result["acc_delta"])

            return score

        except Exception as e:
            return 0.0

    print(f"Starting Optuna optimization with {n_trials} trials...")
    print("=" * 70)

    study = optuna.create_study(direction="maximize")
    # run trials in parallel if n_jobs > 1
    if n_jobs and int(n_jobs) > 1:
        study.optimize(objective, n_trials=n_trials, n_jobs=int(n_jobs), show_progress_bar=True)
    else:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Результаты
    print("\n" + "=" * 70)
    print("OPTUNA OPTIMIZATION COMPLETE")
    print("=" * 70)

    best = study.best_trial
    print(f"\nBest score: {best.value:.4f}")
    print(f"\n=== BEST PARAMETERS ===")
    for key, value in best.params.items():
        print(f"  {key}: {value}")

    print(f"\n=== BEST TRIAL METRICS ===")
    print(f"  Precision: {best.user_attrs.get('precision', 0):.4f}")
    print(f"  Applied: {best.user_attrs.get('applied', 0)}")
    print(f"  Improved: {best.user_attrs.get('improved', 0)}")
    print(f"  Worsened: {best.user_attrs.get('worsened', 0)}")
    print(f"  CER delta: {best.user_attrs.get('cer_delta', 0):.6f}")
    print(f"  Acc delta: {best.user_attrs.get('acc_delta', 0):.6f}")

    # Сохраняем
    trials_df = study.trials_dataframe()
    trials_df.to_csv("simple_optuna_trials.csv", index=False)
    print(f"\nTrials saved to: simple_optuna_trials.csv")

    best_config = config_base.copy()
    best_config.update(best.params)

    return study, best_config


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "optuna":
        n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else CONFIG.get("n_jobs", 1)
        study, best_config = optuna_optimize(n_trials=n_trials, n_jobs=n_jobs)

        print("\n" + "=" * 70)
        print("RUNNING WITH BEST PARAMETERS")
        print("=" * 70)

        corr = SimpleCorrector(best_config)
        df = corr.load_data()
        corr.fit(df)
        result = corr.evaluate(df)
        corr.print_evaluation(result)

    else:
        # Обычный режим
        config = CONFIG.copy()

        corr = SimpleCorrector(config)
        df = corr.load_data()
        corr.fit(df)
        result = corr.evaluate(df)
        corr.print_evaluation(result)

        # Примеры
        print("\n" + "=" * 50)
        print("EXAMPLES")
        print("=" * 50)

        test_words = ["Воличеству", "Енисейскаго", "допосъ", "благопосично"]
        for word in test_words:
            corrected, info = corr.correct_word(word)
            if info:
                print(f"'{word}' -> '{corrected}' (prob={info['prob']:.3f})")
            else:
                print(f"'{word}' -> '{corrected}' (no change)")
