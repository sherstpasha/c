"""
OCR Corrector: Gate Model + Rule-based Correction

Логика:
1. Gate модель решает, нужно ли исправлять слово
2. Если gate пропускает (prob >= threshold), применяется корректор
3. Корректор использует правила OCR→GT для исправления
"""

import pandas as pd
import numpy as np
import re
import Levenshtein
import evaluate

from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve


# ======================================================
# ПАРАМЕТРЫ (настраиваемые)
# ======================================================

CONFIG = {
    # === Пути к данным ===
    "data_path": "pairs.csv",  # путь к файлу с парами (ocr, gt)
    "ocr_column": "incorrect",  # колонка с OCR текстом
    "gt_column": "correct",  # колонка с GT текстом
    "image_column": "image",  # колонка с именем изображения
    # === Gate Model параметры ===
    "gate_ngram_range": (1, 3),  # диапазон n-грамм для gate
    "gate_min_df": 3,  # минимальная частота n-граммы
    "gate_max_iter": 300,  # макс итераций LogReg
    "gate_class_weight": "balanced",  # баланс классов
    "gate_target_precision": 0.7,  # целевая точность для подбора порога
    "gate_threshold": None,  # можно задать фиксированный порог (иначе подберётся)
    # === Rule Extractor параметры ===
    "rule_max_ngram": 3,  # максимальная длина n-граммы правила
    "rule_max_context_dist": 1,  # макс расстояние контекста слева/справа
    "rule_context_size": 3,  # размер контекста для правила
    "rule_min_count": 3,  # минимальная частота правила
    # === Corrector Model параметры ===
    "corr_context_size": 2,  # размер контекста слева/справа
    "corr_max_iter": 300,  # макс итераций LogReg
    "corr_min_rule_freq": 3,  # минимальная частота замены (ocr, gt)
    "corr_negative_ratio": 2.0,  # соотношение negative/positive samples
    "corr_prob_threshold": 0.5,  # порог вероятности для применения правки
    "corr_gate_k": 2.0,  # коэффициент gating (p_change > p_no_change * k)
    # === Общие параметры ===
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
    """Проверка валидности символа"""
    if not s:
        return False
    if not WORD_CHAR_RE.match(s):
        return False
    if DIGIT_RE.search(s):
        return False
    if "_" in s:
        return False
    return True


def is_valid_ngram(s: str) -> bool:
    """Проверка валидности n-граммы"""
    return s and all(is_valid_symbol(c) for c in s)


def is_latin(s: str) -> bool:
    """Содержит ли латиницу"""
    return bool(LATIN_RE.search(s))


def is_case_only(a: str, b: str) -> bool:
    """Отличие только в регистре"""
    return a.lower() == b.lower() and a != b


def is_valid_word(w: str) -> bool:
    """Валидно ли слово для обработки"""
    if DIGIT_RE.search(w):
        return False
    return True


def normalize_word(w: str) -> str:
    """Нормализация слова"""
    return str(w).strip()


def add_position_markers(w: str) -> str:
    """Добавление маркеров начала/конца"""
    return f"^{w}$"


# ======================================================
# GATE MODEL
# ======================================================


class GateModel:
    """
    Gate модель: определяет, нужно ли исправлять слово.
    Использует char n-grams + LogisticRegression.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 3,
        max_iter: int = 300,
        class_weight: str = "balanced",
        target_precision: float = 0.9,
        threshold: Optional[float] = None,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.target_precision = target_precision
        self.threshold = threshold

        self.vectorizer = None
        self.clf = None
        self.fitted_threshold = None

    def fit(self, df: pd.DataFrame) -> "GateModel":
        """
        Обучение gate модели.
        df должен содержать колонки 'ocr' и 'gt'
        """
        # Labels: need_fix = 1 если ocr != gt
        y = (df["ocr"] != df["gt"]).astype(int).values

        # Features: char n-grams с маркерами позиции
        texts = df["ocr"].apply(add_position_markers)

        self.vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )

        X = self.vectorizer.fit_transform(texts)

        self.clf = LogisticRegression(
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            n_jobs=-1,
        )

        self.clf.fit(X, y)

        # Подбор порога
        if self.threshold is not None:
            self.fitted_threshold = self.threshold
        else:
            probs = self.clf.predict_proba(X)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y, probs)

            # Ищем порог с precision >= target И максимальным recall
            valid_idx = np.where(precision >= self.target_precision)[0]
            if len(valid_idx) > 0:
                # Берём первый валидный (с макс recall при заданной precision)
                idx = valid_idx[0]
            else:
                # Если нет подходящего, берём порог с макс F1
                f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
                idx = np.argmax(f1_scores)

            self.fitted_threshold = thresholds[min(idx, len(thresholds) - 1)]

        return self

    def predict_proba(self, words: List[str]) -> np.ndarray:
        """Вероятность того, что слово нужно исправить"""
        texts = [add_position_markers(w) for w in words]
        X = self.vectorizer.transform(texts)
        return self.clf.predict_proba(X)[:, 1]

    def predict(self, words: List[str]) -> np.ndarray:
        """Бинарное решение: исправлять или нет"""
        probs = self.predict_proba(words)
        return (probs >= self.fitted_threshold).astype(int)

    def get_stats(self, df: pd.DataFrame) -> Dict:
        """Статистика качества gate модели"""
        y = (df["ocr"] != df["gt"]).astype(int).values
        probs = self.predict_proba(df["ocr"].tolist())

        roc_auc = roc_auc_score(y, probs)

        precision, recall, _ = precision_recall_curve(y, probs)

        # precision/recall при текущем пороге
        preds = (probs >= self.fitted_threshold).astype(int)
        tp = ((preds == 1) & (y == 1)).sum()
        fp = ((preds == 1) & (y == 0)).sum()
        fn = ((preds == 0) & (y == 1)).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0

        return {
            "roc_auc": roc_auc,
            "threshold": self.fitted_threshold,
            "precision": prec,
            "recall": rec,
        }


# ======================================================
# RULE EXTRACTOR
# ======================================================


class RuleExtractor:
    """
    Извлекает правила OCR→GT из пар слов.
    """

    def __init__(
        self,
        max_ngram: int = 3,
        max_context_dist: int = 1,
        context_size: int = 3,
        min_count: int = 3,
    ):
        self.max_ngram = max_ngram
        self.max_context_dist = max_context_dist
        self.context_size = context_size
        self.min_count = min_count

    def extract_rules_from_pair(self, ocr: str, gt: str) -> List[Dict]:
        """Извлечь правила из одной пары"""
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
                        "left_ctx": ocr[max(0, i - self.context_size) : i],
                        "right_ctx": ocr[i + n : i + n + self.context_size],
                        "left_dist": left_dist,
                        "right_dist": right_dist,
                    }
                )

        return rules

    def build_rule_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Построить таблицу правил из датасета"""
        all_rules = []

        for _, row in df.iterrows():
            ocr = str(row["ocr"])
            gt = str(row["gt"])
            all_rules.extend(self.extract_rules_from_pair(ocr, gt))

        if not all_rules:
            return pd.DataFrame()

        raw_df = pd.DataFrame(all_rules)

        # считаем частоты
        raw_df["count"] = raw_df.groupby(["ocr_ngram", "gt_ngram", "len"])[
            "ocr_ngram"
        ].transform("count")

        # фильтрация по частоте
        filtered_df = raw_df[raw_df["count"] >= self.min_count].copy()

        return filtered_df


# ======================================================
# CORRECTOR MODEL
# ======================================================


class CorrectorModel:
    """
    Модель корректора: применяет правила OCR→GT.
    Использует контекст + LogisticRegression для выбора правильной замены.
    """

    def __init__(
        self,
        context_size: int = 2,
        max_iter: int = 300,
        min_rule_freq: int = 3,
        negative_ratio: float = 2.0,
        prob_threshold: float = 0.6,
        gate_k: float = 3.0,
        random_state: int = 42,
    ):
        self.context_size = context_size
        self.max_iter = max_iter
        self.min_rule_freq = min_rule_freq
        self.negative_ratio = negative_ratio
        self.prob_threshold = prob_threshold
        self.gate_k = gate_k
        self.random_state = random_state

        self.vec = None
        self.clf = None
        self.classes = None
        self.no_change_idx = None

    def _extract_positive_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечь positive samples (реальные OCR ошибки)"""
        pos_rows = []

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

                left = ocr[max(0, i - self.context_size) : i]
                right = ocr[i + 1 : i + 1 + self.context_size]

                pos_rows.append(
                    {
                        "ocr": ch_ocr,
                        "y": ch_gt,
                        "left": left,
                        "right": right,
                    }
                )

        pos_df = pd.DataFrame(pos_rows)

        if pos_df.empty:
            return pos_df

        # фильтр по частоте
        pos_df = pos_df[
            pos_df.groupby(["ocr", "y"])["ocr"].transform("count") >= self.min_rule_freq
        ]

        return pos_df

    def _extract_negative_samples(
        self, df: pd.DataFrame, n_positive: int
    ) -> pd.DataFrame:
        """Извлечь negative samples (не нужно исправлять)"""
        neg_rows = []

        for _, row in df.iterrows():
            ocr = str(row["ocr"])
            gt = str(row["gt"])

            for i in range(min(len(ocr), len(gt))):
                if ocr[i] == gt[i] and is_valid_symbol(ocr[i]):
                    left = ocr[max(0, i - self.context_size) : i]
                    right = ocr[i + 1 : i + 1 + self.context_size]

                    neg_rows.append(
                        {
                            "ocr": ocr[i],
                            "y": NO_CHANGE,
                            "left": left,
                            "right": right,
                        }
                    )

        neg_df = pd.DataFrame(neg_rows)

        # балансировка
        n_neg = int(n_positive * self.negative_ratio)
        if len(neg_df) > n_neg:
            neg_df = neg_df.sample(n=n_neg, random_state=self.random_state)

        return neg_df

    def _make_features(self, row) -> Dict:
        """Создать признаки для одного примера"""
        feats = {f"ocr={row['ocr']}": 1}

        for i, ch in enumerate(row["left"][::-1]):
            if is_valid_symbol(ch):
                feats[f"L{i+1}={ch}"] = 1

        for i, ch in enumerate(row["right"]):
            if is_valid_symbol(ch):
                feats[f"R{i+1}={ch}"] = 1

        return feats

    def fit(self, df: pd.DataFrame) -> "CorrectorModel":
        """Обучить модель корректора"""
        # Positive samples
        pos_df = self._extract_positive_samples(df)
        print(f"Positive samples: {len(pos_df)}")

        # Negative samples
        neg_df = self._extract_negative_samples(df, len(pos_df))
        print(f"Negative samples: {len(neg_df)}")

        # Объединяем
        train_df = pd.concat([pos_df, neg_df]).sample(
            frac=1, random_state=self.random_state
        )
        print(f"Total training samples: {len(train_df)}")

        if train_df.empty:
            raise ValueError("No training samples extracted!")

        # Features
        X = train_df.apply(self._make_features, axis=1)
        y = train_df["y"]

        # Vectorize & train
        self.vec = DictVectorizer(sparse=True)
        X_vec = self.vec.fit_transform(X)

        self.clf = LogisticRegression(max_iter=self.max_iter)
        self.clf.fit(X_vec, y)

        self.classes = list(self.clf.classes_)
        self.no_change_idx = self.classes.index(NO_CHANGE)

        print(f"Model trained. Classes: {len(self.classes)}")

        return self

    def correct_word(self, ocr: str) -> Tuple[str, Optional[Dict]]:
        """Исправить одно слово"""
        best = None

        for pos in range(len(ocr)):
            ch = ocr[pos]
            if not is_valid_symbol(ch):
                continue

            left = ocr[max(0, pos - self.context_size) : pos]
            right = ocr[pos + 1 : pos + 1 + self.context_size]

            feats = {f"ocr={ch}": 1}
            for i, c in enumerate(left[::-1]):
                if is_valid_symbol(c):
                    feats[f"L{i+1}={c}"] = 1
            for i, c in enumerate(right):
                if is_valid_symbol(c):
                    feats[f"R{i+1}={c}"] = 1

            X = self.vec.transform([feats])
            probs = self.clf.predict_proba(X)[0]

            p_no = probs[self.no_change_idx]

            for cls, p in zip(self.classes, probs):
                if cls == NO_CHANGE:
                    continue
                if is_latin(cls):
                    continue

                # gating
                if p < self.prob_threshold:
                    continue
                if p <= p_no * self.gate_k:
                    continue

                if best is None or p > best["prob"]:
                    best = {"pos": pos, "char": cls, "prob": p}

        if best:
            corrected = ocr[: best["pos"]] + best["char"] + ocr[best["pos"] + 1 :]
            return corrected, best

        return ocr, None


# ======================================================
# OCR CORRECTOR (MAIN CLASS)
# ======================================================


class OCRCorrector:
    """
    Главный класс корректора OCR.
    Объединяет Gate модель и Corrector модель.

    Логика:
    1. Gate решает, нужно ли исправлять слово
    2. Если gate пропускает, применяется corrector
    """

    def __init__(self, config: Dict = None):
        self.config = config or CONFIG.copy()

        self.gate = None
        self.corrector = None
        self.df = None

    def load_data(
        self,
        data_path: str = None,
    ) -> pd.DataFrame:
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

    def build_gate_model(self, df: pd.DataFrame = None) -> GateModel:
        """Построить Gate модель"""
        import warnings

        df = df if df is not None else self.df

        self.gate = GateModel(
            ngram_range=self.config["gate_ngram_range"],
            min_df=self.config["gate_min_df"],
            max_iter=self.config["gate_max_iter"],
            class_weight=self.config["gate_class_weight"],
            target_precision=self.config["gate_target_precision"],
            threshold=self.config["gate_threshold"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gate.fit(df)

        stats = self.gate.get_stats(df)
        print(f"Gate Model trained:")
        print(f"  ROC-AUC: {stats['roc_auc']:.4f}")
        print(f"  Threshold: {stats['threshold']:.4f}")
        print(f"  Precision: {stats['precision']:.4f}")
        print(f"  Recall: {stats['recall']:.4f}")

        return self.gate

    def build_corrector_model(self, df: pd.DataFrame = None) -> CorrectorModel:
        """Построить Corrector модель"""
        df = df if df is not None else self.df

        self.corrector = CorrectorModel(
            context_size=self.config["corr_context_size"],
            max_iter=self.config["corr_max_iter"],
            min_rule_freq=self.config["corr_min_rule_freq"],
            negative_ratio=self.config["corr_negative_ratio"],
            prob_threshold=self.config["corr_prob_threshold"],
            gate_k=self.config["corr_gate_k"],
            random_state=self.config["random_state"],
        )

        self.corrector.fit(df)

        return self.corrector

    def build(self, df: pd.DataFrame = None) -> "OCRCorrector":
        """Построить обе модели"""
        df = df if df is not None else self.df

        print("=" * 50)
        print("Building Gate Model...")
        print("=" * 50)
        self.build_gate_model(df)

        print()
        print("=" * 50)
        print("Building Corrector Model...")
        print("=" * 50)
        self.build_corrector_model(df)

        return self

    def correct_word(self, word: str) -> Tuple[str, Dict]:
        """
        Исправить одно слово.

        Returns:
            (corrected_word, info_dict)
            info_dict содержит: gate_prob, gate_decision, correction_info
        """
        # Gate
        gate_prob = self.gate.predict_proba([word])[0]
        gate_decision = gate_prob >= self.gate.fitted_threshold

        info = {
            "gate_prob": gate_prob,
            "gate_decision": gate_decision,
            "correction_info": None,
        }

        # Если gate не пропускает - не исправляем
        if not gate_decision:
            return word, info

        # Corrector
        corrected, correction_info = self.corrector.correct_word(word)
        info["correction_info"] = correction_info

        return corrected, info

    def correct_words(self, words: List[str]) -> List[Tuple[str, Dict]]:
        """Исправить список слов"""
        return [self.correct_word(w) for w in words]

    def evaluate(self, df: pd.DataFrame = None) -> Dict:
        """
        Оценить качество корректора.

        Returns:
            dict с метриками: cer_before, cer_after, acc_before, acc_after, etc.
        """
        df = df if df is not None else self.df

        cer_metric = evaluate.load("cer")

        refs = []
        ocr_preds = []
        corr_preds = []

        applied = 0
        improved = 0
        worsened = 0
        gate_passed = 0

        for _, row in df.iterrows():
            ocr = str(row["ocr"])
            gt = str(row["gt"])

            corrected, info = self.correct_word(ocr)

            refs.append(gt)
            ocr_preds.append(ocr)
            corr_preds.append(corrected)

            if info["gate_decision"]:
                gate_passed += 1

            if info["correction_info"] is not None:
                applied += 1
                dist_before = Levenshtein.distance(ocr, gt)
                dist_after = Levenshtein.distance(corrected, gt)

                if dist_after < dist_before:
                    improved += 1
                elif dist_after > dist_before:
                    worsened += 1

        # Метрики
        cer_before = cer_metric.compute(predictions=ocr_preds, references=refs)
        cer_after = cer_metric.compute(predictions=corr_preds, references=refs)

        acc_before = sum(o == g for o, g in zip(ocr_preds, refs)) / len(refs)
        acc_after = sum(c == g for c, g in zip(corr_preds, refs)) / len(refs)

        results = {
            "cer_before": cer_before,
            "cer_after": cer_after,
            "cer_delta": cer_before - cer_after,
            "acc_before": acc_before,
            "acc_after": acc_after,
            "acc_delta": acc_after - acc_before,
            "total_words": len(df),
            "gate_passed": gate_passed,
            "corrections_applied": applied,
            "improved": improved,
            "worsened": worsened,
            "no_effect": applied - improved - worsened,
            "precision": improved / applied if applied > 0 else 0,
        }

        return results

    def print_evaluation(self, results: Dict = None, df: pd.DataFrame = None):
        """Вывести результаты оценки"""
        if results is None:
            results = self.evaluate(df)

        print()
        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print()
        print(f"Total words:           {results['total_words']}")
        print(
            f"Gate passed:           {results['gate_passed']} ({results['gate_passed']/results['total_words']*100:.1f}%)"
        )
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
        print()

    def save_results(
        self,
        output_path: str = "ocr_corrector_results.csv",
        df: pd.DataFrame = None,
    ):
        """Сохранить результаты в CSV"""
        df = df if df is not None else self.df

        out_rows = []

        for _, row in df.iterrows():
            ocr = str(row["ocr"])
            gt = str(row["gt"])

            corrected, info = self.correct_word(ocr)

            cer_before = Levenshtein.distance(ocr, gt)
            cer_after = Levenshtein.distance(corrected, gt)

            if corrected != ocr:
                if cer_after > cer_before:
                    effect = "WORSENED"
                elif cer_after < cer_before:
                    effect = "IMPROVED"
                else:
                    effect = "NO_EFFECT"
            else:
                effect = "NOT_APPLIED"

            out_rows.append(
                {
                    "image": row.get("image", ""),
                    "ocr": ocr,
                    "corrected": corrected,
                    "gt": gt,
                    "effect": effect,
                    "gate_prob": info["gate_prob"],
                    "gate_decision": info["gate_decision"],
                    "cer_before": cer_before,
                    "cer_after": cer_after,
                    "cer_delta": cer_before - cer_after,
                }
            )

        res_df = pd.DataFrame(out_rows)
        res_df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

        return res_df


# ======================================================
# MAIN
# ======================================================


def optuna_optimize(config_base: Dict = None, n_trials: int = 100):
    """
    Optuna оптимизация гиперпараметров.
    Цель: максимизировать precision корректора.
    """
    import optuna
    import warnings

    warnings.filterwarnings("ignore")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    config_base = config_base or CONFIG.copy()

    # Загружаем данные один раз
    corrector = OCRCorrector(config_base)
    df = corrector.load_data()

    def objective(trial):
        config = config_base.copy()

        # === Gate параметры ===
        config["gate_target_precision"] = trial.suggest_float(
            "gate_target_precision", 0.4, 0.9
        )
        config["gate_ngram_range"] = trial.suggest_categorical(
            "gate_ngram_range", [(1, 2), (1, 3), (2, 3)]
        )
        config["gate_min_df"] = trial.suggest_int("gate_min_df", 2, 10)

        # === Corrector параметры ===
        config["corr_prob_threshold"] = trial.suggest_float(
            "corr_prob_threshold", 0.2, 0.8
        )
        config["corr_gate_k"] = trial.suggest_float("corr_gate_k", 0.5, 10.0)
        config["corr_min_rule_freq"] = trial.suggest_int("corr_min_rule_freq", 2, 10)
        config["corr_negative_ratio"] = trial.suggest_float(
            "corr_negative_ratio", 1.0, 5.0
        )
        config["corr_context_size"] = trial.suggest_int("corr_context_size", 1, 4)

        try:
            corr = OCRCorrector(config)
            corr.df = df

            # Подавляем вывод
            import io
            import sys

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            corr.build(df)
            eval_result = corr.evaluate(df)

            sys.stdout = old_stdout

            applied = eval_result["corrections_applied"]
            precision = eval_result["precision"]
            improved = eval_result["improved"]
            worsened = eval_result["worsened"]

            # Если мало правок - штрафуем
            if applied < 10:
                return 0.0

            # Основная метрика: precision, но учитываем количество
            # Хотим высокую precision И достаточно правок
            score = precision * min(1.0, applied / 50)  # бонус за количество до 50

            # Сохраняем доп инфо
            trial.set_user_attr("applied", applied)
            trial.set_user_attr("improved", improved)
            trial.set_user_attr("worsened", worsened)
            trial.set_user_attr("precision", precision)
            trial.set_user_attr("cer_delta", eval_result["cer_delta"])
            trial.set_user_attr("acc_delta", eval_result["acc_delta"])

            return score

        except Exception as e:
            return 0.0

    # Запуск оптимизации
    print(f"Starting Optuna optimization with {n_trials} trials...")
    print("=" * 70)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Результаты
    print("\n" + "=" * 70)
    print("OPTUNA OPTIMIZATION COMPLETE")
    print("=" * 70)

    best_trial = study.best_trial
    print(f"\nBest score: {best_trial.value:.4f}")
    print(f"\n=== BEST PARAMETERS ===")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    print(f"\n=== BEST TRIAL METRICS ===")
    print(f"  Precision: {best_trial.user_attrs.get('precision', 'N/A'):.4f}")
    print(f"  Applied: {best_trial.user_attrs.get('applied', 'N/A')}")
    print(f"  Improved: {best_trial.user_attrs.get('improved', 'N/A')}")
    print(f"  Worsened: {best_trial.user_attrs.get('worsened', 'N/A')}")
    print(f"  CER delta: {best_trial.user_attrs.get('cer_delta', 'N/A'):.6f}")
    print(f"  Acc delta: {best_trial.user_attrs.get('acc_delta', 'N/A'):.6f}")

    # Сохраняем все trials
    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_trials.csv", index=False)
    print(f"\nTrials saved to: optuna_trials.csv")

    # ТОП-10 trials
    print("\n=== TOP 10 TRIALS ===")
    top_trials = sorted(
        study.trials, key=lambda t: t.value if t.value else 0, reverse=True
    )[:10]
    for i, t in enumerate(top_trials):
        prec = t.user_attrs.get("precision", 0)
        appl = t.user_attrs.get("applied", 0)
        print(f"  {i+1}. score={t.value:.4f}, precision={prec:.4f}, applied={appl}")

    # Возвращаем лучшую конфигурацию
    best_config = config_base.copy()
    best_config.update(best_trial.params)

    return study, best_config


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "optuna":
        # Optuna режим
        n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        study, best_config = optuna_optimize(n_trials=n_trials)

        # Запуск с лучшими параметрами
        print("\n" + "=" * 70)
        print("RUNNING WITH BEST PARAMETERS")
        print("=" * 70)

        corrector = OCRCorrector(best_config)
        df = corrector.load_data()
        corrector.build(df)
        results = corrector.evaluate(df)
        corrector.print_evaluation(results)
        corrector.save_results("ocr_corrector_best_results.csv", df)

    else:
        # Обычный режим
        config = CONFIG.copy()

        # === Примеры изменения параметров ===
        # config["gate_target_precision"] = 0.85
        # config["corr_prob_threshold"] = 0.5
        # config["corr_gate_k"] = 2.0
        # config["rule_min_count"] = 5

        # Создаём корректор
        corrector = OCRCorrector(config)

        # Загружаем данные
        df = corrector.load_data()

        # Строим модели
        corrector.build(df)

        # Оценка
        results = corrector.evaluate(df)
        corrector.print_evaluation(results)

        # Сохранение
        corrector.save_results("ocr_corrector_results.csv", df)

        # === Пример использования на новых словах ===
        print("\n" + "=" * 50)
        print("EXAMPLE CORRECTIONS")
        print("=" * 50)

        test_words = ["Воличеству", "Енисейскаго", "допосъ"]

        for word in test_words:
            corrected, info = corrector.correct_word(word)
            print(f"\n'{word}' -> '{corrected}'")
            print(f"  gate_prob: {info['gate_prob']:.3f}")
            print(f"  gate_decision: {info['gate_decision']}")
            if info["correction_info"]:
                print(
                    f"  correction: pos={info['correction_info']['pos']}, "
                    f"char='{info['correction_info']['char']}', "
                    f"prob={info['correction_info']['prob']:.3f}"
                )
