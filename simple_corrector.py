"""
Simple OCR Corrector: только правила + модель-предиктор
Без Gate модели. Модель можно выбрать: LogisticRegression, RandomForest, GradientBoosting
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report


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
    "model_type": "logistic",  # "logistic", "random_forest", "gradient_boosting"
    # === LogisticRegression параметры ===
    "lr_C": 1.0,
    "lr_max_iter": 300,
    # === RandomForest параметры ===
    "rf_n_estimators": 100,
    "rf_max_depth": 10,
    "rf_min_samples_split": 5,
    "rf_min_samples_leaf": 2,
    # === GradientBoosting параметры ===
    "gb_n_estimators": 100,
    "gb_max_depth": 5,
    "gb_learning_rate": 0.1,
    "gb_min_samples_split": 5,
    # === Corrector параметры ===
    "context_size": 2,
    "min_rule_freq": 3,
    "negative_ratio": 2.0,
    "prob_threshold": 0.5,
    "gate_k": 2.0,  # p_change > p_no_change * gate_k
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
        feats = {f"ocr={row['ocr']}": 1}

        # Контекст слева
        for i, ch in enumerate(row["left"][::-1]):
            if is_valid_symbol(ch):
                feats[f"L{i+1}={ch}"] = 1

        # Контекст справа
        for i, ch in enumerate(row["right"]):
            if is_valid_symbol(ch):
                feats[f"R{i+1}={ch}"] = 1

        # Позиционные признаки
        feats["pos_ratio"] = row["pos_in_word"] / max(row["word_len"], 1)
        feats["is_start"] = 1 if row["pos_in_word"] == 0 else 0
        feats["is_end"] = 1 if row["pos_in_word"] == row["word_len"] - 1 else 0

        return feats

    def _create_model(self):
        """Создать модель по конфигу"""
        model_type = self.config["model_type"]

        if model_type == "logistic":
            return LogisticRegression(
                C=self.config["lr_C"],
                max_iter=self.config["lr_max_iter"],
                random_state=self.config["random_state"],
            )
        elif model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.config["rf_n_estimators"],
                max_depth=self.config["rf_max_depth"],
                min_samples_split=self.config["rf_min_samples_split"],
                min_samples_leaf=self.config["rf_min_samples_leaf"],
                random_state=self.config["random_state"],
                n_jobs=-1,
            )
        elif model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.config["gb_n_estimators"],
                max_depth=self.config["gb_max_depth"],
                learning_rate=self.config["gb_learning_rate"],
                min_samples_split=self.config["gb_min_samples_split"],
                random_state=self.config["random_state"],
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def fit(self, df: pd.DataFrame = None) -> "SimpleCorrector":
        """Обучить модель"""
        df = df if df is not None else self.df

        # Positive samples
        pos_df = self._extract_positive_samples(df)
        print(f"Positive samples: {len(pos_df)}")

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

    def correct_word(self, ocr: str) -> Tuple[str, Optional[Dict]]:
        """Исправить одно слово"""
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

        refs, ocr_preds, corr_preds = [], [], []
        applied, improved, worsened = 0, 0, 0

        for _, row in df.iterrows():
            ocr = str(row["ocr"])
            gt = str(row["gt"])

            corrected, info = self.correct_word(ocr)

            refs.append(gt)
            ocr_preds.append(ocr)
            corr_preds.append(corrected)

            if info is not None:
                applied += 1
                dist_before = Levenshtein.distance(ocr, gt)
                dist_after = Levenshtein.distance(corrected, gt)

                if dist_after < dist_before:
                    improved += 1
                elif dist_after > dist_before:
                    worsened += 1

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


def optuna_optimize(config_base: Dict = None, n_trials: int = 100):
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

        # === Выбор модели ===
        config["model_type"] = trial.suggest_categorical(
            "model_type", ["logistic", "random_forest", "gradient_boosting"]
        )

        # === Параметры по типу модели ===
        if config["model_type"] == "logistic":
            config["lr_C"] = trial.suggest_float("lr_C", 0.01, 10.0, log=True)
            config["lr_max_iter"] = trial.suggest_int("lr_max_iter", 100, 500)

        elif config["model_type"] == "random_forest":
            config["rf_n_estimators"] = trial.suggest_int("rf_n_estimators", 50, 300)
            config["rf_max_depth"] = trial.suggest_int("rf_max_depth", 3, 20)
            config["rf_min_samples_split"] = trial.suggest_int(
                "rf_min_samples_split", 2, 20
            )
            config["rf_min_samples_leaf"] = trial.suggest_int(
                "rf_min_samples_leaf", 1, 10
            )

        elif config["model_type"] == "gradient_boosting":
            config["gb_n_estimators"] = trial.suggest_int("gb_n_estimators", 50, 300)
            config["gb_max_depth"] = trial.suggest_int("gb_max_depth", 2, 10)
            config["gb_learning_rate"] = trial.suggest_float(
                "gb_learning_rate", 0.01, 0.3, log=True
            )
            config["gb_min_samples_split"] = trial.suggest_int(
                "gb_min_samples_split", 2, 20
            )

        # === Общие параметры корректора ===
        config["context_size"] = trial.suggest_int("context_size", 1, 4)
        config["min_rule_freq"] = trial.suggest_int("min_rule_freq", 2, 10)
        config["negative_ratio"] = trial.suggest_float("negative_ratio", 1.0, 5.0)
        config["prob_threshold"] = trial.suggest_float("prob_threshold", 0.2, 0.8)
        config["gate_k"] = trial.suggest_float("gate_k", 0.5, 10.0)

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

            trial.set_user_attr("model_type", config["model_type"])
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
    print(f"  Model: {best.user_attrs.get('model_type', 'N/A')}")
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

    # ТОП по моделям
    print("\n=== TOP BY MODEL TYPE ===")
    for model_type in ["logistic", "random_forest", "gradient_boosting"]:
        model_trials = [
            t
            for t in study.trials
            if t.user_attrs.get("model_type") == model_type and t.value
        ]
        if model_trials:
            best_model = max(model_trials, key=lambda t: t.value)
            print(
                f"  {model_type}: score={best_model.value:.4f}, "
                f"precision={best_model.user_attrs.get('precision', 0):.4f}, "
                f"applied={best_model.user_attrs.get('applied', 0)}"
            )

    best_config = config_base.copy()
    best_config.update(best.params)

    return study, best_config


# ======================================================
# GRID SEARCH (Полный перебор)
# ======================================================

def grid_search_full():
    """Полный перебор всех комбинаций параметров"""
    from itertools import product
    
    config_base = CONFIG.copy()
    
    # Загружаем данные
    corr = SimpleCorrector(config_base)
    df = corr.load_data()
    print(f"Loaded {len(df)} word pairs\n")
    
    # === ПОЛНАЯ СЕТКА ПАРАМЕТРОВ ===
    param_grid = {
        "model_type": ["logistic", "random_forest", "gradient_boosting"],
        
        # Общие параметры
        "context_size": [1, 2, 3, 4],
        "min_rule_freq": [2, 3, 5, 10],
        "negative_ratio": [1.0, 1.5, 2.0, 3.0, 5.0],
        "prob_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "gate_k": [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0],
        
        # LogisticRegression
        "lr_C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "lr_max_iter": [100, 300, 500],
        
        # RandomForest
        "rf_n_estimators": [50, 100, 200],
        "rf_max_depth": [5, 10, 15, None],
        "rf_min_samples_split": [2, 5, 10],
        "rf_min_samples_leaf": [1, 2, 5],
        
        # GradientBoosting
        "gb_n_estimators": [50, 100, 200],
        "gb_max_depth": [3, 5, 7],
        "gb_learning_rate": [0.01, 0.05, 0.1, 0.3],
        "gb_min_samples_split": [2, 5, 10],
    }
    
    # Генерируем комбинации с учётом модели
    all_results = []
    
    for model_type in param_grid["model_type"]:
        # Общие параметры
        common_keys = ["context_size", "min_rule_freq", "negative_ratio", 
                       "prob_threshold", "gate_k"]
        
        # Параметры конкретной модели
        if model_type == "logistic":
            model_keys = ["lr_C", "lr_max_iter"]
        elif model_type == "random_forest":
            model_keys = ["rf_n_estimators", "rf_max_depth", 
                          "rf_min_samples_split", "rf_min_samples_leaf"]
        else:  # gradient_boosting
            model_keys = ["gb_n_estimators", "gb_max_depth", 
                          "gb_learning_rate", "gb_min_samples_split"]
        
        all_keys = ["model_type"] + common_keys + model_keys
        all_values = [[model_type]] + [param_grid[k] for k in common_keys + model_keys]
        
        combinations = list(product(*all_values))
        
        print(f"\n{model_type}: {len(combinations)} combinations")
        
        for i, combo in enumerate(combinations):
            config = config_base.copy()
            
            # Заполняем конфиг
            for key, value in zip(all_keys, combo):
                config[key] = value
            
            try:
                corr = SimpleCorrector(config)
                corr.df = df
                corr.fit(df)
                
                eval_result = corr.evaluate(df)
                
                result = {
                    **{k: v for k, v in zip(all_keys, combo)},
                    "precision": eval_result["precision"],
                    "improved": eval_result["improved"],
                    "worsened": eval_result["worsened"],
                    "applied": eval_result["corrections_applied"],
                    "cer_delta": eval_result["cer_delta"],
                    "acc_delta": eval_result["acc_delta"],
                    "score": eval_result["precision"] * min(1.0, eval_result["corrections_applied"] / 50)
                }
                
                all_results.append(result)
                
                # Прогресс
                if (i + 1) % 50 == 0:
                    best_score = max([r["score"] for r in all_results])
                    print(f"  Progress: {i + 1}/{len(combinations)}, best_score={best_score:.4f}")
                
            except Exception as e:
                print(f"  Error at combo {i}: {e}")
                continue
    
    # Сохраняем результаты
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("score", ascending=False)
    results_df.to_csv("simple_corrector_grid_full.csv", index=False)
    
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)
    print(f"\nTotal combinations tested: {len(results_df)}")
    print(f"Results saved to: simple_corrector_grid_full.csv")
    
    # Лучший результат
    best = results_df.iloc[0]
    print("\n=== BEST CONFIGURATION ===")
    for key in results_df.columns:
        if key not in ["score", "precision", "improved", "worsened", "applied", "cer_delta", "acc_delta"]:
            print(f"  {key}: {best[key]}")
    
    print("\n=== BEST METRICS ===")
    print(f"  Score: {best['score']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Applied: {best['applied']}")
    print(f"  Improved: {best['improved']}")
    print(f"  Worsened: {best['worsened']}")
    print(f"  CER delta: {best['cer_delta']:.6f}")
    print(f"  Acc delta: {best['acc_delta']:.6f}")
    
    # ТОП-20 по каждой модели
    print("\n=== TOP 20 BY MODEL ===")
    for model in ["logistic", "random_forest", "gradient_boosting"]:
        print(f"\n{model}:")
        top = results_df[results_df["model_type"] == model].head(20)
        for i, row in top.iterrows():
            print(f"  {len(top) - len(top) + list(top.index).index(i) + 1}. score={row['score']:.4f}, "
                  f"prec={row['precision']:.4f}, applied={row['applied']}")
    
    return results_df


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "grid":
        # Полный перебор
        results_df = grid_search_full()
        
    elif len(sys.argv) > 1 and sys.argv[1] == "optuna":
        n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        study, best_config = optuna_optimize(n_trials=n_trials)

        print("\n" + "=" * 70)
        print("RUNNING WITH BEST PARAMETERS")
        print("=" * 70)

        corr = SimpleCorrector(best_config)
        df = corr.load_data()
        corr.fit(df)
        result = corr.evaluate(df)
        corr.print_evaluation(result)

    else:
        # Обычный режим - можно задать модель через аргумент
        config = CONFIG.copy()

        if len(sys.argv) > 1:
            config["model_type"] = sys.argv[
                1
            ]  # logistic, random_forest, gradient_boosting

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
