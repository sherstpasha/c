"""
Экспериментальный корректор для тестирования параметров модели
Варьирует параметры окна контекста, пересечения и итераций
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple, Optional
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import time
from dataclasses import dataclass

from CharTransformerMLM.model import CharTransformerMLM
from CharTransformerMLM.vocab import CharVocab


@dataclass
class CorrectionExample:
    """Пример коррекции для логирования"""

    word_original: str
    word_corrected: str
    position: int
    char_changes: List[Tuple[int, str, str, float]]  # (pos, old, new, confidence)


# ================= МЕТРИКИ (из quick_metrics.py) =================


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def calculate_cer(reference, hypothesis):
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return levenshtein_distance(" ".join(ref_words), " ".join(hyp_words)) / len(
        ref_words
    )


def calculate_bow_f1(reference, hypothesis):
    """Bag-of-Words F1 score"""
    ref_words = set(reference.split())
    hyp_words = set(hypothesis.split())

    if len(ref_words) == 0 and len(hyp_words) == 0:
        return 1.0
    if len(ref_words) == 0 or len(hyp_words) == 0:
        return 0.0

    tp = len(ref_words & hyp_words)
    fp = len(hyp_words - ref_words)
    fn = len(ref_words - hyp_words)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_char_ngrams(text, n=3):
    return set([text[i : i + n] for i in range(len(text) - n + 1)])


def calculate_char_ngram_f1(reference, hypothesis, n=3):
    """Character n-gram F1 score"""
    if len(reference) < n and len(hypothesis) < n:
        return 1.0 if reference == hypothesis else 0.0
    if len(reference) < n or len(hypothesis) < n:
        return 0.0

    ref_ngrams = get_char_ngrams(reference, n)
    hyp_ngrams = get_char_ngrams(hypothesis, n)

    if len(ref_ngrams) == 0 and len(hyp_ngrams) == 0:
        return 1.0
    if len(ref_ngrams) == 0 or len(hyp_ngrams) == 0:
        return 0.0

    tp = len(ref_ngrams & hyp_ngrams)
    fp = len(hyp_ngrams - ref_ngrams)
    fn = len(ref_ngrams - hyp_ngrams)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def extract_text(annotations):
    """
    Извлекает текст из аннотаций (порядок уже правильный)
    """
    if not annotations:
        return ""

    texts = []
    for ann in annotations:
        text = ""
        if "attributes" in ann and "transcription" in ann["attributes"]:
            text = ann["attributes"]["transcription"]
        elif "text" in ann:
            text = ann["text"]

        if text is None or (
            isinstance(text, float) and (np.isnan(text) or text != text)
        ):
            text = ""
        else:
            text = str(text).strip()

        if text and text not in ["<NR>", "<nr>", "###", ""]:
            texts.append(text)

    return " ".join(texts)


# ================= КОРРЕКТОР =================


class WordCorrector:
    """
    Корректор OCR ошибок с настраиваемыми параметрами
    """

    def __init__(self, checkpoint_path: str, charset_path: str, words_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Загрузка vocab
        self.vocab = CharVocab(charset_path)

        # Загрузка словаря слов
        self.word_dict = set()
        if words_path and Path(words_path).exists():
            with open(words_path, encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.word_dict.add(word)
            print(f"📚 Загружен словарь: {len(self.word_dict)} слов")

        # Загрузка модели
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
            config = ckpt.get("config", {})
        else:
            state_dict = ckpt
            config = {}

        self.model = CharTransformerMLM(
            vocab_size=len(self.vocab.token_to_id),
            emb_size=config.get("emb_size", 192),
            n_layers=config.get("n_layers", 6),
            n_heads=config.get("n_heads", 6),
            ffn_size=config.get("ffn_size", 768),
            dropout=0.0,
            pad_idx=self.vocab.pad,
            eow_idx=self.vocab.eow,
            ins_idx=self.vocab.ins,
            space_idx=self.vocab.token_to_id.get(" ", 6),
        ).to(self.device)

        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"✅ Модель загружена на {self.device}")

    def process_window_batch(
        self,
        batch_words: List[List[str]],
        confidence_threshold: float = 0.0,
        use_dictionary: bool = False,
    ) -> Tuple[List[List[str]], List[List[CorrectionExample]]]:
        """
        Обработать батч окон из слов с поддержкой confidence threshold.

        Args:
            batch_words: список окон слов
            confidence_threshold: порог уверенности для замены (0.0 = все заменять)
            use_dictionary: применять замены только если слово есть в словаре

        Returns:
            (исправленные окна, примеры коррекций для логирования)
        """
        if not batch_words:
            return [], []

        # Подготовка батча
        all_ids = []
        all_word_boundaries = []
        batch_sizes = []

        for words in batch_words:
            if not words:
                batch_sizes.append(0)
                all_ids.append([])
                all_word_boundaries.append([])
                continue

            ids = []
            word_boundaries = []

            for word in words:
                start = len(ids)
                word_ids = self.vocab.encode(word)
                ids.extend(word_ids)
                ids.append(self.vocab.ins)
                ids.append(self.vocab.eow)
                word_boundaries.append((start, len(ids) - 1))

            batch_sizes.append(len(ids))
            all_ids.append(ids)
            all_word_boundaries.append(word_boundaries)

        # Паддинг до максимальной длины
        max_len = max(batch_sizes) if batch_sizes else 0
        if max_len == 0:
            return [[] for _ in batch_words], [[] for _ in batch_words]

        padded_ids = []
        for ids in all_ids:
            if len(ids) == 0:
                padded_ids.append([self.vocab.pad] * max_len)
            else:
                padded_ids.append(ids + [self.vocab.pad] * (max_len - len(ids)))

        # Прогоняем через модель
        x = torch.tensor(padded_ids, device=self.device)
        y_dummy = torch.full_like(x, -100)

        with torch.no_grad():
            logits, _ = self.model(x, y_dummy)

            # Применяем маски
            eow = self.vocab.eow
            ins = self.vocab.ins

            mask = (x != eow) & (x != ins) & (y_dummy == -100)
            logits[..., eow] -= mask * 1e9

            eow_positions = x == eow
            non_eow_mask = torch.ones_like(logits, dtype=torch.bool)
            non_eow_mask[..., eow] = False
            logits[eow_positions.unsqueeze(-1).expand_as(logits) & non_eow_mask] -= 1e9

            logits[..., ins] -= 1e9

            # Получаем вероятности для threshold
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1).tolist()
            max_probs = probs.max(dim=-1).values.tolist()

        # Собираем исправленные слова для каждого окна
        result_batch = []
        examples_batch = []
        space_id = self.vocab.token_to_id.get(" ", 6)

        for batch_idx, words in enumerate(batch_words):
            if not words or batch_sizes[batch_idx] == 0:
                result_batch.append([])
                examples_batch.append([])
                continue

            result_words = []
            examples = []
            ids = all_ids[batch_idx]
            word_boundaries = all_word_boundaries[batch_idx]
            batch_preds = preds[batch_idx]
            batch_probs = max_probs[batch_idx]

            for word_idx, (start, end) in enumerate(word_boundaries):
                orig_word = words[word_idx]
                input_ids = ids[start : end + 1]
                pred_ids_full = batch_preds[start : end + 1]
                pred_probs_full = batch_probs[start : end + 1]

                pred_word_chars = []
                char_changes = []
                char_pos = 0

                for pos_in_word, (inp_id, pred_id, pred_prob) in enumerate(
                    zip(input_ids, pred_ids_full, pred_probs_full)
                ):
                    if inp_id == self.vocab.eow:
                        continue
                    elif inp_id == self.vocab.ins:
                        if pred_id != space_id and pred_id != self.vocab.ins:
                            new_char = self.vocab.id_to_token[pred_id]
                            # Проверяем threshold для вставки
                            if pred_prob >= confidence_threshold:
                                pred_word_chars.append(new_char)
                                char_changes.append((char_pos, "", new_char, pred_prob))
                                char_pos += 1
                    else:
                        orig_char = self.vocab.id_to_token[inp_id]
                        new_char = self.vocab.id_to_token[pred_id]

                        # Применяем threshold
                        if orig_char != new_char and pred_prob >= confidence_threshold:
                            pred_word_chars.append(new_char)
                            char_changes.append(
                                (char_pos, orig_char, new_char, pred_prob)
                            )
                        else:
                            pred_word_chars.append(orig_char)
                        char_pos += 1

                pred_word = "".join(pred_word_chars)
                result_words.append(pred_word)

                # Логируем если были изменения
                if char_changes and pred_word != orig_word:
                    examples.append(
                        CorrectionExample(
                            word_original=orig_word,
                            word_corrected=pred_word,
                            position=word_idx,
                            char_changes=char_changes,
                        )
                    )

            result_batch.append(result_words)
            examples_batch.append(examples)

        return result_batch, examples_batch

    def correct_text(
        self,
        words: List[str],
        window_size: int = 4,
        overlap: int = 1,
        iterations: int = 1,
        batch_size: int = 16,
        confidence_threshold: float = 0.0,
        use_dictionary: bool = False,
        log_examples: bool = False,
    ) -> Tuple[List[str], List[CorrectionExample], float]:
        """
        Исправить текст (список слов) скользящим окном с батчингом.

        Args:
            words: список слов для исправления
            window_size: размер окна (количество слов)
            overlap: пересечение окон (количество слов)
            iterations: количество итераций обработки
            batch_size: размер батча для обработки
            confidence_threshold: порог уверенности для замены
            use_dictionary: применять замены только если слово есть в словаре
            log_examples: логировать примеры коррекций

        Returns:
            (список исправленных слов, примеры коррекций, время обработки)
        """
        if not words:
            return words, [], 0.0

        start_time = time.time()
        result_words = list(words)
        all_examples = []

        for iteration in range(iterations):
            iteration_changed = False

            # Собираем окна для батча
            windows = []
            window_positions = []
            pos = 0

            while pos < len(result_words):
                end = min(pos + window_size, len(result_words))
                windows.append(result_words[pos:end])
                window_positions.append((pos, end))
                pos += max(1, window_size - overlap)

            # Обрабатываем батчами
            for batch_start in range(0, len(windows), batch_size):
                batch_end = min(batch_start + batch_size, len(windows))
                batch_windows = windows[batch_start:batch_end]
                batch_positions = window_positions[batch_start:batch_end]

                corrected_batch, examples_batch = self.process_window_batch(
                    batch_windows, confidence_threshold, use_dictionary
                )

                # Применяем исправления
                for corrected_words, (pos, end), examples in zip(
                    corrected_batch, batch_positions, examples_batch
                ):
                    if corrected_words != result_words[pos:end]:
                        iteration_changed = True
                        result_words[pos:end] = corrected_words

                        if log_examples and examples:
                            all_examples.extend(examples)

            # Если не было изменений, прерываем итерации
            if not iteration_changed:
                break

        elapsed_time = time.time() - start_time
        return result_words, all_examples, elapsed_time


# ================= ВЫЧИСЛЕНИЕ МЕТРИК (для параллелизации) =================


def compute_metrics_for_file(args):
    """
    Вычислить метрики для одного файла (для параллелизации)
    """
    file_name, gt_text, pred_text_orig, pred_text_corrected = args

    # Метрики для оригинального pred
    cer_orig = (
        calculate_cer(gt_text, pred_text_orig) if gt_text or pred_text_orig else 0.0
    )
    wer_orig = (
        calculate_wer(gt_text, pred_text_orig) if gt_text or pred_text_orig else 0.0
    )
    bow_f1_orig = calculate_bow_f1(gt_text, pred_text_orig)
    char_3gram_f1_orig = calculate_char_ngram_f1(gt_text, pred_text_orig, n=3)

    # Метрики для исправленного pred
    cer_corr = (
        calculate_cer(gt_text, pred_text_corrected)
        if gt_text or pred_text_corrected
        else 0.0
    )
    wer_corr = (
        calculate_wer(gt_text, pred_text_corrected)
        if gt_text or pred_text_corrected
        else 0.0
    )
    bow_f1_corr = calculate_bow_f1(gt_text, pred_text_corrected)
    char_3gram_f1_corr = calculate_char_ngram_f1(gt_text, pred_text_corrected, n=3)

    # Эффективность корректора (на уровне слов)
    gt_words = gt_text.split()
    pred_words_orig = pred_text_orig.split()
    pred_words_corr = pred_text_corrected.split()

    total_fixes = 0
    correct_fixes = 0
    incorrect_fixes = 0

    # Выравниваем по минимальной длине для корректного сравнения
    min_len = min(len(gt_words), len(pred_words_orig), len(pred_words_corr))

    for i in range(min_len):
        gt_word = gt_words[i].lower()
        orig_word = pred_words_orig[i].lower()
        corr_word = pred_words_corr[i].lower()

        # Если слово изменилось - это исправление
        if orig_word != corr_word:
            total_fixes += 1

            # Проверяем, правильное ли это исправление
            # Правильное: если исправленное слово стало ближе к GT
            orig_dist = levenshtein_distance(gt_word, orig_word)
            corr_dist = levenshtein_distance(gt_word, corr_word)

            if corr_dist < orig_dist:
                correct_fixes += 1
            elif corr_dist > orig_dist:
                incorrect_fixes += 1
            # Если равны - не учитываем как правильное

    correction_precision = correct_fixes / total_fixes if total_fixes > 0 else 0.0
    correction_recall = correct_fixes / max(
        1,
        sum(
            1
            for i in range(min_len)
            if gt_words[i].lower() != pred_words_orig[i].lower()
        ),
    )

    return {
        "file_name": file_name,
        "cer_orig": cer_orig,
        "cer_corr": cer_corr,
        "cer_delta": cer_orig - cer_corr,
        "wer_orig": wer_orig,
        "wer_corr": wer_corr,
        "wer_delta": wer_orig - wer_corr,
        "bow_f1_orig": bow_f1_orig,
        "bow_f1_corr": bow_f1_corr,
        "bow_f1_delta": bow_f1_corr - bow_f1_orig,
        "char_3gram_f1_orig": char_3gram_f1_orig,
        "char_3gram_f1_corr": char_3gram_f1_corr,
        "char_3gram_f1_delta": char_3gram_f1_corr - char_3gram_f1_orig,
        "total_fixes": total_fixes,
        "correct_fixes": correct_fixes,
        "incorrect_fixes": incorrect_fixes,
        "correction_precision": correction_precision,
        "correction_recall": correction_recall,
    }


# ================= ЭКСПЕРИМЕНТЫ =================


def run_experiments(
    corrector: WordCorrector,
    gt_path: Path,
    pred_path: Path,
    param_grid: Dict[str, List],
    output_dir: Path,
):
    """
    Запуск экспериментов с различными параметрами

    Args:
        corrector: экземпляр WordCorrector
        gt_path: путь к ground truth JSON
        pred_path: путь к predictions JSON
        param_grid: словарь с параметрами для перебора
        output_dir: директория для сохранения результатов
    """
    output_dir.mkdir(exist_ok=True)

    # Загружаем данные
    print("📂 Загрузка данных...")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    # Подготовка маппингов (как в quick_metrics.py)
    print("🔄 Подготовка маппингов...")
    pred_file_to_first_id = {}
    for img in pred_data["images"]:
        fname = img["file_name"]
        if fname not in pred_file_to_first_id:
            pred_file_to_first_id[fname] = img["id"]

    pred_images_map = {}
    for img in pred_data["images"]:
        fname = img["file_name"]
        if fname not in pred_images_map:
            pred_images_map[fname] = img

    gt_images_map = {img["file_name"]: img for img in gt_data["images"]}
    gt_id_to_file = {img["id"]: img["file_name"] for img in gt_data["images"]}
    pred_id_to_file = {img["id"]: img["file_name"] for img in pred_data["images"]}

    gt_ann_by_file = defaultdict(list)
    for ann in gt_data["annotations"]:
        fname = gt_id_to_file.get(ann["image_id"])
        if fname:
            gt_ann_by_file[fname].append(ann)

    pred_ann_by_file = defaultdict(list)
    for ann in pred_data["annotations"]:
        img_id = ann["image_id"]
        fname = pred_id_to_file.get(img_id)
        if fname:
            if img_id == pred_file_to_first_id[fname]:
                pred_ann_by_file[fname].append(ann)

    print(f"✅ Загружено {len(gt_images_map)} уникальных файлов")

    # Генерация комбинаций параметров
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    print(f"🧪 Всего комбинаций параметров: {len(param_combinations)}")
    print(f"   Параметры: {param_names}")

    # Результаты экспериментов
    all_results = []

    # Прогоняем каждую комбинацию параметров
    for combo_idx, params in enumerate(param_combinations, 1):
        param_dict = dict(zip(param_names, params))
        window_size = param_dict["window_size"]
        overlap = param_dict["overlap"]
        iterations = param_dict["iterations"]
        batch_size = param_dict.get("batch_size", 16)
        confidence_threshold = param_dict.get("confidence_threshold", 0.0)
        use_dictionary = param_dict.get("use_dictionary", False)

        print(f"\n{'='*70}")
        print(f"Эксперимент {combo_idx}/{len(param_combinations)}")
        print(f"  Параметры:")
        print(
            f"    window_size={window_size}, overlap={overlap}, iterations={iterations}"
        )
        print(
            f"    batch_size={batch_size}, confidence_threshold={confidence_threshold:.2f}, use_dict={use_dictionary}"
        )

        file_names = sorted(gt_images_map.keys())[:10]  # Ограничиваем 10 файлами
        total_files = len(file_names)
        print(f"  🔬 Тестируем на {total_files} файлах (подмножество)")

        # ЭТАП 1: Коррекция всех текстов
        print(f"\n  📝 Этап 1: Коррекция текстов...")
        correction_start = time.time()

        corrected_texts = {}
        all_examples = []
        total_words = 0

        for file_idx, file_name in enumerate(file_names, 1):
            if file_idx % 10 == 0:
                print(f"  Корректируем {file_idx}/{total_files} файлов...", end="\r")

            # Pred текст (оригинальный)
            pred_text_orig = extract_text(pred_ann_by_file.get(file_name, []))
            pred_words = pred_text_orig.split()
            total_words += len(pred_words)

            # Корректируем pred текст
            corrected_words, examples, _ = corrector.correct_text(
                pred_words,
                window_size=window_size,
                overlap=overlap,
                iterations=iterations,
                batch_size=batch_size,
                confidence_threshold=confidence_threshold,
                use_dictionary=use_dictionary,
                log_examples=(file_idx <= 5),  # Логируем только первые 5 файлов
            )
            pred_text_corrected = " ".join(corrected_words)
            corrected_texts[file_name] = pred_text_corrected

            if file_idx <= 5 and examples:
                all_examples.extend(
                    [(file_name, ex) for ex in examples[:3]]
                )  # По 3 примера

        correction_time = time.time() - correction_start
        words_per_sec = total_words / correction_time if correction_time > 0 else 0

        print(f"\n  ✅ Коррекция завершена за {correction_time:.2f}с")
        print(f"     Скорость: {words_per_sec:.1f} слов/сек")

        # ЭТАП 2: Параллельное вычисление метрик
        print(f"\n  📊 Этап 2: Вычисление метрик...")
        metrics_start = time.time()

        # Подготавливаем задачи для параллелизации
        tasks = []
        for file_name in file_names:
            gt_text = extract_text(gt_ann_by_file.get(file_name, []))
            pred_text_orig = extract_text(pred_ann_by_file.get(file_name, []))
            pred_text_corrected = corrected_texts[file_name]
            tasks.append((file_name, gt_text, pred_text_orig, pred_text_corrected))

        # Параллельное вычисление
        experiment_metrics = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(compute_metrics_for_file, task) for task in tasks
            ]
            for future in as_completed(futures):
                experiment_metrics.append(future.result())

        metrics_time = time.time() - metrics_start
        print(f"  ✅ Метрики вычислены за {metrics_time:.2f}с")

        # Сортируем результаты по имени файла
        experiment_metrics.sort(key=lambda x: x["file_name"])

        # Агрегированные метрики для этой комбинации параметров
        cer_orig_vals = [m["cer_orig"] for m in experiment_metrics]
        cer_corr_vals = [m["cer_corr"] for m in experiment_metrics]
        cer_delta_vals = [m["cer_delta"] for m in experiment_metrics]

        wer_orig_vals = [m["wer_orig"] for m in experiment_metrics]
        wer_corr_vals = [m["wer_corr"] for m in experiment_metrics]
        wer_delta_vals = [m["wer_delta"] for m in experiment_metrics]

        bow_f1_orig_vals = [m["bow_f1_orig"] for m in experiment_metrics]
        bow_f1_corr_vals = [m["bow_f1_corr"] for m in experiment_metrics]
        bow_f1_delta_vals = [m["bow_f1_delta"] for m in experiment_metrics]

        char_3gram_f1_orig_vals = [m["char_3gram_f1_orig"] for m in experiment_metrics]
        char_3gram_f1_corr_vals = [m["char_3gram_f1_corr"] for m in experiment_metrics]
        char_3gram_f1_delta_vals = [
            m["char_3gram_f1_delta"] for m in experiment_metrics
        ]

        total_fixes_vals = [m["total_fixes"] for m in experiment_metrics]
        correct_fixes_vals = [m["correct_fixes"] for m in experiment_metrics]
        incorrect_fixes_vals = [m["incorrect_fixes"] for m in experiment_metrics]
        correction_precision_vals = [
            m["correction_precision"] for m in experiment_metrics
        ]
        correction_recall_vals = [m["correction_recall"] for m in experiment_metrics]

        summary = {
            "window_size": window_size,
            "overlap": overlap,
            "iterations": iterations,
            "batch_size": batch_size,
            "confidence_threshold": confidence_threshold,
            "use_dictionary": use_dictionary,
            "correction_time": correction_time,
            "metrics_time": metrics_time,
            "total_time": correction_time + metrics_time,
            "words_per_sec": words_per_sec,
            "cer_orig_mean": np.mean(cer_orig_vals),
            "cer_orig_median": np.median(cer_orig_vals),
            "cer_corr_mean": np.mean(cer_corr_vals),
            "cer_corr_median": np.median(cer_corr_vals),
            "cer_delta_mean": np.mean(cer_delta_vals),
            "cer_delta_median": np.median(cer_delta_vals),
            "wer_orig_mean": np.mean(wer_orig_vals),
            "wer_orig_median": np.median(wer_orig_vals),
            "wer_corr_mean": np.mean(wer_corr_vals),
            "wer_corr_median": np.median(wer_corr_vals),
            "wer_delta_mean": np.mean(wer_delta_vals),
            "wer_delta_median": np.median(wer_delta_vals),
            "bow_f1_orig_mean": np.mean(bow_f1_orig_vals),
            "bow_f1_orig_median": np.median(bow_f1_orig_vals),
            "bow_f1_corr_mean": np.mean(bow_f1_corr_vals),
            "bow_f1_corr_median": np.median(bow_f1_corr_vals),
            "bow_f1_delta_mean": np.mean(bow_f1_delta_vals),
            "bow_f1_delta_median": np.median(bow_f1_delta_vals),
            "char_3gram_f1_orig_mean": np.mean(char_3gram_f1_orig_vals),
            "char_3gram_f1_orig_median": np.median(char_3gram_f1_orig_vals),
            "char_3gram_f1_corr_mean": np.mean(char_3gram_f1_corr_vals),
            "char_3gram_f1_corr_median": np.median(char_3gram_f1_corr_vals),
            "char_3gram_f1_delta_mean": np.mean(char_3gram_f1_delta_vals),
            "char_3gram_f1_delta_median": np.median(char_3gram_f1_delta_vals),
            "total_fixes": sum(total_fixes_vals),
            "correct_fixes": sum(correct_fixes_vals),
            "incorrect_fixes": sum(incorrect_fixes_vals),
            "correction_precision_mean": np.mean(correction_precision_vals),
            "correction_precision_median": np.median(correction_precision_vals),
            "correction_recall_mean": np.mean(correction_recall_vals),
            "correction_recall_median": np.median(correction_recall_vals),
        }

        all_results.append(summary)

        # Выводим результаты
        print(f"\n  📊 Результаты:")
        print(
            f"     Время: коррекция={correction_time:.2f}с, метрики={metrics_time:.2f}с"
        )
        print(f"     Скорость: {words_per_sec:.1f} слов/сек")
        print(
            f"     Исправлений: всего={summary['total_fixes']}, правильных={summary['correct_fixes']}, неправильных={summary['incorrect_fixes']}"
        )
        print(
            f"     Precision: {summary['correction_precision_mean']:.4f} (median={summary['correction_precision_median']:.4f})"
        )
        print(
            f"     Recall: {summary['correction_recall_mean']:.4f} (median={summary['correction_recall_median']:.4f})"
        )
        print(
            f"     CER (mean):    {summary['cer_orig_mean']:.4f} → {summary['cer_corr_mean']:.4f} (Δ={summary['cer_delta_mean']:+.4f})"
        )
        print(
            f"     CER (median):  {summary['cer_orig_median']:.4f} → {summary['cer_corr_median']:.4f} (Δ={summary['cer_delta_median']:+.4f})"
        )
        print(
            f"     WER (mean):    {summary['wer_orig_mean']:.4f} → {summary['wer_corr_mean']:.4f} (Δ={summary['wer_delta_mean']:+.4f})"
        )
        print(
            f"     WER (median):  {summary['wer_orig_median']:.4f} → {summary['wer_corr_median']:.4f} (Δ={summary['wer_delta_median']:+.4f})"
        )
        print(
            f"     BoW F1 (mean):   {summary['bow_f1_orig_mean']:.4f} → {summary['bow_f1_corr_mean']:.4f} (Δ={summary['bow_f1_delta_mean']:+.4f})"
        )
        print(
            f"     BoW F1 (median): {summary['bow_f1_orig_median']:.4f} → {summary['bow_f1_corr_median']:.4f} (Δ={summary['bow_f1_delta_median']:+.4f})"
        )

        # Логируем примеры
        if all_examples:
            print(f"\n  💡 Примеры коррекций (первые 5 файлов):")
            for file_name, example in all_examples[:10]:
                print(
                    f"     [{file_name}] '{example.word_original}' → '{example.word_corrected}'"
                )
                for pos, old, new, conf in example.char_changes[:3]:
                    if old:
                        print(f"       pos {pos}: '{old}' → '{new}' (conf={conf:.3f})")
                    else:
                        print(f"       pos {pos}: INSERT '{new}' (conf={conf:.3f})")

        # Находим и логируем худшие случаи (где метрики ухудшились)
        print(f"\n  ⚠️  Топ-5 худших случаев (где коррекция ухудшила метрики):")
        worst_cases = sorted(experiment_metrics, key=lambda x: x["cer_delta"])[:5]

        for idx, case in enumerate(worst_cases, 1):
            if case["cer_delta"] >= 0:  # Показываем только ухудшения
                break
            print(f"\n     {idx}. Файл: {case['file_name']}")
            print(
                f"        CER: {case['cer_orig']:.4f} → {case['cer_corr']:.4f} (Δ={case['cer_delta']:+.4f})"
            )
            print(
                f"        WER: {case['wer_orig']:.4f} → {case['wer_corr']:.4f} (Δ={case['wer_delta']:+.4f})"
            )

        # Сохраняем детальные примеры худших случаев
        worst_examples_file = (
            output_dir
            / f"worst_cases_w{window_size}_o{overlap}_i{iterations}_c{int(confidence_threshold*100)}_d{use_dictionary}.txt"
        )
        with open(worst_examples_file, "w", encoding="utf-8") as f:
            f.write(f"Худшие случаи для конфигурации:\n")
            f.write(
                f"  window_size={window_size}, overlap={overlap}, iterations={iterations}\n"
            )
            f.write(
                f"  confidence_threshold={confidence_threshold}, use_dictionary={use_dictionary}\n\n"
            )
            f.write("=" * 80 + "\n\n")

            for idx, case in enumerate(worst_cases, 1):
                if case["cer_delta"] >= 0:
                    break

                # Получаем тексты для этого файла
                file_name = case["file_name"]
                gt_text = extract_text(gt_ann_by_file.get(file_name, []))
                pred_text_orig = extract_text(pred_ann_by_file.get(file_name, []))
                pred_text_corr = corrected_texts[file_name]

                f.write(f"Случай #{idx}: {file_name}\n")
                f.write(
                    f"CER: {case['cer_orig']:.4f} → {case['cer_corr']:.4f} (Δ={case['cer_delta']:+.4f})\n"
                )
                f.write(
                    f"WER: {case['wer_orig']:.4f} → {case['wer_corr']:.4f} (Δ={case['wer_delta']:+.4f})\n"
                )
                f.write(
                    f"BoW F1: {case['bow_f1_orig']:.4f} → {case['bow_f1_corr']:.4f} (Δ={case['bow_f1_delta']:+.4f})\n\n"
                )

                f.write("ИСТИННЫЙ (Ground Truth):\n")
                f.write(f"{gt_text}\n\n")

                f.write("ПРЕДСКАЗАНИЕ (Original Prediction):\n")
                f.write(f"{pred_text_orig}\n\n")

                f.write("ИСПРАВЛЕННОЕ (Corrected):\n")
                f.write(f"{pred_text_corr}\n\n")

                f.write("=" * 80 + "\n\n")

        # Сохраняем детальные результаты для этой комбинации
        detail_file = (
            output_dir
            / f"detail_w{window_size}_o{overlap}_i{iterations}_c{int(confidence_threshold*100)}.csv"
        )
        with open(detail_file, "w", encoding="utf-8-sig", newline="") as f:
            fieldnames = list(experiment_metrics[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(experiment_metrics)

    # Сохраняем сводную таблицу
    summary_file = output_dir / "summary.csv"
    print(f"\n{'='*70}")
    print(f"💾 Сохранение сводной таблицы: {summary_file}")

    with open(summary_file, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(all_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    # Выводим сводку лучших результатов
    print(f"\n{'='*70}")
    print("🏆 ТОП-3 по улучшению CER (mean delta):")
    sorted_by_cer = sorted(all_results, key=lambda x: x["cer_delta_mean"], reverse=True)
    for i, r in enumerate(sorted_by_cer[:3], 1):
        print(
            f"  {i}. window={r['window_size']}, overlap={r['overlap']}, iter={r['iterations']}, conf={r['confidence_threshold']:.2f}, dict={r['use_dictionary']}"
        )
        print(f"     CER Δ = {r['cer_delta_mean']:+.4f}")

    print(f"\n🏆 ТОП-3 по улучшению WER (mean delta):")
    sorted_by_wer = sorted(all_results, key=lambda x: x["wer_delta_mean"], reverse=True)
    for i, r in enumerate(sorted_by_wer[:3], 1):
        print(
            f"  {i}. window={r['window_size']}, overlap={r['overlap']}, iter={r['iterations']}, conf={r['confidence_threshold']:.2f}"
        )
        print(f"     WER Δ = {r['wer_delta_mean']:+.4f}")

    print(f"\n🏆 ТОП-3 по улучшению BoW F1 (mean delta):")
    sorted_by_bow = sorted(
        all_results, key=lambda x: x["bow_f1_delta_mean"], reverse=True
    )
    for i, r in enumerate(sorted_by_bow[:3], 1):
        print(
            f"  {i}. window={r['window_size']}, overlap={r['overlap']}, iter={r['iterations']}, conf={r['confidence_threshold']:.2f}"
        )
        print(f"     BoW F1 Δ = {r['bow_f1_delta_mean']:+.4f}")

    print(f"\n{'='*70}")
    print("✅ Эксперименты завершены!")


# ================= MAIN =================


def main():
    # Пути
    checkpoint_path = "checkpoints/best.pt"
    charset_path = "data/charset.txt"
    words_path = "data/all_words.txt"
    gt_path = Path("test_sorted.json")
    pred_path = Path("test_predictions.json")
    output_dir = Path("experiments_results")

    # Создаём корректор
    print("🚀 Инициализация корректора...")
    corrector = WordCorrector(checkpoint_path, charset_path, words_path)

    # Параметры для экспериментов
    param_grid = {
        "window_size": [7, 8, 9, 10],  # размер окна контекста
        "overlap": [0],  # пересечение окон
        "iterations": [1],  # повторные итерации
        "batch_size": [64],  # размер батча
        "confidence_threshold": [0, 0.9, 0.95],  # порог уверенности
        "use_dictionary": [True],  # использовать словарь
    }

    print(f"\n📋 Сетка параметров:")
    for param, values in param_grid.items():
        print(f"   {param}: {values}")

    # Запуск экспериментов
    run_experiments(corrector, gt_path, pred_path, param_grid, output_dir)


if __name__ == "__main__":
    main()
