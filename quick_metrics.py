"""
Вычисление метрик с автоматическим исправлением дубликатов
"""

import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
from manuscript.data.structures import Word, Line, Block, Page
from manuscript.utils.sorting import organize_page


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
    """Генерирует символьные n-граммы"""
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
    Извлекает текст из аннотаций с сортировкой через EAST organize_page
    """
    if not annotations:
        return ""

    # Создаем Word объекты из аннотаций
    words = []
    for ann in annotations:
        # Извлекаем текст
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

        # Пропускаем специальные метки
        if not text or text in ["<NR>", "<nr>", "###", ""]:
            continue

        # Извлекаем полигон
        if "segmentation" in ann and ann["segmentation"]:
            flat_polygon = ann["segmentation"][0]
        elif "bbox" in ann:
            x, y, w, h = ann["bbox"]
            flat_polygon = [x, y, x + w, y, x + w, y + h, x, y + h]
        else:
            continue

        # Преобразуем плоский список в список кортежей (x, y)
        polygon = [
            (flat_polygon[i], flat_polygon[i + 1])
            for i in range(0, len(flat_polygon), 2)
        ]

        # Создаем Word с минимальными параметрами
        word = Word(
            polygon=polygon, detection_confidence=ann.get("score", 1.0), text=text
        )
        words.append(word)

    if not words:
        return ""

    # Создаем Page структуру для сортировки
    page = Page(blocks=[Block(lines=[Line(words=words, order=0)], order=0)])

    # Применяем EAST сортировку с теми же параметрами что и в детекторе
    sorted_page = organize_page(page, max_splits=10, use_columns=True)

    # Собираем текст в правильном порядке
    texts = []
    for block in sorted(sorted_page.blocks, key=lambda b: b.order):
        for line in sorted(block.lines, key=lambda l: l.order):
            for word in sorted(line.words, key=lambda w: w.order):
                if word.text:
                    texts.append(word.text)

    return " ".join(texts)


# Пути
gt_path = Path("Archives020525/test_sorted.json")
pred_path = Path(
    "Archives020525/test_predictions.json"
)  # Исправлено: используем существующий файл
output_csv = Path("Archives020525/metrics_results.csv")
output_txt = Path("Archives020525/metrics_results.txt")

# Проверяем что файлы существуют
if not gt_path.exists():
    print(f"❌ ОШИБКА: {gt_path} не найден!")
    print("   Запустите run_resort.bat для создания отсортированного GT файла")
    exit(1)

if not pred_path.exists():
    print(f"❌ ОШИБКА: {pred_path} не найден!")
    exit(1)

print("Загрузка файлов...")
with open(gt_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)
with open(pred_path, "r", encoding="utf-8") as f:
    pred_data = json.load(f)

print(f"GT: {len(gt_data['images'])} images, {len(gt_data['annotations'])} annotations")
print(
    f"Pred: {len(pred_data['images'])} images, {len(pred_data['annotations'])} annotations"
)

# Убираем дубликаты из predictions по file_name
print("\nУдаление дубликатов из predictions...")
# Создаем маппинг file_name -> первый встреченный image_id
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

# Группируем аннотации
print("Группировка аннотаций...")
gt_images_map = {img["file_name"]: img for img in gt_data["images"]}
gt_id_to_file = {img["id"]: img["file_name"] for img in gt_data["images"]}
pred_id_to_file = {img["id"]: img["file_name"] for img in pred_data["images"]}

gt_ann_by_file = defaultdict(list)
for ann in gt_data["annotations"]:
    fname = gt_id_to_file.get(ann["image_id"])
    if fname:
        gt_ann_by_file[fname].append(ann)

# Для predictions берем только аннотации от ПЕРВОГО image_id каждого файла
pred_ann_by_file = defaultdict(list)
for ann in pred_data["annotations"]:
    img_id = ann["image_id"]
    fname = pred_id_to_file.get(img_id)
    if fname:
        # Берем только если это аннотация от первого image_id для этого file_name
        if img_id == pred_file_to_first_id[fname]:
            pred_ann_by_file[fname].append(ann)

print(f"\nУникальных файлов в GT: {len(gt_images_map)}")
print(f"Уникальных файлов в Pred: {len(pred_images_map)}")
print(f"Уникальных аннотаций GT: {sum(len(anns) for anns in gt_ann_by_file.values())}")
print(
    f"Уникальных аннотаций Pred: {sum(len(anns) for anns in pred_ann_by_file.values())}"
)

# Вычисляем метрики
print("\nВычисление метрик...")
results = []
for file_name in sorted(gt_images_map.keys()):
    gt_text = extract_text(gt_ann_by_file.get(file_name, []))
    pred_text = extract_text(pred_ann_by_file.get(file_name, []))

    cer = calculate_cer(gt_text, pred_text) if gt_text or pred_text else 0.0
    wer = calculate_wer(gt_text, pred_text) if gt_text or pred_text else 0.0
    acc = max(0.0, 1.0 - cer)
    bow_f1 = calculate_bow_f1(gt_text, pred_text)
    char_3gram_f1 = calculate_char_ngram_f1(gt_text, pred_text, n=3)

    results.append(
        {
            "file_name": file_name,
            "cer": f"{cer:.4f}",
            "wer": f"{wer:.4f}",
            "accuracy": f"{acc:.4f}",
            "bow_f1": f"{bow_f1:.4f}",
            "char_3gram_f1": f"{char_3gram_f1:.4f}",
            "gt_chars": len(gt_text),
            "pred_chars": len(pred_text),
            "gt_words": len(gt_text.split()),
            "pred_words": len(pred_text.split()),
            "gt_text": gt_text,
            "pred_text": pred_text,
        }
    )

# Сохраняем CSV
print(f"\nСохранение {output_csv}...")
with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    fields = [
        "file_name",
        "cer",
        "wer",
        "accuracy",
        "bow_f1",
        "char_3gram_f1",
        "gt_chars",
        "pred_chars",
        "gt_words",
        "pred_words",
        "gt_text",
        "pred_text",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

# Статистика
cer_vals = [float(r["cer"]) for r in results]
wer_vals = [float(r["wer"]) for r in results]
acc_vals = [float(r["accuracy"]) for r in results]
bow_f1_vals = [float(r["bow_f1"]) for r in results]
char_3gram_f1_vals = [float(r["char_3gram_f1"]) for r in results]

print("\n" + "=" * 70)
print(f"Обработано: {len(results)} изображений")
print("\nМетрики (среднее ± std | медиана):")
print(
    f"  CER:           {np.mean(cer_vals):.4f} ± {np.std(cer_vals):.4f} | {np.median(cer_vals):.4f}"
)
print(
    f"  WER:           {np.mean(wer_vals):.4f} ± {np.std(wer_vals):.4f} | {np.median(wer_vals):.4f}"
)
print(
    f"  Accuracy:      {np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f} | {np.median(acc_vals):.4f}"
)
print(
    f"  BoW F1:        {np.mean(bow_f1_vals):.4f} ± {np.std(bow_f1_vals):.4f} | {np.median(bow_f1_vals):.4f}"
)
print(
    f"  Char 3-gram F1: {np.mean(char_3gram_f1_vals):.4f} ± {np.std(char_3gram_f1_vals):.4f} | {np.median(char_3gram_f1_vals):.4f}"
)
print("=" * 70)
print(f"\nРезультаты сохранены в {output_csv}")

# Сохраняем TXT с детальными результатами
print(f"Сохранение {output_txt}...")
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write(f"Обработано: {len(results)} изображений\n\n")
    f.write("Метрики (среднее ± std | медиана):\n")
    f.write(
        f"  CER:           {np.mean(cer_vals):.4f} ± {np.std(cer_vals):.4f} | {np.median(cer_vals):.4f}\n"
    )
    f.write(
        f"  WER:           {np.mean(wer_vals):.4f} ± {np.std(wer_vals):.4f} | {np.median(wer_vals):.4f}\n"
    )
    f.write(
        f"  Accuracy:      {np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f} | {np.median(acc_vals):.4f}\n"
    )
    f.write(
        f"  BoW F1:        {np.mean(bow_f1_vals):.4f} ± {np.std(bow_f1_vals):.4f} | {np.median(bow_f1_vals):.4f}\n"
    )
    f.write(
        f"  Char 3-gram F1: {np.mean(char_3gram_f1_vals):.4f} ± {np.std(char_3gram_f1_vals):.4f} | {np.median(char_3gram_f1_vals):.4f}\n"
    )
    f.write("=" * 70 + "\n\n")

    # Детали по каждому изображению
    for r in results:
        f.write(f"\n{'='*70}\n")
        f.write(f"Файл: {r['file_name']}\n")
        f.write(f"  CER: {r['cer']}, WER: {r['wer']}, Accuracy: {r['accuracy']}\n")
        f.write(f"  BoW F1: {r['bow_f1']}, Char 3-gram F1: {r['char_3gram_f1']}\n")
        f.write(f"  GT: {r['gt_chars']} символов, {r['gt_words']} слов\n")
        f.write(f"  Pred: {r['pred_chars']} символов, {r['pred_words']} слов\n")
        f.write(f"\n  GT текст:\n  {r['gt_text']}\n")
        f.write(f"\n  Pred текст:\n  {r['pred_text']}\n")

print(f"Детальные результаты сохранены в {output_txt}")
