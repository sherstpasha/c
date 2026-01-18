import os
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from CharTransformerMLM.model import CharTransformerEdit, EditVocab
from CharTransformerMLM.dataset import CharOCREditDataset
from CharTransformerMLM.vocab import CharVocab
from CharTransformerMLM.utils.collate import collate_edit


class TxtLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_train_val_split(text_path, pairs_csv_path, val_ratio=0.05, seed=42):
    """Создаёт train/val split для строк текста и OCR пар

    Returns:
        (val_indices_lines, val_indices_pairs) - множества индексов для validation
    """
    random.seed(seed)

    # Читаем строки текста
    with open(text_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    n_lines = len(lines)
    n_val_lines = int(n_lines * val_ratio)

    # Случайно выбираем индексы для validation
    all_line_indices = list(range(n_lines))
    random.shuffle(all_line_indices)
    val_indices_lines = set(all_line_indices[:n_val_lines])

    # Читаем OCR пары (если есть)
    val_indices_pairs = set()
    if pairs_csv_path is not None:
        import csv

        with open(pairs_csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            pairs = list(reader)

        # Разделяем на правильные и неправильные пары
        error_indices = []  # incorrect != correct
        correct_indices = []  # incorrect == correct

        for i, row in enumerate(pairs[1:], start=1):  # skip header
            if len(row) >= 3:
                incorrect, correct = row[1].strip(), row[2].strip()
                if incorrect != correct:
                    error_indices.append(i)
                else:
                    correct_indices.append(i)

        # Гарантируем, что в validation попадут неправильные пары
        n_val_pairs = int(len(pairs) * val_ratio)
        n_val_errors = int(len(error_indices) * val_ratio)
        n_val_correct = n_val_pairs - n_val_errors

        # Случайно выбираем из каждой категории
        random.shuffle(error_indices)
        random.shuffle(correct_indices)

        val_indices_pairs = set(error_indices[:n_val_errors])
        if n_val_correct > 0 and correct_indices:
            val_indices_pairs.update(correct_indices[:n_val_correct])

        print(f"Train/Val split: {n_lines - n_val_lines}/{n_val_lines} lines")
        print(
            f"  OCR pairs: {len(error_indices) - n_val_errors}/{n_val_errors} error pairs, "
            f"{len(correct_indices) - n_val_correct}/{n_val_correct} correct pairs"
        )

    return val_indices_lines, val_indices_pairs


def save_augmentation_examples(dataset, vocab, edit_vocab, save_dir, n_samples=3):
    """Сохраняет примеры аугментаций и JSON с окончаниями"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем JSON с окончаниями
    if dataset.top_endings:
        endings_data = {
            "top_endings": dataset.top_endings,
            "total_endings": len(dataset.top_endings),
            "ending_swap_prob_min": dataset.ending_swap_prob_min,
            "ending_swap_prob_max": dataset.ending_swap_prob_max,
        }
        with open(save_dir / "endings.json", "w", encoding="utf-8") as f:
            json.dump(endings_data, f, ensure_ascii=False, indent=2)

    # Генерируем примеры для каждой аугментации
    examples = []
    examples.append("=" * 80)
    examples.append("AUGMENTATION EXAMPLES")
    examples.append("=" * 80)

    augmentation_methods = [
        ("SYNTHETIC NOISE", dataset.make_synthetic_noise),
        ("ENDING SWAP", dataset.make_swap_ending),
        ("EXTRA PUNCTUATION", dataset.make_extra_punct),
        ("HYPHEN BREAK", dataset.make_hyphen_break),
        ("COMMA PREFIX", dataset.make_comma_prefix),
        ("REPEAT ENDING", dataset.make_repeat_ending),
        ("REPEAT BEGINNING", dataset.make_repeat_beginning),
    ]

    # Добавляем real OCR если доступен
    if dataset.pairs_by_image and dataset.error_refs:
        augmentation_methods.insert(0, ("REAL OCR", dataset.make_real_ocr))

    for aug_name, aug_method in augmentation_methods:
        examples.append(f"\n{'=' * 80}")
        examples.append(f"{aug_name}")
        examples.append(f"{'=' * 80}")

        for i in range(n_samples):
            try:
                x, y = aug_method()
                x_ids = x.tolist()
                y_ids = y.tolist()

                noisy = "".join(vocab.id_to_token[i] for i in x_ids)
                target = apply_edit_ops(vocab, edit_vocab, x_ids, y_ids)

                examples.append(f"\n--- Sample {i + 1} ---")
                examples.append(f"NOISY:  {noisy}")
                examples.append(f"TARGET: {target}")
            except Exception as e:
                examples.append(f"\n--- Sample {i + 1} ---")
                examples.append(f"ERROR: {str(e)}")

    # Сохраняем в файл
    with open(save_dir / "augmentation_examples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(examples))

    print(f"Saved augmentation examples to {save_dir / 'augmentation_examples.txt'}")
    if dataset.top_endings:
        print(f"Saved endings to {save_dir / 'endings.json'}")


def edit_accuracy(logits, targets):
    mask = targets != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    preds = logits.argmax(dim=-1)
    return (preds[mask] == targets[mask]).float().mean()


def edit_detection_f1(logits, y, edit_vocab):
    """Метрика для обнаружения редактирований (precision, recall, F1)"""
    copy_id = edit_vocab.COPY
    pred = logits.argmax(-1)

    y_edit = (y != -100) & (y != copy_id)
    p_edit = pred != copy_id

    tp = (p_edit & y_edit).sum().float()
    fp = (p_edit & ~y_edit & (y != -100)).sum().float()
    fn = (~p_edit & y_edit).sum().float()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


@torch.no_grad()
def apply_edit_ops(vocab, edit_vocab, x_ids, op_ids):
    out = []

    for xi, oi in zip(x_ids, op_ids):
        ch = vocab.id_to_token.get(xi, "")

        if oi == -100 or edit_vocab.id_to_op[oi] == "COPY":
            out.append(ch)

        elif edit_vocab.id_to_op[oi] == "DELETE":
            continue

        elif edit_vocab.id_to_op[oi].startswith("REPLACE_"):
            out.append(edit_vocab.id_to_op[oi].replace("REPLACE_", ""))

        elif edit_vocab.id_to_op[oi].startswith("INSERT_"):
            ins = edit_vocab.id_to_op[oi].replace("INSERT_", "")
            out.append(ins)
            out.append(ch)

    return "".join(out)


def compute_cer(pred: str, target: str) -> float:
    """Compute Character Error Rate (Levenshtein distance normalized)"""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    # Levenshtein distance
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / len(target)


def generate_fixed_val_set(dataset, vocab, edit_vocab, save_dir, n_samples_per_aug=50):
    """Генерирует фиксированный validation набор ОДИН РАЗ"""
    save_dir = Path(save_dir)
    val_dir = save_dir / "val"
    val_dir.mkdir(parents=True, exist_ok=True)

    # Проверяем, существует ли уже val набор
    marker_file = val_dir / ".generated"
    if marker_file.exists():
        print(f"Validation set already exists in {val_dir}, skipping generation")
        return

    print(
        f"Generating fixed validation set with {n_samples_per_aug} samples per augmentation..."
    )

    augmentation_configs = [
        ("synthetic_noise", dataset.make_synthetic_noise),
        ("ending_swap", dataset.make_swap_ending),
        ("extra_punctuation", dataset.make_extra_punct),
        ("hyphen_break", dataset.make_hyphen_break),
        ("comma_prefix", dataset.make_comma_prefix),
        ("repeat_ending", dataset.make_repeat_ending),
        ("repeat_beginning", dataset.make_repeat_beginning),
    ]

    # Real OCR если доступен
    if dataset.pairs_by_image and dataset.error_refs:
        augmentation_configs.insert(0, ("real_ocr", dataset.make_real_ocr))

    for aug_name, aug_method in augmentation_configs:
        samples = []

        for i in range(n_samples_per_aug):
            try:
                x, y = aug_method()
                x_ids = x.tolist()
                y_ops = y.tolist()

                noisy = "".join(vocab.id_to_token[idx] for idx in x_ids)
                target = apply_edit_ops(vocab, edit_vocab, x_ids, y_ops)

                samples.append(
                    {
                        "noisy": noisy,
                        "target": target,
                        "x_ids": x_ids,
                        "y_ops": y_ops,
                    }
                )
            except Exception as e:
                print(f"Warning: failed to generate {aug_name} sample {i}: {e}")
                continue

        # Сохраняем в JSON
        output_path = val_dir / f"{aug_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"Generated {len(samples)} samples for {aug_name}")

    # Создаем маркер что набор сгенерирован
    marker_file.write_text("generated")
    print(f"Fixed validation set saved to {val_dir}")


@torch.no_grad()
def validate_on_fixed_set(model, vocab, edit_vocab, val_dir, device):
    """Валидация на фиксированном наборе с метриками качества"""
    val_dir = Path(val_dir)

    if not val_dir.exists():
        print(f"Warning: validation directory {val_dir} does not exist")
        return {}

    model.eval()

    total_metrics = {
        "exact_match": 0,
        "total": 0,
        "cer_sum": 0.0,
        "fixed": 0,  # ошибка была → исправлена
        "had_error": 0,  # сколько было с ошибкой
        "prec_sum": 0.0,
        "rec_sum": 0.0,
        "f1_sum": 0.0,
    }

    aug_metrics = {}

    # Проходим по всем JSON файлам
    for json_file in sorted(val_dir.glob("*.json")):
        aug_name = json_file.stem

        with open(json_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        aug_exact = 0
        aug_cer_sum = 0.0
        aug_fixed = 0
        aug_had_error = 0
        aug_prec_sum = 0.0
        aug_rec_sum = 0.0
        aug_f1_sum = 0.0

        for sample in samples:
            x_ids = (
                torch.tensor(sample["x_ids"], dtype=torch.long).unsqueeze(0).to(device)
            )
            y_ops = (
                torch.tensor(sample["y_ops"], dtype=torch.long).unsqueeze(0).to(device)
            )
            target = sample["target"]
            noisy = sample["noisy"]

            # Предсказание
            logits, _ = model(x_ids)
            pred_ops = logits.argmax(dim=-1)[0].tolist()
            pred = apply_edit_ops(vocab, edit_vocab, sample["x_ids"], pred_ops)

            # Метрики
            is_exact = pred == target
            cer = compute_cer(pred, target)
            precision, recall, f1 = edit_detection_f1(logits, y_ops, edit_vocab)

            aug_exact += int(is_exact)
            aug_cer_sum += cer
            aug_prec_sum += precision.item()
            aug_rec_sum += recall.item()
            aug_f1_sum += f1.item()

            # Fix rate: была ошибка в noisy и мы её исправили
            if noisy != target:
                aug_had_error += 1
                if pred == target:
                    aug_fixed += 1

        # Агрегируем по аугментации
        n = len(samples)
        aug_metrics[aug_name] = {
            "exact_match": aug_exact / n if n > 0 else 0.0,
            "cer": aug_cer_sum / n if n > 0 else 0.0,
            "fix_rate": aug_fixed / aug_had_error if aug_had_error > 0 else 0.0,
            "precision": aug_prec_sum / n if n > 0 else 0.0,
            "recall": aug_rec_sum / n if n > 0 else 0.0,
            "f1": aug_f1_sum / n if n > 0 else 0.0,
            "n_samples": n,
        }

        # Добавляем к общим метрикам
        total_metrics["exact_match"] += aug_exact
        total_metrics["total"] += n
        total_metrics["cer_sum"] += aug_cer_sum
        total_metrics["fixed"] += aug_fixed
        total_metrics["had_error"] += aug_had_error
        total_metrics["prec_sum"] += aug_prec_sum
        total_metrics["rec_sum"] += aug_rec_sum
        total_metrics["f1_sum"] += aug_f1_sum

    # Средние метрики
    total = total_metrics["total"]
    overall_metrics = {
        "exact_match_rate": total_metrics["exact_match"] / total if total > 0 else 0.0,
        "avg_cer": total_metrics["cer_sum"] / total if total > 0 else 0.0,
        "fix_rate": (
            total_metrics["fixed"] / total_metrics["had_error"]
            if total_metrics["had_error"] > 0
            else 0.0
        ),
        "precision": total_metrics["prec_sum"] / total if total > 0 else 0.0,
        "recall": total_metrics["rec_sum"] / total if total > 0 else 0.0,
        "f1": total_metrics["f1_sum"] / total if total > 0 else 0.0,
    }

    return {
        "overall": overall_metrics,
        "by_augmentation": aug_metrics,
    }


@torch.no_grad()
def inspect_predictions(model, vocab, edit_vocab, dataset, device, n_samples=3):
    model.eval()
    print("\n=== EDIT MODEL CHECK (NOISY / TARGET / PRED) ===")

    for _ in range(n_samples):
        x, y = dataset[random.randrange(len(dataset))]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        logits, _ = model(x)
        preds = logits.argmax(dim=-1)

        x_ids = x[0].tolist()
        y_ids = y[0].tolist()
        p_ids = preds[0].tolist()

        noisy = "".join(vocab.id_to_token[i] for i in x_ids)

        # TARGET reconstruction (from GT edit ops)
        target = apply_edit_ops(vocab, edit_vocab, x_ids, y_ids)

        # PRED reconstruction
        pred = apply_edit_ops(vocab, edit_vocab, x_ids, p_ids)

        print("\n--- SAMPLE ---")
        print("NOISY:  ", noisy)
        print("TARGET: ", target)
        print("PRED:   ", pred)


def train(config):
    set_seed(config["seed"])
    device = config["device"]

    os.makedirs(config["save_dir"], exist_ok=True)
    logger = TxtLogger(Path(config["save_dir"]) / "train.log")

    vocab = CharVocab(config["charset_path"])
    edit_vocab = EditVocab(vocab)

    # Создаём train/val split
    val_indices_lines, val_indices_pairs = create_train_val_split(
        config["text_path"],
        config.get("pairs_csv"),
        val_ratio=config.get("val_ratio", 0.05),
        seed=config["seed"],
    )

    # TRAIN dataset
    dataset = CharOCREditDataset(
        text_path=config["text_path"],
        vocab=vocab,
        edit_vocab=edit_vocab,
        pairs_csv_path=config.get("pairs_csv"),
        words_path=config.get("words_path"),
        max_len=config["max_len"],
        max_words=config["max_words"],
        noise_prob=config["noise_prob"],
        ocr_window=config["ocr_window"],
        # Augmentation probabilities
        p_real_ocr=config["p_real_ocr"],
        p_synthetic_noise=config["p_synthetic_noise"],
        p_ending_swap=config["p_ending_swap"],
        p_extra_punct=config["p_extra_punct"],
        p_hyphen_break=config["p_hyphen_break"],
        p_comma_prefix=config["p_comma_prefix"],
        p_repeat_ending=config["p_repeat_ending"],
        p_repeat_beginning=config["p_repeat_beginning"],
        ending_swap_prob_min=config["ending_swap_prob_min"],
        ending_swap_prob_max=config["ending_swap_prob_max"],
        split="train",
        val_indices_lines=val_indices_lines,
        val_indices_pairs=val_indices_pairs,
    )

    # VAL dataset (для генерации фиксированного val набора)
    val_dataset = CharOCREditDataset(
        text_path=config["text_path"],
        vocab=vocab,
        edit_vocab=edit_vocab,
        pairs_csv_path=config.get("pairs_csv"),
        words_path=config.get("words_path"),
        max_len=config["max_len"],
        max_words=config["max_words"],
        noise_prob=config["noise_prob"],
        ocr_window=config["ocr_window"],
        # Augmentation probabilities
        p_real_ocr=config["p_real_ocr"],
        p_synthetic_noise=config["p_synthetic_noise"],
        p_ending_swap=config["p_ending_swap"],
        p_extra_punct=config["p_extra_punct"],
        p_hyphen_break=config["p_hyphen_break"],
        p_comma_prefix=config["p_comma_prefix"],
        p_repeat_ending=config["p_repeat_ending"],
        p_repeat_beginning=config["p_repeat_beginning"],
        ending_swap_prob_min=config["ending_swap_prob_min"],
        ending_swap_prob_max=config["ending_swap_prob_max"],
        split="val",
        val_indices_lines=val_indices_lines,
        val_indices_pairs=val_indices_pairs,
    )

    # Сохраняем примеры аугментаций и окончания при старте
    save_augmentation_examples(dataset, vocab, edit_vocab, config["save_dir"])

    # Генерируем фиксированный validation набор ОДИН РАЗ из VAL данных
    generate_fixed_val_set(
        val_dataset,  # используем val_dataset!
        vocab,
        edit_vocab,
        config["save_dir"],
        n_samples_per_aug=config.get("val_samples_per_aug", 50),
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_edit(b, vocab.pad),
    )

    model = CharTransformerEdit(
        vocab_size=len(vocab.token_to_id),
        edit_vocab_size=edit_vocab.size,
        emb_size=config["emb_size"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        ffn_size=config["ffn_size"],
        dropout=config["dropout"],
        pad_idx=vocab.pad,
        eow_idx=vocab.eow,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=config["lr"] * 0.01,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_loss = float("inf")
    best_cer = float("inf")
    best_em = 0.0

    val_dir = Path(config["save_dir"]) / "val"

    for epoch in range(config["epochs"]):
        model.train()
        run_loss = 0.0
        run_acc = 0.0
        run_prec = 0.0
        run_rec = 0.0
        run_f1 = 0.0

        pbar = tqdm(loader, desc=f"epoch {epoch}", ncols=100)

        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            logits, _ = model(x, y)

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            acc = edit_accuracy(logits, y)
            precision, recall, f1 = edit_detection_f1(logits, y, edit_vocab)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            run_loss += loss.item()
            run_acc += acc.item()
            run_prec += precision.item()
            run_rec += recall.item()
            run_f1 += f1.item()

            pbar.set_postfix(
                loss=f"{run_loss/(pbar.n+1):.3f}",
                f1=f"{run_f1/(pbar.n+1):.3f}",
            )

        scheduler.step()

        avg_loss = run_loss / len(loader)
        avg_acc = run_acc / len(loader)
        avg_prec = run_prec / len(loader)
        avg_rec = run_rec / len(loader)
        avg_f1 = run_f1 / len(loader)

        logger.log(
            f"epoch {epoch} | loss={avg_loss:.4f} acc={avg_acc:.3f} "
            f"P={avg_prec:.3f} R={avg_rec:.3f} F1={avg_f1:.3f}"
        )

        # Валидация на фиксированном наборе
        val_metrics = validate_on_fixed_set(model, vocab, edit_vocab, val_dir, device)

        if val_metrics:
            overall = val_metrics["overall"]
            logger.log(
                f"VAL | EM={overall['exact_match_rate']:.3f} "
                f"CER={overall['avg_cer']:.4f} "
                f"Fix={overall['fix_rate']:.3f} "
                f"P={overall['precision']:.3f} "
                f"R={overall['recall']:.3f} "
                f"F1={overall['f1']:.3f}"
            )

            # Логируем по аугментациям
            for aug_name, metrics in val_metrics["by_augmentation"].items():
                logger.log(
                    f"  {aug_name:20s} | EM={metrics['exact_match']:.3f} "
                    f"CER={metrics['cer']:.4f} "
                    f"F1={metrics['f1']:.3f}"
                )

            # Сохраняем лучшие модели
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    model.state_dict(), Path(config["save_dir"]) / "best_loss.pt"
                )
                logger.log(f"Saved best_loss.pt (loss={best_loss:.4f})")

            if overall["avg_cer"] < best_cer:
                best_cer = overall["avg_cer"]
                torch.save(model.state_dict(), Path(config["save_dir"]) / "best_cer.pt")
                logger.log(f"Saved best_cer.pt (CER={best_cer:.4f})")

            if overall["exact_match_rate"] > best_em:
                best_em = overall["exact_match_rate"]
                torch.save(model.state_dict(), Path(config["save_dir"]) / "best_em.pt")
                logger.log(f"Saved best_em.pt (EM={best_em:.3f})")

        if (epoch + 1) % config["eval_every_epochs"] == 0:
            inspect_predictions(model, vocab, edit_vocab, dataset, device)

    torch.save(model.state_dict(), Path(config["save_dir"]) / "last.pt")


CONFIG = {
    # ===== GENERAL =====
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # ===== DATA =====
    "charset_path": "data/charset.txt",
    "text_path": "data/extracted_texts_cleaned.txt",
    "pairs_csv": "data\\pairs_combined_extracted.csv",  # None если не используем real OCR
    "words_path": "data/all_words.txt",  # для извлечения окончаний
    # ===== DATASET PARAMETERS =====
    "max_len": 32,  # максимальная длина последовательности
    "max_words": 3,  # максимальное количество слов в семпле
    "ocr_window": 3,  # размер окна для real OCR пар
    # ===== AUGMENTATION PROBABILITIES =====
    # Доля каждой аугментации (должны в сумме давать ~1.0)
    "p_real_ocr": 1.0,  # real OCR window (30%)
    "p_synthetic_noise": 0.0,  # synthetic noise (20%)
    "p_ending_swap": 0.0,  # ending swap (15%)
    "p_extra_punct": 0.0,  # extra punctuation (1%)
    "p_hyphen_break": 0.0,  # hyphen break (1%)
    "p_comma_prefix": 0.0,  # comma prefix (1%)
    "p_repeat_ending": 0.0,  # repeat ending (7.5%)
    "p_repeat_beginning": 0.0,  # repeat beginning (7.5%)
    # Параметры для конкретных аугментаций
    "noise_prob": 0.0,  # вероятность замены символа в synthetic noise
    "ending_swap_prob_min": 0.0,  # минимальная вероятность ending swap внутри аугментации
    "ending_swap_prob_max": 0.0,  # максимальная вероятность ending swap внутри аугментации
    # ===== MODEL ARCHITECTURE =====
    "emb_size": 192,
    "n_layers": 5,
    "n_heads": 6,
    "ffn_size": 768,
    "dropout": 0.1,
    # ===== TRAINING HYPERPARAMETERS =====
    "epochs": 50,
    "batch_size": 128,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    # ===== LOGGING & CHECKPOINTS =====
    "eval_every_epochs": 1,
    "save_dir": "checkpoints_edit3",
    "val_samples_per_aug": 100,  # количество примеров на каждую аугментацию в val наборе
    "val_ratio": 0.1,  # доля данных для validation (5%)
    "resume": None,
}


if __name__ == "__main__":
    train(CONFIG)
