import json
import random
import Levenshtein
import torch
from datetime import datetime
import torch.nn.functional as F
from statistics import mean
from tqdm import tqdm


def evaluate_ocr_confidence(
    model,
    eval_pairs,
    c2i,
    device,
    max_len,
):
    model.eval()

    correct_confs = []
    incorrect_confs = []

    for inc, cor in tqdm(
        eval_pairs,
        desc="OCR confidence eval",
        leave=False,
    ):
        chars_inc = list(inc)
        chars_cor = list(cor)
        L = min(len(chars_inc), max_len)

        for i in range(L):
            # маскируем текущую позицию
            unk = c2i["<UNK>"]
            mask = c2i["<MASK>"]

            ids = [(c2i.get(chars_inc[j], unk) if j != i else mask) for j in range(L)]
            ids += [c2i["<PAD>"]] * (max_len - L)

            x = torch.tensor(ids, device=device).unsqueeze(0)

            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits[0, i], dim=-1)

            unk = c2i["<UNK>"]
            cur_char = chars_inc[i]
            cur_id = c2i.get(cur_char, unk)
            conf = probs[cur_id].item()

            if chars_inc[i] == chars_cor[i]:
                correct_confs.append(conf)
            else:
                incorrect_confs.append(conf)

    def stats(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        n = len(xs)
        return {
            "n": n,
            "mean": sum(xs) / n,
            "p25": xs[int(0.25 * n)],
            "median": xs[int(0.5 * n)],
            "p75": xs[int(0.75 * n)],
        }

    return stats(correct_confs), stats(incorrect_confs)


import json


def format_trace(trace):
    if not trace:
        return "[]"
    return json.dumps(trace, ensure_ascii=False)


def reconstruct_word_thresholded_batch(
    model,
    word: str,
    c2i,
    i2c,
    device,
    max_len: int,
    mask_threshold: float,
    apply_threshold: float,
    max_edits: int,
    return_trace: bool = False,
    return_p_cur: bool = False,
):
    """
    Batch-masked OCR correction.
    Делает ОДИН forward для всех позиций слова.

    Поведение идентично reconstruct_word_thresholded,
    но в ~L раз быстрее.
    """
    model.eval()

    chars = list(word[:max_len])
    L = len(chars)
    trace = []

    if L == 0:
        if return_trace or return_p_cur:
            return word, [], []
        return word

    unk = c2i["<UNK>"]
    mask = c2i["<MASK>"]
    pad = c2i["<PAD>"]

    # -------------------------------------------------
    # 1) строим batch из L masked-вариантов
    # -------------------------------------------------
    batch = []

    for i in range(L):
        ids = [(c2i.get(ch, unk) if j != i else mask) for j, ch in enumerate(chars)]
        ids += [pad] * (max_len - len(ids))
        batch.append(ids)

    x = torch.tensor(batch, device=device)  # [L, T]

    # -------------------------------------------------
    # 2) один forward
    # -------------------------------------------------
    with torch.no_grad():
        logits = model(x)  # [L, T, V]
        probs = torch.softmax(logits, dim=-1)

    # -------------------------------------------------
    # 3) считаем p_cur и собираем confidences
    # -------------------------------------------------
    confidences = []

    for i in range(L):
        cur_id = c2i.get(chars[i], unk)
        p_cur = probs[i, i, cur_id].item()
        confidences.append((i, p_cur, probs[i, i]))

    # -------------------------------------------------
    # 4) выбираем кандидатов
    # -------------------------------------------------
    candidates = [
        (i, p_cur, prob_vec)
        for (i, p_cur, prob_vec) in confidences
        if p_cur < mask_threshold
    ]

    candidates.sort(key=lambda x: x[1])  # самые неуверенные первые

    # -------------------------------------------------
    # 5) применяем исправления
    # -------------------------------------------------
    edits = 0

    for i, p_cur, prob_vec in candidates:
        if edits >= max_edits:
            break

        best_id = prob_vec.argmax().item()
        best_p = prob_vec[best_id].item()
        best_char = i2c[best_id]

        applied = (best_p >= apply_threshold) and (best_char != chars[i])

        trace.append(
            {
                "pos": i,
                "old": chars[i],
                "best": best_char,
                "p_cur": round(p_cur, 4),
                "p_best": round(best_p, 4),
                "applied": applied,
            }
        )

        if applied:
            chars[i] = best_char
            edits += 1

    result = "".join(chars)

    if return_trace or return_p_cur:
        return result, trace, confidences
    return result


def cer(pred: str, target: str) -> float:
    if not target:
        return 0.0
    return Levenshtein.distance(pred, target) / len(target)


def safe_mean(xs):
    return mean(xs) if xs else 0.0


def per_char_p_cur(model, word: str, c2i, device, max_len: int):
    """
    Для каждого i: маскируем позицию i и считаем p(cur_char | контекст).
    Возвращает list[float] длины L (L = min(len(word), max_len)).
    """
    model.eval()
    chars = list(word[:max_len])
    L = len(chars)
    p_list = []

    for i in range(L):
        unk = c2i["<UNK>"]
        mask = c2i["<MASK>"]

        ids = [(c2i.get(ch, unk) if j != i else mask) for j, ch in enumerate(chars)]
        ids += [c2i["<PAD>"]] * (max_len - len(ids))
        x = torch.tensor(ids, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits[0, i], dim=-1)

        unk = c2i["<UNK>"]
        cur_id = c2i.get(chars[i], unk)
        p_list.append(float(probs[cur_id].item()))

    return p_list


def format_p_list(p_list, ndigits=4):
    """Чтобы CSV был компактным: округляем и сериализуем в JSON-строку."""
    return json.dumps([round(p, ndigits) for p in p_list], ensure_ascii=False)


def evaluate_ocr_with_cer(
    model,
    eval_pairs,
    c2i,
    i2c,
    device,
    max_len,
    mask_threshold,
    apply_threshold,
    max_edits,
    csv_path=None,
    lexicon=None,
):
    """
    OCR evaluation (single-pass):

    - exact match
    - CER before / after
    - delta CER
    - improved / worsened / unchanged %
    - optional CSV with p_cur + trace

    IMPORTANT:
    - does NOT call per_char_p_cur
    - uses confidences returned by reconstruct_word_thresholded
    """

    if not eval_pairs:
        return {
            "exact_match": 0.0,
            "cer_before": 0.0,
            "cer_after": 0.0,
            "delta": 0.0,
            "improved_pct": 0.0,
            "worsened_pct": 0.0,
            "unchanged_pct": 0.0,
        }

    model.eval()

    exact = 0
    cer_before = []
    cer_after = []
    rows = []

    with torch.no_grad():
        for inc, cor in tqdm(
            eval_pairs,
            desc="OCR CER eval",
            leave=False,
        ):
            cb = cer(inc, cor)

            # ---------- shortcut: already a valid lexicon word ----------
            if lexicon is not None and inc in lexicon:
                pred = inc
                ca = cb
                label = "unchanged"
                p_cur_str = format_p_list([], ndigits=4)
                trace = []

            else:
                pred, trace, confidences = reconstruct_word_thresholded_batch(
                    model,
                    inc,
                    c2i,
                    i2c,
                    device,
                    max_len,
                    mask_threshold,
                    apply_threshold,
                    max_edits,
                    return_trace=True,
                    return_p_cur=True,
                )

                ca = cer(pred, cor)

                if ca < cb:
                    label = "improved"
                elif ca > cb:
                    label = "worsened"
                else:
                    label = "unchanged"

                # извлекаем p_cur из confidences
                p_cur_list = [p for (_, p, _) in confidences]
                p_cur_str = format_p_list(p_cur_list, ndigits=4)

            cer_before.append(cb)
            cer_after.append(ca)

            if pred == cor:
                exact += 1

            rows.append(
                {
                    "incorrect": inc,
                    "predicted": pred,
                    "correct": cor,
                    "label": label,
                    "p_cur": p_cur_str,
                    "trace": format_trace(trace),
                }
            )

    # ---------- sorting for readable CSV ----------
    LABEL_ORDER = {"improved": 0, "worsened": 1, "unchanged": 2}
    rows.sort(key=lambda r: LABEL_ORDER.get(r["label"], 999))

    if csv_path:
        write_ocr_csv(rows, csv_path)

    stats = {
        "exact_match": exact / len(eval_pairs),
        "cer_before": safe_mean(cer_before),
        "cer_after": safe_mean(cer_after),
        "delta": safe_mean(cer_after) - safe_mean(cer_before),
        "improved_pct": 100.0 * sum(r["label"] == "improved" for r in rows) / len(rows),
        "worsened_pct": 100.0 * sum(r["label"] == "worsened" for r in rows) / len(rows),
        "unchanged_pct": 100.0
        * sum(r["label"] == "unchanged" for r in rows)
        / len(rows),
    }

    return stats


import csv


def write_ocr_csv(rows, path):
    """
    rows: list of dicts with keys:
      incorrect, predicted, correct, label
    """
    if not rows:
        return

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "incorrect",
                "predicted",
                "correct",
                "label",
                "p_cur",
                "trace",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def log_random_examples(model, batch, logits, c2i, i2c, logger, k=3):
    x, y = batch
    preds = logits.argmax(dim=-1)

    B, T = x.shape
    indices = random.sample(range(B), min(k, B))

    pad_id = c2i["<PAD>"]

    logger.log("---- MLM examples ----")
    for b in indices:
        x_ids = x[b].tolist()
        y_ids = y[b].tolist()
        p_ids = preds[b].tolist()

        inp = []
        tgt = []
        prd = []

        for i in range(T):
            if x_ids[i] == pad_id:
                break  # <-- КЛЮЧЕВО

            if y_ids[i] != -100:
                inp.append("<MASK>")
                tgt.append(i2c[y_ids[i]])
                prd.append(i2c[p_ids[i]])
            else:
                ch = i2c[x_ids[i]]
                inp.append(ch)
                tgt.append(ch)
                prd.append(ch)

        logger.log(f"INPUT   : {''.join(inp)}")
        logger.log(f"TARGET  : {''.join(tgt)}")
        logger.log(f"PREDICT : {''.join(prd)}")
        logger.log("")


class Logger:
    """Простой логгер в файл и консоль."""

    def __init__(self, path: str = None):
        self.path = path
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"=== Training started: {datetime.now()} ===\n")

    def log(self, msg: str):
        print(msg)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


def build_vocab(
    words: list[str], include_space: bool = True
) -> tuple[dict, dict, list]:
    """Построить словарь символов."""
    chars = set()
    for w in words:
        chars.update(w)
    if include_space:
        chars.add(" ")

    chars = ["<PAD>", "<MASK>", "<UNK>"] + sorted(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    return c2i, i2c, chars


def encode_str(s: str, c2i: dict, max_len: int) -> list[int]:
    """Закодировать строку в список индексов с паддингом."""
    unk = c2i["<UNK>"]
    pad = c2i["<PAD>"]
    ids = [c2i.get(ch, unk) for ch in s[:max_len]]
    return ids + [pad] * (max_len - len(ids))


def choose_spans(
    L: int, span_min: int, span_max: int, num_spans_min: int, num_spans_max: int
) -> list[int]:
    """Выбрать позиции для span masking (избегая краёв)."""
    if L <= 3:
        return []
    n_spans = random.randint(num_spans_min, num_spans_max)
    positions = set()
    for _ in range(n_spans):
        span_len = random.randint(span_min, span_max)
        start_max = min(L - 2, L - 1 - span_len)
        if start_max < 1:
            continue
        start = random.randint(1, start_max)
        for p in range(start, start + span_len):
            if 1 <= p <= L - 2:
                positions.add(p)
    return sorted(positions)


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Accuracy по маскированным позициям."""
    with torch.no_grad():
        mask = targets != -100
        if mask.sum().item() == 0:
            return 0.0
        preds = logits.argmax(dim=-1)
        return (preds[mask] == targets[mask]).float().mean().item()


def load_allowed_chars(charset_path: str) -> set[str]:
    """Загрузить разрешённые символы из charset.txt (только буквы)."""
    allowed = set()
    with open(charset_path, encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n\r")
            if len(ch) == 1 and ch.isalpha():
                allowed.add(ch)
    return allowed


def filter_words(
    words: list[str], min_len: int = 1, allowed_chars: set[str] = None
) -> list[str]:
    """Фильтрация слов: только валидные слова из букв."""
    if allowed_chars is None:
        allowed_chars = get_allowed_chars()
    result = []
    for w in words:
        cleaned = "".join(ch for ch in w if ch in allowed_chars)
        if len(cleaned) >= min_len:
            result.append(cleaned)
    return result
