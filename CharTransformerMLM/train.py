import os
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from CharTransformerMLM.model import CharTransformerMLM
from CharTransformerMLM.dataset import CharOCRDenoiseDataset
from CharTransformerMLM.vocab import CharVocab
from CharTransformerMLM.utils.collate import collate_denoise


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


CONFIG = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # data
    "charset_path": "data/charset.txt",
    "text_path": "data/extracted_texts_cleaned.txt",
    "pairs_csv": "data/pairs_with_errors.csv",
    "words_path": "data/all_words.txt",
    "max_words": 3,
    "p_real_start": 0.1,
    "p_real_end": 0.3,
    "p_ending_swap": 0.03,
    "p_extra_punct": 0.02,
    # model
    "emb_size": 160,
    "n_layers": 4,
    "n_heads": 5,
    "ffn_size": 640,
    "dropout": 0.1,
    # training
    "epochs": 10,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    # logging
    "eval_every_epochs": 1,
    "save_dir": "checkpoints",
    "resume": None,
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mlm_accuracy(logits, targets):
    mask = targets != -100
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    preds = logits.argmax(dim=-1)
    return (preds[mask] == targets[mask]).float().mean()


@torch.no_grad()
def inspect_denoise_predictions(model, vocab, dataset, device, n_samples=3):
    model.eval()
    print("\n=== DENOISING PREDICTIONS CHECK ===")

    def pretty_full(ids):
        s = vocab.decode(ids, collapse_ins=False)
        return s.replace("<EOW>", " | ").replace("<INS>", "·")

    def show_sample(x, y, label="SAMPLE"):
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        logits, _ = model(x, y)

        eow = vocab.eow
        ins = vocab.ins

        # Маска для копирования: не EOW и не INS (если target=-100)
        mask = (x != eow) & (x != ins) & (y == -100)
        logits[..., eow] -= mask * 1e9

        # EOW позиции - запрещаем не-EOW
        eow_positions = x == eow
        non_eow_mask = torch.ones_like(logits, dtype=torch.bool)
        non_eow_mask[..., eow] = False
        logits[eow_positions.unsqueeze(-1).expand_as(logits) & non_eow_mask] -= 1e9

        # <INS> никогда не должен быть выходным символом (только входной слот)
        logits[..., ins] -= 1e9

        preds = logits.argmax(dim=-1)

        x_ids = x[0].tolist()
        y_ids = y[0].tolist()
        p_ids = preds[0].tolist()

        noisy_full = pretty_full(x_ids)
        pred_full = pretty_full(p_ids)

        tgt_full = []
        for xi, yi in zip(x_ids, y_ids):
            if yi != -100:
                tgt_full.append(vocab.id_to_token[yi])
            else:
                tgt_full.append(vocab.id_to_token[xi])
        tgt_full = "".join(tgt_full).replace("<EOW>", " | ").replace("<INS>", "·")

        fix_noisy = []
        fix_tgt = []
        fix_pred = []

        for xi, yi, pi in zip(x_ids, y_ids, p_ids):
            if yi != -100:
                fix_noisy.append(vocab.id_to_token[xi])
                fix_tgt.append(vocab.id_to_token[yi])
                fix_pred.append(vocab.id_to_token[pi])

        print(f"\n--- {label} ---")
        print("NOISY FULL:  ", noisy_full)
        print("TARGET FULL: ", tgt_full)
        print("PRED FULL:   ", pred_full)

        if fix_noisy:
            print("\nFIX ONLY:")
            print("NOISY:  ", "".join(fix_noisy).replace("<INS>", "·"))
            print("TARGET: ", "".join(fix_tgt))
            print("PRED:   ", "".join(fix_pred))

    # Сначала показываем пример из реальных OCR-пар (если есть)
    real_sample = dataset.get_real_sample()
    if real_sample is not None:
        x, y = real_sample
        show_sample(x, y, label="REAL OCR PAIR")

    # Затем показываем синтетические примеры
    for i in range(n_samples):
        x, y = dataset.get_synthetic_sample()
        show_sample(x, y, label=f"SYNTHETIC {i+1}")


@torch.no_grad()
def inspect_word_embeddings(model, vocab, dataset, device, n_words=3, top_k=5):
    model.eval()
    print("\n=== WORD EMBEDDINGS CHECK ===")

    words = []
    embeddings = []

    sample_lines = random.sample(dataset.lines, min(100, len(dataset.lines)))

    for line in sample_lines:
        line_words = line.split()
        for word in line_words:
            if len(word) < 3 or len(word) > 15:
                continue
            if not all(ch in vocab.token_to_id for ch in word):
                continue
            if not any(ch.isalpha() for ch in word):
                continue

            ids = vocab.encode(word) + [vocab.eow]
            x = torch.tensor([ids], device=device)

            _, hidden = model(x)
            word_embs = model.extract_word_embeddings(hidden, x)

            if len(word_embs[0]) > 0:
                words.append(word)
                embeddings.append(word_embs[0][0])

        if len(words) >= 150:
            break

    if len(words) < n_words + top_k:
        print("Недостаточно слов для анализа")
        return

    emb_matrix = torch.stack(embeddings)
    emb_matrix = emb_matrix / emb_matrix.norm(dim=-1, keepdim=True)

    indices = random.sample(range(len(words)), n_words)

    for idx in indices:
        query_word = words[idx]
        query_emb = emb_matrix[idx : idx + 1]  # [1, D]

        similarities = (emb_matrix @ query_emb.T).squeeze()  # [N]

        top_indices = similarities.argsort(descending=True)[: top_k + 1]

        neighbors = []
        for ti in top_indices:
            if ti != idx:
                neighbors.append(f"{words[ti]} ({similarities[ti]:.3f})")

        print(f"\n'{query_word}' → {', '.join(neighbors[:top_k])}")


def save_augmentation_samples(dataset, vocab, save_dir: Path, n_samples: int = 10):

    def pretty_full(ids):
        s = vocab.decode(ids)
        return s.replace("<EOW>", " | ")

    def detect_augmentation_types(x_ids, y_ids, vocab):
        """Определить типы примененных аугментаций"""
        types = []
        space_id = vocab.token_to_id.get(" ")
        comma_id = vocab.token_to_id.get(",")
        hyphen_id = vocab.token_to_id.get("-")

        for i, (xi, yi) in enumerate(zip(x_ids, y_ids)):
            if yi != -100:
                # 1. Дефис-запятая (-, в конце слова)
                if (
                    i > 0
                    and xi == comma_id
                    and x_ids[i - 1] == hyphen_id
                    and yi == space_id
                ):
                    if "Артефакты переносов: -," not in types:
                        types.append("Артефакты переносов: -,")

                # 2. Запятая в начале слова
                if (
                    xi == comma_id
                    and yi == space_id
                    and i > 0
                    and x_ids[i - 1] == vocab.eow
                ):
                    if "Артефакты переносов: , в начале" not in types:
                        types.append("Артефакты переносов: , в начале")

                # 3. Одиночный дефис в конце слова
                if (
                    xi == hyphen_id
                    and yi == space_id
                    and i + 1 < len(x_ids)
                    and x_ids[i + 1] == vocab.eow
                ):
                    if "Разрыв слова: одиночный -" not in types:
                        types.append("Разрыв слова: одиночный -")

                # 4. Лишняя пунктуация (замена на знак препинания)
                if xi in [
                    comma_id,
                    vocab.token_to_id.get("."),
                    vocab.token_to_id.get(";"),
                ]:
                    if yi != space_id and "Лишняя пунктуация" not in types:
                        types.append("Лишняя пунктуация")

                # 5. Синтетический шум (обычные замены символов)
                if (
                    yi != space_id
                    and xi != comma_id
                    and xi != hyphen_id
                    and xi != vocab.eow
                    and "Синтетический шум" not in types
                ):
                    types.append("Синтетический шум")

        if not types:
            types.append("Без изменений")

        return types

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("AUGMENTATION SAMPLES")
    output_lines.append("=" * 80)
    output_lines.append("")

    for sample_idx in range(n_samples):
        x, y = dataset[sample_idx]

        x_ids = x.tolist()
        y_ids = y.tolist()

        # Определяем типы аугментаций
        aug_types = detect_augmentation_types(x_ids, y_ids, vocab)

        noisy_full = pretty_full(x_ids)

        tgt_full = []
        for xi, yi in zip(x_ids, y_ids):
            if yi != -100:
                tgt_full.append(vocab.id_to_token[yi])
            else:
                tgt_full.append(vocab.id_to_token[xi])
        tgt_full = "".join(tgt_full).replace("<EOW>", " | ")

        fix_noisy = []
        fix_tgt = []

        for xi, yi in zip(x_ids, y_ids):
            if yi != -100:
                fix_noisy.append(vocab.id_to_token[xi])
                fix_tgt.append(vocab.id_to_token[yi])

        if fix_noisy:
            output_lines.append(f"--- SAMPLE {sample_idx + 1} ---")
            output_lines.append(f"АУГМЕНТАЦИИ: {', '.join(aug_types)}")
            output_lines.append(f"NOISY FULL:  {noisy_full}")
            output_lines.append(f"TARGET FULL: {tgt_full}")
            output_lines.append("")
            output_lines.append("FIX ONLY:")
            output_lines.append(f"NOISY:  {''.join(fix_noisy)}")
            output_lines.append(f"TARGET: {''.join(fix_tgt)}")
            output_lines.append("")

    aug_file = save_dir / "augmentation_samples.txt"
    with open(aug_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Сохранено {n_samples} примеров аугментаций в {aug_file}")


def save_all_augmentation_types(dataset, vocab, save_dir: Path):
    """Сохранить примеры ВСЕХ типов аугментаций принудительно"""

    def pretty_full(ids):
        s = vocab.decode(ids, collapse_ins=False)
        return s.replace("<EOW>", " | ").replace("<INS>", "·")

    def encode_with_ins(words, vocab):
        """Кодировать слова с добавлением <INS> перед <EOW>"""
        ids = []
        for w in words:
            ids.extend(vocab.encode(w))
            ids.append(vocab.ins)  # <INS> перед <EOW>
            ids.append(vocab.eow)
        return ids

    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("ВСЕ ТИПЫ АУГМЕНТАЦИЙ (ПРИНУДИТЕЛЬНО)")
    output_lines.append("· = <INS> (позиция для вставки символа)")
    output_lines.append("=" * 80)
    output_lines.append("")

    # 1. Синтетический шум
    output_lines.append("--- ТИП 1: СИНТЕТИЧЕСКИЙ ШУМ ---")
    for _ in range(3):
        line = dataset.lines[random.randrange(len(dataset.lines))]
        words = line.split()[: dataset.max_words]
        ids = encode_with_ins(words, vocab)

        x, y = dataset.force_synthetic_noise(ids)

        output_lines.append(f"NOISY FULL:  {pretty_full(x)}")
        output_lines.append(f"TARGET FULL: {pretty_full(ids)}")

        fix_indices = [i for i, yi in enumerate(y) if yi != -100]
        if fix_indices:
            noisy_chars = "".join([vocab.id_to_token[x[i]] for i in fix_indices])
            target_chars = "".join([vocab.id_to_token[y[i]] for i in fix_indices])
            output_lines.append(f"FIX: {noisy_chars} → {target_chars}")
        output_lines.append("")

    # 2. Замена окончаний
    output_lines.append("--- ТИП 2: ЗАМЕНА ОКОНЧАНИЙ ---")
    attempts = 0
    found = 0
    while found < 3 and attempts < 50:
        attempts += 1
        line = dataset.lines[random.randrange(len(dataset.lines))]
        words = line.split()
        if len(words) < 3:
            continue

        result = dataset.force_ending_swap(words[: dataset.max_words])
        if result:
            words_noisy, word_idx, orig_word, noisy_word = result
            found += 1

            ids_clean = encode_with_ins(words[: dataset.max_words], vocab)
            ids_noisy = encode_with_ins(words_noisy, vocab)

            output_lines.append(f"NOISY FULL:  {pretty_full(ids_noisy)}")
            output_lines.append(f"TARGET FULL: {pretty_full(ids_clean)}")
            output_lines.append(f"ЗАМЕНА: {noisy_word} → {orig_word}")
            output_lines.append("")

    # 3. Лишняя пунктуация
    output_lines.append("--- ТИП 3: ЛИШНЯЯ ПУНКТУАЦИЯ ---")
    for _ in range(3):
        line = dataset.lines[random.randrange(len(dataset.lines))]
        words = line.split()[: dataset.max_words]
        ids = encode_with_ins(words, vocab)

        x, y = dataset.force_extra_punct(ids)

        output_lines.append(f"NOISY FULL:  {pretty_full(x)}")
        output_lines.append(f"TARGET FULL: {pretty_full(ids)}")

        fix_indices = [i for i, yi in enumerate(y) if yi != -100]
        if fix_indices:
            noisy_chars = "".join([vocab.id_to_token[x[i]] for i in fix_indices])
            target_chars = "".join([vocab.id_to_token[y[i]] for i in fix_indices])
            output_lines.append(f"FIX: {noisy_chars} → {target_chars}")
        output_lines.append("")

    # 4. Дефис-запятая в конце слова
    output_lines.append("--- ТИП 4: ДЕФИС-ЗАПЯТАЯ (-,) ---")
    for _ in range(3):
        line = dataset.lines[random.randrange(len(dataset.lines))]
        words = line.split()[: dataset.max_words]
        ids = encode_with_ins(words, vocab)

        x, y = dataset.force_hyphen_comma(ids)

        output_lines.append(f"NOISY FULL:  {pretty_full(x)}")
        output_lines.append(f"TARGET FULL: {pretty_full(ids)}")

        fix_indices = [i for i, yi in enumerate(y) if yi != -100]
        if fix_indices:
            noisy_chars = "".join([vocab.id_to_token[x[i]] for i in fix_indices])
            target_chars = "".join([vocab.id_to_token[ids[i]] for i in fix_indices])
            output_lines.append(f"FIX: -, → {target_chars}")
        output_lines.append("")

    # 5. Запятая в начале слова
    output_lines.append("--- ТИП 5: ЗАПЯТАЯ В НАЧАЛЕ СЛОВА ---")
    for _ in range(3):
        line = dataset.lines[random.randrange(len(dataset.lines))]
        words = line.split()[: dataset.max_words]
        ids = encode_with_ins(words, vocab)

        x, y = dataset.force_comma_prefix(ids)

        output_lines.append(f"NOISY FULL:  {pretty_full(x)}")
        output_lines.append(f"TARGET FULL: {pretty_full(ids)}")

        fix_indices = [i for i, yi in enumerate(y) if yi != -100]
        if fix_indices:
            target_char = vocab.id_to_token[y[fix_indices[0]]]
            output_lines.append(f"FIX: , → {target_char}")
        output_lines.append("")

    # 6. Повторы окончаний (отключено с <INS>)
    output_lines.append("--- ТИП 6: ПОВТОРЫ ОКОНЧАНИЙ (ОТКЛЮЧЕНО) ---")
    output_lines.append("Сложно реализовать с <INS>, пока отключено")
    output_lines.append("")

    # 7. Одиночный дефис-разрыв с использованием <INS>
    output_lines.append("--- ТИП 7: ОДИНОЧНЫЙ ДЕФИС-РАЗРЫВ (с <INS>) ---")
    attempts = 0
    found = 0
    while found < 3 and attempts < 50:
        attempts += 1
        line = dataset.lines[random.randrange(len(dataset.lines))]
        words = line.split()[: dataset.max_words]
        if len(words) < 2:
            continue
        ids = encode_with_ins(words, vocab)

        x, y = dataset.force_single_hyphen(ids)

        fix_indices = [i for i, yi in enumerate(y) if yi != -100]
        if not fix_indices:
            continue

        found += 1
        output_lines.append(f"NOISY FULL:  {pretty_full(x)}")
        output_lines.append(f"TARGET FULL: {pretty_full(ids)}")

        # Показываем что происходит
        ins_filled = [i for i in fix_indices if x[i] == vocab.ins]
        if ins_filled:
            filled_char = vocab.id_to_token[y[ins_filled[0]]]
            output_lines.append(f"FIX: - → восстановлен, · → {filled_char} (вставка)")
        else:
            output_lines.append(f"FIX: - → восстановлен")
        output_lines.append("")

    aug_file = save_dir / "all_augmentation_types.txt"
    with open(aug_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\nСохранены примеры ВСЕХ 7 типов аугментаций в {aug_file}")


def train(config):
    set_seed(config["seed"])
    device = config["device"]

    os.makedirs(config["save_dir"], exist_ok=True)
    logger = TxtLogger(Path(config["save_dir"]) / "train.log")

    vocab = CharVocab(config["charset_path"])

    dataset = CharOCRDenoiseDataset(
        text_path=config["text_path"],
        pairs_csv_path=config["pairs_csv"],
        vocab=vocab,
        words_path=config.get("words_path"),
        max_words=config["max_words"],
        noise_prob=config.get("noise_prob", 0.15),
        p_ending_swap=config.get("p_ending_swap", 0.03),
        p_extra_punct=config.get("p_extra_punct", 0.02),
        p_hyphen_comma=config.get("p_hyphen_comma", 0.015),
        p_comma_prefix=config.get("p_comma_prefix", 0.01),
        p_repeat_ending=config.get("p_repeat_ending", 0.01),
        p_single_hyphen=config.get("p_single_hyphen", 0.02),
    )

    if dataset.top_endings:
        endings_path = Path(config["save_dir"]) / "top_endings.json"
        with open(endings_path, "w", encoding="utf-8") as f:
            json.dump(dataset.top_endings, f, ensure_ascii=False, indent=2)
        logger.log(f"Сохранено {len(dataset.top_endings)} окончаний в {endings_path}")

    # Сохранить примеры ВСЕХ типов аугментаций принудительно
    save_all_augmentation_types(dataset, vocab, Path(config["save_dir"]))

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_denoise(b, vocab.pad),
    )

    model = CharTransformerMLM(
        vocab_size=len(vocab.token_to_id),
        emb_size=config["emb_size"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        ffn_size=config["ffn_size"],
        dropout=config["dropout"],
        pad_idx=vocab.pad,
        eow_idx=vocab.eow,
        ins_idx=vocab.ins,
        space_idx=vocab.token_to_id.get(" ", 6),
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

    ce_loss = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    best_acc = 0.0
    start_epoch = 0

    if config["resume"]:
        ckpt = torch.load(config["resume"], map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        best_acc = ckpt["best_acc"]
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, config["epochs"]):
        progress = epoch / max(1, config["epochs"] - 1)
        dataset.p_real = config["p_real_start"] + progress * (
            config["p_real_end"] - config["p_real_start"]
        )

        model.train()

        run_loss = 0.0
        run_acc = 0.0

        pbar = tqdm(loader, desc=f"epoch {epoch}", ncols=100)

        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            logits, _ = model(x, y)

            # Loss вычисляем на чистых logits (copy-bias уже применён в модели)
            loss = ce_loss(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            # Для accuracy применяем маски чтобы получить правильные предсказания
            eow = vocab.eow
            ins = vocab.ins

            with torch.no_grad():
                masked_logits = logits.clone()

                # <INS> никогда не должен быть выходным символом
                masked_logits[..., ins] -= 1e9

                # At non-EOW/INS positions where y == -100, penalize EOW token
                copy_mask = (x != eow) & (x != ins) & (y == -100)
                masked_logits[:, :, eow] -= copy_mask.float() * 1e9

                # At EOW positions, penalize all non-EOW tokens
                eow_positions = x == eow
                penalty = torch.zeros_like(masked_logits)
                penalty[:, :, :eow] = eow_positions.unsqueeze(-1).float() * 1e9
                penalty[:, :, eow + 1 :] = eow_positions.unsqueeze(-1).float() * 1e9
                masked_logits = masked_logits - penalty

                acc = mlm_accuracy(masked_logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()

            run_loss += loss.item()
            run_acc += acc.item()

            pbar.set_postfix(
                loss=f"{run_loss/(pbar.n+1):.3f}",
                acc=f"{run_acc/(pbar.n+1):.3f}",
            )

        ep_loss = run_loss / len(loader)
        ep_acc = run_acc / len(loader)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        logger.log(
            f"epoch {epoch} | loss={ep_loss:.4f} acc={ep_acc:.3f} lr={current_lr:.2e} p_real={dataset.p_real:.2f}"
        )

        if (epoch + 1) % config["eval_every_epochs"] == 0:
            inspect_denoise_predictions(model, vocab, dataset, device)
            inspect_word_embeddings(model, vocab, dataset, device)

        save_dir = Path(config["save_dir"])

        if ep_acc > best_acc:
            best_acc = ep_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": config,
                },
                save_dir / "best.pt",
            )

            torch.save(
                model.state_dict(),
                save_dir / "best_weights.pt",
            )

            with open(save_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            logger.log(f"Saved BEST checkpoint and weights (acc={best_acc:.3f})")

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "best_acc": best_acc,
                "config": config,
            },
            save_dir / "last.pt",
        )


if __name__ == "__main__":
    train(CONFIG)
