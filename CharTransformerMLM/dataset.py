import csv
import random
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


def is_good_pair(src: str, tgt: str) -> bool:
    # Требуем одинаковую длину для корректного выравнивания
    if len(src) != len(tgt):
        return False

    if not any(ch.isalpha() for ch in src + tgt):
        return False

    digit_ratio = sum(ch.isdigit() for ch in src + tgt) / max(1, len(src) + len(tgt))
    if digit_ratio > 0.3:
        return False

    return True


def extract_top_endings(
    words_path: str, top_n: int = 50, min_len: int = 2, max_len: int = 4
) -> List[str]:
    """Извлечь топ окончаний из словаря"""
    endings_counter = Counter()

    with open(words_path, encoding="utf-8") as f:
        for line in f:
            word = line.strip().lower()
            if len(word) >= 4:
                for end_len in range(min_len, min(max_len + 1, len(word))):
                    endings_counter[word[-end_len:]] += 1

    # Берём только частые окончания (>100 раз)
    top_endings = [e for e, cnt in endings_counter.most_common(top_n * 3) if cnt > 100]
    return top_endings[:top_n]


class CharOCRDenoiseDataset(Dataset):
    def __init__(
        self,
        text_path: str,
        pairs_csv_path: str,
        vocab,
        words_path: str = None,
        max_len: int = 128,
        max_words: int = 3,
        noise_prob: float = 0.15,
        p_real: float = 0.4,
        p_ending_swap: float = 0.03,
        p_extra_punct: float = 0.02,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.max_words = max_words
        self.noise_prob = noise_prob
        self.p_real = p_real
        self.ocr_window = 2
        self.p_ending_swap = p_ending_swap
        self.p_extra_punct = p_extra_punct

        with open(text_path, encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if l.strip()]

        self.pairs: List[Tuple[str, str]] = []
        with open(pairs_csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                _, src, tgt = row[0], row[1].strip(), row[2].strip()
                if src and tgt and is_good_pair(src, tgt):
                    self.pairs.append((src, tgt))

        if not self.pairs:
            raise ValueError("Нет валидных OCR-пар после фильтрации")

        self.replace_ids = vocab.mlm_replace_ids

        # Загрузить топ окончаний для замены
        self.top_endings = []
        if words_path:
            try:
                self.top_endings = extract_top_endings(words_path)
            except Exception:
                pass

        # Пунктуация для случайной вставки
        self.extra_punct = [",", ".", ";", "-"]
        self.punct_ids = [
            vocab.token_to_id.get(p) for p in self.extra_punct if p in vocab.token_to_id
        ]

    def __len__(self):
        return max(len(self.lines), len(self.pairs))

    def _apply_synthetic_noise(self, ids: List[int]):
        x = ids.copy()
        y = [-100] * len(ids)

        for i in range(len(ids)):
            if ids[i] == self.vocab.eow:
                continue
            if random.random() < self.noise_prob:
                y[i] = ids[i]
                x[i] = random.choice(self.replace_ids)

        return x, y

    def _real_pair_window(self):
        i = random.randint(0, len(self.pairs) - self.ocr_window)

        xs = []
        ys = []

        for j in range(self.ocr_window):
            src, tgt = self.pairs[i + j]

            x = self.vocab.encode(src)
            y = self.vocab.encode(tgt)

            # Пары уже отфильтрованы по длине в is_good_pair
            if len(x) == 0:
                continue

            # y = -100 где символы совпадают (как в синтетическом шуме)
            for xi, yi in zip(x, y):
                xs.append(xi)
                ys.append(yi if xi != yi else -100)

            xs.append(self.vocab.eow)
            ys.append(-100)  # EOW не нужно исправлять

        xs = xs[: self.max_len]
        ys = ys[: self.max_len]

        return xs, ys

    def _maybe_swap_ending(self, word: str) -> Tuple[str, str]:
        """Иногда заменяет окончание слова на другое из топ-окончаний"""
        if not self.top_endings or len(word) < 5:
            return word, word

        if random.random() > self.p_ending_swap:
            return word, word

        # Найти подходящее окончание в слове
        for ending in self.top_endings:
            if word.endswith(ending) and len(word) > len(ending) + 1:
                # Заменить на другое окончание той же длины
                same_len_endings = [
                    e for e in self.top_endings if len(e) == len(ending) and e != ending
                ]
                if same_len_endings:
                    new_ending = random.choice(same_len_endings)
                    noisy = word[: -len(ending)] + new_ending
                    return noisy, word
                break

        return word, word

    def _maybe_add_punct(
        self, ids: List[int], targets: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Иногда заменяет символ на пунктуацию (лишняя запятая и т.д.)"""
        if not self.punct_ids:
            return ids, targets

        x = ids.copy()
        y = targets.copy()

        for i in range(1, len(x) - 1):  # Не первый и не последний
            if x[i] != self.vocab.eow and random.random() < self.p_extra_punct:
                y[i] = ids[i]  # Оригинальный символ - target
                x[i] = random.choice(self.punct_ids)  # Заменяем на пунктуацию

        return x, y

    def __getitem__(self, idx):
        if random.random() < self.p_real:
            x, y = self._real_pair_window()
            return (
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
            )

        line = self.lines[idx % len(self.lines)]
        words = line.split()

        if not words:
            return self.__getitem__(idx + 1)

        if len(words) > self.max_words:
            start = random.randint(0, len(words) - self.max_words)
            words = words[start : start + self.max_words]

        ids = []
        targets = []

        for i, w in enumerate(words):
            # Для средних слов иногда меняем окончание
            if 0 < i < len(words) - 1:
                noisy_w, clean_w = self._maybe_swap_ending(w)
            else:
                noisy_w, clean_w = w, w

            noisy_ids = self.vocab.encode(noisy_w)
            clean_ids = self.vocab.encode(clean_w)

            # Если длины совпадают - добавляем с учётом различий
            if len(noisy_ids) == len(clean_ids):
                for ni, ci in zip(noisy_ids, clean_ids):
                    ids.append(ni)
                    targets.append(ci if ni != ci else -100)
            else:
                # Длины разные - просто добавляем без target
                ids.extend(self.vocab.encode(w))
                targets.extend([-100] * len(self.vocab.encode(w)))

            ids.append(self.vocab.eow)
            targets.append(-100)

        ids = ids[: self.max_len]
        targets = targets[: self.max_len]

        # Применяем синтетический шум
        x, y = self._apply_synthetic_noise(ids)

        # Объединяем targets от окончаний с синтетическим шумом
        for i in range(len(y)):
            if targets[i] != -100:
                y[i] = targets[i]

        # Иногда добавляем лишнюю пунктуацию
        x, y = self._maybe_add_punct(x, y)

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
