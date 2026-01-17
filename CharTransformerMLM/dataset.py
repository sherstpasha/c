import csv
import random
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


def is_good_pair(src: str, tgt: str) -> bool:
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
        p_hyphen_comma: float = 0.015,
        p_comma_prefix: float = 0.01,
        p_repeat_ending: float = 0.01,
        p_single_hyphen: float = 0.02,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.max_words = max_words
        self.noise_prob = noise_prob
        self.p_real = p_real
        self.ocr_window = 2
        self.p_ending_swap = p_ending_swap
        self.p_extra_punct = p_extra_punct
        self.p_hyphen_comma = p_hyphen_comma
        self.p_comma_prefix = p_comma_prefix
        self.p_repeat_ending = p_repeat_ending
        self.p_single_hyphen = p_single_hyphen

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

        self.top_endings = []
        if words_path:
            try:
                self.top_endings = extract_top_endings(words_path)
            except Exception:
                pass

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

            if len(x) == 0:
                continue

            for xi, yi in zip(x, y):
                xs.append(xi)
                ys.append(yi if xi != yi else -100)

            xs.append(self.vocab.eow)
            ys.append(-100)

        xs = xs[: self.max_len]
        ys = ys[: self.max_len]

        return xs, ys

    def _maybe_swap_ending(self, word: str) -> Tuple[str, str]:
        """Иногда заменяет окончание слова на другое из топ-окончаний"""
        if not self.top_endings or len(word) < 5:
            return word, word

        if random.random() > self.p_ending_swap:
            return word, word

        for ending in self.top_endings:
            if word.endswith(ending) and len(word) > len(ending) + 1:
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
        if not self.punct_ids:
            return ids, targets

        x = ids.copy()
        y = targets.copy()

        for i in range(1, len(x) - 1):
            if x[i] != self.vocab.eow and random.random() < self.p_extra_punct:
                y[i] = ids[i]
                x[i] = random.choice(self.punct_ids)

        return x, y

    def _apply_hyphen_artifacts(
        self, ids: List[int], targets: List[int]
    ) -> Tuple[List[int], List[int]]:

        if " " not in self.vocab.token_to_id:
            return ids, targets

        space_id = self.vocab.token_to_id[" "]
        comma_id = self.vocab.token_to_id.get(",")
        hyphen_id = self.vocab.token_to_id.get("-")

        if comma_id is None or hyphen_id is None:
            return ids, targets

        x = ids.copy()
        y = targets.copy()

        eow_positions = [
            i for i, token_id in enumerate(ids) if token_id == self.vocab.eow
        ]

        for eow_idx in eow_positions:
            if eow_idx >= 3 and random.random() < self.p_hyphen_comma:
                if (
                    x[eow_idx - 1] != self.vocab.eow
                    and x[eow_idx - 2] != self.vocab.eow
                    and x[eow_idx - 3] != self.vocab.eow
                ):
                    y[eow_idx - 2] = space_id
                    y[eow_idx - 1] = space_id
                    x[eow_idx - 2] = hyphen_id
                    x[eow_idx - 1] = comma_id

            if (
                eow_idx + 1 < len(x)
                and x[eow_idx + 1] != self.vocab.eow
                and random.random() < self.p_comma_prefix
            ):
                y[eow_idx + 1] = space_id
                x[eow_idx + 1] = comma_id

        for eow_idx in eow_positions[:-1]:
            if random.random() < self.p_repeat_ending and eow_idx >= 4:
                dup_len = random.randint(2, 3)
                if eow_idx >= dup_len + 1:
                    next_word_start = eow_idx + 1
                    if next_word_start + dup_len < len(x):
                        for j in range(dup_len):
                            orig_id = x[eow_idx - dup_len + j]
                            if orig_id != self.vocab.eow and next_word_start + j < len(
                                x
                            ):
                                x[next_word_start + j] = orig_id
                                y[next_word_start + j] = space_id

        for eow_idx in eow_positions[:-1]:
            if eow_idx >= 3 and random.random() < self.p_single_hyphen:
                if (
                    x[eow_idx - 1] != hyphen_id
                    and x[eow_idx - 1] != self.vocab.eow
                    and x[eow_idx - 2] != self.vocab.eow
                ):
                    y[eow_idx - 1] = space_id
                    x[eow_idx - 1] = hyphen_id

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
            if 0 < i < len(words) - 1:
                noisy_w, clean_w = self._maybe_swap_ending(w)
            else:
                noisy_w, clean_w = w, w

            noisy_ids = self.vocab.encode(noisy_w)
            clean_ids = self.vocab.encode(clean_w)

            if len(noisy_ids) == len(clean_ids):
                for ni, ci in zip(noisy_ids, clean_ids):
                    ids.append(ni)
                    targets.append(ci if ni != ci else -100)
            else:
                ids.extend(self.vocab.encode(w))
                targets.extend([-100] * len(self.vocab.encode(w)))

            ids.append(self.vocab.eow)
            targets.append(-100)

        ids = ids[: self.max_len]
        targets = targets[: self.max_len]

        x, y = self._apply_synthetic_noise(ids)

        for i in range(len(y)):
            if targets[i] != -100:
                y[i] = targets[i]

        x, y = self._maybe_add_punct(x, y)

        x, y = self._apply_hyphen_artifacts(x, y)

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )

    def force_synthetic_noise(self, ids):
        """Принудительно применить синтетический шум"""
        x = ids.copy()
        y = [-100] * len(ids)

        candidates = [i for i, tid in enumerate(ids) if tid != self.vocab.eow]
        if candidates:
            pos = random.choice(candidates)
            noise_char = random.choice(list(self.vocab.token_to_id.values()))
            if noise_char != ids[pos]:
                x[pos] = noise_char
                y[pos] = ids[pos]

        return x, y

    def force_ending_swap(self, words):
        """Принудительно применить замену окончания"""
        if len(words) < 3 or not self.top_endings:
            return None

        # Выбираем случайное слово (не первое и не последнее)
        word_idx = random.randint(1, len(words) - 2)
        word = words[word_idx]

        if len(word) < 4:
            return None

        # Пробуем заменить окончание
        for end_len in range(3, 1, -1):
            if len(word) <= end_len:
                continue
            ending = word[-end_len:]

            # Ищем альтернативные окончания той же длины
            alternatives = [
                e for e in self.top_endings if len(e) == end_len and e != ending
            ]

            if alternatives:
                new_ending = random.choice(alternatives)
                noisy_word = word[:-end_len] + new_ending
                words_copy = words.copy()
                words_copy[word_idx] = noisy_word
                return words_copy, word_idx, word, noisy_word

    def force_extra_punct(self, ids):
        """Принудительно добавить лишнюю пунктуацию"""
        x = ids.copy()
        y = [-100] * len(ids)

        punct_ids = [
            self.vocab.token_to_id.get(p)
            for p in [",", ".", ";", ":", "!", "?"]
            if p in self.vocab.token_to_id
        ]

        candidates = [i for i, tid in enumerate(ids) if tid != self.vocab.eow]
        if candidates and punct_ids:
            pos = random.choice(candidates)
            punct_id = random.choice(punct_ids)
            y[pos] = ids[pos]
            x[pos] = punct_id

        return x, y

    def force_hyphen_comma(self, ids):
        """Принудительно добавить дефис-запятую в конце слова"""
        x = ids.copy()
        y = [-100] * len(ids)

        hyphen_id = self.vocab.token_to_id.get("-")
        comma_id = self.vocab.token_to_id.get(",")
        space_id = self.vocab.token_to_id.get(" ")

        if not all([hyphen_id, comma_id, space_id]):
            return x, y

        eow_positions = [i for i, tid in enumerate(ids) if tid == self.vocab.eow]

        valid_positions = [
            eow_idx
            for eow_idx in eow_positions
            if eow_idx >= 3
            and ids[eow_idx - 1] != self.vocab.eow
            and ids[eow_idx - 2] != self.vocab.eow
            and ids[eow_idx - 3] != self.vocab.eow
        ]

        if valid_positions:
            eow_idx = random.choice(valid_positions)
            y[eow_idx - 2] = space_id
            y[eow_idx - 1] = space_id
            x[eow_idx - 2] = hyphen_id
            x[eow_idx - 1] = comma_id

        return x, y

    def force_comma_prefix(self, ids):
        """Принудительно добавить запятую в начале слова"""
        x = ids.copy()
        y = [-100] * len(ids)

        comma_id = self.vocab.token_to_id.get(",")
        space_id = self.vocab.token_to_id.get(" ")

        if not comma_id or not space_id:
            return x, y

        eow_positions = [i for i, tid in enumerate(ids) if tid == self.vocab.eow]

        valid_positions = [
            eow_idx
            for eow_idx in eow_positions
            if eow_idx + 1 < len(ids) and ids[eow_idx + 1] != self.vocab.eow
        ]

        if valid_positions:
            eow_idx = random.choice(valid_positions)
            y[eow_idx + 1] = space_id
            x[eow_idx + 1] = comma_id

        return x, y

    def force_repeat_ending(self, ids):
        """Принудительно добавить повтор окончания"""
        x = ids.copy()
        y = [-100] * len(ids)

        space_id = self.vocab.token_to_id.get(" ")
        if not space_id:
            return x, y

        eow_positions = [i for i, tid in enumerate(ids) if tid == self.vocab.eow]

        valid_positions = [eow_idx for eow_idx in eow_positions[:-1] if eow_idx >= 4]

        if valid_positions:
            eow_idx = random.choice(valid_positions)
            dup_len = random.randint(2, 3)

            if eow_idx >= dup_len + 1:
                next_word_start = eow_idx + 1
                if next_word_start + dup_len < len(ids):
                    for j in range(dup_len):
                        orig_id = ids[eow_idx - dup_len + j]
                        if orig_id != self.vocab.eow and next_word_start + j < len(ids):
                            x[next_word_start + j] = orig_id
                            y[next_word_start + j] = space_id

        return x, y

    def force_single_hyphen(self, ids):
        """Принудительно добавить одиночный дефис-разрыв"""
        x = ids.copy()
        y = [-100] * len(ids)

        hyphen_id = self.vocab.token_to_id.get("-")
        space_id = self.vocab.token_to_id.get(" ")

        if not hyphen_id or not space_id:
            return x, y

        eow_positions = [i for i, tid in enumerate(ids) if tid == self.vocab.eow]

        valid_positions = [
            eow_idx
            for eow_idx in eow_positions[:-1]
            if eow_idx >= 3
            and ids[eow_idx - 1] != hyphen_id
            and ids[eow_idx - 1] != self.vocab.eow
            and ids[eow_idx - 2] != self.vocab.eow
        ]

        if valid_positions:
            eow_idx = random.choice(valid_positions)
            y[eow_idx - 1] = space_id
            x[eow_idx - 1] = hyphen_id

        return x, y
