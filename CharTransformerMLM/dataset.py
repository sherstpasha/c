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
        pairs_csv_path: str = None,
        vocab=None,
        words_path: str = None,
        max_len: int = 128,
        max_words: int = 3,
        noise_prob: float = 0.15,
        p_real: float = 0.4,
        ocr_window: int = 2,
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
        self.ocr_window = ocr_window
        self.p_ending_swap = p_ending_swap
        self.p_extra_punct = p_extra_punct
        self.p_hyphen_comma = p_hyphen_comma
        self.p_comma_prefix = p_comma_prefix
        self.p_repeat_ending = p_repeat_ending
        self.p_single_hyphen = p_single_hyphen

        with open(text_path, encoding="utf-8") as f:
            self.lines = [l.strip() for l in f if l.strip()]

        self.pairs: List[Tuple[str, str]] = []
        if pairs_csv_path:
            with open(pairs_csv_path, encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) < 3:
                        continue
                    _, src, tgt = row[0], row[1].strip(), row[2].strip()
                    if src and tgt and is_good_pair(src, tgt):
                        self.pairs.append((src, tgt))

        # Если нет пар - отключаем p_real
        if not self.pairs:
            self.p_real = 0.0

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
        if self.pairs:
            return max(len(self.lines), len(self.pairs))
        return len(self.lines)

    def _apply_synthetic_noise(self, ids: List[int]):
        x = ids.copy()
        y = [-100] * len(ids)

        for i in range(len(ids)):
            # Пропускаем специальные токены
            if ids[i] == self.vocab.eow or ids[i] == self.vocab.ins:
                continue
            if random.random() < self.noise_prob:
                y[i] = ids[i]
                x[i] = random.choice(self.replace_ids)

        return x, y

    def _real_pair_window(self):
        # Рандомим количество слов от 1 до self.ocr_window
        window_size = random.randint(1, self.ocr_window)
        i = random.randint(0, len(self.pairs) - window_size)

        xs = []
        ys = []

        for j in range(window_size):
            src, tgt = self.pairs[i + j]

            x = self.vocab.encode(src)
            y = self.vocab.encode(tgt)

            if len(x) == 0:
                continue

            for xi, yi in zip(x, y):
                xs.append(xi)
                ys.append(yi if xi != yi else -100)

            # Добавляем <INS> перед <EOW>
            # Явно учим модель выводить пробел на <INS> (= "ничего не вставлять")
            xs.append(self.vocab.ins)
            space_id = self.vocab.token_to_id.get(" ")
            ys.append(space_id if space_id else -100)

            xs.append(self.vocab.eow)
            ys.append(-100)

        xs = xs[: self.max_len]
        ys = ys[: self.max_len]

        return xs, ys

    def get_real_sample(self):
        """Получить пример из реальных OCR-пар (для логирования)"""
        if not self.pairs:
            return None

        x, y = self._real_pair_window()
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )

    def get_synthetic_sample(self):
        """Получить синтетический пример (для логирования)"""
        # Выбираем случайную строку
        line = random.choice(self.lines)
        words = line.split()

        if not words:
            return self.get_synthetic_sample()

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

            # Добавляем <INS> перед <EOW>
            ids.append(self.vocab.ins)
            space_id = self.vocab.token_to_id.get(" ")
            targets.append(space_id if space_id else -100)

            ids.append(self.vocab.eow)
            targets.append(-100)

        # Обрезаем до max_len
        ids = ids[: self.max_len]
        targets = targets[: self.max_len]

        # Применяем синтетические аугментации (как в __getitem__)
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
            # Пропускаем специальные токены
            if x[i] == self.vocab.eow or x[i] == self.vocab.ins:
                continue
            if random.random() < self.p_extra_punct:
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

        space_id = self.vocab.token_to_id.get(" ")
        ins_id = self.vocab.ins

        # Находим позиции <EOW> (структура: ...символы + <INS> + <EOW>...)
        eow_positions = [
            i for i, token_id in enumerate(ids) if token_id == self.vocab.eow
        ]

        # Дефис-запятая: слов- , → слово (- и , заменяются на пробелы)
        for eow_idx in eow_positions:
            # Позиция <INS> прямо перед <EOW>
            ins_idx = eow_idx - 1
            if ins_idx < 0 or x[ins_idx] != ins_id:
                continue

            # Позиции символов перед <INS>
            if ins_idx >= 2 and random.random() < self.p_hyphen_comma:
                char_before_ins = ins_idx - 1
                char_before_that = ins_idx - 2

                if (
                    x[char_before_ins] != self.vocab.eow
                    and x[char_before_ins] != ins_id
                    and x[char_before_that] != self.vocab.eow
                    and x[char_before_that] != ins_id
                ):
                    # Сохраняем оригинальные символы для восстановления
                    orig_char1 = x[char_before_that]
                    orig_char2 = x[char_before_ins]

                    # Заменяем на -,
                    x[char_before_that] = hyphen_id
                    x[char_before_ins] = comma_id

                    # Учим восстанавливать: - → оригинал, , → оригинал
                    y[char_before_that] = orig_char1
                    y[char_before_ins] = orig_char2

        # Запятая в начале слова: ,слово → слово
        for eow_idx in eow_positions:
            # После <EOW> идёт первый символ следующего слова
            next_char_idx = eow_idx + 1
            if (
                next_char_idx < len(x)
                and x[next_char_idx] != self.vocab.eow
                and x[next_char_idx] != ins_id
                and random.random() < self.p_comma_prefix
            ):
                orig_char = x[next_char_idx]
                x[next_char_idx] = comma_id
                y[next_char_idx] = orig_char

        # Одиночный дефис-разрыв: слов- → слово (модель использует <INS> для вставки)
        for eow_idx in eow_positions[:-1]:
            ins_idx = eow_idx - 1
            if ins_idx < 1 or x[ins_idx] != ins_id:
                continue

            char_before_ins = ins_idx - 1

            if (
                random.random() < self.p_single_hyphen
                and x[char_before_ins] != hyphen_id
                and x[char_before_ins] != self.vocab.eow
                and x[char_before_ins] != ins_id
            ):
                # Находим первый символ следующего слова
                next_word_start = eow_idx + 1
                if next_word_start < len(x) and x[next_word_start] != self.vocab.eow:
                    next_char = x[next_word_start]

                    # Заменяем последний символ на дефис
                    orig_char = x[char_before_ins]
                    x[char_before_ins] = hyphen_id
                    y[char_before_ins] = orig_char

                    # <INS> должен заполниться первым символом следующего слова
                    y[ins_idx] = next_char

                    # Первый символ следующего слова → пробел (удаляется)
                    y[next_word_start] = space_id

        return x, y

    def __getitem__(self, idx):
        if self.pairs and random.random() < self.p_real:
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

            # Добавляем <INS> перед <EOW> - резерв для вставки символа
            # Явно учим модель выводить пробел на <INS> (= "ничего не вставлять")
            ids.append(self.vocab.ins)
            space_id = self.vocab.token_to_id.get(" ")
            targets.append(space_id if space_id else -100)

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

        # Исключаем <EOW> и <INS>
        candidates = [
            i
            for i, tid in enumerate(ids)
            if tid != self.vocab.eow and tid != self.vocab.ins
        ]
        if candidates:
            pos = random.choice(candidates)
            noise_char = random.choice(list(self.vocab.token_to_id.values()))
            if noise_char != ids[pos] and noise_char not in self.vocab.special_ids:
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

        # Исключаем <EOW> и <INS>
        candidates = [
            i
            for i, tid in enumerate(ids)
            if tid != self.vocab.eow and tid != self.vocab.ins
        ]
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
        ins_id = self.vocab.ins

        if not all([hyphen_id, comma_id]):
            return x, y

        eow_positions = [i for i, tid in enumerate(ids) if tid == self.vocab.eow]

        valid_positions = []
        for eow_idx in eow_positions:
            # Структура: ...char char <INS> <EOW>
            ins_idx = eow_idx - 1
            if ins_idx < 2 or ids[ins_idx] != ins_id:
                continue
            char_before_ins = ins_idx - 1
            char_before_that = ins_idx - 2
            if (
                ids[char_before_ins] != self.vocab.eow
                and ids[char_before_ins] != ins_id
                and ids[char_before_that] != self.vocab.eow
                and ids[char_before_that] != ins_id
            ):
                valid_positions.append(eow_idx)

        if valid_positions:
            eow_idx = random.choice(valid_positions)
            ins_idx = eow_idx - 1
            char_before_ins = ins_idx - 1
            char_before_that = ins_idx - 2

            # Сохраняем оригиналы
            orig_char1 = ids[char_before_that]
            orig_char2 = ids[char_before_ins]

            # Заменяем на -,
            x[char_before_that] = hyphen_id
            x[char_before_ins] = comma_id

            # Учим восстанавливать
            y[char_before_that] = orig_char1
            y[char_before_ins] = orig_char2

        return x, y

    def force_comma_prefix(self, ids):
        """Принудительно добавить запятую в начале слова"""
        x = ids.copy()
        y = [-100] * len(ids)

        comma_id = self.vocab.token_to_id.get(",")
        ins_id = self.vocab.ins

        if not comma_id:
            return x, y

        eow_positions = [i for i, tid in enumerate(ids) if tid == self.vocab.eow]

        valid_positions = []
        for eow_idx in eow_positions:
            # После <EOW> идёт первый символ следующего слова
            next_char_idx = eow_idx + 1
            if (
                next_char_idx < len(ids)
                and ids[next_char_idx] != self.vocab.eow
                and ids[next_char_idx] != ins_id
            ):
                valid_positions.append(eow_idx)

        if valid_positions:
            eow_idx = random.choice(valid_positions)
            next_char_idx = eow_idx + 1
            orig_char = ids[next_char_idx]
            x[next_char_idx] = comma_id
            y[next_char_idx] = orig_char

        return x, y

    def force_repeat_ending(self, ids):
        """Принудительно добавить повтор окончания - УБРАНО, сложно с <INS>"""
        # Эта аугментация сложна с новой структурой, пока отключаем
        return ids.copy(), [-100] * len(ids)

    def force_single_hyphen(self, ids):
        """Принудительно добавить одиночный дефис-разрыв с использованием <INS>"""
        x = ids.copy()
        y = [-100] * len(ids)

        hyphen_id = self.vocab.token_to_id.get("-")
        space_id = self.vocab.token_to_id.get(" ")
        ins_id = self.vocab.ins

        if not hyphen_id or not space_id:
            return x, y

        eow_positions = [i for i, tid in enumerate(ids) if tid == self.vocab.eow]

        valid_positions = []
        for eow_idx in eow_positions[:-1]:
            # Структура: ...char <INS> <EOW> next_char...
            ins_idx = eow_idx - 1
            if ins_idx < 1 or ids[ins_idx] != ins_id:
                continue
            char_before_ins = ins_idx - 1

            # Проверяем что следующее слово существует
            next_word_start = eow_idx + 1
            if next_word_start >= len(ids):
                continue
            if ids[next_word_start] == self.vocab.eow or ids[next_word_start] == ins_id:
                continue

            if (
                ids[char_before_ins] != hyphen_id
                and ids[char_before_ins] != self.vocab.eow
                and ids[char_before_ins] != ins_id
            ):
                valid_positions.append(eow_idx)

        if valid_positions:
            eow_idx = random.choice(valid_positions)
            ins_idx = eow_idx - 1
            char_before_ins = ins_idx - 1
            next_word_start = eow_idx + 1

            # Сохраняем оригинальный символ
            orig_char = ids[char_before_ins]
            next_char = ids[next_word_start]

            # Заменяем последний символ на дефис
            x[char_before_ins] = hyphen_id
            y[char_before_ins] = orig_char

            # <INS> заполняется первым символом следующего слова
            y[ins_idx] = next_char

            # Первый символ следующего слова → пробел (удаляется)
            y[next_word_start] = space_id

        return x, y
