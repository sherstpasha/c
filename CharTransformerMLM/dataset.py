import csv
import random
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset


# ============================================================
# Endings (optional)
# ============================================================


def extract_top_endings(
    words_path: str,
    min_len: int = 2,
    max_len: int = 5,
    min_freq: int = 30,
    top_k: int = 120,
) -> List[str]:
    counter = Counter()
    with open(words_path, encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if len(w) < min_len + 1:
                continue
            for L in range(min_len, min(max_len + 1, len(w))):
                end = w[-L:]
                # только буквенные окончания (для дореформенной орфографии это нормально)
                if end.isalpha():
                    counter[end] += 1
    endings = [e for e, c in counter.items() if c >= min_freq]
    endings.sort(key=lambda e: counter[e], reverse=True)
    return endings[:top_k]


# ============================================================
# Levenshtein -> edit ops
# ============================================================


def levenshtein_ops(src: List[int], tgt: List[int], edit_vocab) -> List[int]:
    """
    src, tgt: char ids (already encoded by CharVocab; digits -> <DIGIT>, word boundaries -> <EOW>)
    returns: op ids, length == len(src)
    """
    n, m = len(src), len(tgt)
    if n == 0:
        return []

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
        back[i][0] = "DEL"
    for j in range(m + 1):
        dp[0][j] = j
        back[0][j] = "INS"
    back[0][0] = None

    for i in range(1, n + 1):
        si = src[i - 1]
        for j in range(1, m + 1):
            tj = tgt[j - 1]
            if si == tj:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "COPY"
            else:
                choices = [
                    (dp[i - 1][j] + 1, "DEL"),
                    (dp[i][j - 1] + 1, "INS"),
                    (dp[i - 1][j - 1] + 1, "REP"),
                ]
                dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])

    ops = []
    i, j = n, m

    char_vocab = edit_vocab.char_vocab
    digit_id = char_vocab.token_to_id.get("<DIGIT>")

    while i > 0:
        op = back[i][j]

        if op == "COPY":
            ops.append(edit_vocab.COPY)
            i -= 1
            j -= 1

        elif op == "REP":
            tgt_id = tgt[j - 1]
            ch = char_vocab.id_to_token[tgt_id]

            # Запрещаем правки цифр, <EOW>, и всего, чего нет в EditVocab
            rep_key = f"REPLACE_{ch}"
            if (
                tgt_id == digit_id
                or ch == "<EOW>"
                or rep_key not in edit_vocab.op_to_id
            ):
                ops.append(edit_vocab.COPY)
            else:
                ops.append(edit_vocab.op_to_id[rep_key])

            i -= 1
            j -= 1

        elif op == "DEL":
            ops.append(edit_vocab.DELETE)
            i -= 1

        elif op == "INS":
            tgt_id = tgt[j - 1]
            ch = char_vocab.id_to_token[tgt_id]

            ins_key = f"INSERT_{ch}"
            # цифры и <EOW> не вставляем, и не вставляем то, чего нет в EditVocab
            if (
                tgt_id == digit_id
                or ch == "<EOW>"
                or ins_key not in edit_vocab.op_to_id
            ):
                j -= 1
                continue
            else:
                ops.append(edit_vocab.op_to_id[ins_key])
                j -= 1

        else:
            # на всякий случай (не должно случаться)
            ops.append(edit_vocab.COPY)
            i -= 1

    ops.reverse()
    # длина ops может стать меньше n (из-за пропущенных INS по <EOW>/<DIGIT>), добиваем COPY
    if len(ops) < n:
        ops = [edit_vocab.COPY] * (n - len(ops)) + ops
    return ops[:n]


# ============================================================
# Dataset
# ============================================================


class CharOCREditDataset(Dataset):
    """
    Два источника:
    1) clean text -> аугментации
    2) real OCR pairs (grouped by image) -> окно с гарантированной ошибкой
    """

    def __init__(
        self,
        text_path: str,
        vocab,
        edit_vocab,
        pairs_csv_path: Optional[str] = None,
        words_path: Optional[str] = None,
        max_len: int = 128,
        max_words: int = 8,
        noise_prob: float = 0.25,
        ocr_window: Optional[int] = None,
        # probabilities (normalised inside __getitem__)
        p_real_ocr: float = 0.30,
        p_synthetic_noise: float = 0.20,
        p_ending_swap: float = 0.15,
        p_extra_punct: float = 0.10,
        p_hyphen_break: float = 0.10,
        p_comma_prefix: float = 0.10,
        p_repeat_ending: float = 0.05,
        p_repeat_beginning: float = 0.05,
        ending_swap_prob_min: float = 0.08,
        ending_swap_prob_max: float = 0.12,
        # train/val split
        split: str = "train",
        val_indices_lines: Optional[set] = None,
        val_indices_pairs: Optional[set] = None,
    ):
        self.vocab = vocab
        self.edit_vocab = edit_vocab
        self.max_len = max_len
        self.max_words = max_words
        self.noise_prob = noise_prob
        self.ocr_window = ocr_window if ocr_window is not None else max_words

        # aug probs
        self.p_real_ocr = p_real_ocr
        self.p_synthetic_noise = p_synthetic_noise
        self.p_ending_swap = p_ending_swap
        self.p_extra_punct = p_extra_punct
        self.p_hyphen_break = p_hyphen_break
        self.p_comma_prefix = p_comma_prefix
        self.p_repeat_ending = p_repeat_ending
        self.p_repeat_beginning = p_repeat_beginning
        self.ending_swap_prob_min = ending_swap_prob_min
        self.ending_swap_prob_max = ending_swap_prob_max
        
        # train/val split
        self.split = split
        self.val_indices_lines = val_indices_lines or set()
        self.val_indices_pairs = val_indices_pairs or set()

        # clean text - загружаем все, потом фильтруем
        with open(text_path, encoding="utf-8") as f:
            all_lines = [l.strip() for l in f if l.strip()]
        
        # Фильтруем по split
        if split == "train":
            self.lines = [l for i, l in enumerate(all_lines) if i not in self.val_indices_lines]
        elif split == "val":
            self.lines = [l for i, l in enumerate(all_lines) if i in self.val_indices_lines]
        else:  # "all"
            self.lines = all_lines
            
        if not self.lines:
            raise ValueError(f"text_path пустой: нет строк для split={split}")

        # endings
        self.top_endings = []
        if words_path:
            try:
                self.top_endings = extract_top_endings(words_path)
            except Exception:
                self.top_endings = []

        # real OCR pairs
        self.pairs_by_image: dict[str, List[Tuple[str, str]]] = {}
        self.error_refs: List[Tuple[str, int]] = []  # (image, idx_in_image)
        self.all_pairs_list: List[Tuple[str, str, str]] = []  # (image, src, tgt) для фильтрации по split

        if pairs_csv_path:
            self._load_pairs_csv(pairs_csv_path)

    # -----------------------------
    # CSV loading (robust)
    # -----------------------------

    def _load_pairs_csv(self, pairs_csv_path: str):
        grouped = defaultdict(list)

        with open(pairs_csv_path, encoding="utf-8") as f:
            # ждём header: image,incorrect,correct
            reader = csv.DictReader(f)
            # если header не распознался (странный csv) — fallback на обычный reader
            if reader.fieldnames is None or not {
                "image",
                "incorrect",
                "correct",
            }.issubset(set(reader.fieldnames)):
                f.seek(0)
                r = csv.reader(f)
                for row in r:
                    if len(row) < 3:
                        continue
                    # heuristics: skip header row
                    if row[0].lower() == "image" and row[1].lower() == "incorrect":
                        continue
                    img, src, tgt = row[0], row[1], row[2]
                    if self._is_valid_pair(src, tgt):
                        grouped[img].append((src, tgt))
            else:
                for row in reader:
                    img = (row.get("image") or "").strip()
                    src = (row.get("incorrect") or "").strip()
                    tgt = (row.get("correct") or "").strip()
                    if not img:
                        continue
                    if self._is_valid_pair(src, tgt):
                        grouped[img].append((src, tgt))

        # Сохраняем все пары с индексами для фильтрации
        pair_idx = 0
        for img, pairs in grouped.items():
            for src, tgt in pairs:
                self.all_pairs_list.append((img, src, tgt, pair_idx))
                pair_idx += 1
        
        # Фильтруем по split
        if self.split == "train":
            filtered_pairs = [(img, src, tgt) for img, src, tgt, idx in self.all_pairs_list if idx not in self.val_indices_pairs]
        elif self.split == "val":
            filtered_pairs = [(img, src, tgt) for img, src, tgt, idx in self.all_pairs_list if idx in self.val_indices_pairs]
        else:  # "all"
            filtered_pairs = [(img, src, tgt) for img, src, tgt, _ in self.all_pairs_list]
        
        # Перестраиваем groups после фильтрации
        grouped_filtered = defaultdict(list)
        for img, src, tgt in filtered_pairs:
            grouped_filtered[img].append((src, tgt))
        
        # чистим пустые группы и собираем error refs
        self.pairs_by_image = {img: pairs for img, pairs in grouped_filtered.items() if pairs}
        self.error_refs = []
        for img, pairs in self.pairs_by_image.items():
            for i, (s, t) in enumerate(pairs):
                if s != t:
                    self.error_refs.append((img, i))

        # если real OCR недоступен — просто отключим p_real_ocr в __getitem__ нормализацией
        # (без исключения, чтобы можно было тренить чисто на синтетике)
        # но полезно предупредить:
        if pairs_csv_path and not self.error_refs:
            print(
                f"[dataset split={self.split}] Warning: no real OCR errors found after filtering (src != tgt). Real OCR disabled."
            )

    def _is_valid_pair(self, src: str, tgt: str) -> bool:
        # пустые/NaN строки — в мусор
        if not src or not tgt:
            return False
        if src.strip().lower() == "nan" or tgt.strip().lower() == "nan":
            return False
        # иногда из pandas пролезает "None"
        if src.strip().lower() == "none" or tgt.strip().lower() == "none":
            return False

        # запрещаем угловые скобки, чтобы не ловить REPLACE_< и т.п.
        if "<" in src or ">" in src or "<" in tgt or ">" in tgt:
            return False

        # все нецифровые символы должны быть в charset
        for ch in src + tgt:
            if ch.isdigit():
                continue
            if ch not in self.vocab.token_to_id:
                return False
        return True

    # -----------------------------

    def __len__(self):
        # делаем "бесконечность" через max из источников
        real_len = (
            sum(len(v) for v in self.pairs_by_image.values())
            if self.pairs_by_image
            else 0
        )
        return max(len(self.lines), real_len) if real_len > 0 else len(self.lines)

    # ==================================================
    # Sampling
    # ==================================================

    def __getitem__(self, idx):
        augs = []

        # real OCR только если реально есть ошибки
        if self.error_refs:
            augs.append(("real_ocr", self.p_real_ocr))

        augs.extend(
            [
                ("synthetic_noise", self.p_synthetic_noise),
                ("ending_swap", self.p_ending_swap),
                ("extra_punct", self.p_extra_punct),
                ("hyphen_break", self.p_hyphen_break),
                ("comma_prefix", self.p_comma_prefix),
                ("repeat_ending", self.p_repeat_ending),
                ("repeat_beginning", self.p_repeat_beginning),
            ]
        )

        total = sum(p for _, p in augs)
        if total <= 0:
            return self.make_synthetic_noise()

        r = random.random() * total
        s = 0.0
        choice = augs[-1][0]
        for name, p in augs:
            s += p
            if r <= s:
                choice = name
                break

        if choice == "real_ocr":
            return self.make_real_ocr()
        if choice == "ending_swap":
            return self.make_swap_ending()
        if choice == "extra_punct":
            return self.make_extra_punct()
        if choice == "hyphen_break":
            return self.make_hyphen_break()
        if choice == "comma_prefix":
            return self.make_comma_prefix()
        if choice == "repeat_ending":
            return self.make_repeat_ending()
        if choice == "repeat_beginning":
            return self.make_repeat_beginning()
        return self.make_synthetic_noise()

    # ==================================================
    # REAL OCR WINDOW (inside same image)
    # ==================================================

    def make_real_ocr(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Берём одну ошибочную пару (src!=tgt) как anchor,
        и добираем контекст слева/справа В ТОЙ ЖЕ image-группе.
        Контекст можно брать любой (и correct, и error),
        но anchor гарантирует "внутри каждого примера есть ошибка".
        """
        img, err_i = random.choice(self.error_refs)
        pairs = self.pairs_by_image[img]
        k = self.ocr_window

        # ставим ошибку не всегда в центр
        left = random.randint(0, k - 1)
        start = max(0, err_i - left)
        end = min(len(pairs), start + k)
        start = max(0, end - k)

        # гарантируем, что err_i попал
        if not (start <= err_i < end):
            start = max(0, min(err_i, len(pairs) - k))
            end = min(len(pairs), start + k)

        src_ids: List[int] = []
        tgt_ids: List[int] = []
        for i in range(start, end):
            s, t = pairs[i]
            src_ids.extend(self.vocab.encode(s))
            src_ids.append(self.vocab.eow)
            tgt_ids.extend(self.vocab.encode(t))
            tgt_ids.append(self.vocab.eow)

        return self._finalize(src_ids, tgt_ids)

    # ==================================================
    # SYNTHETIC BASE
    # ==================================================

    def _sample_clean_ids(self) -> List[int]:
        line = random.choice(self.lines)
        words = line.split()[: self.max_words]
        ids: List[int] = []
        for w in words:
            ids.extend(self.vocab.encode(w))
            ids.append(self.vocab.eow)
        return ids[: self.max_len]

    def make_synthetic_noise(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt = self._sample_clean_ids()
        src = tgt.copy()

        digit_id = self.vocab.digit
        for i in range(len(src)):
            if src[i] == self.vocab.eow:
                continue
            if src[i] == digit_id:
                continue
            if random.random() < self.noise_prob:
                src[i] = random.choice(self.vocab.mlm_replace_ids)

        return self._finalize(src, tgt)

    # ==================================================
    # AUGS (you can tune later)
    # ==================================================

    def make_extra_punct(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt = self._sample_clean_ids()
        src = tgt.copy()
        punct_ids = [
            self.vocab.token_to_id.get(p)
            for p in [",", ".", ":", ";", "!", "?"]
            if p in self.vocab.token_to_id
        ]
        if not punct_ids:
            return self.make_synthetic_noise()

        # заменяем 1 символ пунктуацией
        for i in range(len(src)):
            if src[i] == self.vocab.eow:
                continue
            if src[i] == self.vocab.digit:
                continue
            src[i] = random.choice(punct_ids)
            break

        return self._finalize(src, tgt)

    def make_comma_prefix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt = self._sample_clean_ids()
        src = tgt.copy()
        comma_id = self.vocab.token_to_id.get(",")
        if comma_id is None:
            return self.make_synthetic_noise()

        # вставляем запятую после <EOW> (в начало слова)
        for i in range(len(src) - 1):
            if src[i] == self.vocab.eow and src[i + 1] != self.vocab.eow:
                src.insert(i + 1, comma_id)
                break

        return self._finalize(src, tgt)

    def make_hyphen_break(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt = self._sample_clean_ids()
        src = tgt.copy()
        hy_id = self.vocab.token_to_id.get("-")
        if hy_id is None:
            return self.make_synthetic_noise()

        # вставляем дефис перед <EOW> в конце слова
        # word...X <EOW> -> word...X - <EOW>
        for i in range(1, len(src)):
            if src[i] == self.vocab.eow and src[i - 1] not in (
                self.vocab.eow,
                self.vocab.digit,
            ):
                src.insert(i, hy_id)
                break

        return self._finalize(src, tgt)

    def _split_words(self, ids: List[int]) -> List[List[int]]:
        words, cur = [], []
        for tid in ids:
            if tid == self.vocab.eow:
                if cur:
                    words.append(cur)
                cur = []
            else:
                cur.append(tid)
        return words

    def make_repeat_ending(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt = self._sample_clean_ids()
        words = self._split_words(tgt)
        if not words:
            return self._finalize(tgt, tgt)

        w = random.choice(words)
        if len(w) < 2:
            return self._finalize(tgt, tgt)

        tail = w[-2:]
        src = tgt.copy()

        # найдём позицию конца выбранного слова
        pos = 0
        for ww in words:
            pos += len(ww)
            if ww is w:
                break
            pos += 1  # EOW

        # вставим хвост (иногда с пробелом)
        if random.random() < 0.5 and " " in self.vocab.token_to_id:
            src[pos:pos] = [self.vocab.token_to_id[" "]] + tail
        else:
            src[pos:pos] = tail

        return self._finalize(src, tgt)

    def make_repeat_beginning(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt = self._sample_clean_ids()
        words = self._split_words(tgt)
        if not words:
            return self._finalize(tgt, tgt)

        w = random.choice(words)
        if len(w) < 2:
            return self._finalize(tgt, tgt)

        head = w[:2]
        src = tgt.copy()

        # найдём начало выбранного слова
        pos = 0
        for ww in words:
            if ww is w:
                break
            pos += len(ww) + 1  # +EOW

        if random.random() < 0.5 and " " in self.vocab.token_to_id:
            src[pos:pos] = head + [self.vocab.token_to_id[" "]]
        else:
            src[pos:pos] = head

        return self._finalize(src, tgt)

    def make_swap_ending(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tgt = self._sample_clean_ids()
        if not self.top_endings:
            return self._finalize(tgt, tgt)

        src = tgt.copy()
        words = self._split_words(tgt)
        if len(words) < 3:
            return self._finalize(tgt, tgt)

        # вероятность внутри аугментации (как ты и хотел)
        p = random.uniform(self.ending_swap_prob_min, self.ending_swap_prob_max)

        for wi, w in enumerate(words):
            if wi == 0 or wi == len(words) - 1:
                continue
            if random.random() > p:
                continue

            word_str = "".join(self.vocab.id_to_token[c] for c in w)
            for end in self.top_endings:
                if word_str.endswith(end) and len(word_str) > len(end) + 1:
                    same_len = [
                        e for e in self.top_endings if len(e) == len(end) and e != end
                    ]
                    if not same_len:
                        break
                    new_end = random.choice(same_len)
                    new_word = word_str[: -len(end)] + new_end

                    # посчитаем позиции в src (по словам)
                    start_pos = sum(len(words[j]) + 1 for j in range(wi))  # +EOWs
                    end_pos = start_pos + len(w)
                    src = (
                        src[:start_pos]
                        + self.vocab.encode(new_word)
                        + [self.vocab.eow]
                        + src[end_pos + 1 :]
                    )
                    return self._finalize(src, tgt)

        return self._finalize(tgt, tgt)

    # ==================================================
    # Finalize -> ops
    # ==================================================

    def _finalize(
        self, src_ids: List[int], tgt_ids: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_ids = src_ids[: self.max_len]
        tgt_ids = tgt_ids[: self.max_len * 2]

        ops = levenshtein_ops(src_ids, tgt_ids, self.edit_vocab)
        ops = ops[: len(src_ids)]

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(ops, dtype=torch.long),
        )
