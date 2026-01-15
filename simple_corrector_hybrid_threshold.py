"""Простой корректор на базе CharLM с использованием эмбеддингов.

ГИБРИДНАЯ ВЕРСИЯ: Комбинация косинусного расстояния и расстояния Левенштейна
С контекстуальными эмбеддингами (alpha=0.7) + фильтрация слов + порог замены
"""

import json
import re
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from tqdm import tqdm
from CharLM.model import CharTransformerMLM
from CharLM.utils import encode_str


def should_skip(word: str) -> bool:
    """
    Проверить, нужно ли пропустить слово (не исправлять).

    Пропускаем если:
    - Длина < 4 символов
    - Содержит цифры или спецсимволы (*)
    - Нет кириллических букв

    Args:
        word: Слово для проверки

    Returns:
        True если слово нужно пропустить
    """
    # Пропускаем короткие слова
    if len(word) < 4:
        return True

    # Пропускаем слова с цифрами или спецсимволами
    if re.search(r"[0-9*]", word):
        return True

    # Если слово в нижнем регистре - не пропускаем (могут быть ошибки)
    if word.lower() == word and word.isupper() is False:
        return False

    # Пропускаем если нет кириллицы
    if not re.search(r"[а-яА-ЯёЁ]", word):
        return True

    return False


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Вычислить расстояние Левенштейна между двумя строками.

    Args:
        s1: Первая строка
        s2: Вторая строка

    Returns:
        Расстояние Левенштейна
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 вместо j так как previous_row и current_row имеют +1 элемент
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalize_levenshtein(s1: str, s2: str) -> float:
    """
    Нормализованное расстояние Левенштейна (от 0 до 1).

    0 = идентичные строки
    1 = максимально различные

    Args:
        s1: Первая строка
        s2: Вторая строка

    Returns:
        Нормализованное расстояние (0-1)
    """
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return dist / max_len


class SimpleCorrector:
    """
    Корректор орфографии на основе векторных представлений слов.

    1. Загружает модель и создает эмбеддинги для всех слов из словаря
    2. Для входного слова находит топ-K ближайших по косинусному расстоянию
    3. Возвращает топ-K кандидатов и делает замену на самый близкий
    """

    def __init__(
        self,
        model_path: str,
        config_path: str,
        vocab_path: str,
        lexicon_path: str,
        device: str = "cuda",
    ):
        """
        Args:
            model_path: Путь к .pt файлу модели
            config_path: Путь к config.json
            vocab_path: Путь к vocab.json (словарь символов)
            lexicon_path: Путь к all_words.txt (список всех слов)
            device: "cuda" или "cpu"
        """
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        print(f"Используется устройство: {self.device}")

        # Загрузка конфигурации
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Загрузка словаря символов
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            # vocab.json это список символов, создаем словари
            self.i2c = {i: c for i, c in enumerate(vocab)}
            self.c2i = {c: i for i, c in enumerate(vocab)}

        # Инициализация модели
        self.model = CharTransformerMLM(
            vocab_size=len(self.c2i),
            emb_size=self.config["emb_size"],
            max_len=self.config["max_len"],
            n_layers=self.config["n_layers"],
            n_heads=self.config["n_heads"],
            ffn_size=self.config["ffn_size"],
            dropout=0.0,  # Для inference
            pad_idx=self.c2i["<PAD>"],
        )

        # Загрузка весов модели
        print(f"Загрузка модели из {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Загрузка словаря слов
        print(f"Загрузка словаря слов из {lexicon_path}...")
        with open(lexicon_path, "r", encoding="utf-8") as f:
            self.words = [line.strip() for line in f if line.strip()]
        print(f"Загружено {len(self.words)} слов")

        # FAISS индекс и эмбеддинги
        self.embeddings = None
        self.faiss_index = None

    def encode_word(self, word: str) -> torch.Tensor:
        """Закодировать слово в тензор индексов."""
        ids = encode_str(word, self.c2i, self.config["max_len"])
        return torch.tensor(ids, dtype=torch.long)

    def get_word_embedding_with_context(
        self,
        word: str,
        left: str | None = None,
        right: str | None = None,
    ) -> torch.Tensor:
        """
        Контекстный эмбеддинг слова.
        Формирует строку: "left word right"
        и возвращает embedding ТОЛЬКО позиции word.
        """
        with torch.no_grad():
            parts = []
            if left:
                parts.append(left)
            parts.append(word)
            if right:
                parts.append(right)

            seq = " ".join(parts)

            ids = encode_str(seq, self.c2i, self.config["max_len"])
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)

            B, T = x.shape
            pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

            h = self.model.emb(x) + self.model.pos(pos_ids)
            h = self.model.encoder(h, src_key_padding_mask=(x == self.model.pad_idx))

            # вычисляем span слова
            prefix_len = 0
            if left:
                prefix_len = len(left) + 1  # пробел

            start = prefix_len
            end = min(start + len(word), self.config["max_len"])

            if start >= end:
                return self.get_word_embedding(word)

            word_h = h[0, start:end]  # [L, D]
            emb = word_h.mean(dim=0)

            return emb

    def get_word_embedding(self, word: str) -> torch.Tensor:
        """
        Получить эмбеддинг слова (mean pooling по всем позициям).

        Returns:
            Tensor размера [emb_size]
        """
        with torch.no_grad():
            # Кодируем слово
            x = self.encode_word(word).unsqueeze(0).to(self.device)  # [1, max_len]

            # Получаем hidden states из encoder
            B, T = x.shape
            pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            h = self.model.emb(x) + self.model.pos(pos_ids)  # [1, max_len, emb_size]
            h = self.model.encoder(
                h, src_key_padding_mask=(x == self.model.pad_idx)
            )  # [1, max_len, emb_size]

            # Mean pooling (только по непустым позициям)
            mask = (x != self.model.pad_idx).float().unsqueeze(-1)  # [1, max_len, 1]
            word_emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                min=1
            )  # [1, emb_size]

            return word_emb.squeeze(0)  # [emb_size]

    def build_index(self, batch_size: int = 512):
        """
        Создать индекс эмбеддингов для всех слов из словаря.

        Args:
            batch_size: Размер батча для обработки
        """
        print(
            f"Создание эмбеддингов для {len(self.words)} слов (batch_size={batch_size})..."
        )

        all_embeddings = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(self.words), batch_size), desc="Создание индекса"
            ):
                batch_words = self.words[i : i + batch_size]

                # Кодируем батч
                batch_ids = [
                    encode_str(w, self.c2i, self.config["max_len"]) for w in batch_words
                ]
                x = torch.tensor(batch_ids, dtype=torch.long).to(
                    self.device
                )  # [B, max_len]

                # Получаем hidden states
                B, T = x.shape
                pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                h = self.model.emb(x) + self.model.pos(
                    pos_ids
                )  # [B, max_len, emb_size]
                h = self.model.encoder(
                    h, src_key_padding_mask=(x == self.model.pad_idx)
                )  # [B, max_len, emb_size]

                # Mean pooling для каждого слова в батче
                mask = (
                    (x != self.model.pad_idx).float().unsqueeze(-1)
                )  # [B, max_len, 1]
                batch_emb = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                    min=1
                )  # [B, emb_size]

                all_embeddings.append(batch_emb.cpu())

        # Объединяем все батчи
        self.embeddings = torch.cat(all_embeddings, dim=0)  # [num_words, emb_size]

        # Нормализуем для косинусного расстояния
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)

        print(f"Индекс создан: {self.embeddings.shape}")

        # Создаем FAISS индекс
        self._build_faiss_index()

    def _build_faiss_index(self):
        """Создать FAISS индекс из эмбеддингов."""
        print("Создание FAISS индекса...")
        emb_dim = self.embeddings.shape[1]

        # Используем HNSW индекс для быстрого поиска
        # M=32 - количество связей на уровень (влияет на качество и скорость)
        # efConstruction=40 - параметр построения индекса
        self.faiss_index = faiss.IndexHNSWFlat(emb_dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.faiss_index.hnsw.efSearch = (
            64  # параметр поиска (больше = точнее но медленнее)
        )

        print(f"Добавление {self.embeddings.shape[0]} векторов в HNSW индекс...")
        self.faiss_index.add(self.embeddings.numpy())

        print(f"FAISS HNSW индекс создан: {self.faiss_index.ntotal} векторов")

    def save_index(self, save_dir: str):
        """
        Сохранить индекс (эмбеддинги и FAISS индекс) на диск.

        Args:
            save_dir: Директория для сохранения
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        # Сохраняем эмбеддинги
        embeddings_path = save_dir / "embeddings.pt"
        torch.save(self.embeddings, embeddings_path)
        print(f"Эмбеддинги сохранены: {embeddings_path}")

        # Сохраняем FAISS индекс
        faiss_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.faiss_index, str(faiss_path))
        print(f"FAISS индекс сохранен: {faiss_path}")

        # Сохраняем список слов
        words_path = save_dir / "words_list.txt"
        with open(words_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.words))
        print(f"Список слов сохранен: {words_path}")

    def load_index(self, save_dir: str):
        """
        Загрузить сохраненный индекс с диска.

        Args:
            save_dir: Директория с сохраненным индексом

        Returns:
            True если загрузка успешна, False если файлы не найдены
        """
        save_dir = Path(save_dir)

        embeddings_path = save_dir / "embeddings.pt"
        faiss_path = save_dir / "faiss_index.bin"
        words_path = save_dir / "words_list.txt"

        # Проверяем наличие всех файлов
        if not (
            embeddings_path.exists() and faiss_path.exists() and words_path.exists()
        ):
            return False

        print(f"Загрузка индекса из {save_dir}...")

        # Загружаем эмбеддинги
        self.embeddings = torch.load(embeddings_path, map_location="cpu")
        print(f"Эмбеддинги загружены: {self.embeddings.shape}")

        # Загружаем FAISS индекс
        self.faiss_index = faiss.read_index(str(faiss_path))
        print(f"FAISS индекс загружен: {self.faiss_index.ntotal} векторов")

        # Загружаем список слов
        with open(words_path, "r", encoding="utf-8") as f:
            self.words = [line.strip() for line in f if line.strip()]
        print(f"Список слов загружен: {len(self.words)} слов")

        return True

    def find_closest(self, word: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Найти top_k ближайших слов по косинусному расстоянию (используя FAISS).

        Args:
            word: Входное слово
            top_k: Количество ближайших кандидатов

        Returns:
            Список кортежей (слово, similarity_score)
        """
        if self.faiss_index is None:
            raise ValueError(
                "Индекс не создан! Вызовите build_index() или load_index() сначала."
            )

        # Получаем эмбеддинг входного слова
        query_emb = self.get_word_embedding(word).cpu()  # [emb_size]
        query_emb = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)  # [1, emb_size]

        # Поиск в FAISS
        top_scores, top_indices = self.faiss_index.search(
            query_emb.numpy(), min(top_k, len(self.words))
        )

        results = []
        for idx, score in zip(top_indices[0], top_scores[0]):
            results.append((self.words[idx], float(score)))

        return results

    def find_closest_hybrid(
        self, word: str, top_k: int = 5, alpha: float = 0.5, retrieval_size: int = 100
    ) -> list[tuple[str, float]]:
        """
        Найти top_k ближайших слов используя гибридный подход:
        комбинация косинусного сходства и расстояния Левенштейна.

        Итоговый score = alpha * cosine_sim + (1 - alpha) * (1 - normalized_levenshtein)

        Args:
            word: Входное слово
            top_k: Количество ближайших кандидатов для возврата
            alpha: Вес косинусного сходства (0-1).
                   alpha=1.0 - только косинусное
                   alpha=0.0 - только Левенштейн
            retrieval_size: Сколько кандидатов извлечь из FAISS для ре-ранжирования

        Returns:
            Список кортежей (слово, hybrid_score)
        """
        if self.faiss_index is None:
            raise ValueError(
                "Индекс не создан! Вызовите build_index() или load_index() сначала."
            )

        # Шаг 1: Получаем больше кандидатов из FAISS (только косинусное)
        query_emb = self.get_word_embedding(word).cpu()
        query_emb = F.normalize(query_emb.unsqueeze(0), p=2, dim=1)

        faiss_k = min(retrieval_size, len(self.words))
        cosine_scores, indices = self.faiss_index.search(query_emb.numpy(), faiss_k)

        # Шаг 2: Вычисляем расстояния Левенштейна для всех кандидатов
        hybrid_results = []

        for idx, cos_sim in zip(indices[0], cosine_scores[0]):
            candidate_word = self.words[idx]

            # Нормализованное расстояние Левенштейна (0-1)
            lev_dist = normalize_levenshtein(word, candidate_word)

            # Левенштейн similarity (инвертируем: 1 = идентичные, 0 = очень разные)
            lev_sim = 1.0 - lev_dist

            # Гибридный score
            hybrid_score = alpha * float(cos_sim) + (1.0 - alpha) * lev_sim

            hybrid_results.append((candidate_word, hybrid_score))

        # Шаг 3: Сортируем по гибридному score и берём топ-k
        hybrid_results.sort(key=lambda x: x[1], reverse=True)

        return hybrid_results[:top_k]

    def correct(self, word: str, top_k: int = 5, verbose: bool = True) -> str:
        """
        Исправить слово, найдя ближайшее в словаре.

        Args:
            word: Входное слово для коррекции
            top_k: Сколько кандидатов показать
            verbose: Печатать ли топ-K кандидатов

        Returns:
            Самое близкое слово (замена)
        """
        candidates = self.find_closest(word, top_k)

        if verbose:
            print(f"\nВходное слово: '{word}'")
            print(f"Топ-{top_k} кандидатов:")
            for i, (candidate, score) in enumerate(candidates, 1):
                print(f"  {i}. {candidate} (score: {score:.4f})")

        # Самое близкое слово
        best_word = candidates[0][0]

        if verbose:
            print(f"\nЗамена: '{word}' -> '{best_word}'")

        return best_word


def extract_words(text: str, min_length: int = 4) -> list[tuple[str, int, int]]:
    """
    Извлечь слова из текста (только буквы, кириллица/латиница).

    Args:
        text: Входной текст
        min_length: Минимальная длина слова для извлечения

    Returns:
        Список кортежей (слово, start_pos, end_pos) для слов длиннее min_length символов
    """
    # Находим все последовательности букв (кириллица и латиница)
    pattern = r"[а-яА-ЯёЁa-zA-ZІіѢѣѲѳѵ]+"
    words = []

    for match in re.finditer(pattern, text):
        word = match.group()
        if len(word) >= min_length:
            words.append((word, match.start(), match.end()))

    return words


def correct_text(
    corrector: SimpleCorrector, text: str, min_word_length: int = 4
) -> str:
    """
    Исправить текст: заменить слова на их топ-1 кандидатов из корректора.

    Args:
        corrector: Экземпляр SimpleCorrector
        text: Исходный текст
        min_word_length: Минимальная длина слова для коррекции

    Returns:
        Исправленный текст
    """
    # Извлекаем слова с позициями
    words_with_pos = extract_words(text, min_length=min_word_length)

    if not words_with_pos:
        return text

    # Создаем список замен (в обратном порядке, чтобы позиции не сбивались)
    replacements = []
    for word, start, end in reversed(words_with_pos):
        # Получаем топ-1 кандидата
        candidates = corrector.find_closest(word, top_k=1)
        if candidates:
            replacement = candidates[0][0]
            replacements.append((start, end, word, replacement))

    # Применяем замены
    corrected_text = text
    for start, end, word, replacement in replacements:
        corrected_text = corrected_text[:start] + replacement + corrected_text[end:]

    return corrected_text


def evaluate_on_pairs(
    corrector: SimpleCorrector,
    csv_path: str,
    min_word_length: int = 4,
    max_samples: int = None,
    top_k: int = 15,
    alpha: float = 0.7,
    threshold: float = 0.7,
    retrieval_size: int = 100,
    verbose: bool = True,
    save_results: bool = False,
    results_path: str = "correction_results.csv",
) -> dict:
    """
    Оценить корректор на датасете pairs.csv.

    Для каждой строки:
    1. Берет incorrect текст
    2. Исправляет все буквенные слова (длиннее min_word_length)
    3. Сравнивает исправленный текст с correct

    Метрики:
    - total_edits: все сделанные правки (когда топ-1 != слово)
    - useful_edits_@k: правки где правильный ответ есть в топ-k
    - precision@k: useful_edits@k / total_edits

    Args:
        corrector: Экземпляр SimpleCorrector
        csv_path: Путь к pairs.csv
        min_word_length: Минимальная длина слова для коррекции
        max_samples: Максимальное количество примеров (для отладки)
        top_k: Количество топ кандидатов для извлечения
        alpha: Вес косинусного сходства в гибридном подходе (0-1)
        retrieval_size: Размер кандидатов для ре-ранжирования
        verbose: Печатать ли примеры

    Returns:
        Словарь с метриками
    """
    print(f"\nОценка корректора на {csv_path}...")
    print(f"Гибридный режим: alpha={alpha:.2f}, threshold={threshold:.2f}")
    print(f"Косинус={alpha:.0%}, Левенштейн={1-alpha:.0%}")

    # Загружаем данные
    df = pd.read_csv(csv_path)
    if max_samples:
        df = df.head(max_samples)

    print(f"Загружено {len(df)} примеров")

    total_edits = 0
    total_skipped = 0  # Счетчик пропущенных слов

    # Для каждого k считаем сколько раз правильный ответ был в топ-k
    useful_edits_at_k = {k: 0 for k in [1, 5, 10, 15, 1000]}

    examples = []
    all_corrections = []  # Для сохранения всех результатов

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Оценка"):
        incorrect = str(row["incorrect"])
        correct = str(row["correct"])

        # Извлекаем слова из incorrect (только буквы)
        words_with_pos = extract_words(incorrect, min_length=min_word_length)

        if not words_with_pos:
            continue

        # Применяем корректор к каждому слову
        for word, start, end in words_with_pos:
            # Проверяем, нужно ли пропустить слово
            if should_skip(word):
                total_skipped += 1
                continue

            # Получаем топ-K кандидатов (гибридный поиск)
            candidates = corrector.find_closest_hybrid(
                word, top_k=max(top_k, 1000), alpha=alpha, retrieval_size=retrieval_size
            )
            if not candidates:
                continue

            top1_replacement = candidates[0][0]
            top1_score = candidates[0][1]

            # Применяем порог: если score ниже threshold, не меняем слово
            if top1_score < threshold:
                top1_replacement = word  # Оставляем как есть

            # Сохраняем информацию о коррекции
            correction_info = {
                "original_word": word,
                "corrected_word": top1_replacement,
                "top1_score": top1_score,
                "changed": top1_replacement != word,
                "useful": top1_replacement in correct,
                "incorrect_text": incorrect,
                "correct_text": correct,
                "top5_candidates": [c[0] for c in candidates[:5]],
            }
            all_corrections.append(correction_info)

            # Если топ-1 изменил слово - это правка
            if top1_replacement != word:
                total_edits += 1

                # Проверяем для каждого k: есть ли правильный ответ в топ-k?
                for k in useful_edits_at_k.keys():
                    if k > len(candidates):
                        continue

                    # Берем топ-k кандидатов
                    topk_words = [c[0] for c in candidates[:k]]

                    # Проверяем, есть ли хотя бы один из топ-k в correct
                    if any(cand in correct for cand in topk_words):
                        useful_edits_at_k[k] += 1

                # Сохраняем примеры
                if len(examples) < 20:
                    # Проверяем топ-1
                    useful_top1 = top1_replacement in correct
                    examples.append(
                        {
                            "original": word,
                            "top1": top1_replacement,
                            "top_k_candidates": [
                                c[0] for c in candidates[:5]
                            ],  # первые 5
                            "incorrect_full": incorrect,
                            "correct_full": correct,
                            "useful_top1": useful_top1,
                        }
                    )

    # Считаем метрики
    results = {
        "total_edits": total_edits,
        "total_skipped": total_skipped,
        "alpha": alpha,
        "threshold": threshold,
    }

    for k in useful_edits_at_k.keys():
        results[f"useful_edits@{k}"] = useful_edits_at_k[k]
        results[f"precision@{k}"] = (
            useful_edits_at_k[k] / total_edits if total_edits > 0 else 0.0
        )

    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ ОЦЕНКИ (alpha={alpha:.2f}, threshold={threshold:.2f})")
    print(f"{'='*60}")
    print(f"Пропущено слов (фильтрация): {total_skipped}")
    print(f"Всего правок (когда топ-1 != слово): {total_edits}")
    print()
    for k in [1, 5, 10, 15, 1000]:
        useful = results[f"useful_edits@{k}"]
        precision = results[f"precision@{k}"]
        print(f"Precision@{k:2d}: {precision:.2%}  (полезных: {useful}/{total_edits})")
    print(f"{'='*60}")

    if verbose and examples:
        print(f"\nПримеры правок (первые {len(examples)}):")
        for i, ex in enumerate(examples, 1):
            status = "✓" if ex["useful_top1"] else "✗"
            print(f"\n{i}. [{status}] '{ex['original']}' -> топ-1: '{ex['top1']}'")
            print(f"   Топ-5 кандидатов: {ex['top_k_candidates']}")

    # Сохраняем результаты если нужно
    if save_results and all_corrections:
        # Разделяем на категории
        changed_corrections = [c for c in all_corrections if c["changed"]]
        unchanged_corrections = [c for c in all_corrections if not c["changed"]]

        # Ошибки: изменено, но неправильно (changed=True, useful=False)
        errors = [c for c in changed_corrections if not c["useful"]]

        # Правильные исправления: изменено и правильно (changed=True, useful=True)
        correct_corrections = [c for c in changed_corrections if c["useful"]]

        # Сохраняем все результаты (сначала исправленные, потом неизмененные)
        results_df = pd.DataFrame(changed_corrections + unchanged_corrections)
        results_df.to_csv(results_path, index=False, encoding="utf-8")
        print(f"\n✓ Все результаты сохранены в {results_path}")
        print(f"  - Исправлено слов: {len(changed_corrections)}")
        print(f"  - Не изменено слов: {len(unchanged_corrections)}")

        # Сохраняем только ошибки
        if errors:
            errors_path = results_path.replace(".csv", "_errors.csv")
            errors_df = pd.DataFrame(errors)
            errors_df.to_csv(errors_path, index=False, encoding="utf-8")
            print(f"\n✓ Ошибки сохранены в {errors_path}")
            print(f"  - Неправильных исправлений: {len(errors)}")

        # Сохраняем только правильные исправления
        if correct_corrections:
            correct_path = results_path.replace(".csv", "_correct.csv")
            correct_df = pd.DataFrame(correct_corrections)
            correct_df.to_csv(correct_path, index=False, encoding="utf-8")
            print(f"\n✓ Правильные исправления сохранены в {correct_path}")
            print(f"  - Правильных исправлений: {len(correct_corrections)}")

    return results


def grid_search_threshold(
    corrector: SimpleCorrector,
    csv_path: str,
    threshold_values: list[float] = None,
    alpha: float = 0.7,
    min_word_length: int = 4,
    max_samples: int = None,
    retrieval_size: int = 100,
) -> pd.DataFrame:
    """
    Перебор различных значений threshold для поиска оптимального порога замены.

    Args:
        corrector: Экземпляр SimpleCorrector
        csv_path: Путь к pairs.csv
        threshold_values: Список значений threshold для тестирования
        alpha: Фиксированное значение alpha
        min_word_length: Минимальная длина слова
        max_samples: Максимальное количество примеров
        retrieval_size: Размер кандидатов для ре-ранжирования

    Returns:
        DataFrame с результатами для каждого threshold
    """
    if threshold_values is None:
        # По умолчанию от 0.5 до 0.95 с шагом 0.1
        threshold_values = [round(0.5 + i * 0.05, 2) for i in range(10)]  # 0.5-0.95

    results_list = []

    for threshold in threshold_values:
        print(f"\n{'='*70}")
        print(f"Тестирование threshold={threshold:.2f}")
        print(f"{'='*70}")

        results = evaluate_on_pairs(
            corrector=corrector,
            csv_path=csv_path,
            min_word_length=min_word_length,
            max_samples=max_samples,
            top_k=15,
            alpha=alpha,
            threshold=threshold,
            retrieval_size=retrieval_size,
            verbose=False,
            save_results=False,
        )

        # Добавляем результаты в список
        row = {
            "threshold": threshold,
            "alpha": alpha,
            "total_edits": results["total_edits"],
            "total_skipped": results["total_skipped"],
        }

        # Добавляем precision для каждого k
        for k in [1, 5, 10, 15]:
            row[f"precision@{k}"] = results[f"precision@{k}"]

        results_list.append(row)

    # Создаем DataFrame
    df_results = pd.DataFrame(results_list)

    # Выводим сводку
    print(f"\n{'='*70}")
    print("СВОДКА GRID SEARCH ПО THRESHOLD")
    print(f"{'='*70}")
    print(df_results.to_string(index=False))
    print(f"{'='*70}")

    # Находим лучший threshold по precision@1
    best_row = df_results.loc[df_results["precision@1"].idxmax()]
    print(f"\nЛучший threshold: {best_row['threshold']:.2f}")
    print(f"Precision@1: {best_row['precision@1']:.2%}")

    return df_results


def main():
    """Тестирование гибридного корректора с порогом и фильтрацией."""

    # Пути к файлам
    model_path = "exp1/model_a.pt"
    config_path = "exp1/config.json"
    vocab_path = "exp1/vocab.json"
    lexicon_path = "all_words.txt"
    index_dir = "exp1/index_hybrid"  # Отдельная директория для гибридной версии

    # Создание корректора
    corrector = SimpleCorrector(
        model_path=model_path,
        config_path=config_path,
        vocab_path=vocab_path,
        lexicon_path=lexicon_path,
        device="cuda",
    )

    # Попытка загрузить индекс, если не получилось - создаем
    if not corrector.load_index(index_dir):
        print("Индекс не найден, создаем новый...")
        corrector.build_index(batch_size=512)
        corrector.save_index(index_dir)
    else:
        print("Индекс успешно загружен!")

    # Тестовое слово
    print("\n" + "=" * 70)
    print("ТЕСТ НА ОДНОМ СЛОВЕ (alpha=0.7, threshold=0.9)")
    print("=" * 70)
    test_word = "Августь"
    candidates = corrector.find_closest_hybrid(test_word, top_k=5, alpha=0.7)
    print(f"\nВходное слово: '{test_word}'")
    print(f"Топ-5 кандидатов (alpha=0.7):")
    for i, (candidate, score) in enumerate(candidates, 1):
        print(f"  {i}. {candidate} (score: {score:.4f})")

    # Оценка на pairs.csv с порогом 0.9
    print("\n" + "=" * 70)
    print("ОЦЕНКА НА ДАТАСЕТЕ (alpha=0.7, threshold=0.9)")
    print("=" * 70)
    results = evaluate_on_pairs(
        corrector=corrector,
        csv_path="pairs.csv",
        min_word_length=4,
        max_samples=None,
        top_k=3,
        alpha=0.7,
        threshold=0.9,  # Фиксированный порог
        retrieval_size=3,
        verbose=True,
        save_results=True,
        results_path="hybrid_threshold_results.csv",
    )


if __name__ == "__main__":
    main()
