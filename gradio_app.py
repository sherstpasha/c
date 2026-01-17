"""
Gradio интерфейс для тестирования OCR-денойзинг модели
"""

import re
import torch
import gradio as gr
from pathlib import Path

from CharTransformerMLM.model import CharTransformerMLM
from CharTransformerMLM.vocab import CharVocab


# Пути по умолчанию
DEFAULT_CHECKPOINT = "checkpoints/best.pt"
DEFAULT_CHARSET = "data/charset.txt"
DEFAULT_WORDS = "data/all_words.txt"


def preprocess_hyphen_breaks(text: str) -> tuple[str, list[tuple[str, str]]]:
    """
    Препроцессинг для артефактов переносов строк:
    - "слово-," → "слово"  (убираем -,)
    - ",слово" → "слово"   (убираем , в начале)
    - "слово-" → "слово"   (убираем - в конце если следующее слово с маленькой)

    Возвращает: (обработанный текст, список замен для статистики)
    """
    changes = []
    words = text.split()
    result_words = []

    for i, word in enumerate(words):
        new_word = word

        # Убираем -,  в конце
        if new_word.endswith("-,"):
            new_word = new_word[:-2]
            changes.append((word, new_word))
        elif new_word.endswith(",-"):
            new_word = new_word[:-2]
            changes.append((word, new_word))

        # Убираем , в начале (если это артефакт переноса)
        if new_word.startswith(",") and len(new_word) > 1 and new_word[1].isalpha():
            new_word = new_word[1:]
            changes.append((word, new_word))

        # Убираем одиночный - в конце если следующее слово начинается с маленькой буквы
        if new_word.endswith("-") and i + 1 < len(words):
            next_word = words[i + 1]
            # Убираем , из начала следующего для проверки
            next_clean = next_word.lstrip(",")
            if next_clean and next_clean[0].islower():
                new_word = new_word[:-1]
                changes.append((word, new_word))

        result_words.append(new_word)

    return " ".join(result_words), changes


class OCRDenoiser:
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

        # Загрузка модели
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Проверяем формат чекпоинта
        if isinstance(ckpt, dict) and "model" in ckpt:
            # Полный чекпоинт с конфигом
            state_dict = ckpt["model"]
            config = ckpt.get("config", {})
        else:
            # Только веса (OrderedDict)
            state_dict = ckpt
            config = {}  # Используем дефолтные значения

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

        print(f"Модель загружена: {checkpoint_path}")
        print(f"Словарь: {len(self.word_dict)} слов")

    def process_window(self, text: str) -> tuple[str, list[tuple[int, int, str, str]]]:
        """
        Обработать окно текста.
        Возвращает: (исправленный текст, список изменений [(word_idx, char_idx, было, стало)])
        """
        words = text.split()
        if not words:
            return text, []

        # Кодируем: слово + <INS> + <EOW> для каждого
        ids = []
        word_boundaries = []

        for word in words:
            start = len(ids)
            word_ids = self.vocab.encode(word)
            ids.extend(word_ids)
            ids.append(self.vocab.ins)  # <INS> перед <EOW>
            ids.append(self.vocab.eow)
            word_boundaries.append((start, len(ids) - 1))

        if not ids:
            return text, []

        # Прогоняем через модель
        x = torch.tensor([ids], device=self.device)
        y_dummy = torch.full_like(x, -100)

        with torch.no_grad():
            logits, _ = self.model(x, y_dummy)

            # Применяем маски
            eow = self.vocab.eow
            ins = self.vocab.ins

            # Запрещаем <EOW> на не-EOW позициях
            mask = (x != eow) & (x != ins) & (y_dummy == -100)
            logits[..., eow] -= mask * 1e9

            # Разрешаем только <EOW> на EOW позициях
            eow_positions = x == eow
            non_eow_mask = torch.ones_like(logits, dtype=torch.bool)
            non_eow_mask[..., eow] = False
            logits[eow_positions.unsqueeze(-1).expand_as(logits) & non_eow_mask] -= 1e9

            # <INS> никогда не должен быть выходным символом
            logits[..., ins] -= 1e9

            preds = logits.argmax(dim=-1)[0].tolist()

        # Собираем исправленные слова
        changes = []
        result_words = []
        space_id = self.vocab.token_to_id.get(" ", 6)

        for word_idx, (start, end) in enumerate(word_boundaries):
            orig_word = words[word_idx]
            orig_ids = self.vocab.encode(orig_word)
            input_ids = ids[start : end + 1]  # включая <INS> и <EOW>
            pred_ids_full = preds[start : end + 1]

            # Обрабатываем предсказания с учётом <INS>
            # <INS> на входе → если предсказан пробел, это "ничего" (пропускаем)
            # <INS> на входе → если предсказан символ, это вставка
            pred_word_chars = []
            for inp_id, pred_id in zip(input_ids, pred_ids_full):
                if inp_id == self.vocab.eow:
                    # EOW позиция - пропускаем
                    continue
                elif inp_id == self.vocab.ins:
                    # <INS> позиция - если предсказан пробел, пропускаем (ничего не вставляем)
                    # если предсказан символ - вставляем его
                    if pred_id != space_id and pred_id != self.vocab.ins:
                        pred_word_chars.append(self.vocab.id_to_token[pred_id])
                else:
                    # Обычная позиция
                    pred_word_chars.append(self.vocab.id_to_token[pred_id])

            pred_word = "".join(pred_word_chars)

            # Сравниваем изменения (только для обычных позиций)
            pred_idx = 0
            for i, oi in enumerate(orig_ids):
                if pred_idx < len(pred_word_chars):
                    pred_char = pred_word_chars[pred_idx]
                    orig_char = self.vocab.id_to_token[oi]
                    if orig_char != pred_char:
                        changes.append((word_idx, i, orig_char, pred_char))
                pred_idx += 1

            result_words.append(pred_word)

        return " ".join(result_words), changes

    def process_text(
        self,
        text: str,
        window_size: int = 4,
        overlap: int = 1,
        iterations: int = 1,
        check_dictionary: bool = False,
    ) -> tuple[str, str, str]:
        """
        Обработать текст скользящим окном.
        Возвращает: (оригинал_html, результат_html, статистика)
        """
        original_words = text.split()
        if not original_words:
            return text, text, "Пустой текст"

        result_words = list(original_words)
        all_changes = []  # (word_idx, char_idx, old, new)
        actual_iterations = 0

        # Повторные итерации
        for iteration in range(iterations):
            actual_iterations = iteration + 1
            iteration_changes = []
            pos = 0
            attempts = 0
            max_attempts = len(result_words) * 2

            while pos < len(result_words) and attempts < max_attempts:
                attempts += 1

                end = min(pos + window_size, len(result_words))
                window_words = result_words[pos:end]
                window_text = " ".join(window_words)

                corrected_text, changes = self.process_window(window_text)
                corrected_words = corrected_text.split()

                if len(corrected_words) != len(window_words):
                    pos += 1
                    continue

                for word_idx, char_idx, old_char, new_char in changes:
                    global_word_idx = pos + word_idx
                    new_word = corrected_words[word_idx]

                    if check_dictionary and self.word_dict:
                        if new_word.lower().strip(".,;:!?\"'()") not in self.word_dict:
                            continue

                    result_words[global_word_idx] = new_word
                    iteration_changes.append(
                        (global_word_idx, char_idx, old_char, new_char)
                    )

                pos += window_size - overlap
                if pos >= len(result_words):
                    break

            all_changes.extend(iteration_changes)

            # Если в этой итерации не было изменений - выходим
            if not iteration_changes:
                break

        # Формируем HTML для оригинала (красная подсветка - что было)
        orig_html_parts = []
        changed_word_indices = set(c[0] for c in all_changes)

        for i, word in enumerate(original_words):
            if i in changed_word_indices:
                word_changes = [(c[1], c[2], c[3]) for c in all_changes if c[0] == i]
                highlighted = self._highlight_word_original(word, word_changes)
                orig_html_parts.append(highlighted)
            else:
                orig_html_parts.append(word)

        orig_html = " ".join(orig_html_parts)

        # Формируем HTML для результата (зелёная подсветка - что стало)
        result_html_parts = []
        for i, word in enumerate(result_words):
            if i in changed_word_indices:
                word_changes = [(c[1], c[2], c[3]) for c in all_changes if c[0] == i]
                highlighted = self._highlight_word_result(word, word_changes)
                result_html_parts.append(highlighted)
            else:
                result_html_parts.append(word)

        result_html = " ".join(result_html_parts)

        # Статистика
        stats = f"Обработано слов: {len(original_words)}\n"
        stats += f"Итераций: {actual_iterations}\n"
        stats += f"Изменений: {len(all_changes)}\n"
        if all_changes:
            stats += "\nЗамены:\n"
            for word_idx, char_idx, old_char, new_char in all_changes[:30]:
                stats += f"  '{old_char}' → '{new_char}' (слово #{word_idx})\n"
            if len(all_changes) > 30:
                stats += f"  ... и ещё {len(all_changes) - 30}\n"

        return orig_html, result_html, stats

    def _highlight_word_original(self, word: str, changes: list) -> str:
        """Подсветить в оригинале что было заменено (красным)"""
        if not changes:
            return word

        # changes: [(char_idx, old, new), ...]
        change_map = {c[0]: c[1] for c in changes}  # char_idx -> old_char

        result = []
        for i, char in enumerate(word):
            if i in change_map:
                result.append(
                    f'<span style="background-color: #FFB6C1; font-weight: bold; text-decoration: line-through;">{char}</span>'
                )
            else:
                result.append(char)

        return "".join(result)

    def _highlight_word_result(self, word: str, changes: list) -> str:
        """Подсветить в результате что стало (зелёным)"""
        if not changes:
            return word

        changed_positions = {c[0] for c in changes}

        result = []
        for i, char in enumerate(word):
            if i in changed_positions:
                result.append(
                    f'<span style="background-color: #90EE90; font-weight: bold;">{char}</span>'
                )
            else:
                result.append(char)

        return "".join(result)


# Глобальный экземпляр
denoiser = None


def load_model():
    global denoiser
    try:
        denoiser = OCRDenoiser(DEFAULT_CHECKPOINT, DEFAULT_CHARSET, DEFAULT_WORDS)
        return f"✅ Модель загружена!\nУстройство: {denoiser.device}\nСловарь: {len(denoiser.word_dict)} слов"
    except Exception as e:
        return f"❌ Ошибка загрузки: {str(e)}"


def process(text, window_size, overlap, iterations, check_dict, fix_hyphens):
    if denoiser is None:
        return "Модель не загружена!", "", ""

    try:
        preprocess_stats = ""

        # Препроцессинг переносов
        if fix_hyphens:
            text, hyphen_changes = preprocess_hyphen_breaks(text)
            if hyphen_changes:
                preprocess_stats = (
                    f"Препроцессинг переносов: {len(hyphen_changes)} замен\n"
                )
                for old, new in hyphen_changes[:10]:
                    preprocess_stats += f"  '{old}' → '{new}'\n"
                if len(hyphen_changes) > 10:
                    preprocess_stats += f"  ... и ещё {len(hyphen_changes) - 10}\n"
                preprocess_stats += "\n"

        orig_html, result_html, stats = denoiser.process_text(
            text,
            window_size=int(window_size),
            overlap=int(overlap),
            iterations=int(iterations),
            check_dictionary=check_dict,
        )
        return orig_html, result_html, preprocess_stats + stats
    except Exception as e:
        return f"Ошибка: {str(e)}", "", ""


def create_app():
    # Загружаем модель при старте
    status = load_model()
    print(status)

    with gr.Blocks(title="OCR Denoiser") as app:
        gr.Markdown("# 🔧 OCR Denoising Model Tester")

        # Статус модели
        gr.Markdown(f"**Статус:** {status}")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Настройки")
                window_size = gr.Slider(
                    minimum=1, maximum=10, value=7, step=1, label="Размер окна (слов)"
                )
                overlap = gr.Slider(
                    minimum=0, maximum=5, value=2, step=1, label="Пересечение окон"
                )
                iterations = gr.Slider(
                    minimum=1, maximum=5, value=1, step=1, label="Повторные итерации"
                )
                check_dict = gr.Checkbox(
                    label="🔍 Проверять слова по словарю", value=False
                )
                fix_hyphens = gr.Checkbox(
                    label="✂️ Исправлять артефакты переносов (-,  ,слово)", value=True
                )

        gr.Markdown("---")

        input_text = gr.Textbox(
            label="📝 Входной текст (с ошибками OCR)",
            lines=5,
            placeholder="Вставьте текст с ошибками OCR...",
        )

        process_btn = gr.Button("🚀 Исправить", variant="primary", size="lg")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### ❌ Оригинал (красным — что заменено)")
                orig_output = gr.HTML(elem_classes=["output-html"])
            with gr.Column():
                gr.Markdown("### ✅ Результат (зелёным — исправления)")
                result_output = gr.HTML(elem_classes=["output-html"])

        with gr.Row():
            stats_output = gr.Textbox(label="📊 Статистика", lines=10)

        # События
        process_btn.click(
            process,
            inputs=[
                input_text,
                window_size,
                overlap,
                iterations,
                check_dict,
                fix_hyphens,
            ],
            outputs=[orig_output, result_output, stats_output],
        )

        # Примеры
        gr.Markdown("### 📋 Примеры")
        gr.Examples(
            examples=[
                ["Директоръ обсужЭаетъ теплѮцу въ саду"],
                ["Баянчикъ вскрикнулъ вскочилъ ХуPѵо"],
                ["фицер-, ,скихъ чиновъ было много"],
                ["занимается ается кресть-, евъ мазуровъ"],
                ["сло- во разорвано переносомъ строки"],
                ["развивая- ясь понемногу"],
                ["году въ Губерніи нахxдилозь много людей и животныхъ"],
                ["стороны. Незнакоіецъ въ темЩомалиновомГ платьѣ"],
            ],
            inputs=input_text,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False, css=".output-html { font-size: 16px; line-height: 1.8; }")
