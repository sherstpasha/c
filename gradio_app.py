"""
Gradio app for the OCR edit model (CharTransformerEdit).
"""

import difflib
import html
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import torch

from CharTransformerMLM.model import CharTransformerEdit, EditVocab
from CharTransformerMLM.vocab import CharVocab


DEFAULT_CHECKPOINT = "checkpoints_edit/best_em.pt"
DEFAULT_CHARSET = "data/charset.txt"
DEFAULT_WORDS = "data/all_words.txt"


def load_training_config() -> dict:
    try:
        from train import CONFIG as TRAIN_CONFIG

        return TRAIN_CONFIG
    except Exception:
        return {}


def preprocess_hyphen_breaks(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Normalize some common OCR hyphen/comma split artifacts.
    """
    changes: List[Tuple[str, str]] = []
    words = text.split()
    result_words: List[str] = []

    for i, word in enumerate(words):
        new_word = word

        if new_word.endswith("-,"):
            new_word = new_word[:-2]
            changes.append((word, new_word))
        elif new_word.endswith(",-"):
            new_word = new_word[:-2]
            changes.append((word, new_word))

        if new_word.startswith(",") and len(new_word) > 1 and new_word[1].isalpha():
            new_word = new_word[1:]
            changes.append((word, new_word))

        if new_word.endswith("-") and i + 1 < len(words):
            next_word = words[i + 1]
            next_clean = next_word.lstrip(",")
            if next_clean and next_clean[0].islower():
                new_word = new_word[:-1]
                changes.append((word, new_word))

        result_words.append(new_word)

    return " ".join(result_words), changes


def apply_edit_ops(vocab: CharVocab, edit_vocab: EditVocab, x_ids, op_ids) -> str:
    """
    Apply edit operations to noisy input to reconstruct target string.
    """
    out: List[str] = []

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


def _wrap_deleted(text: str) -> str:
    return (
        '<span style="background-color: #FFB6C1; '
        'font-weight: bold; text-decoration: line-through;">'
        f"{html.escape(text)}"
        "</span>"
    )


def _wrap_inserted(text: str) -> str:
    return (
        '<span style="background-color: #90EE90; font-weight: bold;">'
        f"{html.escape(text)}"
        "</span>"
    )


def diff_word(orig: str, new: str) -> Tuple[str, str, int]:
    matcher = difflib.SequenceMatcher(None, orig, new)
    orig_parts: List[str] = []
    new_parts: List[str] = []
    change_chars = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            orig_parts.append(html.escape(orig[i1:i2]))
            new_parts.append(html.escape(new[j1:j2]))
        elif tag == "replace":
            change_chars += max(i2 - i1, j2 - j1)
            orig_parts.append(_wrap_deleted(orig[i1:i2]))
            new_parts.append(_wrap_inserted(new[j1:j2]))
        elif tag == "delete":
            change_chars += i2 - i1
            orig_parts.append(_wrap_deleted(orig[i1:i2]))
        elif tag == "insert":
            change_chars += j2 - j1
            new_parts.append(_wrap_inserted(new[j1:j2]))

    return "".join(orig_parts), "".join(new_parts), change_chars


def diff_words(
    original_words: List[str], result_words: List[str]
) -> Tuple[str, str, int, int]:
    if len(original_words) != len(result_words):
        orig_html = html.escape(" ".join(original_words))
        res_html = html.escape(" ".join(result_words))
        return orig_html, res_html, 0, 0

    orig_parts: List[str] = []
    res_parts: List[str] = []
    changed_words = 0
    changed_chars = 0

    for orig_word, res_word in zip(original_words, result_words):
        if orig_word == res_word:
            orig_parts.append(html.escape(orig_word))
            res_parts.append(html.escape(res_word))
            continue
        changed_words += 1
        o_html, r_html, c_count = diff_word(orig_word, res_word)
        changed_chars += c_count
        orig_parts.append(o_html)
        res_parts.append(r_html)

    return " ".join(orig_parts), " ".join(res_parts), changed_words, changed_chars


class OCRDenoiser:
    def __init__(
        self,
        checkpoint_path: str,
        charset_path: str,
        words_path: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vocab = CharVocab(charset_path)
        self.edit_vocab = EditVocab(self.vocab)

        self.word_dict = set()
        if words_path and Path(words_path).exists():
            with open(words_path, encoding="utf-8") as f:
                for line in f:
                    word = line.strip().lower()
                    if word:
                        self.word_dict.add(word)

        self.model, self.max_len = self._load_model(checkpoint_path, config or {})

    def _load_model(self, checkpoint_path: str, config: dict):
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
            config = {**config, **ckpt.get("config", {})}
        else:
            state_dict = ckpt

        emb_size = state_dict["char_emb.weight"].shape[1]
        max_len = state_dict["pos_emb.weight"].shape[0]

        n_layers = config.get("n_layers", 6)
        n_heads = config.get("n_heads", 6)
        ffn_size = config.get("ffn_size", 768)
        dropout = config.get("dropout", 0.1)

        model = CharTransformerEdit(
            vocab_size=len(self.vocab.token_to_id),
            edit_vocab_size=self.edit_vocab.size,
            emb_size=emb_size,
            max_len=max_len,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_size=ffn_size,
            dropout=dropout,
            pad_idx=self.vocab.pad,
            eow_idx=self.vocab.eow,
        ).to(self.device)

        model.load_state_dict(state_dict)
        model.eval()

        return model, max_len

    def _encode_words(self, words: List[str]) -> List[int]:
        ids: List[int] = []
        for word in words:
            ids.extend(self.vocab.encode(word))
            ids.append(self.vocab.eow)
        return ids

    def _predict_window(self, words: List[str], ids: List[int]) -> Optional[List[str]]:
        if not ids:
            return words

        x = torch.tensor([ids], device=self.device)
        with torch.no_grad():
            logits, _ = self.model(x)

        pred_ops = logits.argmax(dim=-1)[0].tolist()
        pred_text = apply_edit_ops(self.vocab, self.edit_vocab, ids, pred_ops)
        pred_words = [w for w in pred_text.split("<EOW>") if w]

        if len(pred_words) != len(words):
            return None

        return pred_words

    def process_text(
        self,
        text: str,
        window_size: int = 4,
        overlap: int = 1,
        iterations: int = 1,
        check_dictionary: bool = False,
    ) -> Tuple[str, str, str]:
        original_words = text.split()
        if not original_words:
            return text, text, "Empty input."

        result_words = list(original_words)
        actual_iterations = 0

        for iteration in range(iterations):
            actual_iterations = iteration + 1
            iteration_changes = 0
            pos = 0
            attempts = 0
            max_attempts = len(result_words) * 2

            while pos < len(result_words) and attempts < max_attempts:
                attempts += 1
                window_end = min(pos + window_size, len(result_words))
                window_words = result_words[pos:window_end]
                ids = self._encode_words(window_words)

                while window_words and len(ids) > self.max_len:
                    window_end -= 1
                    window_words = result_words[pos:window_end]
                    ids = self._encode_words(window_words)

                if not window_words or len(ids) > self.max_len:
                    pos += 1
                    continue

                corrected_words = self._predict_window(window_words, ids)
                if corrected_words is None or len(corrected_words) != len(window_words):
                    pos += 1
                    continue

                for i, new_word in enumerate(corrected_words):
                    global_idx = pos + i
                    if check_dictionary and self.word_dict:
                        cleaned = new_word.lower().strip(".,;:!?\"'()[]{}")
                        if cleaned and cleaned not in self.word_dict:
                            continue
                    if new_word != result_words[global_idx]:
                        iteration_changes += 1
                        result_words[global_idx] = new_word

                step = max(len(window_words) - overlap, 1)
                pos += step

            if iteration_changes == 0:
                break

        orig_html, result_html, changed_words, changed_chars = diff_words(
            original_words, result_words
        )

        stats = f"Words: {len(original_words)}\n"
        stats += f"Iterations: {actual_iterations}\n"
        stats += f"Changed words: {changed_words}\n"
        stats += f"Changed chars: {changed_chars}\n"

        return orig_html, result_html, stats


denoiser = None


def load_model():
    global denoiser
    try:
        config = load_training_config()
        denoiser = OCRDenoiser(DEFAULT_CHECKPOINT, DEFAULT_CHARSET, DEFAULT_WORDS, config)
        return (
            "OK: model loaded\n"
            f"Device: {denoiser.device}\n"
            f"Dictionary size: {len(denoiser.word_dict)}\n"
            f"Max length: {denoiser.max_len}"
        )
    except Exception as e:
        return f"ERROR: failed to load model: {str(e)}"


def process(text, window_size, overlap, iterations, check_dict, fix_hyphens):
    if denoiser is None:
        return "Model is not loaded.", "", ""

    try:
        preprocess_stats = ""

        if fix_hyphens:
            text, hyphen_changes = preprocess_hyphen_breaks(text)
            if hyphen_changes:
                preprocess_stats = (
                    f"Preprocess fixes: {len(hyphen_changes)}\n"
                )
                for old, new in hyphen_changes[:10]:
                    preprocess_stats += f"  '{old}' -> '{new}'\n"
                if len(hyphen_changes) > 10:
                    preprocess_stats += (
                        f"  ... plus {len(hyphen_changes) - 10} more\n"
                    )
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
        return f"ERROR: {str(e)}", "", ""


def create_app():
    status = load_model()
    print(status)

    with gr.Blocks(
        title="OCR Denoiser (Edit Model)",
        css=".output-html { font-size: 16px; line-height: 1.8; }",
    ) as app:
        gr.Markdown("# OCR Edit Model Tester")
        gr.Markdown(f"**Model status:** {status}")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                window_size = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=7,
                    step=1,
                    label="Window size (words)",
                )
                overlap = gr.Slider(
                    minimum=0,
                    maximum=6,
                    value=2,
                    step=1,
                    label="Window overlap (words)",
                )
                iterations = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Iterations",
                )
                check_dict = gr.Checkbox(
                    label="Filter edits by dictionary", value=False
                )
                fix_hyphens = gr.Checkbox(
                    label="Preprocess hyphen/comma breaks", value=True
                )

        gr.Markdown("---")

        input_text = gr.Textbox(
            label="Input text",
            lines=5,
            placeholder="Paste OCR text here...",
        )

        process_btn = gr.Button("Process", variant="primary", size="lg")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Original (highlighted)")
                orig_output = gr.HTML(elem_classes=["output-html"])
            with gr.Column():
                gr.Markdown("### Corrected (highlighted)")
                result_output = gr.HTML(elem_classes=["output-html"])

        with gr.Row():
            stats_output = gr.Textbox(label="Stats", lines=10)

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

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
