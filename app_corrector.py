import gradio as gr
import torch
import json
import os
from charlm.model import CharTransformerMLM
from charlm.utils import CharLMCorrector


def load_model(checkpoint_path, vocab_path, device="cuda"):
    with open(vocab_path, encoding="utf-8") as f:
        chars = json.load(f)
    
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    vocab_size = len(chars)
    
    model = CharTransformerMLM(vocab_size=vocab_size, emb_size=192, max_len=32, 
                               n_layers=6, n_heads=6, ffn_size=768, dropout=0.1, 
                               pad_idx=c2i["<PAD>"])
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    return model, c2i, i2c


def load_corrector(checkpoint_path, vocab_path, words_path, device):
    model, c2i, i2c = load_model(checkpoint_path, vocab_path, device)
    
    lexicon = None
    if os.path.exists(words_path):
        with open(words_path, encoding="utf-8") as f:
            lexicon = set(w.strip().lower() for w in f if w.strip())
    
    substitutions = {}
    sub_path = os.path.join(os.path.dirname(checkpoint_path), "..", "substitutions.json")
    sub_path = os.path.normpath(sub_path)
    if os.path.exists(sub_path):
        with open(sub_path, encoding="utf-8") as f:
            substitutions = json.load(f)
    
    return CharLMCorrector(
        model, c2i, i2c, device, max_len=32,
        mask_threshold=0.05, apply_threshold=0.95, max_edits=1,
        lexicon=lexicon, min_word_len=4,
        substitutions=substitutions, sub_threshold=100
    )


def highlight_differences(original, corrected):
    html = []
    i, j = 0, 0
    
    while i < len(original) or j < len(corrected):
        if i < len(original) and j < len(corrected):
            if original[i] == corrected[j]:
                html.append(original[i])
                i += 1
                j += 1
            else:
                html.append(f'<span style="background-color: #FFB6C6; text-decoration: line-through;">{original[i]}</span>')
                html.append(f'<span style="background-color: #90EE90; font-weight: bold;">{corrected[j]}</span>')
                i += 1
                j += 1
        elif i < len(original):
            html.append(f'<span style="background-color: #FFB6C6; text-decoration: line-through;">{original[i]}</span>')
            i += 1
        else:
            html.append(f'<span style="background-color: #90EE90; font-weight: bold;">{corrected[j]}</span>')
            j += 1
    
    return ''.join(html)


def correct_text(text, mask_threshold, apply_threshold, max_edits, sub_threshold):
    if not text.strip():
        return "", ""
    
    corrector.mask_threshold = mask_threshold
    corrector.apply_threshold = apply_threshold
    corrector.max_edits = max_edits
    corrector.sub_threshold = sub_threshold
    
    corrected = corrector.correct_word(text)
    highlighted = highlight_differences(text, corrected)
    
    return corrected, highlighted


device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "exp_stage_a5/checkpoints/charlm_epoch_58.pt"
vocab_path = "exp_stage_a5/vocab.json"
words_path = "data/words.txt"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint: {checkpoint_path}")
if not os.path.exists(vocab_path):
    raise FileNotFoundError(f"Vocab: {vocab_path}")

corrector = load_corrector(checkpoint_path, vocab_path, words_path, device)

with gr.Blocks(title="CharLM Corrector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔤 CharLM OCR Corrector")
    gr.Markdown("Коррекция текста на основе частот замен и языковой модели")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Входной текст",
                placeholder="Введите текст с возможными ошибками...",
                lines=5
            )
            
            with gr.Accordion("Параметры", open=False):
                mask_threshold = gr.Slider(0.0, 1.0, value=0.05, step=0.01, 
                                          label="Порог маскирования")
                apply_threshold = gr.Slider(0.0, 1.0, value=0.95, step=0.01, 
                                           label="Порог применения")
                max_edits = gr.Slider(1, 5, value=1, step=1, 
                                     label="Максимум правок")
                sub_threshold = gr.Slider(0, 500, value=100, step=10,
                                         label="Порог частоты замены")
            
            btn = gr.Button("Исправить", variant="primary", size="lg")
        
        with gr.Column():
            output_text = gr.Textbox(label="Исправленный текст", lines=5)
            highlighted = gr.HTML(label="Визуализация (🔴 удалено, 🟢 добавлено)")
    
    btn.click(
        correct_text,
        inputs=[input_text, mask_threshold, apply_threshold, max_edits, sub_threshold],
        outputs=[output_text, highlighted]
    )
    
    gr.Markdown("---")
    gr.Markdown(
        "**Параметры:**\n"
        "- **mask_threshold**: порог уверенности для кандидатов на исправление\n"
        "- **apply_threshold**: минимальная уверенность для применения исправления\n"
        "- **max_edits**: максимум исправлений в одном слове\n"
        "- **sub_threshold**: минимальная частота замены в обучающих данных\n\n"
        "🔴 Розовым с зачёркиванием — удалённые символы  \n"
        "🟢 Зелёным — новые символы"
    )

if __name__ == "__main__":
    demo.launch()
