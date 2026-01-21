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


def create_corrector(model, c2i, i2c, device, checkpoint_path, mask_threshold=0.01, apply_threshold=0.95, max_edits=1, sub_threshold=100):
    substitutions = {}
    sub_path = os.path.join(os.path.dirname(checkpoint_path), "..", "substitutions.json")
    sub_path = os.path.normpath(sub_path)
    if os.path.exists(sub_path):
        with open(sub_path, encoding="utf-8") as f:
            substitutions = json.load(f)
    
    return CharLMCorrector(model, c2i, i2c, device, max_len=32, 
                          mask_threshold=mask_threshold, 
                          apply_threshold=apply_threshold, 
                          max_edits=max_edits,
                          substitutions=substitutions,
                          sub_threshold=sub_threshold)


def highlight_changes(original, corrected, trace):
    html = "<div style='font-family: monospace; font-size: 16px; line-height: 1.8;'>"
    
    changes = {item["pos"]: item for item in trace if item["applied"]}
    
    for i, char in enumerate(corrected):
        if i in changes:
            change = changes[i]
            html += f"<span style='background-color: #90EE90; padding: 2px 4px; border-radius: 3px;' " \
                    f"title='было: {change['old']} | p_cur: {change['p_cur']} | p_best: {change['p_best']}'>{char}</span>"
        else:
            html += char
    
    html += "</div>"
    return html


def format_trace_table(trace):
    if not trace:
        return "Изменений не было"
    
    html = "<table style='width:100%; border-collapse: collapse; font-size: 14px;'>"
    html += "<tr style='background-color: #f0f0f0;'>"
    html += "<th style='padding: 8px; border: 1px solid #ddd;'>Позиция</th>"
    html += "<th style='padding: 8px; border: 1px solid #ddd;'>Было</th>"
    html += "<th style='padding: 8px; border: 1px solid #ddd;'>Стало</th>"
    html += "<th style='padding: 8px; border: 1px solid #ddd;'>P(текущий)</th>"
    html += "<th style='padding: 8px; border: 1px solid #ddd;'>P(лучший)</th>"
    html += "<th style='padding: 8px; border: 1px solid #ddd;'>Применено</th>"
    html += "</tr>"
    
    for item in trace:
        bg = "#e8f5e9" if item["applied"] else "#fff"
        html += f"<tr style='background-color: {bg};'>"
        html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{item['pos']}</td>"
        html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{item['old']}</td>"
        html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{item['best']}</td>"
        html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{item['p_cur']:.4f}</td>"
        html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{item['p_best']:.4f}</td>"
        html += f"<td style='padding: 8px; border: 1px solid #ddd; text-align: center;'>{'✓' if item['applied'] else '✗'}</td>"
        html += "</tr>"
    
    html += "</table>"
    return html


def format_confidences(confidences):
    if not confidences:
        return "Нет данных"
    
    html = "<div style='font-family: monospace; font-size: 14px;'>"
    html += "Уверенность модели по позициям:<br><br>"
    
    for i, p_cur, _ in confidences:
        color = "#90EE90" if p_cur >= 0.7 else ("#FFD700" if p_cur >= 0.3 else "#FFA07A")
        html += f"<span style='background-color: {color}; padding: 2px 6px; margin: 2px; " \
                f"border-radius: 3px; display: inline-block;'>{i}: {p_cur:.4f}</span>"
    
    html += "</div>"
    return html


def correct_text(text, mask_threshold, apply_threshold, max_edits, sub_threshold):
    if not text.strip():
        return "", "", "", ""
    
    corrector.mask_threshold = mask_threshold
    corrector.apply_threshold = apply_threshold
    corrector.max_edits = max_edits
    corrector.sub_threshold = sub_threshold
    
    corrected, trace, confidences = corrector.correct_word(
        text.lower().strip(), return_trace=True, return_p_cur=True
    )
    
    highlighted = highlight_changes(text, corrected, trace)
    trace_table = format_trace_table(trace)
    conf_html = format_confidences(confidences)
    
    return corrected, highlighted, trace_table, conf_html


device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "exp_stage_a3/checkpoints/charlm_epoch_30.pt"
vocab_path = "exp_stage_a3/vocab.json"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
if not os.path.exists(vocab_path):
    raise FileNotFoundError(f"Vocab not found: {vocab_path}")

model, c2i, i2c = load_model(checkpoint_path, vocab_path, device)
corrector = create_corrector(model, c2i, i2c, device, checkpoint_path)

with gr.Blocks(title="CharLM OCR Corrector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔤 CharLM OCR Corrector")
    gr.Markdown("Коррекция OCR-ошибок с помощью Transformer модели на уровне символов")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Входной текст",
                placeholder="Введите текст с возможными OCR-ошибками...",
                lines=3
            )
            
            with gr.Accordion("Параметры коррекции", open=False):
                mask_threshold = gr.Slider(0.0, 1.0, value=0.01, step=0.01, 
                                          label="Порог маскирования (mask_threshold)")
                apply_threshold = gr.Slider(0.0, 1.0, value=0.95, step=0.01, 
                                           label="Порог применения (apply_threshold)")
                max_edits = gr.Slider(1, 10, value=1, step=1, 
                                     label="Максимум правок (max_edits)")
                sub_threshold = gr.Slider(0, 1000, value=100, step=10,
                                         label="Порог частоты замены (sub_threshold)")
            
            btn = gr.Button("Исправить", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Исправленный текст", lines=3)
            highlighted = gr.HTML(label="Визуализация изменений")
    
    with gr.Row():
        with gr.Column():
            trace_html = gr.HTML(label="Детали исправлений")
        with gr.Column():
            conf_html = gr.HTML(label="Уверенность модели")
    
    btn.click(
        correct_text,
        inputs=[input_text, mask_threshold, apply_threshold, max_edits, sub_threshold],
        outputs=[output_text, highlighted, trace_html, conf_html]
    )
    
    gr.Markdown("---")
    gr.Markdown(
        "**Как это работает:**\n"
        "- **mask_threshold**: если уверенность модели в символе < порога, он становится кандидатом на исправление\n"
        "- **apply_threshold**: исправление применяется только если уверенность в новом символе > порога\n"
        "- **max_edits**: максимальное число символов, которые можно исправить за один проход\n\n"
        "🟢 Зелёным подсвечиваются изменённые символы"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
