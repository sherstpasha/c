import os
import json
import torch
import pandas as pd
import Levenshtein
from tqdm import tqdm
from charlm.model import CharTransformerMLM
from charlm.utils import CharLMCorrector
import itertools
import time


def load_model(checkpoint_path, vocab_path, device="cuda"):
    with open(vocab_path, encoding="utf-8") as f:
        chars = json.load(f)
    c2i = {c: i for i, c in enumerate(chars)}
    i2c = {i: c for c, i in c2i.items()}
    vocab_size = len(chars)
    
    model = CharTransformerMLM(
        vocab_size=vocab_size, emb_size=192, max_len=32, n_layers=8,
        n_heads=6, ffn_size=1024, dropout=0.1, pad_idx=c2i["<PAD>"]
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, c2i, i2c


def cer(pred, target):
    if not target:
        return 0.0
    return Levenshtein.distance(pred, target) / len(target)


def accuracy(pred, target):
    return 1.0 if pred == target else 0.0


def normalize_text(text):
    text = text.replace('i', 'і')  # латинская i -> кириллическая і
    text = text.replace('I', 'І')
    return text


def evaluate_predictions(gt_df, pred_df, corrector):
    results = []
    gt_dict = dict(zip(gt_df['filename'], gt_df['text']))
    
    total_chars_improved = 0
    total_chars_worsened = 0
    total_words = 0
    total_time = 0.0
    
    for _, row in pred_df.iterrows():
        filename = row['image']
        pred = normalize_text(str(row['prediction']).lower().strip())
        
        if filename not in gt_dict:
            continue
        
        gt = normalize_text(str(gt_dict[filename]).lower().strip())
        
        cer_before = cer(pred, gt)
        
        start_time = time.time()
        corrected = corrector.correct_word(pred)
        end_time = time.time()
        total_time += (end_time - start_time)
        total_words += 1
        
        cer_after = cer(corrected, gt)
        
        # Вычисляем изменение в символах
        dist_before = Levenshtein.distance(pred, gt)
        dist_after = Levenshtein.distance(corrected, gt)
        chars_delta = dist_before - dist_after
        
        if chars_delta > 0:
            total_chars_improved += chars_delta
        elif chars_delta < 0:
            total_chars_worsened += abs(chars_delta)
        
        results.append({
            'cer_before': cer_before,
            'cer_after': cer_after,
            'improved': cer_after < cer_before,
            'worsened': cer_after > cer_before,
            'chars_delta': chars_delta,
        })
    
    words_per_second = total_words / total_time if total_time > 0 else 0
    
    return results, total_chars_improved, total_chars_worsened, words_per_second


def evaluate_config(model, c2i, i2c, device, lexicon, gt_df_hwr, pred_df_hwr, 
                    gt_df_prt, pred_df_prt, mask_threshold, apply_threshold, 
                    max_edits, min_word_len):
    """Оценка одной конфигурации гиперпараметров"""
    
    corrector = CharLMCorrector(
        model, c2i, i2c, device, max_len=32,
        mask_threshold=mask_threshold, 
        apply_threshold=apply_threshold, 
        max_edits=max_edits,
        lexicon=lexicon, 
        min_word_len=min_word_len
    )
    
    # Оценка на HWR trba_lite_g1
    results_hwr, chars_imp_hwr, chars_wor_hwr, wps_hwr = evaluate_predictions(
        gt_df_hwr, pred_df_hwr, corrector
    )
    
    # Оценка на PRT trba_lite_g1
    results_prt, chars_imp_prt, chars_wor_prt, wps_prt = evaluate_predictions(
        gt_df_prt, pred_df_prt, corrector
    )
    
    # Объединяем результаты
    all_results = results_hwr + results_prt
    
    improved = sum(1 for r in all_results if r['improved'])
    worsened = sum(1 for r in all_results if r['worsened'])
    total = len(all_results)
    
    improved_pct = (improved / total * 100) if total > 0 else 0
    worsened_pct = (worsened / total * 100) if total > 0 else 0
    
    chars_improved = chars_imp_hwr + chars_imp_prt
    chars_worsened = chars_wor_hwr + chars_wor_prt
    
    # Средняя скорость
    avg_wps = (wps_hwr + wps_prt) / 2
    
    # NED (Normalized Edit Distance) = 1 - CER
    cer_before_avg = sum(r['cer_before'] for r in all_results) / total if total > 0 else 0
    cer_after_avg = sum(r['cer_after'] for r in all_results) / total if total > 0 else 0
    ned_before = 1 - cer_before_avg
    ned_after = 1 - cer_after_avg
    ned_delta = ned_after - ned_before
    
    # Целевая метрика: максимизируем разницу (improved - worsened)
    score = improved - worsened
    ratio = improved / (worsened + 1)
    
    return {
        'mask_threshold': mask_threshold,
        'apply_threshold': apply_threshold,
        'max_edits': max_edits,
        'min_word_len': min_word_len,
        'improved': improved,
        'worsened': worsened,
        'improved_pct': improved_pct,
        'worsened_pct': worsened_pct,
        'chars_improved': chars_improved,
        'chars_worsened': chars_worsened,
        'ned_before': ned_before,
        'ned_after': ned_after,
        'ned_delta': ned_delta,
        'words_per_sec': avg_wps,
        'score': score,
        'ratio': ratio,
        'total': total
    }


def main():
    checkpoint_path = "exp_last/checkpoints/charlm_epoch_50.pt"
    vocab_path = "exp_last/vocab.json"
    words_path = "data/words.txt"
    
    # Тестируем на обоих устройствах
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    
    for device in devices:
        print(f"\n{'='*80}")
        print(f"RUNNING GRID SEARCH ON: {device.upper()}")
        print(f"{'='*80}\n")
        
        print(f"Loading model from {checkpoint_path}")
        model, c2i, i2c = load_model(checkpoint_path, vocab_path, device)
        
        print(f"Loading lexicon from {words_path}")
        with open(words_path, encoding="utf-8") as f:
            lexicon = set(w.strip().lower() for w in f if w.strip())
        print(f"Lexicon size: {len(lexicon):,}")
        
        # Загружаем данные для оценки
        print("\nLoading evaluation datasets...")
        gt_df_hwr = pd.read_csv("exp/YeniseiGovReports-HWR_gt.csv")
        pred_df_hwr = pd.read_csv("exp/YeniseiGovReports-HWR_trba_lite_g1.csv")
        
        gt_df_prt = pd.read_csv("exp/YeniseiGovReports-PRT_gt.csv")
        pred_df_prt = pd.read_csv("exp/YeniseiGovReports-PRT_trba_lite_g1.csv")
        
        print(f"HWR samples: {len(pred_df_hwr)}")
        print(f"PRT samples: {len(pred_df_prt)}")
        
        # Определяем сетку гиперпараметров
        param_grid = {
            'mask_threshold': [0.01, 0.05, 0.1],
            'apply_threshold': [0.9, 0.95, 0.98],
            'max_edits': [1, 2, 3],
            'min_word_len': [3, 4]
        }
        
        # Генерируем все комбинации
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        print(f"\nTotal combinations to test: {len(combinations)}")
        print(f"Starting grid search on {device}...\n")
        
        results = []
        
        # Проходим по всем комбинациям с прогресс-баром
        for combination in tqdm(combinations, desc=f"Grid Search ({device})"):
            params = dict(zip(keys, combination))
            
            result = evaluate_config(
                model, c2i, i2c, device, lexicon, 
                gt_df_hwr, pred_df_hwr, 
                gt_df_prt, pred_df_prt,
                **params
            )
            
            results.append(result)
        
        # Сохраняем все результаты
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        results_path = f"exp/hyperparameter_search_results_{device}.csv"
        results_df.to_csv(results_path, index=False)
        
        print(f"\n{'='*80}")
        print(f"GRID SEARCH COMPLETED ON {device.upper()}")
        print(f"{'='*80}")
        print(f"\nTop 10 configurations by score (improved - worsened):\n")
        print(results_df.head(10).to_string(index=False))
        
        print(f"\n\nTop 10 configurations by ratio (improved / (worsened + 1)):\n")
        results_df_ratio = results_df.sort_values('ratio', ascending=False)
        print(results_df_ratio.head(10).to_string(index=False))
        
        # Лучшая конфигурация
        best = results_df.iloc[0]
        print(f"\n{'='*80}")
        print(f"BEST CONFIGURATION ON {device.upper()} (by score):")
        print(f"{'='*80}")
        print(f"mask_threshold:   {best['mask_threshold']}")
        print(f"apply_threshold:  {best['apply_threshold']}")
        print(f"max_edits:        {int(best['max_edits'])}")
        print(f"min_word_len:     {int(best['min_word_len'])}")
        print(f"\nResults:")
        print(f"  NED before:    {best['ned_before']:.4f}")
        print(f"  NED after:     {best['ned_after']:.4f}")
        print(f"  NED delta:     {best['ned_delta']:.4f}")
        print(f"  Improved:      {int(best['improved'])} ({best['improved_pct']:.2f}%) | {int(best['chars_improved'])} chars")
        print(f"  Worsened:      {int(best['worsened'])} ({best['worsened_pct']:.2f}%) | {int(best['chars_worsened'])} chars")
        print(f"  Score:         {int(best['score'])} (improved - worsened)")
        print(f"  Ratio:         {best['ratio']:.2f} (improved / (worsened + 1))")
        print(f"  Speed:         {best['words_per_sec']:.2f} words/sec")
        
        print(f"\n\nAll results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
