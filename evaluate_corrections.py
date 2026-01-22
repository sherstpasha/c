import os
import json
import torch
import pandas as pd
import Levenshtein
from tqdm import tqdm
from charlm.model import CharTransformerMLM
from charlm.utils import CharLMCorrector
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


def evaluate_predictions(gt_df, pred_df, corrector=None):
    results = []
    gt_dict = dict(zip(gt_df['filename'], gt_df['text']))
    
    total_chars_improved = 0
    total_chars_worsened = 0
    total_words = 0
    total_time = 0.0
    
    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df), desc="Evaluating"):
        filename = row['image']
        pred = normalize_text(str(row['prediction']).lower().strip())
        
        if filename not in gt_dict:
            continue
        
        gt = normalize_text(str(gt_dict[filename]).lower().strip())
        
        cer_before = cer(pred, gt)
        acc_before = accuracy(pred, gt)
        
        if corrector:
            start_time = time.time()
            corrected = corrector.correct_word(pred)
            end_time = time.time()
            total_time += (end_time - start_time)
            total_words += 1
            
            cer_after = cer(corrected, gt)
            acc_after = accuracy(corrected, gt)
            
            # Вычисляем изменение в символах
            dist_before = Levenshtein.distance(pred, gt)
            dist_after = Levenshtein.distance(corrected, gt)
            chars_delta = dist_before - dist_after
            
            if chars_delta > 0:
                total_chars_improved += chars_delta
            elif chars_delta < 0:
                total_chars_worsened += abs(chars_delta)
        else:
            corrected = pred
            cer_after = cer_before
            acc_after = acc_before
            chars_delta = 0
        
        results.append({
            'filename': filename,
            'ground_truth': gt,
            'prediction': pred,
            'corrected': corrected,
            'cer_before': cer_before,
            'cer_after': cer_after,
            'acc_before': acc_before,
            'acc_after': acc_after,
            'improved': cer_after < cer_before,
            'worsened': cer_after > cer_before,
            'unchanged': cer_after == cer_before,
            'chars_delta': chars_delta,
        })
    
    words_per_second = total_words / total_time if total_time > 0 else 0
    
    return pd.DataFrame(results), total_chars_improved, total_chars_worsened, words_per_second


def print_statistics(df, model_name, chars_improved, chars_worsened, words_per_sec):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"\nBefore correction:")
    print(f"  CER:      {df['cer_before'].mean():.4f}")
    print(f"  NED:      {1 - df['cer_before'].mean():.4f}")
    print(f"  Accuracy: {df['acc_before'].mean():.4f}")
    print(f"\nAfter correction:")
    print(f"  CER:      {df['cer_after'].mean():.4f}")
    print(f"  NED:      {1 - df['cer_after'].mean():.4f}")
    print(f"  Accuracy: {df['acc_after'].mean():.4f}")
    print(f"\nDelta:")
    print(f"  CER:      {df['cer_after'].mean() - df['cer_before'].mean():.4f}")
    print(f"  NED:      {(1 - df['cer_after'].mean()) - (1 - df['cer_before'].mean()):.4f}")
    print(f"  Accuracy: {df['acc_after'].mean() - df['acc_before'].mean():.4f}")
    print(f"\nChanges:")
    print(f"  Improved:  {df['improved'].sum()} ({df['improved'].mean()*100:.2f}%) | {chars_improved} chars")
    print(f"  Worsened:  {df['worsened'].sum()} ({df['worsened'].mean()*100:.2f}%) | {chars_worsened} chars")
    print(f"  Unchanged: {df['unchanged'].sum()} ({df['unchanged'].mean()*100:.2f}%)")
    print(f"\nPerformance:")
    print(f"  Speed: {words_per_sec:.2f} words/sec")


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
        print(f"\n{'#'*80}")
        print(f"# RUNNING ON: {device.upper()}")
        print(f"{'#'*80}\n")
        
        print(f"Loading model from {checkpoint_path}")
        model, c2i, i2c = load_model(checkpoint_path, vocab_path, device)
        
        print(f"Loading lexicon from {words_path}")
        with open(words_path, encoding="utf-8") as f:
            lexicon = set(w.strip().lower() for w in f if w.strip())
        print(f"Lexicon size: {len(lexicon):,}")
        
        corrector = CharLMCorrector(
            model, c2i, i2c, device, max_len=32,
            mask_threshold=0.01, apply_threshold=0.9, max_edits=1,
            lexicon=lexicon, min_word_len=4
        )
        
        datasets = [
            ("YeniseiGovReports-HWR", [
                "YeniseiGovReports-HWR_dialecticstackmixDomino",
                "YeniseiGovReports-HWR_trba_lite_g1",
                "YeniseiGovReports-HWR_trba_base_g1",
                "YeniseiGovReports-HWR_trocr_ru_1700s"
            ]),
            ("YeniseiGovReports-PRT", [
                "YeniseiGovReports-PRT_dialecticstackmixDomino",
                "YeniseiGovReports-PRT_trba_lite_g1",
                "YeniseiGovReports-PRT_trba_base_g1",
                "YeniseiGovReports-PRT_trocr_ru_1700s"
            ])
        ]
        
        all_results = []
        
        for dataset_name, model_names in datasets:
            gt_path = f"exp/{dataset_name}_gt.csv"
            gt_df = pd.read_csv(gt_path)
            
            print(f"\n\n{'#'*60}")
            print(f"# Dataset: {dataset_name}")
            print(f"{'#'*60}")
            
            for model_name in model_names:
                pred_path = f"exp/{model_name}.csv"
                
                if not os.path.exists(pred_path):
                    print(f"\nSkipping {model_name} (file not found)")
                    continue
                
                pred_df = pd.read_csv(pred_path)
                
                results_df, chars_improved, chars_worsened, words_per_sec = evaluate_predictions(
                    gt_df, pred_df, corrector
                )
                
                print_statistics(results_df, model_name, chars_improved, chars_worsened, words_per_sec)
                
                output_path = f"exp/{model_name}_corrected_{device}.csv"
                results_df.to_csv(output_path, index=False)
                print(f"\nSaved detailed results to: {output_path}")
                
                worsened_df = results_df[results_df['worsened'] == True]
                if len(worsened_df) > 0:
                    worsened_path = f"exp/{model_name}_worsened_{device}.csv"
                    worsened_df.to_csv(worsened_path, index=False)
                    print(f"Saved worsened cases to: {worsened_path}")
                
                all_results.append({
                    'device': device,
                    'dataset': dataset_name,
                    'model': model_name,
                    'cer_before': results_df['cer_before'].mean(),
                    'cer_after': results_df['cer_after'].mean(),
                    'ned_before': 1 - results_df['cer_before'].mean(),
                    'ned_after': 1 - results_df['cer_after'].mean(),
                    'acc_before': results_df['acc_before'].mean(),
                    'acc_after': results_df['acc_after'].mean(),
                    'improved_pct': results_df['improved'].mean() * 100,
                    'worsened_pct': results_df['worsened'].mean() * 100,
                    'improved_count': results_df['improved'].sum(),
                    'worsened_count': results_df['worsened'].sum(),
                    'chars_improved': chars_improved,
                    'chars_worsened': chars_worsened,
                    'words_per_sec': words_per_sec,
                })
        
        summary_df = pd.DataFrame(all_results)
        summary_path = f"exp/correction_summary_{device}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n\nSummary saved to: {summary_path}")
        print("\n" + "="*80)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
