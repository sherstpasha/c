"""
OCR Spell Correction Experiment

This script evaluates the OCR spell corrector on a test dataset,
computing accuracy metrics before and after correction, as well as
the correction rate.

Usage:
    python experiment.py

Configuration can be adjusted in the CONFIG section below.
"""

import time
import pandas as pd
from evaluate import load
from tqdm import tqdm
from corrector import OCRSpellCorrector


# ================= CONFIGURATION =================
CONFIG = {
    # Data paths
    "gt_path": "YeniseiGovReports-HWR_gt_mapped.csv",
    "pred_path": "YeniseiGovReports-HWR_trba_lite_g1_mapped.csv",
    "dict_path": "all_words_with_gt.txt",
    # Data processing
    "limit": 5000,  # Number of samples to process (None for all)
    # Corrector parameters
    "corrector_params": {
        "max_edit": 2,
        "min_token_len": 3,
        "symspell_prefix_len": 5,
        "protect_prefix_len": 1,
        "max_candidates": 5,
        "skip_hyphen_tokens": True,
        "char_map": {
            "i": "і",
            "I": "І",
        },
        "forbidden_suffix_changes": [
            ("е", "и"),
            ("и", "е"),
            ("ть", "те"),
            ("ть", "т"),
            ("й", "я"),
            ("я", "й"),
        ],
    },
    # Output
    "log_path": "corrections_log_experiment.csv",
}


def load_data(gt_path, pred_path, dict_path, limit=None):
    """
    Load ground truth, predictions, and dictionary.

    Parameters
    ----------
    gt_path : str
        Path to ground truth CSV
    pred_path : str
        Path to predictions CSV
    dict_path : str
        Path to dictionary file
    limit : int, optional
        Maximum number of samples to load

    Returns
    -------
    tuple
        (gt_texts, pred_texts, words_set)
    """
    print("Loading data...")

    # Load ground truth
    gt_df = pd.read_csv(gt_path)
    gt_texts = gt_df["text"].astype(str).tolist()

    # Load predictions
    pred_df = pd.read_csv(pred_path)
    pred_texts = pred_df["prediction"].astype(str).tolist()

    # Apply limit if specified
    if limit is not None:
        gt_texts = gt_texts[:limit]
        pred_texts = pred_texts[:limit]

    # Load dictionary
    with open(dict_path, encoding="utf-8") as f:
        words = set(w.strip() for w in f if w.strip())

    print(f"  Loaded {len(gt_texts)} samples")
    print(f"  Dictionary size: {len(words)} words")

    return gt_texts, pred_texts, words


def compute_metrics(predictions, references, cer_metric, wer_metric):
    """
    Compute CER and WER metrics.

    Parameters
    ----------
    predictions : list
        List of prediction strings
    references : list
        List of reference strings
    cer_metric : evaluate.Metric
        CER metric instance
    wer_metric : evaluate.Metric
        WER metric instance

    Returns
    -------
    dict
        Dictionary with 'CER' and 'WER' keys
    """
    cer = cer_metric.compute(predictions=predictions, references=references)
    wer = wer_metric.compute(predictions=predictions, references=references)
    return {"CER": cer, "WER": wer}


def run_experiment(config):
    """
    Run the spell correction experiment.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Results dictionary with metrics and statistics
    """
    print("\n" + "=" * 60)
    print("OCR SPELL CORRECTION EXPERIMENT")
    print("=" * 60)

    # Load data
    gt_texts, pred_texts, words = load_data(
        config["gt_path"],
        config["pred_path"],
        config["dict_path"],
        config.get("limit"),
    )

    # Load metrics
    print("\nLoading metrics...")
    cer_metric = load("cer")
    wer_metric = load("wer")

    # Compute baseline metrics (before correction)
    print("\nComputing baseline metrics (before correction)...")
    baseline_metrics = compute_metrics(pred_texts, gt_texts, cer_metric, wer_metric)
    print(f"  Baseline CER: {baseline_metrics['CER']:.5f}")
    print(f"  Baseline WER: {baseline_metrics['WER']:.5f}")

    # Initialize corrector
    print("\nInitializing corrector...")
    print(f"  Parameters: {config['corrector_params']}")

    start_time = time.time()
    corrector = OCRSpellCorrector(words=words, **config["corrector_params"])
    init_time = time.time() - start_time
    print(f"  Initialization time: {init_time:.2f}s")

    # Apply corrections
    print("\nApplying corrections...")
    start_time = time.time()

    corrected_texts = []
    for i, pred in tqdm(
        enumerate(pred_texts), total=len(pred_texts), desc="Correcting"
    ):
        corrected = corrector.correct_text(pred, row_id=i, gt_text=gt_texts[i])
        corrected_texts.append(corrected)

    correction_time = time.time() - start_time
    print(f"  Correction time: {correction_time:.2f}s")
    print(f"  Speed: {len(pred_texts) / correction_time:.1f} texts/sec")

    # Get correction statistics
    stats = corrector.get_statistics()
    print("\nCorrection statistics:")
    print(f"  Tokens processed: {stats['tokens_processed']}")
    print(f"  Tokens corrected: {stats['tokens_corrected']}")
    print(f"  Tokens skipped: {stats['tokens_skipped']}")
    print(f"  Correction rate: {stats['correction_rate']:.2%}")
    print(f"  Cache size: {stats['cache_size']}")

    # Compute metrics after correction
    print("\nComputing metrics after correction...")
    corrected_metrics = compute_metrics(
        corrected_texts, gt_texts, cer_metric, wer_metric
    )
    print(f"  Corrected CER: {corrected_metrics['CER']:.5f}")
    print(f"  Corrected WER: {corrected_metrics['WER']:.5f}")

    # Compute improvements
    cer_improvement = baseline_metrics["CER"] - corrected_metrics["CER"]
    wer_improvement = baseline_metrics["WER"] - corrected_metrics["WER"]
    cer_relative = (
        (cer_improvement / baseline_metrics["CER"]) * 100
        if baseline_metrics["CER"] > 0
        else 0
    )
    wer_relative = (
        (wer_improvement / baseline_metrics["WER"]) * 100
        if baseline_metrics["WER"] > 0
        else 0
    )

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(
        f"CER: {baseline_metrics['CER']:.5f} → {corrected_metrics['CER']:.5f} "
        f"(Δ {cer_improvement:+.5f}, {cer_relative:+.2f}%)"
    )
    print(
        f"WER: {baseline_metrics['WER']:.5f} → {corrected_metrics['WER']:.5f} "
        f"(Δ {wer_improvement:+.5f}, {wer_relative:+.2f}%)"
    )
    print(
        f"\nCorrection rate: {stats['correction_rate']:.2%} "
        f"({stats['tokens_corrected']}/{stats['tokens_processed']} tokens)"
    )
    print(f"Processing speed: {len(pred_texts) / correction_time:.1f} texts/sec")
    print("=" * 60)

    # Save correction log
    if config.get("log_path"):
        print(f"\nSaving correction log to {config['log_path']}...")
        corrector.save_log(config["log_path"])

    # Return results
    return {
        "baseline": baseline_metrics,
        "corrected": corrected_metrics,
        "improvement": {
            "CER_absolute": cer_improvement,
            "WER_absolute": wer_improvement,
            "CER_relative": cer_relative,
            "WER_relative": wer_relative,
        },
        "statistics": stats,
        "timing": {
            "init_time": init_time,
            "correction_time": correction_time,
            "texts_per_second": len(pred_texts) / correction_time,
        },
    }


def main():
    """Main entry point."""
    results = run_experiment(CONFIG)

    # Optionally save results to JSON
    import json

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nFull results saved to experiment_results.json")


if __name__ == "__main__":
    main()
