#!/usr/bin/env python3
"""
Convert HistGradientBoosting predictor to 2D lookup table pkl.

Loads the 6 HistGradientBoosting models, pre-computes predictions for
a grid of (uncached, cached) combinations, then aggregates into a 2D
lookup table pkl compatible with the existing RequestLevelTimePredictor.

Usage:
  python3 scripts/convert_qwen_predictor_to_lookup.py \
    --models-dir /path/to/qwen36_plus_hit_binary_base_through_20260609_h04 \
    --output /path/to/predictor_lookup.pkl
"""

import sys, os, argparse
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")


def load_models(models_dir):
    """Load all 6 HistGradientBoosting models."""
    models = {}
    for bucket in ["0_32k", "32_256k", "256kplus"]:
        for split in ["zero_hit", "has_hit"]:
            filename = f"qwen_request_latency_{bucket}_{split}.joblib"
            path = os.path.join(models_dir, filename)
            data = joblib.load(path)
            models[(bucket, split)] = data["model"]
    print("Loaded %d models" % len(models))
    return models


def select_model(models, uncached, cached):
    """Select model based on total_tokens bucket and cached split."""
    total = uncached + cached
    if total < 32000:
        bucket = "0_32k"
    elif total < 256000:
        bucket = "32_256k"
    else:
        bucket = "256kplus"
    split = "zero_hit" if cached <= 0 else "has_hit"
    return models[(bucket, split)]


def predict_one(models, uncached, cached):
    """Predict latency for a single (uncached, cached) pair."""
    model = select_model(models, uncached, cached)
    inlen = uncached + cached
    hit = cached
    cr = cached / max(1, inlen)
    attn = uncached * (2 * cached + uncached) / 1e6
    features = [inlen, uncached, hit, cr, 0, 0, 0, 0, attn]
    log1p_ms = model.predict([features])[0]
    ms = np.expm1(log1p_ms)
    return max(ms, 0.1)  # minimum 0.1ms


def build_lookup_table(models):
    """Build 2D lookup table by sampling and aggregating."""
    # Define bins
    uncached_bins = [
        (0, 128), (128, 256), (256, 512), (512, 1024),
        (1024, 2048), (2048, 3072), (3072, 4096), (4096, 5120),
        (5120, 6144), (6144, 7168), (7168, 8192),
        (8192, 12288), (12288, 16384), (16384, 24576), (24576, 32768),
        (32768, 49152), (49152, 65536), (65536, 98304), (98304, 131072),
        (131072, 200000), (200000, 300000), (300000, 500000),
        (500000, 999999999),
    ]
    cached_bins = [
        (0, 1), (1, 1000), (1000, 5000), (5000, 10000),
        (10000, 50000), (50000, 100000), (100000, 999999999),
    ]

    # Sample points within each bin
    np.random.seed(42)
    table = {}
    total_samples = 0

    for u_lo, u_hi in uncached_bins:
        for c_lo, c_hi in cached_bins:
            # Generate sample points within this bin
            u_range = u_hi - u_lo
            c_range = c_hi - c_lo

            # Number of samples proportional to bin size
            n_samples = 50
            uncached_vals = np.random.randint(u_lo, min(u_hi, u_lo + 100000), n_samples)
            cached_vals = np.random.randint(c_lo, min(c_hi, c_lo + 100000), n_samples)

            predictions = []
            for u, c in zip(uncached_vals, cached_vals):
                ms = predict_one(models, int(u), int(c))
                if ms > 0:
                    predictions.append(ms)

            if predictions:
                median_ms = float(np.median(predictions))
                table[(u_lo, u_hi, c_lo, c_hi)] = median_ms
                total_samples += len(predictions)

    print("Table entries: %d (from %d samples)" % (len(table), total_samples))
    return table, uncached_bins, cached_bins


def evaluate_table(table, models, n_test=5000):
    """Evaluate lookup table accuracy vs original model."""
    np.random.seed(123)
    errors = []

    for _ in range(n_test):
        uncached = int(np.random.lognormal(mean=7, sigma=2))
        uncached = min(uncached, 500000)
        cached = int(np.random.lognormal(mean=6, sigma=3))
        cached = min(cached, 500000)

        original_ms = predict_one(models, uncached, cached)

        # Lookup
        predicted_ms = None
        for (u_lo, u_hi, c_lo, c_hi), val in table.items():
            if u_lo <= uncached < u_hi and c_lo <= cached < c_hi:
                predicted_ms = val
                break
        if predicted_ms is None:
            for (u_lo, u_hi, c_lo, c_hi), val in table.items():
                if u_lo <= uncached < u_hi:
                    predicted_ms = val
                    break

        if predicted_ms and original_ms > 0:
            ape = abs(predicted_ms - original_ms) / original_ms * 100
            errors.append(ape)

    if errors:
        mape = np.mean(errors)
        p50 = np.median(errors)
        p90 = np.percentile(errors, 90)
        print("\nLookup table vs original model (n=%d):" % len(errors))
        print("  MAPE: %.2f%%" % mape)
        print("  P50:  %.2f%%" % p50)
        print("  P90:  %.2f%%" % p90)


def main():
    p = argparse.ArgumentParser(description="Convert Qwen HistGB predictor to lookup table")
    p.add_argument("--models-dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    args = p.parse_args()

    print("Loading models...")
    models = load_models(args.models_dir)

    print("\nBuilding lookup table...")
    table, uncached_bins, cached_bins = build_lookup_table(models)

    print("\nEvaluating...")
    evaluate_table(table, models)

    # Save
    out = {
        "train_table": table,
        "uncached_bins": uncached_bins,
        "cached_bins": cached_bins,
        "version": "qwen_lookup_v1",
        "mode": "2d",
        "source": "HistGradientBoosting",
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(out, args.output)
    print("\nSaved: %s (%d bytes)" % (args.output, os.path.getsize(args.output)))


if __name__ == "__main__":
    main()
