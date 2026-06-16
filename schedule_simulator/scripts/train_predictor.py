#!/usr/bin/env python3
"""训练 Request 级时间预测器（支持 1D/2D Lookup Table）

用法:
  # 1D 预测器（仅 uncached）
  python3 scripts/train_predictor.py \
    --input /nfs_3820/.../h10.timeline.jsonl.zst \
    --output predictor_1d.pkl

  # 2D 预测器（uncached + cached）
  python3 scripts/train_predictor.py \
    --input /nfs_3820/.../h10.jsonl.zst \
    --output predictor_2d.pkl \
    --mode 2d

  # 多文件输入
  python3 scripts/train_predictor.py \
    --input /nfs_3820/.../h10.jsonl.zst /nfs_3820/.../h11.jsonl.zst \
    --output predictor.pkl --max-records 100000
"""

import json, os, sys, subprocess, argparse
import numpy as np
import joblib


def load_timeline_data(input_paths, max_records=None):
    records = []
    for path in input_paths:
        if path.endswith(".zst"):
            proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            fh = proc.stdout
        else:
            fh = open(path, "rb")

        for raw_line in fh:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue

            pf = d.get("prefill")
            if not pf or not isinstance(pf, dict):
                continue

            first_lat = pf.get("first_latency_ms")
            queue_ms = pf.get("queue_ms", 0)
            input_len = d.get("input_length", 0)
            cached = pf.get("cached_input_tokens", 0)

            if first_lat is None or first_lat <= 0 or input_len <= 0:
                continue

            records.append({
                "uncached": max(input_len - cached, 0),
                "cached": cached,
                "first_latency_ms": first_lat,
                "queue_ms": queue_ms,
            })

            if max_records and len(records) >= max_records:
                break
        fh.close()
        if max_records and len(records) >= max_records:
            break
    return records


def train_lookup_table_1d(records, percentile_lo=5, percentile_hi=95):
    clean = [r for r in records if r["queue_ms"] == 0]
    print(f"清洗: {len(records)} -> {len(clean)} (queue=0)")
    if len(clean) < 100:
        print("WARNING: queue=0 数据太少，使用全部数据")
        clean = records

    bins = [
        (0, 256), (256, 512), (512, 1024), (1024, 2048),
        (2048, 3072), (3072, 4096), (4096, 5120), (5120, 6144),
        (6144, 7168), (7168, 8192),
        (8192, 12288), (12288, 16384), (16384, 24576), (24576, 32768),
        (32768, 49152), (49152, 65536), (65536, 98304), (98304, 131072),
        (131072, 200000), (200000, 999999999),
    ]

    table = {}
    for lo, hi in bins:
        values = [r["first_latency_ms"] for r in clean if lo <= r["uncached"] < hi]
        if len(values) < 5:
            continue
        p_lo = np.percentile(values, percentile_lo)
        p_hi = np.percentile(values, percentile_hi)
        trimmed = [v for v in values if p_lo <= v <= p_hi]
        if trimmed:
            table[(lo, hi)] = float(np.median(trimmed))
            print(f"  [{lo:>7}, {hi:>7}): n={len(values):>5}, median={table[(lo, hi)]:>8.1f} ms")
    return table, bins


def train_lookup_table_2d(records, percentile_lo=5, percentile_hi=95):
    clean = [r for r in records if r["queue_ms"] == 0]
    print(f"清洗: {len(records)} -> {len(clean)} (queue=0)")
    if len(clean) < 100:
        print("WARNING: queue=0 数据太少，使用全部数据")
        clean = records

    uncached_bins = [
        (0, 256), (256, 512), (512, 1024), (1024, 2048),
        (2048, 3072), (3072, 4096), (4096, 5120), (5120, 6144),
        (6144, 7168), (7168, 8192),
        (8192, 12288), (12288, 16384), (16384, 24576), (24576, 32768),
        (32768, 49152), (49152, 65536), (65536, 98304), (98304, 131072),
        (131072, 200000), (200000, 999999999),
    ]
    
    # Cached bins: 0, (0, 1000), (1000, 10000), (10000, 100000), (100000, inf)
    cached_bins = [(0, 1), (1, 1000), (1000, 10000), (10000, 100000), (100000, 999999999)]

    table = {}
    for u_lo, u_hi in uncached_bins:
        for c_lo, c_hi in cached_bins:
            values = [r["first_latency_ms"] for r in clean 
                     if u_lo <= r["uncached"] < u_hi and c_lo <= r["cached"] < c_hi]
            if len(values) < 5:
                continue
            p_lo = np.percentile(values, percentile_lo)
            p_hi = np.percentile(values, percentile_hi)
            trimmed = [v for v in values if p_lo <= v <= p_hi]
            if trimmed:
                table[(u_lo, u_hi, c_lo, c_hi)] = float(np.median(trimmed))
    
    print(f"  2D table entries: {len(table)}")
    return table, uncached_bins, cached_bins


def evaluate_1d(records, table):
    errors = []
    for r in records:
        unc = r["uncached"]
        actual = r["first_latency_ms"]
        predicted = None
        for (lo, hi), val in table.items():
            if lo <= unc < hi:
                predicted = val
                break
        if predicted is None:
            continue
        errors.append(abs(predicted - actual) / max(actual, 1e-6) * 100)

    if errors:
        print(f"\n评估: n={len(errors)}, MAPE={np.mean(errors):.1f}%, "
              f"P50={np.median(errors):.1f}%, P90={np.percentile(errors, 90):.1f}%")


def evaluate_2d(records, table):
    errors = []
    for r in records:
        unc = r["uncached"]
        cached = r["cached"]
        actual = r["first_latency_ms"]
        predicted = None
        for key, val in table.items():
            u_lo, u_hi, c_lo, c_hi = key
            if u_lo <= unc < u_hi and c_lo <= cached < c_hi:
                predicted = val
                break
        if predicted is None:
            # Fallback: find closest uncached bin
            for key, val in table.items():
                u_lo, u_hi, c_lo, c_hi = key
                if u_lo <= unc < u_hi:
                    predicted = val
                    break
        if predicted is None:
            continue
        errors.append(abs(predicted - actual) / max(actual, 1e-6) * 100)

    if errors:
        print(f"\n评估: n={len(errors)}, MAPE={np.mean(errors):.1f}%, "
              f"P50={np.median(errors):.1f}%, P90={np.percentile(errors, 90):.1f}%")


def main():
    p = argparse.ArgumentParser(description="训练 Request 级时间预测器")
    p.add_argument("--input", type=str, nargs="+", required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--mode", type=str, choices=["1d", "2d"], default="1d",
                   help="1d: 仅 uncached; 2d: uncached + cached")
    args = p.parse_args()

    print("加载数据 ...")
    records = load_timeline_data(args.input, args.max_records)
    print(f"加载 {len(records)} 条记录")
    if not records:
        print("ERROR: 没有有效记录"); sys.exit(1)

    unc = [r["uncached"] for r in records]
    cached = [r["cached"] for r in records]
    lat = [r["first_latency_ms"] for r in records]
    print(f"uncached: [{min(unc)}, {max(unc)}], median={np.median(unc):.0f}")
    print(f"cached:   [{min(cached)}, {max(cached)}], median={np.median(cached):.0f}")
    print(f"latency:  [{min(lat):.0f}, {max(lat):.0f}] ms, median={np.median(lat):.0f} ms")

    print(f"\n训练 {args.mode.upper()} ...")
    if args.mode == "1d":
        table, bins = train_lookup_table_1d(records)
        evaluate_1d(records, table)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        joblib.dump({"train_table": table, "bins": bins, "version": "v1", 
                     "mode": "1d", "n_records": len(records)}, args.output)
    else:  # 2d
        table, uncached_bins, cached_bins = train_lookup_table_2d(records)
        evaluate_2d(records, table)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        joblib.dump({"train_table": table, "uncached_bins": uncached_bins, 
                     "cached_bins": cached_bins, "version": "v1",
                     "mode": "2d", "n_records": len(records)}, args.output)
    
    print(f"\n保存: {args.output} ({os.path.getsize(args.output)} bytes)")


if __name__ == "__main__":
    main()
