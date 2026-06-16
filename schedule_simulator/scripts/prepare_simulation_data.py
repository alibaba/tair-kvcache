#!/usr/bin/env python3
"""
从 Qwen3.6-Plus timeline 原始数据生成仿真输入文件。

用法:
  # 处理单个文件
  python prepare_simulation_data.py \
    --input /nfs_3820/users/linsiyuan.lsy/qwen36_timeline/downloaded/qwen3.6-plus.20260603.h10.timeline.jsonl.zst \
    --output output.jsonl

  # 处理整个目录（自动合并多个小时文件）
  python prepare_simulation_data.py \
    --input-dir /nfs_3820/users/linsiyuan.lsy/qwen36_timeline/downloaded/ \
    --output main_svc_full.jsonl

  # 指定部署前缀过滤
  python prepare_simulation_data.py \
    --input-dir /nfs_3820/users/linsiyuan.lsy/qwen36_timeline/downloaded/ \
    --output main_svc_e02sg.jsonl \
    --pod-prefix e02-sg

  # 限制输出条数
  python prepare_simulation_data.py \
    --input-dir /nfs_3820/users/linsiyuan.lsy/qwen36_timeline/downloaded/ \
    --output main_svc_50k.jsonl \
    --pod-prefix e02-sg \
    --max-records 50000

  # 指定 service 过滤
  python prepare_simulation_data.py \
    --input-dir /nfs_3820/users/linsiyuan.lsy/qwen36_timeline/downloaded/ \
    --output main_svc_full.jsonl \
    --service-keyword think-model-e1b8

  # 按时间范围过滤（从数据开始时间起的前N分钟）
  python prepare_simulation_data.py \
    --input-dir /nfs_3820/users/linsiyuan.lsy/qwen36_timeline/downloaded/ \
    --output main_svc_15min.jsonl \
    --pod-prefix e02-sg \
    --time-range 15min

  # 按绝对时间范围过滤
  python prepare_simulation_data.py \
    --input-dir /nfs_3820/users/linsiyuan.lsy/qwen36_timeline/downloaded/ \
    --output main_svc_timerange.jsonl \
    --start-time 1717456800 \
    --end-time 1717457700

原始数据字段:
  request_id, timestamp(epoch秒), model_name, input_length, type,
  input_block_hash_ids(hex字符串列表), service_names(列表), pods(列表), prefill

输出数据字段:
  timestamp(epoch毫秒), input_length, output_length(=1), block_ids(int列表), instance_id
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter


def parse_time_range(time_str):
    """解析时间范围字符串，返回秒数"""
    if not time_str:
        return None

    # 支持格式: 15min, 30m, 1h, 2hour, 90s, 1d
    match = re.match(r'^(\d+(?:\.\d+)?)(s|sec|m|min|h|hour|d|day)$', time_str.lower())
    if not match:
        raise ValueError(f"无法解析时间范围: {time_str}")

    value = float(match.group(1))
    unit = match.group(2)

    multipliers = {
        's': 1, 'sec': 1,
        'm': 60, 'min': 60,
        'h': 3600, 'hour': 3600,
        'd': 86400, 'day': 86400,
    }

    return value * multipliers[unit]


def open_input(path):
    """打开输入文件，自动处理 .zst 压缩"""
    if path.endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-dc", path], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        return proc.stdout
    else:
        return open(path, "rb")


def hex_to_int(hex_str):
    """将 hex 字符串转为 int64（用于 block_ids）"""
    return int(hex_str, 16) % (2**63)


def process_file(input_path, output_file, args, stats, time_filter):
    """处理单个输入文件"""
    fh = open_input(input_path)
    for raw_line in fh:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue

        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            stats["parse_errors"] += 1
            continue

        stats["total_read"] += 1

        # timestamp: 秒
        ts = d.get("timestamp", 0)

        # 时间过滤
        if time_filter:
            start_ts, end_ts = time_filter
            if ts < start_ts or ts > end_ts:
                stats["filtered_time"] += 1
                continue

        # service 过滤
        if args.service_keyword:
            services = d.get("service_names", [])
            if not any(args.service_keyword in s for s in services):
                stats["filtered_service"] += 1
                continue

        # pod 信息
        pods = d.get("pods", [])
        instance_id = pods[0] if pods else d.get("instance_id", "")

        # pod 前缀过滤
        if args.pod_prefix and not instance_id.startswith(args.pod_prefix):
            stats["filtered_pod"] += 1
            continue

        # block_ids: 支持两种格式
        # 1. 原始 timeline: input_block_hash_ids (hex 字符串列表) → 转 int
        # 2. 已处理文件: block_ids (int 列表) → 直接使用
        if 'input_block_hash_ids' in d:
            raw_block_ids = d['input_block_hash_ids']
            block_ids = []
            for h in raw_block_ids:
                try:
                    block_ids.append(hex_to_int(h))
                except (ValueError, TypeError):
                    pass
        elif 'block_ids' in d:
            block_ids = d['block_ids']
        else:
            block_ids = []

        # timestamp: 秒 → 毫秒
        ts_ms = ts * 1000

        out = {
            "timestamp": ts_ms,
            "input_length": d.get("input_length", 0),
            "output_length": 1,
            "block_ids": block_ids,
            "instance_id": instance_id,
        }
        output_file.write(json.dumps(out) + "\n")
        stats["written"] += 1

        # service/pod 统计
        for s in d.get("service_names", []):
            stats["services"][s] += 1
        stats["pods"].add(instance_id)

        if args.max_records and stats["written"] >= args.max_records:
            break

    if hasattr(fh, "close"):
        fh.close()

    return args.max_records and stats["written"] >= args.max_records


def scan_time_range(input_files):
    """扫描所有文件，找到最小和最大 timestamp"""
    print("扫描数据时间范围...")
    min_ts = float('inf')
    max_ts = 0

    for input_file in input_files:
        fh = open_input(input_file)
        for raw_line in fh:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ts = d.get("timestamp", 0)
                if ts > 0:
                    min_ts = min(min_ts, ts)
                    max_ts = max(max_ts, ts)
            except json.JSONDecodeError:
                continue
        if hasattr(fh, "close"):
            fh.close()

    if min_ts == float('inf'):
        print("错误: 未找到有效的时间戳数据", file=sys.stderr)
        sys.exit(1)

    print(f"  最小时间: {min_ts} ({min_ts:.0f}s)")
    print(f"  最大时间: {max_ts} ({max_ts:.0f}s)")
    print(f"  时间跨度: {max_ts - min_ts:.1f}s ({(max_ts - min_ts)/60:.1f}min)")

    return min_ts, max_ts


def main():
    p = argparse.ArgumentParser(
        description="从 timeline 原始数据生成仿真输入 JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", type=str, default=None, help="单个输入文件路径（支持 .jsonl 和 .jsonl.zst）")
    p.add_argument("--input-dir", type=str, default=None, help="输入目录（处理目录下所有 .jsonl/.jsonl.zst 文件）")
    p.add_argument("--output", type=str, required=True, help="输出 JSONL 文件路径")
    p.add_argument("--pod-prefix", type=str, default=None, help="按 pod 名前缀过滤（如 e02-sg）")
    p.add_argument("--service-keyword", type=str, default=None, help="按 service_name 关键词过滤")
    p.add_argument("--max-records", type=int, default=None, help="最大输出记录数")
    p.add_argument("--time-range", type=str, default=None,
                   help="时间范围过滤（从数据开始时间起），如: 15min, 1h, 30m, 2hour")
    p.add_argument("--start-time", type=float, default=None,
                   help="起始时间（epoch 秒），与 --end-time 配合使用")
    p.add_argument("--end-time", type=float, default=None,
                   help="结束时间（epoch 秒），与 --start-time 配合使用")
    p.add_argument("--sort-by-time", action="store_true", help="按 timestamp 排序输出（需要额外内存）")
    args = p.parse_args()

    if not args.input and not args.input_dir:
        p.error("必须指定 --input 或 --input-dir")

    # 收集输入文件
    input_files = []
    if args.input:
        input_files.append(args.input)
    if args.input_dir:
        for f in sorted(os.listdir(args.input_dir)):
            if f.endswith(".jsonl") or f.endswith(".jsonl.zst"):
                input_files.append(os.path.join(args.input_dir, f))

    if not input_files:
        print("Error: 未找到输入文件", file=sys.stderr)
        sys.exit(1)

    print(f"输入文件: {len(input_files)} 个")
    for f in input_files:
        print(f"  {os.path.basename(f)}")

    if args.pod_prefix:
        print(f"Pod 过滤: {args.pod_prefix}*")
    if args.service_keyword:
        print(f"Service 过滤: *{args.service_keyword}*")
    if args.max_records:
        print(f"最大记录数: {args.max_records}")

    # 时间过滤
    time_filter = None
    if args.time_range:
        duration = parse_time_range(args.time_range)
        print(f"时间范围过滤: 前 {args.time_range} ({duration}s)")
        data_min_ts, data_max_ts = scan_time_range(input_files)
        # Convert duration from seconds to milliseconds (timestamps are in ms)
        time_filter = (data_min_ts, data_min_ts + duration * 1000)
        print(f"  过滤区间: [{time_filter[0]:.0f}, {time_filter[1]:.0f}]")
    elif args.start_time or args.end_time:
        start = args.start_time if args.start_time else 0
        end = args.end_time if args.end_time else float('inf')
        time_filter = (start, end)
        print(f"绝对时间过滤: [{start}, {end}]")

    print()

    stats = {
        "total_read": 0,
        "written": 0,
        "filtered_service": 0,
        "filtered_pod": 0,
        "filtered_time": 0,
        "parse_errors": 0,
        "services": Counter(),
        "pods": set(),
    }

    with open(args.output, "w") as fout:
        for i, input_file in enumerate(input_files):
            print(f"[{i+1}/{len(input_files)}] 处理 {os.path.basename(input_file)} ...")
            reached_limit = process_file(input_file, fout, args, stats, time_filter)
            print(f"  累计: 读取 {stats['total_read']}, 写入 {stats['written']}")
            if reached_limit:
                print(f"  已达到最大记录数 {args.max_records}，停止处理")
                break

    # 按时间排序（可选）
    if args.sort_by_time and stats["written"] > 0:
        print("\n按 timestamp 排序 ...")
        records = []
        with open(args.output) as f:
            for line in f:
                records.append(json.loads(line))
        records.sort(key=lambda r: r["timestamp"])
        with open(args.output, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"排序完成: {len(records)} 条")

    # 输出统计
    print(f"\n{'='*50}")
    print(f"处理完成")
    print(f"{'='*50}")
    print(f"总读取: {stats['total_read']}")
    print(f"写入:   {stats['written']}")
    print(f"过滤（service）: {stats['filtered_service']}")
    print(f"过滤（pod）:     {stats['filtered_pod']}")
    print(f"过滤（时间）:    {stats['filtered_time']}")
    print(f"解析错误:         {stats['parse_errors']}")
    print(f"唯一 Pod 数:     {len(stats['pods'])}")
    print(f"\nService 分布:")
    for svc, cnt in stats["services"].most_common():
        print(f"  {svc}: {cnt}")
    print(f"\n输出文件: {args.output}")
    file_size = os.path.getsize(args.output)
    if file_size > 1e9:
        print(f"文件大小: {file_size/1e9:.1f} GB")
    else:
        print(f"文件大小: {file_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
