#!/usr/bin/env python3
"""
将 enriched.jsonl 数据转换为 schedule_simulator 仿真输入格式。

处理流程:
  enriched.jsonl → (1) 按 service_names 过滤 → (2) 格式转换 → sim.jsonl

使用示例:
  # 完整流程（过滤 + 转换）:
  python3 convert_enriched_to_sim.py \
      --input /path/to/qwen3.6-plus.2026-06-08.h21.enriched.jsonl \
      --output /path/to/h21_e1b8_sim.jsonl \
      --service-name qwen3.6-plus-2026-04-02-think-model-e1b8

  # 仅转换（不过滤，已是单服务数据）:
  python3 convert_enriched_to_sim.py \
      --input /path/to/h21_e1b8.jsonl \
      --output /path/to/h21_e1b8_sim.jsonl \
      --no-filter

字段映射:
  enriched.jsonl 字段        →  sim.jsonl 字段
  ─────────────────────────────────────────────
  timestamp (float, 秒)     →  timestamp (int, 毫秒)
  input_length              →  input_length (不变)
  (不存在)                  →  output_length = 1 (prefill-only)
  input_block_hash_ids      →  block_ids (hex string → int64)
    (hex string list)           int(hex, 16) % 2^63
  pods[0]                   →  instance_id (字符串)
  service_names             →  (用于过滤，不输出)
"""

import argparse
import json
import time
import sys


def hex_to_int64(hex_str: str) -> int:
    """将 hex 字符串转为 int64（取模 2^63 保证正数）"""
    return int(hex_str, 16) % (2**63)


def convert_record(record: dict) -> dict:
    """将单条 enriched 记录转换为仿真器格式"""
    # timestamp: seconds (float) -> milliseconds (int)
    ts_ms = int(record['timestamp'] * 1000)

    # input_block_hash_ids: hex strings -> int64 list
    raw_bids = record.get('input_block_hash_ids', [])
    block_ids = [hex_to_int64(h) for h in raw_bids] if raw_bids else []

    # instance_id from pods[0]
    pods = record.get('pods', [])
    instance_id = pods[0] if pods else None

    out = {
        'timestamp': ts_ms,
        'input_length': record['input_length'],
        'output_length': 1,  # prefill-only 模式
        'block_ids': block_ids,
    }
    if instance_id:
        out['instance_id'] = instance_id

    return out


def should_include(record: dict, target_service: str) -> bool:
    """检查记录的 service_names 是否包含目标服务"""
    service_names = record.get('service_names', [])
    return target_service in service_names


def main():
    parser = argparse.ArgumentParser(
        description='将 enriched.jsonl 转换为 schedule_simulator 仿真输入格式')
    parser.add_argument('--input', '-i', required=True,
                        help='输入 enriched.jsonl 文件路径')
    parser.add_argument('--output', '-o', required=True,
                        help='输出 sim.jsonl 文件路径')
    parser.add_argument('--service-name', '-s', default=None,
                        help='按 service_names 过滤的目标服务名 '
                             '(例: qwen3.6-plus-2026-04-02-think-model-e1b8)')
    parser.add_argument('--no-filter', action='store_true',
                        help='不过滤，直接转换所有记录')
    args = parser.parse_args()

    if not args.no_filter and not args.service_name:
        print("错误: 必须指定 --service-name 或 --no-filter", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    total_read = 0
    total_written = 0
    pods_seen = set()

    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            total_read += 1

            # 过滤
            if not args.no_filter:
                if not should_include(record, args.service_name):
                    continue

            # 转换
            out_rec = convert_record(record)
            fout.write(json.dumps(out_rec) + '\n')
            total_written += 1

            # 统计 pods
            if out_rec.get('instance_id'):
                pods_seen.add(out_rec['instance_id'])

            if total_written % 100000 == 0:
                print(f'  已写入 {total_written} 条...', flush=True)

    elapsed = time.time() - t0

    print(f'\n{"="*50}')
    print(f'转换完成:')
    print(f'  输入文件: {args.input}')
    print(f'  输出文件: {args.output}')
    print(f'  读取记录: {total_read:,}')
    print(f'  写入记录: {total_written:,}')
    print(f'  Pod 数量: {len(pods_seen)}')
    print(f'  耗时: {elapsed:.1f}s')
    if not args.no_filter:
        print(f'  过滤条件: service_names 包含 "{args.service_name}"')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
