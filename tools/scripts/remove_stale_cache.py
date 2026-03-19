#!/usr/bin/env python3
"""
批量删除stale KVCache - 读取scan_redis_stale_keys.py生成的stale.json，调用removeCache接口逐条删除。
instance_id和block_key均从key字段自动提取。

用法:
    python3 remove_stale_cache.py --file stale.json
    python3 remove_stale_cache.py --file stale.json --url http://10.0.0.1:6492
    python3 remove_stale_cache.py --file stale.json --batch 50 --dry-run
"""

import argparse
import json
import sys
import uuid
from collections import defaultdict

try:
    import requests
except ImportError:
    print("请先安装requests: pip install requests", file=sys.stderr)
    sys.exit(1)


def parse_key(key_str: str) -> tuple[str, int] | None:
    """从Redis key中解析instance_id和block_key。
    格式: kvcache:instance_<instance_id>:cache_<block_key>
    返回 (instance_id, block_key) 或 None。
    """
    parts = key_str.split(":")
    if len(parts) != 3:
        return None
    _, inst_part, cache_part = parts
    if not inst_part.startswith("instance_") or not cache_part.startswith("cache_"):
        return None
    try:
        instance_id = inst_part[len("instance_"):]
        block_key = int(cache_part[len("cache_"):])
        return instance_id, block_key
    except ValueError:
        return None


def remove_cache(url: str, instance_id: str, block_keys: list[int], trace_id: str,
                 timeout: float = 10.0) -> dict:
    """调用removeCache接口"""
    payload = {
        "trace_id": trace_id,
        "instance_id": instance_id,
        "block_keys": block_keys,
        "token_ids": [],
        "block_mask": {
            "offset": 0
        }
    }
    resp = requests.post(
        f"{url}/api/removeCache",
        json=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="批量调用removeCache接口删除stale KVCache")
    parser.add_argument("--file", "-f", default="stale.json", help="stale.json文件路径 (default: stale.json)")
    parser.add_argument("--url", "-u", default="http://localhost:6492", help="服务地址 (default: http://localhost:6492)")
    parser.add_argument("--batch", "-b", type=int, default=1, help="每次请求携带的block_key数量 (default: 1)")
    parser.add_argument("--timeout", "-t", type=float, default=10.0, help="请求超时时间(秒) (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="仅打印请求内容，不实际调用接口")
    args = parser.parse_args()

    if args.batch <= 0:
        print(f"--batch 必须为正整数，当前值: {args.batch}", file=sys.stderr)
        sys.exit(1)

    # 读取stale.json
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        print(f"文件不存在: {args.file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}", file=sys.stderr)
        sys.exit(1)

    # 按instance_id分组解析block_keys
    instance_keys: dict[str, list[int]] = defaultdict(list)
    skipped = 0
    for entry in entries:
        key_str = entry.get("key", "")
        parsed = parse_key(key_str)
        if parsed is not None:
            instance_id, block_key = parsed
            instance_keys[instance_id].append(block_key)
        else:
            skipped += 1
            print(f"跳过无法解析的key: {key_str}", file=sys.stderr)

    # 去重
    total_parsed = sum(len(v) for v in instance_keys.values())
    for inst_id in instance_keys:
        instance_keys[inst_id] = list(dict.fromkeys(instance_keys[inst_id]))
    total_unique = sum(len(v) for v in instance_keys.values())

    print(f"共读取 {len(entries)} 条记录，解析出 {total_parsed} 个block_key"
          f"(去重后 {total_unique} 个)，跳过 {skipped} 条")
    print(f"涉及 {len(instance_keys)} 个instance: {list(instance_keys.keys())}")

    if not instance_keys:
        print("没有需要删除的block_key")
        return

    # 按instance分组，再按batch拆分
    all_batches: list[tuple[str, list[int]]] = []
    for inst_id, keys in instance_keys.items():
        for i in range(0, len(keys), args.batch):
            all_batches.append((inst_id, keys[i:i + args.batch]))

    print(f"将分 {len(all_batches)} 批请求发送 (batch_size={args.batch})")
    print("-" * 60)

    success = 0
    failed = 0
    for idx, (inst_id, batch_keys) in enumerate(all_batches, 1):
        trace_id = f"remove_stale_{uuid.uuid4().hex[:12]}"
        if args.dry_run:
            print(f"[{idx}/{len(all_batches)}] [DRY-RUN] instance={inst_id} "
                  f"trace_id={trace_id} block_keys={batch_keys}")
            success += 1
            continue

        try:
            result = remove_cache(args.url, inst_id, batch_keys, trace_id, args.timeout)
            print(f"[{idx}/{len(all_batches)}] OK instance={inst_id} "
                  f"trace_id={trace_id} block_keys={batch_keys} resp={result}")
            success += 1
        except requests.RequestException as e:
            print(f"[{idx}/{len(all_batches)}] FAIL instance={inst_id} "
                  f"trace_id={trace_id} block_keys={batch_keys} error={e}",
                  file=sys.stderr)
            failed += 1

    print("-" * 60)
    print(f"完成: 成功 {success}, 失败 {failed}")


if __name__ == "__main__":
    main()
