#!/usr/bin/env python3
"""
Redis扫描脚本 - 检查status为2或4且超过1小时未更新的key

用法:
    python scan_redis_stale_keys.py --host <host> --port <port> [--password <password>]

示例:
    python scan_redis_stale_keys.py -H localhost -p 6379
    python scan_redis_stale_keys.py -H r-xxx.redis.rds.aliyuncs.com -p 6379 -a your_password
    python scan_redis_stale_keys.py -H localhost -p 6379 -o json > result.json
    python scan_redis_stale_keys.py -H localhost -p 6379 -o csv > result.csv
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime

try:
    import redis
except ImportError:
    print("请先安装redis: pip install redis")
    sys.exit(1)


def format_timestamp(microseconds: int) -> str:
    """将微秒时间戳格式化为可读时间"""
    seconds = microseconds / 1_000_000
    return datetime.fromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S')


def format_duration(hours: float) -> str:
    """格式化时间差"""
    if hours < 24:
        return f"{hours:.2f} 小时"
    days = hours / 24
    return f"{days:.2f} 天"


def scan_stale_keys(r: redis.Redis, key_pattern: str = "*", hours_threshold: float = 1.0,
                    quiet: bool = False):
    """扫描Redis中status为2或4且超过指定时间的key"""
    
    current_time_us = int(time.time() * 1_000_000)
    threshold_us = int(hours_threshold * 3600 * 1_000_000)
    
    target_statuses = {2, 4}
    results = []
    scanned = 0
    matched = 0
    
    if not quiet:
        print(f"开始扫描，key模式: {key_pattern}")
        print(f"目标status: {target_statuses}")
        print(f"时间阈值: {hours_threshold} 小时")
        print("-" * 80)
    
    cursor = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=key_pattern, count=1000)
        if not keys:
            if cursor == 0:
                break
            continue
        
        scanned += len(keys)
        
        # pipeline批量获取type
        type_pipe = r.pipeline(transaction=False)
        for key in keys:
            type_pipe.type(key)
        types = type_pipe.execute()
        
        # 筛选hash类型的key
        hash_keys = [key for key, t in zip(keys, types) if t == b'hash']
        
        if hash_keys:
            # pipeline批量获取__uri__和__lru_time__字段
            hmget_pipe = r.pipeline(transaction=False)
            for key in hash_keys:
                hmget_pipe.hmget(key, '__uri__', '__lru_time__')
            hmget_results = hmget_pipe.execute()
            
            for key, fields in zip(hash_keys, hmget_results):
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                uri_value = fields[0]
                lru_time_value = fields[1]
                
                if not uri_value or not lru_time_value:
                    continue
                
                uri_str = uri_value.decode('utf-8') if isinstance(uri_value, bytes) else uri_value
                lru_time_str = lru_time_value.decode('utf-8') if isinstance(lru_time_value, bytes) else lru_time_value
                
                try:
                    lru_time_us = int(lru_time_str)
                    uri_data = json.loads(uri_str)
                except (ValueError, json.JSONDecodeError):
                    continue
                
                # 检查时间是否超过阈值
                time_diff_us = current_time_us - lru_time_us
                if time_diff_us < threshold_us:
                    continue
                
                # 检查每个item的status
                for item_key, item_value in uri_data.items():
                    if isinstance(item_value, dict):
                        status = item_value.get('status')
                        if status in target_statuses:
                            hours_ago = time_diff_us / (3600 * 1_000_000)
                            result = {
                                'key': key_str,
                                'status': status,
                                'lru_time': lru_time_us,
                                'lru_time_formatted': format_timestamp(lru_time_us),
                                'hours_ago': hours_ago,
                                'uri_parsed': uri_data
                            }
                            results.append(result)
                            matched += 1
                            
                            if not quiet:
                                print(f"[{matched}] Key: {key_str}")
                                print(f"    Status: {status}")
                                print(f"    LRU Time: {format_timestamp(lru_time_us)} ({format_duration(hours_ago)}前)")
                                print(f"    URI: {json.dumps(uri_data, ensure_ascii=False)}")
                                print()
                            break  # 一个key只打印一次
        
        if scanned % 10000 == 0 and scanned > 0:
            print(f"... 已扫描 {scanned} 个key，匹配 {matched} 个", file=sys.stderr)
        
        if cursor == 0:
            break
    
    return results, scanned


def output_csv(results):
    """以CSV格式输出结果"""
    writer = csv.writer(sys.stdout)
    writer.writerow(['key', 'status', 'lru_time', 'lru_time_formatted', 'hours_ago', 'uri_parsed'])
    for r in results:
        writer.writerow([
            r['key'],
            r['status'],
            r['lru_time'],
            r['lru_time_formatted'],
            f"{r['hours_ago']:.2f}",
            json.dumps(r['uri_parsed'], ensure_ascii=False)
        ])


def main():
    parser = argparse.ArgumentParser(
        description='扫描Redis中status为2或4且超过1小时的key'
    )
    parser.add_argument('--host', '-H', default='localhost', help='Redis host')
    parser.add_argument('--port', '-p', type=int, default=6379, help='Redis port')
    parser.add_argument('--password', '-a', default=None, help='Redis password')
    parser.add_argument('--db', '-n', type=int, default=0, help='Redis database number')
    parser.add_argument('--pattern', '-k', default='kvcache:*', help='Key匹配模式 (default: kvcache:*)')
    parser.add_argument('--hours', '-t', type=float, default=1.0, help='时间阈值(小时) (default: 1.0)')
    parser.add_argument('--output', '-o', choices=['table', 'json', 'csv'], default='table', help='输出格式')
    
    args = parser.parse_args()
    
    quiet = args.output in ('json', 'csv')
    
    if not quiet:
        print(f"连接 Redis: {args.host}:{args.port} db={args.db}")
    
    try:
        r = redis.Redis(
            host=args.host,
            port=args.port,
            password=args.password,
            db=args.db,
            socket_timeout=10,
            socket_connect_timeout=10
        )
        r.ping()
        if not quiet:
            print("连接成功\n")
    except redis.ConnectionError as e:
        print(f"连接失败: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        results, scanned = scan_stale_keys(r, args.pattern, args.hours, quiet=quiet)
        
        if args.output == 'json':
            print(json.dumps(results, indent=2, ensure_ascii=False))
        elif args.output == 'csv':
            output_csv(results)
        else:
            print("-" * 80)
            print(f"\n扫描完成!")
            print(f"  扫描key总数: {scanned}")
            print(f"  匹配key数量: {len(results)}")
            
    except KeyboardInterrupt:
        print("\n\n用户中断扫描", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
