#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trace 文件合并与排序工具

职责:
- 合并多个JSONL文件（支持单文件，包括input==output场景）
- 流式归并：利用已排序文件，内存O(k)
- 直接拼接：无排序，更快

前提: converter保证输出文件已按timestamp_us排序
"""

import json
import heapq
import shutil
import tempfile
from pathlib import Path
from typing import List, Iterator, Tuple, Any


def _trace_iterator(file_path: Path) -> Iterator[Tuple[int, dict]]:
    """生成 (timestamp_us, trace) 元组"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            trace = json.loads(line)
            timestamp = trace.get('timestamp_us', 0)
            yield (timestamp, trace)


def _sorted_merge_impl(input_files: List[Path], output_file: Path) -> int:
    """
    流式归并（假设input != output且所有文件已排序）
    内存: O(k)，k=文件数
    """
    iterators = [_trace_iterator(f) for f in input_files]
    
    trace_count = 0
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for timestamp, trace in heapq.merge(*iterators, key=lambda x: x[0]):
            f_out.write(json.dumps(trace, ensure_ascii=False) + '\n')
            trace_count += 1
    
    return trace_count


def _sorted_merge(input_files: List[Path], output_file: Path) -> int:
    """流式归并（自动处理input==output场景）"""
    output_resolved = output_file.resolve()
    has_overlap = any(f.resolve() == output_resolved for f in input_files)
    
    if has_overlap:
        # 使用临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', 
            encoding='utf-8',
            delete=False, 
            suffix='.jsonl',
            dir=output_file.parent
        ) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            trace_count = _sorted_merge_impl(input_files, tmp_path)
            shutil.move(str(tmp_path), str(output_file))
            return trace_count
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    else:
        return _sorted_merge_impl(input_files, output_file)


def _concat_files_impl(input_files: List[Path], output_file: Path) -> int:
    """流式拼接（假设input != output）"""
    trace_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for input_file in input_files:
            with open(input_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    line = line.strip()
                    if line:
                        f_out.write(line + '\n')
                        trace_count += 1
    
    return trace_count


def _concat_files(input_files: List[Path], output_file: Path) -> int:
    """直接拼接（自动处理input==output场景）"""
    output_resolved = output_file.resolve()
    has_overlap = any(f.resolve() == output_resolved for f in input_files)
    
    if has_overlap:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8', 
            delete=False,
            suffix='.jsonl',
            dir=output_file.parent
        ) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            trace_count = _concat_files_impl(input_files, tmp_path)
            shutil.move(str(tmp_path), str(output_file))
            return trace_count
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
    else:
        return _concat_files_impl(input_files, output_file)


def merge_jsonl_files(
    input_files: List[Path],
    output_file: Path,
    sort_by_timestamp: bool = True
) -> int:
    """
    合并JSONL文件
    
    前提（sort模式）: 所有input文件已按timestamp_us排序
    """
    if not input_files:
        raise ValueError("input_files cannot be empty")
    
    for f in input_files:
        if not f.exists():
            raise FileNotFoundError(f"Input file not found: {f}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if sort_by_timestamp:
        return _sorted_merge(input_files, output_file) 
    else:
        return _concat_files(input_files, output_file)


def count_traces_in_file(file_path: Path) -> int:
    """统计JSONL文件中的trace数量"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())
