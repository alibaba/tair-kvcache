"""Text trace converter - 文本对话转trace (整合tokenizer + anonymizer)"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from converters.base import BaseConverter
from utils.prefix_hash import tokens_to_block_ids
from utils.tokenizer_loader import get_tokenizer, smart_tokenize


# ========================================
# 进程级 tokenizer 缓存
# ========================================
_process_tokenizer_cache_text = {}


def _get_cached_tokenizer_text(tokenizer_path: str, model_name: Optional[str]) -> Any:
    """获取缓存的 tokenizer（进程级缓存）"""
    global _process_tokenizer_cache_text
    
    cache_key = (tokenizer_path or '', model_name or '')
    
    if cache_key not in _process_tokenizer_cache_text:
        # 只在第一次加载时打印（每个进程一次）
        import os
        pid = os.getpid()
        print(f"[Process {pid}] Loading tokenizer for text format...")
        
        _process_tokenizer_cache_text[cache_key] = get_tokenizer(
            tokenizer_path=tokenizer_path,
            model_name=model_name
        )
        
        print(f"[Process {pid}] Tokenizer loaded successfully")
    
    return _process_tokenizer_cache_text[cache_key]


def _process_chunk_text(
    lines: List[str],
    tokenizer_path: str,
    model_name: Optional[str],
    block_size: int,
    mode: str,
    default_instance_id: str,
    time_field: str,
    content_field: str
) -> List[dict]:
    """
    处理一个chunk的文本行（在独立进程中运行）
    
    Args:
        lines: 文本行列表
        tokenizer_path: tokenizer路径
        model_name: 模型名称
        block_size: block大小
        mode: 输出模式
        default_instance_id: 默认instance ID
        time_field: 时间字段名
        content_field: 内容字段名
    
    Returns:
        trace列表
    """
    # 获取缓存的 tokenizer（每个进程只加载一次）
    tokenizer = _get_cached_tokenizer_text(tokenizer_path=tokenizer_path, model_name=model_name)
    
    # 创建临时converter（用于生成trace）
    converter = TextTraceConverter(
        default_instance_id=default_instance_id,
        instance_block_sizes={default_instance_id: block_size},
        mode=mode,
        tokenizer_path=tokenizer_path,
        model_name=model_name,
        time_field=time_field,
        content_field=content_field,
        num_workers=1  # 子进程不再并行
    )
    converter.tokenizer = tokenizer  # 复用已加载的tokenizer
    
    traces = []
    for line in lines:
        if not line:
            continue
        
        try:
            data = json.loads(line)
            
            # 提取时间戳
            ts_str = data.get(time_field, '')
            ts_str = ts_str.replace(',', '.')
            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
            timestamp_us = int(dt.timestamp() * 1000000)
            
            # 提取内容并tokenize
            content = data.get(content_field)
            if content is None:
                continue
            
            token_ids = smart_tokenize(tokenizer, content, use_chat_template=True)
            
            # 转换为block IDs
            block_ids = tokens_to_block_ids(token_ids, block_size=block_size)
            
            # 根据模式生成traces
            # tokens字段由BaseConverter根据keep_tokens参数自动控制
            if mode == 'optimizer':
                # Optimizer模式: Get+Write
                get_trace = converter._create_get_trace(
                    timestamp_us=timestamp_us,
                    keys=block_ids,
                    instance_id=default_instance_id,
                    tokens=token_ids
                )
                write_trace = converter._create_write_trace(
                    timestamp_us=timestamp_us + 1,
                    keys=block_ids,
                    instance_id=default_instance_id,
                    tokens=token_ids
                )
                traces.extend([get_trace, write_trace])
            else:
                # Inference模式: DialogTurn
                input_len = len(block_ids) * block_size
                dialog_trace = converter._create_dialog_trace(
                    timestamp_us=timestamp_us,
                    keys=block_ids,
                    input_len=input_len,
                    output_len=0,
                    total_keys=block_ids,
                    instance_id=default_instance_id,
                    tokens=token_ids
                )
                traces.append(dialog_trace)
        
        except Exception:
            # 静默跳过错误行（避免进程间输出混乱）
            continue
    
    return traces


class TextTraceConverter(BaseConverter):
    """
    文本trace转换器

    整合了tokenizer + anonymizer功能

    注意: 文本trace一般本身没有instance概念,所有trace使用default_instance_id。如果输入文件有instance信息,则使用instance_block_sizes进行分组。
    """

    def __init__(
        self,
        default_instance_id: str = 'instance',
        instance_block_sizes: Dict[str, int] = None,
        mode: str = 'optimizer',
        tokenizer_path: str = None,
        model_name: Optional[str] = None,
        time_field: str = 'time',
        content_field: str = 'prompt_messages',
        num_workers: int = None,
        keep_tokens: bool = False,
        **kwargs  # 忽略其他参数（如 model_mapping）
    ):
        super().__init__(default_instance_id, instance_block_sizes, mode, keep_tokens)
        self.block_size = self.get_block_size(default_instance_id)  # 获取该instance的block_size
        self.time_field = time_field
        self.content_field = content_field
        self.tokenizer_path = tokenizer_path
        self.model_name = model_name
        
        # 多进程配置
        self.num_workers = num_workers if num_workers else max(1, cpu_count() - 1)

        # 加载tokenizer (自动使用HF_ENDPOINT环境变量)
        self.tokenizer = get_tokenizer(
            tokenizer_path=tokenizer_path,
            model_name=model_name
        )

    def convert_to_traces(self, input_file: str) -> list:
        """转换文本trace为traces列表（支持多进程加速）"""
        # 读取所有行
        with open(input_file, 'r', encoding='utf-8') as f_in:
            lines = [line.strip() for line in f_in if line.strip()]
        
        total_lines = len(lines)
        print(f"Total lines: {total_lines}")
        print(f"Using {self.num_workers} workers for parallel tokenization")
        
        # 分块处理（确保 chunk 数 = worker 数）
        num_chunks = min(self.num_workers, total_lines)
        chunk_size = total_lines // num_chunks
        
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            # 最后一个chunk包含剩余的所有行
            end = total_lines if i == num_chunks - 1 else (i + 1) * chunk_size
            chunks.append(lines[start:end])
        
        print(f"Split into {len(chunks)} chunks for {self.num_workers} workers (avg {chunk_size} records/chunk)")
        
        # 多进程处理
        process_func = functools.partial(
            _process_chunk_text,
            tokenizer_path=self.tokenizer_path,
            model_name=self.model_name,
            block_size=self.block_size,
            mode=self.mode,
            default_instance_id=self.default_instance_id,
            time_field=self.time_field,
            content_field=self.content_field
        )
        
        print(f"\nProcessing {total_lines} records with {self.num_workers} workers...")
        print(f"(Each worker processes ~{chunk_size} records)\n")
        
        # 使用 ProcessPoolExecutor 替代 Pool，方便检查进度
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import time
        
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(process_func, chunk): i 
                      for i, chunk in enumerate(chunks)}
            
            # 收集结果（显示进度）
            results = [None] * len(chunks)
            completed = 0
            
            for future in as_completed(futures):
                chunk_idx = futures[future]
                results[chunk_idx] = future.result()
                completed += 1
                
                elapsed = time.time() - start_time
                percent = completed / len(chunks) * 100
                speed = (completed * chunk_size) / elapsed if elapsed > 0 else 0
                
                print(f"  [{completed}/{len(chunks)} workers] {percent:.0f}% "
                      f"({elapsed:.1f}s, ~{speed:.0f} records/s)", 
                      end='\r')
            
            print()  # 换行
        
        # 合并结果
        print("\nMerging traces from all workers...")
        all_traces = []
        for chunk_traces in tqdm(results, desc="Merging", unit="chunk", ncols=80, leave=False):
            all_traces.extend(chunk_traces)
        
        # 按timestamp排序（保证输出有序，尤其多进程时）
        print("Sorting traces by timestamp...")
        all_traces.sort(key=lambda t: t.get('timestamp_us', 0))
        
        total_elapsed = time.time() - start_time
        print(f"\n✅ Completed: {len(all_traces)} traces in {total_elapsed:.1f}s "
              f"({len(all_traces)/total_elapsed:.0f} traces/s)")
        
        return all_traces

    def _extract_timestamp(self, data: dict) -> int:
        """提取并转换时间戳"""
        ts_str = data.get(self.time_field, '')

        # 支持格式: "YYYY-MM-DD HH:MM:SS.ffffff"
        ts_str = ts_str.replace(',', '.')
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")

        return int(dt.timestamp() * 1000000)

    def _extract_content(self, data: dict):
        """提取内容字段"""
        content = data.get(self.content_field)
        if content is None:
            raise ValueError(f"Missing field: {self.content_field}")
        return content

    def _tokenize_content(self, content) -> list:
        """对内容进行tokenize (使用智能tokenization)"""
        return smart_tokenize(self.tokenizer, content, use_chat_template=True)

    def _generate_optimizer_traces(self, timestamp_us: int, block_ids: list, tokens: list = None) -> list:
        """
        生成Optimizer格式的Get+Write traces
        
        Args:
            tokens: token IDs列表（由base层根据keep_tokens决定是否保留）
        """
        tokens = tokens or []
        
        # Get trace - 显式使用default_instance_id
        get_trace = self._create_get_trace(
            timestamp_us=timestamp_us,
            keys=block_ids,
            instance_id=self.default_instance_id,
            tokens=tokens
        )

        # Write trace (时间戳+1微秒) - 显式使用default_instance_id
        write_trace = self._create_write_trace(
            timestamp_us=timestamp_us + 1,
            keys=block_ids,
            instance_id=self.default_instance_id,
            tokens=tokens
        )

        return [get_trace, write_trace]

    def _generate_inference_trace(self, timestamp_us: int, block_ids: list, tokens: list = None) -> dict:
        """
        生成Inference格式的DialogTurn trace
        
        Args:
            tokens: token IDs列表（由base层根据keep_tokens决定是否保留）
        """
        tokens = tokens or []
        input_len = len(block_ids) * self.block_size

        # DialogTurn trace - 显式使用default_instance_id
        dialog_trace = self._create_dialog_trace(
            timestamp_us=timestamp_us,
            keys=block_ids,
            input_len=input_len,
            output_len=0,  # 文本输入没有输出
            total_keys=block_ids,
            instance_id=self.default_instance_id,
            tokens=tokens
        )

        return dialog_trace
