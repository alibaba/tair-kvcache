"""Qwen Bailian converter - 转换Qwen Bailian开源数据集格式"""

import json
import sys
from pathlib import Path
from typing import Dict
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from converters.base import BaseConverter
from utils.prefix_hash import apply_prefix_hash


class QwenBailianConverter(BaseConverter):
    """
    Qwen Bailian数据集转换器

    输入格式示例:
    {
        "timestamp": 1704110400.0,
        "hash_ids": [123, 456, 789],  # 只包含input部分的hash
        "input_length": 48,
        "output_length": 32
    }

    注意:
    - 数据集本身没有instance概念,所有trace使用default_instance_id
    - hash_ids只包含input部分,output_length仅用于统计(无对应KV Cache)
    """

    def __init__(self, default_instance_id: str = 'instance',
                 instance_block_sizes: Dict[str, int] = None,
                 mode: str = 'optimizer',
                 **kwargs):  # 忽略其他参数
        super().__init__(default_instance_id, instance_block_sizes, mode)

    def convert(self, input_file: str, output_file: str) -> int:
        trace_count = 0

        with open(input_file, 'r', encoding='utf-8') as f_in:
            # 计算总行数用于进度条
            total_lines = sum(1 for _ in f_in)
            f_in.seek(0)

            with open(output_file, 'w', encoding='utf-8') as f_out:
                for line in tqdm(f_in, total=total_lines, desc="Converting Qwen Bailian"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)

                        # 提取字段
                        timestamp = data.get('timestamp', 0.0)
                        hash_ids = data.get('hash_ids', [])
                        input_length = data.get('input_length', 0)
                        output_length = data.get('output_length', 0)

                        # 应用前缀哈希转换
                        block_keys = apply_prefix_hash(hash_ids)

                        # 根据模式生成不同格式
                        if self.mode == 'optimizer':
                            # Optimizer模式: 生成Get+Write
                            traces = self._generate_optimizer_traces(
                                timestamp, block_keys, input_length, output_length
                            )
                            for trace in traces:
                                f_out.write(json.dumps(trace) + '\n')
                                trace_count += 1
                        else:
                            # Inference模式: 生成DialogTurn
                            trace = self._generate_inference_trace(
                                timestamp, block_keys, input_length, output_length
                            )
                            f_out.write(json.dumps(trace) + '\n')
                            trace_count += 1

                    except json.JSONDecodeError as e:
                        # 提供更详细的错误诊断
                        line_preview = line[:100] + '...' if len(line) > 100 else line
                        print(f"\n⚠️  Warning: JSON parse error at position {e.pos}")
                        print(f"   Line length: {len(line)} chars")
                        print(f"   Error: {e.msg}")
                        print(f"   Preview: {line_preview}")
                        continue
                    except Exception as e:
                        line_preview = line[:100] + '...' if len(line) > 100 else line
                        print(f"\n⚠️  Warning: Failed to parse line: {e}")
                        print(f"   Preview: {line_preview}")
                        continue

        return trace_count

    def _generate_optimizer_traces(
        self,
        timestamp: float,
        block_keys: list,
        input_length: int,
        output_length: int
    ) -> list:
        """
        生成Optimizer格式的Get+Write traces

        Args:
            timestamp: 原始时间戳(秒)
            block_keys: block keys (hash_ids已经是input部分,直接使用)
            input_length: 输入token数 (未使用,仅作记录)
            output_length: 输出token数 (未使用,仅作记录)

        Returns:
            [Get trace, Write trace]
        """
        base_timestamp_us = int(timestamp * 1000000)

        # hash_ids本身就是input的完整block keys,直接使用
        # Get trace (prefill阶段) - 显式使用default_instance_id
        get_trace = self._create_get_trace(
            timestamp_us=base_timestamp_us,
            keys=block_keys,
            instance_id=self.default_instance_id
        )

        # Write trace (prefill阶段, 时间戳+1微秒) - 显式使用default_instance_id
        write_trace = self._create_write_trace(
            timestamp_us=base_timestamp_us + 1,
            keys=block_keys,
            instance_id=self.default_instance_id
        )

        return [get_trace, write_trace]

    def _generate_inference_trace(
        self,
        timestamp: float,
        block_keys: list,
        input_length: int,
        output_length: int
    ) -> dict:
        """
        生成Inference格式的DialogTurn trace

        Args:
            timestamp: 原始时间戳(秒)
            block_keys: block keys (hash_ids已经是input部分)
            input_length: 输入token数
            output_length: 输出token数

        Returns:
            DialogTurn trace
        """
        base_timestamp_us = int(timestamp * 1000000)

        # hash_ids本身就是input的完整block keys,直接使用
        # DialogTurn trace - 显式使用default_instance_id
        dialog_trace = self._create_dialog_trace(
            timestamp_us=base_timestamp_us,
            keys=block_keys,
            input_len=input_length,
            output_len=output_length,
            total_keys=block_keys,
            instance_id=self.default_instance_id
        )

        return dialog_trace
