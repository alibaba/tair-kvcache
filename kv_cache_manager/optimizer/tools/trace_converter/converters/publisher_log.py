"""Publisher Log converter - 转换KVCacheManager Event Publisher日志"""

import json
import sys
from pathlib import Path
from typing import Dict
from collections import defaultdict
from tqdm import tqdm

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from converters.base import BaseConverter


class PublisherLogConverter(BaseConverter):
    """
    Publisher Log转换器

    处理GetCacheLocation、StartWriteCache、FinishWriteCache事件
    并进行事件关联生成完整的DialogTurnSchemaTrace (inference模式)
    或Get+Write traces (optimizer模式)

    自动识别日志中的所有instance (通过source字段)
    """

    def __init__(self, default_instance_id: str = 'instance',
                 instance_block_sizes: Dict[str, int] = None,
                 mode: str = 'optimizer',
                 keep_tokens: bool = False,
                 **kwargs):  # 忽略其他参数
        super().__init__(default_instance_id, instance_block_sizes, mode, keep_tokens)
        # 动态发现的instance -> block_size映射
        self.discovered_instances = {}
        self.pending_write_sessions = {}
        self.pending_get_location_traces = []

    def convert_to_traces(self, input_file: str) -> list:
        """转换Publisher日志为traces列表"""
        traces = []

        # 第一遍: 解析所有事件
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
            f.seek(0)

            for line in tqdm(f, total=total_lines, desc="Parsing Publisher Log"):
                result = self._parse_log_line(line.strip())
                if result:
                    # _parse_log_line可能返回单个trace(dict)或多个traces(list)
                    if isinstance(result, list):
                        traces.extend(result)
                    else:
                        traces.append(result)

        # 处理未匹配的pending事件
        self._check_and_convert_pending_write_event(traces)
        self._convert_pending_get_location_event(traces)

        # 按timestamp排序（保证输出有序）
        traces.sort(key=lambda t: t.get('timestamp_us', 0))

        # 打印统计信息
        if self.discovered_instances:
            print(f"\n✅ Discovered {len(self.discovered_instances)} instance(s):")
            for inst_id, blk_size in sorted(self.discovered_instances.items()):
                inst_traces = sum(1 for t in traces if isinstance(t, dict) and t.get('instance_id') == inst_id)
                print(f"   - {inst_id}: block_size={blk_size}, traces={inst_traces}")

        return traces

    def _parse_log_line(self, line: str):
        """解析单行日志"""
        if not line:
            return None

        try:
            data = json.loads(line)
            event_type = data.get('type', '')

            if event_type == 'GetCacheLocation':
                return self._convert_get_location_event(data)
            elif event_type == 'StartWriteCache':
                return self._convert_start_write_cache_event(data)
            elif event_type == 'FinishWriteCache':
                return self._convert_finish_write_cache_event(data)

        except json.JSONDecodeError as e:
            # 提供更详细的错误诊断
            line_preview = line[:100] + '...' if len(line) > 100 else line
            print(f"\n⚠️  Warning: JSON parse error at position {e.pos}")
            print(f"   Line length: {len(line)} chars")
            print(f"   Error: {e.msg}")
            print(f"   Preview: {line_preview}")
        except Exception as e:
            line_preview = line[:100] + '...' if len(line) > 100 else line
            print(f"\n⚠️  Warning: Failed to parse line: {e}")
            print(f"   Preview: {line_preview}")

        return None

    def _convert_get_location_event(self, data: dict):
        """转换GetCacheLocation事件"""
        # 从日志中提取instance_id
        instance_id = data.get('source', self.default_instance_id)

        # 自动记录新发现的instance
        if instance_id not in self.discovered_instances:
            block_size = self.get_block_size(instance_id)
            self.discovered_instances[instance_id] = block_size
            print(f"📌 Discovered instance: {instance_id} (block_size={block_size})")

        trace = {
            'instance_id': instance_id,
            'timestamp_us': data.get('trigger_time_us', 0),
            'keys': data.get('keys', []),
            'tokens': data.get('tokens', []),  # 提取tokens字段
            'query_type': data.get('query_type', 'prefix_match'),
            'location_spec_names': data.get('location_spec_names', []),
            'block_mask': data.get('block_mask', []),
            'sw_size': data.get('sw_size', 0),
        }

        self.pending_get_location_traces.append(trace)
        return None  # 不直接输出,等待匹配

    def _convert_start_write_cache_event(self, data: dict):
        """转换StartWriteCache事件 - 仅保存到pending,不输出"""
        write_session_id = data.get('write_session_id', '')

        # 从日志中提取instance_id
        instance_id = data.get('source', self.default_instance_id)

        # 自动记录新发现的instance
        if instance_id not in self.discovered_instances:
            block_size = self.get_block_size(instance_id)
            self.discovered_instances[instance_id] = block_size
            print(f"📌 Discovered instance: {instance_id} (block_size={block_size})")

        write_trace = {
            'instance_id': instance_id,
            'timestamp_us': data.get('trigger_time_us', 0),
            'keys': data.get('keys', []),
            'tokens': data.get('tokens', []),
        }

        if write_session_id:
            # 使用 (instance_id, session_id) 作为key，避免不同instance的session_id冲突
            key = (instance_id, write_session_id)
            self.pending_write_sessions[key] = write_trace

        return None

    def _convert_finish_write_cache_event(self, data: dict):
        """转换FinishWriteCache事件 - 更新时间戳并匹配Get"""
        write_session_id = data.get('write_session_id', '')
        instance_id = data.get('source', self.default_instance_id)
        
        if not write_session_id:
            return None
        
        # 使用 (instance_id, session_id) 作为key查找
        key = (instance_id, write_session_id)
        if key not in self.pending_write_sessions:
            return None
        
        write_trace = self.pending_write_sessions.pop(key)
        # 更新为FinishWrite的时间戳
        write_trace['timestamp_us'] = data.get('trigger_time_us', 0)
        
        # 尝试匹配GetLocation并生成traces
        return self._find_matching_get_location_trace(write_trace)

    def _find_matching_get_location_trace(self, write_trace: dict):
        """匹配GetLocation和WriteCache生成DialogTurn或Get+Write"""
        write_keys = write_trace['keys']
        write_timestamp = write_trace['timestamp_us']
        write_instance = write_trace['instance_id']

        # 反向查找匹配的GetLocation
        for i in range(len(self.pending_get_location_traces) - 1, -1, -1):
            get_trace = self.pending_get_location_traces[i]

            # 检查instance_id和时间戳
            if get_trace['instance_id'] != write_instance:
                continue
            if get_trace['timestamp_us'] > write_timestamp:
                continue

            get_keys = get_trace['keys']
            if not get_keys:
                continue

            # 检查key前缀匹配
            match_len = len(get_keys) - 1
            if len(write_keys) >= match_len:
                if write_keys[:match_len] == get_keys[:match_len]:
                    # 找到匹配
                    block_size = self.discovered_instances.get(write_instance, self.default_block_size)

                    if self.mode == 'optimizer':
                        # Optimizer模式: 生成Get+Write
                        result = self._generate_optimizer_traces_from_match(
                            get_trace, write_trace, block_size
                        )
                    else:
                        # Inference模式: 生成DialogTurn
                        result = self._generate_inference_trace_from_match(
                            get_trace, write_trace, block_size
                        )

                    # 从pending中移除
                    del self.pending_get_location_traces[i]
                    return result

        return None

    def _generate_optimizer_traces_from_match(
        self,
        get_trace: dict,
        write_trace: dict,
        block_size: int
    ) -> list:
        """从匹配的Get和Write生成Optimizer格式traces"""
        instance_id = get_trace['instance_id']

        # Get trace (保留原始时间戳和instance_id)
        get_result = self._create_get_trace(
            timestamp_us=get_trace['timestamp_us'],
            keys=get_trace['keys'],
            instance_id=instance_id,
            tokens=get_trace.get('tokens', []),
            query_type=get_trace.get('query_type', 'prefix_match'),
            block_mask=get_trace.get('block_mask', []),
            sw_size=get_trace.get('sw_size', 0),
            location_spec_names=get_trace.get('location_spec_names', [])
        )

        # Write trace (保留原始时间戳和instance_id)
        write_result = self._create_write_trace(
            timestamp_us=write_trace['timestamp_us'],
            keys=write_trace['keys'],
            instance_id=instance_id,
            tokens=write_trace.get('tokens', [])
        )

        return [get_result, write_result]

    def _generate_inference_trace_from_match(
        self,
        get_trace: dict,
        write_trace: dict,
        block_size: int
    ) -> dict:
        """从匹配的Get和Write生成Inference格式DialogTurn"""
        instance_id = get_trace['instance_id']
        get_keys = get_trace['keys']
        write_keys = write_trace['keys']
        match_len = len(get_keys) - 1

        dialog_trace = self._create_dialog_trace(
            timestamp_us=write_trace['timestamp_us'],
            keys=get_keys,
            input_len=len(get_keys) * block_size,
            output_len=(len(write_keys) - match_len) * block_size,
            total_keys=write_keys,
            instance_id=instance_id,
            tokens=get_trace.get('tokens', []),
            query_type=get_trace.get('query_type', 'prefix_match'),
            block_mask=get_trace.get('block_mask', []),
            sw_size=get_trace.get('sw_size', 0),
            location_spec_names=get_trace.get('location_spec_names', [])
        )

        return dialog_trace

    def _check_and_convert_pending_write_event(self, traces: list):
        """处理未完成的write事件 - 尝试匹配Get trace"""
        for write_trace in self.pending_write_sessions.values():
            # 假设结束时间比开始时间晚100ms
            write_trace['timestamp_us'] += 100000

            # ✅ 尝试匹配Get trace
            result = self._find_matching_get_location_trace(write_trace)

            if result:
                # 成功匹配,添加Get+Write或DialogTurn
                if isinstance(result, list):
                    traces.extend(result)
                else:
                    traces.append(result)
            else:
                # 未匹配到Get,只输出Write trace (仅Optimizer模式)
                if self.mode == 'optimizer':
                    write_only = self._create_write_trace(
                        timestamp_us=write_trace['timestamp_us'],
                        keys=write_trace['keys'],
                        instance_id=write_trace['instance_id'],
                        tokens=write_trace.get('tokens', [])
                    )
                    traces.append(write_only)
                # Inference模式下,单独的write没有意义,跳过

        self.pending_write_sessions.clear()

    def _convert_pending_get_location_event(self, traces: list):
        """处理未匹配的GetLocation事件"""
        for get_trace in self.pending_get_location_traces:
            instance_id = get_trace['instance_id']
            block_size = self.discovered_instances.get(instance_id, self.default_block_size)

            if self.mode == 'optimizer':
                # Optimizer模式: 只生成Get trace
                result = self._create_get_trace(
                    timestamp_us=get_trace['timestamp_us'],
                    keys=get_trace['keys'],
                    instance_id=instance_id,
                    tokens=get_trace.get('tokens', []),
                    query_type=get_trace.get('query_type', 'prefix_match'),
                    block_mask=get_trace.get('block_mask', []),
                    sw_size=get_trace.get('sw_size', 0),
                    location_spec_names=get_trace.get('location_spec_names', [])
                )
                traces.append(result)
            else:
                # Inference模式: 生成DialogTurn (output_len=0)
                dialog_trace = self._create_dialog_trace(
                    timestamp_us=get_trace['timestamp_us'],
                    keys=get_trace['keys'],
                    input_len=len(get_trace['keys']) * block_size,
                    output_len=0,
                    total_keys=get_trace['keys'],
                    instance_id=instance_id,
                    tokens=get_trace.get('tokens', []),
                    query_type=get_trace.get('query_type', 'prefix_match'),
                    block_mask=get_trace.get('block_mask', []),
                    sw_size=get_trace.get('sw_size', 0),
                    location_spec_names=get_trace.get('location_spec_names', [])
                )
                traces.append(dialog_trace)

        self.pending_get_location_traces.clear()
