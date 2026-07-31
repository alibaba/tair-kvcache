"""Base converter interface for all trace converters"""

from abc import ABC
from typing import Dict, Any, Set


class BaseConverter(ABC):
    """所有trace转换器的基类"""

    def __init__(self, default_instance_id: str = 'instance',
                 instance_block_sizes: Dict[str, int] = None,
                 mode: str = 'optimizer',
                 **kwargs):
        """
        Args:
            default_instance_id: 默认实例ID (当输入格式没有instance信息时使用)
            instance_block_sizes: 每个instance的block_size映射 {instance_id: block_size}
            mode: 输出模式 ('optimizer')
        """
        self.default_instance_id = default_instance_id
        self.default_block_size = 16  # 硬编码默认值
        self.instance_block_sizes = instance_block_sizes or {}
        self.mode = mode
        # 按instance分组管理时间戳,避免不同instance间的冲突
        self.used_timestamps: Dict[str, Set[int]] = {}

    def get_block_size(self, instance_id: str) -> int:
        """
        获取指定instance的block_size

        Args:
            instance_id: 实例ID

        Returns:
            该instance的block_size,未指定则返回默认值16
        """
        return self.instance_block_sizes.get(instance_id, self.default_block_size)

    def _truncate_keys_to_full_blocks(self, keys: list, input_len: int, instance_id: str) -> list:
        block_size = self.get_block_size(instance_id)
        if block_size <= 0:
            raise ValueError(f"block_size must be positive for instance {instance_id}")
        return list(keys[:input_len // block_size])

    def convert_to_traces(self, input_file: str) -> list:
        """
        转换 trace 文件为 traces 列表。

        Converter 需要实现 convert_to_traces() 或 convert_to_trace_jsonl()
        之一。实现本方法时,文件写出由 trace_converter.py 和
        merge_utils.py 统一处理。

        Args:
            input_file: 输入文件路径

        Returns:
            traces 列表
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement convert_to_traces() "
            "or convert_to_trace_jsonl()"
        )

    def convert_to_trace_jsonl(self, input_file: str, output_file: str) -> int:
        """
        流式转换 trace 文件并直接写出标准 trace JSONL。

        Converter 需要实现 convert_to_traces() 或 convert_to_trace_jsonl()
        之一。实现本方法时,converter 负责完整写出 output_file,每行一个
        Optimizer trace JSON object。

        trace_converter.py 不会对本方法写出的单文件内容做额外排序或顺序校验。
        如果调用方需要特定顺序,由 converter 实现自行保证。
        trace_converter.py 可能传入临时 output_file,并在本方法成功返回后
        原子替换到最终输出路径;converter 只应写入传入的 output_file。

        返回值 contract:
        - 必须返回非负 int,表示写出的 trace 条数。
        - 返回其他类型会被视为 converter contract 错误。

        Args:
            input_file: 输入文件路径或 URL
            output_file: 输出 JSONL 文件路径
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement convert_to_traces() "
            "or convert_to_trace_jsonl()"
        )

    def _allocate_timestamp(self, instance_id: str, base_timestamp: int) -> int:
        """
        按instance分配时间戳,自动处理冲突

        Args:
            instance_id: 实例ID
            base_timestamp: 基础时间戳（纳秒）

        Returns:
            无冲突的时间戳
        """
        # 为新instance初始化时间戳集合
        if instance_id not in self.used_timestamps:
            self.used_timestamps[instance_id] = set()

        timestamp = base_timestamp
        while timestamp in self.used_timestamps[instance_id]:
            timestamp += 1
        self.used_timestamps[instance_id].add(timestamp)
        return timestamp

    def _create_get_trace(
        self,
        timestamp_ns: int,
        keys: list,
        instance_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建GetLocationSchemaTrace (optimizer模式)

        Args:
            timestamp_ns: 纳秒时间戳
            keys: block ID列表
            instance_id: 实例ID (None则使用default_instance_id)
            **kwargs: 其他可选字段

        Returns:
            GetLocationSchemaTrace字典
        """
        # 使用指定的instance_id或默认值
        if instance_id is None:
            instance_id = self.default_instance_id

        # 分配无冲突的时间戳
        timestamp_ns = self._allocate_timestamp(instance_id, timestamp_ns)

        if 'input_len' not in kwargs:
            raise ValueError("optimizer get trace requires input_len")
        input_len = int(kwargs['input_len'])
        if input_len <= 0:
            raise ValueError("optimizer get trace input_len must be positive")
        keys = self._truncate_keys_to_full_blocks(keys, input_len, instance_id)
        block_mask = kwargs.get('block_mask', [])
        if isinstance(block_mask, list) and len(block_mask) > len(keys):
            block_mask = block_mask[:len(keys)]

        trace = {
            'type': 'get',  # 显式标记为Get trace
            # OptimizerSchemaTrace 基础字段
            'instance_id': instance_id,
            'trace_id': f"trace_{instance_id}_{timestamp_ns}",
            'timestamp_ns': timestamp_ns,
            'keys': keys,
            'input_len': input_len,

            # GetLocationSchemaTrace 字段
            'query_type': kwargs.get('query_type', 'prefix_match'),
            'block_mask': block_mask,
            'sw_size': kwargs.get('sw_size', 0),
            'location_spec_names': kwargs.get('location_spec_names', []),
        }

        return trace

    def _create_write_trace(
        self,
        timestamp_ns: int,
        keys: list,
        instance_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建WriteCacheSchemaTrace (optimizer模式)

        Args:
            timestamp_ns: 纳秒时间戳
            keys: block ID列表
            instance_id: 实例ID (None则使用default_instance_id)
            **kwargs: 其他可选字段

        Returns:
            WriteCacheSchemaTrace字典
        """
        # 使用指定的instance_id或默认值
        if instance_id is None:
            instance_id = self.default_instance_id

        # 分配无冲突的时间戳
        timestamp_ns = self._allocate_timestamp(instance_id, timestamp_ns)

        trace = {
            'type': 'write',  # 显式标记为Write trace
            # OptimizerSchemaTrace 基础字段
            'instance_id': instance_id,
            'trace_id': f"trace_{instance_id}_{timestamp_ns}",
            'timestamp_ns': timestamp_ns,
            'keys': keys,
        }

        return trace
