"""Base converter interface for all trace converters"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Set


class BaseConverter(ABC):
    """所有trace转换器的基类"""

    def __init__(self, default_instance_id: str = 'instance',
                 instance_block_sizes: Dict[str, int] = None,
                 mode: str = 'optimizer'):
        """
        Args:
            default_instance_id: 默认实例ID (当输入格式没有instance信息时使用)
            instance_block_sizes: 每个instance的block_size映射 {instance_id: block_size}
            mode: 输出模式 ('optimizer' 或 'inference')
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

    @abstractmethod
    def convert(self, input_file: str, output_file: str) -> int:
        """
        转换trace文件为标准格式

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径

        Returns:
            转换的trace数量
        """
        pass

    def _allocate_timestamp(self, instance_id: str, base_timestamp: int) -> int:
        """
        按instance分配时间戳,自动处理冲突

        Args:
            instance_id: 实例ID
            base_timestamp: 基础时间戳

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
        timestamp_us: int,
        keys: list,
        instance_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建GetLocationSchemaTrace (optimizer模式)

        Args:
            timestamp_us: 微秒时间戳
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
        timestamp_us = self._allocate_timestamp(instance_id, timestamp_us)

        trace = {
            'type': 'get',  # 显式标记为Get trace
            # OptimizerSchemaTrace 基础字段
            'instance_id': instance_id,
            'trace_id': f"trace_{instance_id}_{timestamp_us}",
            'timestamp_us': timestamp_us,
            'tokens': kwargs.get('tokens', []),
            'keys': keys,

            # GetLocationSchemaTrace 字段
            'query_type': kwargs.get('query_type', 'prefix_match'),
            'block_mask': kwargs.get('block_mask', []),
            'sw_size': kwargs.get('sw_size', 0),
            'location_spec_names': kwargs.get('location_spec_names', []),
        }

        return trace

    def _create_write_trace(
        self,
        timestamp_us: int,
        keys: list,
        instance_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建WriteCacheSchemaTrace (optimizer模式)

        Args:
            timestamp_us: 微秒时间戳
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
        timestamp_us = self._allocate_timestamp(instance_id, timestamp_us)

        trace = {
            'type': 'write',  # 显式标记为Write trace
            # OptimizerSchemaTrace 基础字段
            'instance_id': instance_id,
            'trace_id': f"trace_{instance_id}_{timestamp_us}",
            'timestamp_us': timestamp_us,
            'tokens': kwargs.get('tokens', []),
            'keys': keys,
        }

        return trace

    def _create_dialog_trace(
        self,
        timestamp_us: int,
        keys: list,
        input_len: int,
        output_len: int,
        total_keys: list,
        instance_id: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建DialogTurnSchemaTrace (inference模式)

        Args:
            timestamp_us: 微秒时间戳
            keys: prefill阶段的block ID列表
            input_len: 输入token数
            output_len: 输出token数
            total_keys: 完整的key列表 (prefill + decode)
            instance_id: 实例ID (None则使用default_instance_id)
            **kwargs: 其他可选字段

        Returns:
            DialogTurnSchemaTrace字典
        """
        # 使用指定的instance_id或默认值
        if instance_id is None:
            instance_id = self.default_instance_id

        # 分配无冲突的时间戳
        timestamp_us = self._allocate_timestamp(instance_id, timestamp_us)

        trace = {
            'type': 'dialog',  # 显式标记为DialogTurn trace
            # OptimizerSchemaTrace 基础字段
            'instance_id': instance_id,
            'trace_id': f"trace_{instance_id}_{timestamp_us}",
            'timestamp_us': timestamp_us,
            'tokens': kwargs.get('tokens', []),
            'keys': keys,

            # GetLocationSchemaTrace 字段
            'query_type': kwargs.get('query_type', 'prefix_match'),
            'block_mask': kwargs.get('block_mask', []),
            'sw_size': kwargs.get('sw_size', 0),
            'location_spec_names': kwargs.get('location_spec_names', []),

            # DialogTurnSchemaTrace 字段
            'input_len': input_len,
            'output_len': output_len,
            'total_keys': total_keys,
        }

        return trace
