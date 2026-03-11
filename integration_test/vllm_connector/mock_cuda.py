"""CUDA 操作 Mock，支持 CPU 模式运行

该模块提供 CUDA 相关操作的 Mock 实现，使得 vLLM Connector 可以在没有 GPU 的环境下运行测试。
"""

from unittest.mock import MagicMock


class MockCudaStream:
    """Mock torch.cuda.Stream
    
    提供 CUDA Stream 的基本接口，但不执行任何实际操作。
    """
    
    def __init__(self, device=None, priority=0):
        self._device = device
        self._priority = priority
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def synchronize(self):
        """同步操作，在 Mock 中为空操作"""
        pass
    
    def wait_event(self, event):
        """等待事件，在 Mock 中为空操作"""
        pass
    
    def wait_stream(self, stream):
        """等待另一个 stream，在 Mock 中为空操作"""
        pass
    
    def record_event(self, event=None):
        """记录事件"""
        if event is None:
            event = MockCudaEvent()
        return event
    
    def query(self):
        """查询 stream 是否完成，总是返回 True"""
        return True


class MockCudaEvent:
    """Mock torch.cuda.Event
    
    提供 CUDA Event 的基本接口，但不执行任何实际操作。
    """
    
    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        self._enable_timing = enable_timing
        self._blocking = blocking
        self._interprocess = interprocess
        self._recorded = False
    
    def record(self, stream=None):
        """记录事件"""
        self._recorded = True
    
    def wait(self, stream=None):
        """等待事件完成，在 Mock 中为空操作"""
        pass
    
    def synchronize(self):
        """同步等待事件完成，在 Mock 中为空操作"""
        pass
    
    def query(self):
        """查询事件是否完成，总是返回 True"""
        return True
    
    def elapsed_time(self, end_event):
        """返回两个事件之间的时间，在 Mock 中返回 0"""
        return 0.0


def apply_cuda_patches():
    """应用 CUDA patches，使代码可以在 CPU 模式下运行
    
    该函数应该在导入任何使用 CUDA 的模块之前调用。
    它会替换 torch.cuda 模块中的关键类和函数。
    """
    import torch
    
    # 替换 Stream 和 Event 类
    torch.cuda.Stream = MockCudaStream
    torch.cuda.Event = MockCudaEvent
    
    # 替换常用函数
    torch.cuda.current_stream = MagicMock(return_value=MockCudaStream())
    torch.cuda.default_stream = MagicMock(return_value=MockCudaStream())
    torch.cuda.set_device = MagicMock()
    torch.cuda.get_device_properties = MagicMock(return_value=MagicMock(
        name="MockGPU",
        total_memory=16 * 1024 * 1024 * 1024,  # 16GB
        major=8,
        minor=0,
    ))
    
    # 设置 CUDA 不可用，但某些代码可能需要它返回 True
    # 这里我们保持它返回 False，让代码走 CPU 路径
    torch.cuda.is_available = MagicMock(return_value=False)
    
    # Mock synchronize
    torch.cuda.synchronize = MagicMock()


def get_mock_tensor_model_parallel_rank():
    """返回 Mock 的 tensor model parallel rank"""
    return 0


def apply_distributed_patches():
    """应用分布式相关的 patches
    
    替换 vLLM 的分布式函数，使其在单机测试环境下工作。
    """
    try:
        from vllm import distributed
        distributed.get_tensor_model_parallel_rank = get_mock_tensor_model_parallel_rank
    except ImportError:
        pass
