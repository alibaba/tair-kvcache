"""CUDA 操作 Mock，支持 CPU 模式运行

该模块提供 CUDA 相关操作的 Mock 实现，使得 vLLM Connector 可以在没有 GPU 的环境下运行测试。
"""

import sys
import types
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


class _TritonSubModule(types.ModuleType):
    """自动创建子模块的 triton mock 模块

    当访问不存在的属性时，自动创建并注册一个新的子模块，
    以支持 torch 内部 ``import triton.backends.compiler`` 等深层导入。
    """

    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []  # 标记为 package，允许子模块导入

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            if item == "__version__":
                return "0.0.0"  # torch._inductor 检查 triton.__version__
            raise AttributeError(item)
        full_name = f"{self.__name__}.{item}"
        import sys as _sys
        if full_name in _sys.modules:
            return _sys.modules[full_name]
        sub = _TritonSubModule(full_name)
        _sys.modules[full_name] = sub
        return sub

    def __call__(self, *args, **kwargs):
        """让 mock 模块可调用，inspect.signature 等场景需要"""
        return None


class _TritonImportFinder:
    """Meta path finder：自动为任何 triton.* 子模块创建 mock"""

    def find_module(self, fullname, path=None):
        if fullname == "triton" or fullname.startswith("triton."):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        mod = _TritonSubModule(fullname)
        sys.modules[fullname] = mod
        # 确保父模块引用子模块
        parts = fullname.rsplit(".", 1)
        if len(parts) == 2:
            parent_name, child_name = parts
            parent = sys.modules.get(parent_name)
            if parent is not None and not hasattr(parent, child_name):
                setattr(parent, child_name, mod)
        return mod


def _mock_triton():
    # """Mock triton 模块（仅在 triton 未安装时）
    #
    # 如果系统已安装真实的 triton，则跳过 mock，让真实 triton 正常工作。
    # 仅在 triton 不可用时提供最小化的 mock 以使导入链不中断。
    # """
    # try:
    import importlib
    importlib.import_module("triton")
    # 真实 triton 可用，无需 mock
    return
    # except ImportError:
    #     pass
    #
    # # triton 不可用，安装 mock
    # sys.meta_path.insert(0, _TritonImportFinder())
    #
    # triton_mod = _TritonSubModule("triton")
    # triton_lang = _TritonSubModule("triton.language")
    #
    # def _jit_decorator(fn=None, **kwargs):
    #     if fn is not None:
    #         return fn
    #     return lambda f: f
    #
    # triton_mod.jit = _jit_decorator
    # triton_mod.cdiv = lambda a, b: (a + b - 1) // b
    # triton_mod.language = triton_lang
    #
    # triton_lang.constexpr = int
    # triton_lang.dtype = type("dtype", (), {})
    # triton_lang.static_print = lambda *args, **kwargs: None
    # triton_lang.program_id = MagicMock(return_value=0)
    # triton_lang.num_programs = MagicMock(return_value=1)
    # triton_lang.arange = MagicMock()
    # triton_lang.load = MagicMock()
    # triton_lang.store = MagicMock()
    #
    # sys.modules["triton"] = triton_mod
    # sys.modules["triton.language"] = triton_lang


def _mock_batch_gather_scatter():
    """Mock batch_gather_scatter_helper 模块中的 GPU 内核函数为空操作"""
    mod_name = "kv_cache_manager.py_connector.kernel.batch_gather_scatter_helper"

    def batch_scatter_kv_caches(*args, **kwargs):
        """空操作：在 CPU 环境下不执行实际的 GPU scatter"""
        pass

    def batch_gather_kv_caches(*args, **kwargs):
        """空操作：在 CPU 环境下不执行实际的 GPU gather"""
        pass

    # 创建 mock 模块并注册到 sys.modules
    mock_mod = types.ModuleType(mod_name)
    mock_mod.batch_scatter_kv_caches = batch_scatter_kv_caches
    mock_mod.batch_gather_kv_caches = batch_gather_kv_caches
    sys.modules[mod_name] = mock_mod

    # 同时确保父包路径存在
    parent = "kv_cache_manager.py_connector.kernel"
    if parent in sys.modules:
        sys.modules[parent].batch_gather_scatter_helper = mock_mod


def apply_cuda_patches():
    """应用 CUDA patches，使代码可以在 CPU 模式下运行

    该函数应该在导入任何使用 CUDA 的模块之前调用。
    它会替换 torch.cuda 模块中的关键类和函数。
    """
    # Mock triton 模块（batch_gather_scatter_helper 依赖 triton）
    # 必须在 import torch/vllm 触发 torch._inductor 导入 triton 之前完成
    _mock_triton()

    # Mock batch_gather_scatter_helper 中的 GPU 内核函数为空操作
    _mock_batch_gather_scatter()

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
