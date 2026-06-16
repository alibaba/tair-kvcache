"""
Compatibility layer for optional dependencies (kunlun_commons, deepestim).

When these packages are not installed, provides None placeholders.
Modules using these should check availability before use.
"""
import logging

_logger = logging.getLogger("schedule_simulator")

try:
    from kunlun_commons.model_info import ModelInfo
    from kunlun_commons.hardwares.accelerator import AcceleratorInfo
    from kunlun_commons.data_type import DataType
    from kunlun_commons.system_info import AcceleratorInfo as SysAcceleratorInfo
    from kunlun_commons.utils import get_logger
    from kunlun_commons.utils.logger import get_logger as get_logger_v2
    HAS_KUNLUN_COMMONS = True
except ImportError:
    ModelInfo = None
    AcceleratorInfo = None
    SysAcceleratorInfo = None
    DataType = None
    HAS_KUNLUN_COMMONS = False

    def get_logger(name="schedule_simulator"):
        return logging.getLogger(name)

    get_logger_v2 = get_logger

try:
    from deepestim.llmperf import TheoreticalExecutionAnalyzer, RequestInfo, InferenceConfig
    from deepestim.llmperf.analyzers.theoretical_execution_analyzer import (
        TheoreticalExecutionAnalyzer as TEA_v2,
    )
    from deepestim.llmperf.types import InferenceConfig as IC_v2
    HAS_DEEPESTIM = True
except ImportError:
    TheoreticalExecutionAnalyzer = None
    RequestInfo = None
    InferenceConfig = None
    TEA_v2 = None
    IC_v2 = None
    HAS_DEEPESTIM = False
