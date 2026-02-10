import json

from hisim.spec import ModelInfo, AcceleratorInfo, DataType
from hisim.simulation.types import PlatformConfig, SchedulerConfig
from hisim.simulation.manager import Envs
from hisim.simulation.utils import (
    calc_kv_cache_cell_elems,
    calc_kv_cache_per_layer_elems,
)
from hisim.time_predictor import (
    InferTimePredictor,
    AIConfiguratorTimePredictor,
)
from hisim.utils import get_logger


logger = get_logger()


class ConfigManager:
    _model_info: ModelInfo = None
    _platform_config: PlatformConfig = None
    _scheduler_config: SchedulerConfig = None

    @classmethod
    def set_model_info(cls, model: ModelInfo):
        cls._model_info = model

    @classmethod
    def get_model_info(cls, hf_config: dict | None) -> ModelInfo:
        if hf_config is not None:
            model = ModelInfo.from_config(hf_config)
            if model is None:
                logger.error(
                    f"Failed to initialize model information with configuration: {hf_config}"
                )
        else:
            with open(Envs.config_path()) as f:
                config: dict = json.load(f)
            model = ModelInfo.find_by_model_name(config.get("model", {}).get("name"))

        return model

    @classmethod
    def get_accelerator_info(cls) -> AcceleratorInfo:
        with open(Envs.config_path()) as f:
            config: dict = json.load(f)
        platform_config = config.get("platform", {})
        device_name = platform_config.get("accelerator", {}).get("name")
        hw = AcceleratorInfo.find_by_hw_name(device_name)
        if hw is None:
            logger.error(
                f"Failed to initialize device info with {device_name}. All available devices are: {AcceleratorInfo.list_all_hws().keys()}"
            )
            raise ValueError(f"Failed to initialize device info with {device_name}")
        else:
            logger
        return hw

    @classmethod
    def get_platform_config(cls) -> PlatformConfig:
        if cls._platform_config is None:
            hw = cls.get_accelerator_info()
            with open(Envs.config_path()) as f:
                config: dict = json.load(f)
            platform_config = config.get("platform", {})
            cls._platform_config = PlatformConfig(
                device=hw,
                disk_read_bandwidth_gb=platform_config.get("disk_read_bandwidth_gb"),
                disk_write_bandwidth_gb=platform_config.get("disk_write_bandwidth_gb"),
                memory_read_bandwidth_gb=platform_config.get(
                    "memory_read_bandwidth_gb"
                ),
                memory_write_bandwidth_gb=platform_config.get(
                    "memory_write_bandwidth_gb"
                ),
                num_device_per_node=platform_config.get("num_device_per_node"),
            )

            logger.info(
                f"Platform configuration initialized successfully. {cls._platform_config}"
            )

        return cls._platform_config

    @classmethod
    def set_scheduler_config(cls, config: SchedulerConfig):
        cls._scheduler_config = config

    @classmethod
    def get_kv_cache_bytes(cls) -> int:
        model = cls._model_info
        scheduler_config = cls._scheduler_config
        return (
            calc_kv_cache_cell_elems(
                model, scheduler_config.tp_size, scheduler_config.pp_size
            )
            * scheduler_config.data_type.bytes
        )

    @classmethod
    def get_kv_cache_bytes_per_layer(cls) -> int:
        model = cls._model_info
        scheduler_config = cls._scheduler_config
        return (
            calc_kv_cache_per_layer_elems(
                model, scheduler_config.tp_size, scheduler_config.pp_size
            )
            * scheduler_config.data_type.bytes
        )

    @classmethod
    def get_scheduler_config(
        cls, server_args: dict, backend: str, hf_config: dict | None = None
    ):
        model = ConfigManager.get_model_info(hf_config)

        internal_config = cls._parse_server_args(server_args, backend)

        with open(Envs.config_path()) as f:
            config: dict = json.load(f)
        scheduler_config = config.get("scheduler", {})

        tp_size = scheduler_config.get("tp_size")
        if tp_size is None:
            tp_size = internal_config.tp_size
        ep_size = scheduler_config.get("ep_size")
        if ep_size is None:
            ep_size = internal_config.ep_size
        dp_size = scheduler_config.get("dp_size")
        if dp_size is None:
            dp_size = internal_config.dp_size
        dtype = scheduler_config.get("data_type")
        if dtype is not None:
            dtype = DataType(dtype.upper())
        else:
            dtype = DataType.from_torch_dtype(model.torch_dtype)

        kv_cache_dtype = scheduler_config.get("kv_cache_data_type")
        if kv_cache_dtype is not None:
            kv_cache_dtype = DataType(kv_cache_dtype)
        else:
            kv_cache_dtype = dtype

        sched_config = SchedulerConfig(
            model=model,
            max_prefill_tokens=internal_config.max_prefill_tokens,
            chunked_prefill_size=internal_config.chunked_prefill_size,
            mem_fraction_static=internal_config.mem_fraction_static,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            # TODO: initialize with the runtime data type.
            data_type=dtype,
            kv_cache_data_type=kv_cache_dtype,
            page_size=internal_config.page_size,
            backend_name=backend,
            backend_version=scheduler_config.get("backend_version"),
        )
        return sched_config

    @classmethod
    def _parse_server_args(cls, server_args: dict, backend: str) -> SchedulerConfig:
        if backend == "sglang":
            return SchedulerConfig(
                model=None,
                tp_size=server_args.get("tp_size", 1),
                ep_size=server_args.get("ep_size", 1),
                dp_size=server_args.get("dp_size", 1),
                max_prefill_tokens=server_args.get("max_prefill_tokens"),
                chunked_prefill_size=server_args.get("chunked_prefill_size"),
                mem_fraction_static=server_args.get("mem_fraction_static"),
                page_size=server_args.get("page_size"),
                backend_name="sglang",
            )
        else:
            raise RuntimeError(f"Unsupported backend[{backend}] server args parser.")

    @classmethod
    def get_inference_time_predictor(
        cls, model: ModelInfo, hw: AcceleratorInfo, sched_config: SchedulerConfig
    ) -> InferTimePredictor:
        with open(Envs.config_path()) as f:
            config: dict = json.load(f)
        predictor_config = config.get("predictor", {})
        if predictor_config.get("name") == "aiconfigurator":
            device_name = predictor_config.get("device_name")
            hw.name = device_name
            database_mode = predictor_config.get("database_mode", "SILICON")
            prefill_scale_factor = predictor_config.get("prefill_scale_factor", 1)
            decode_scale_factor = predictor_config.get("decode_scale_factor", 1)
            xgb_model_path = predictor_config.get("xgb_model_path", None)
            return AIConfiguratorTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                database_path=predictor_config.get("database_path"),
                database_mode=database_mode,
                prefill_scale_factor=prefill_scale_factor,
                decode_scale_factor=decode_scale_factor,
                xgb_model_path=xgb_model_path,
            )
        else:
            raise ValueError(f"Unknown predictor name: {predictor_config.get('name')}")
