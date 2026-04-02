import argparse
import dataclasses
import json
from typing import Optional


@dataclasses.dataclass
class AcceleratorConfig:
    name: str = "H20"


@dataclasses.dataclass
class SimPlatformConfig:
    accelerator: AcceleratorConfig = dataclasses.field(
        default_factory=AcceleratorConfig
    )
    disk_read_bandwidth_gb: float = 4.0
    disk_write_bandwidth_gb: float = 4.0
    memory_read_bandwidth_gb: float = 64.0
    memory_write_bandwidth_gb: float = 64.0


@dataclasses.dataclass
class PredictorConfig:
    name: str = "aiconfigurator"
    database_path: Optional[str] = None
    device_name: Optional[str] = None
    prefill_scale_factor: float = 1.0
    decode_scale_factor: float = 1.0


@dataclasses.dataclass
class SimSchedulerConfig:
    tp_size: int = 1
    ep_size: int = 1
    dp_size: int = 1
    data_type: str = "FP16"
    kv_cache_data_type: str = "FP16"
    backend_name: str = "sglang"
    backend_version: Optional[str] = None


def _apply_cli_args(ns: argparse.Namespace, target, mappings: list[tuple[str, str]]):
    for cli_arg, field_name in mappings:
        value = getattr(ns, cli_arg, None)
        if value is not None:
            setattr(target, field_name, value)


@dataclasses.dataclass
class SimulationArgs:
    config_path: Optional[str] = None

    platform: SimPlatformConfig = dataclasses.field(default_factory=SimPlatformConfig)
    predictor: PredictorConfig = dataclasses.field(default_factory=PredictorConfig)
    scheduler: SimSchedulerConfig = dataclasses.field(
        default_factory=SimSchedulerConfig
    )

    def add_cli_args(parser: argparse.ArgumentParser):
        prefix = "sim-"

        parser.add_argument(
            f"--{prefix}config-path",
            dest="sim_config_path",
            type=str,
            default=None,
            help="Path to simulation JSON config (same as HISIM_CONFIG_PATH).",
        )

        parser.add_argument(
            f"--{prefix}accelerator-name",
            dest="sim_accelerator_name",
            type=str,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}disk-read-bandwidth-gb",
            dest="sim_disk_read_bandwidth_gb",
            type=float,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}disk-write-bandwidth-gb",
            dest="sim_disk_write_bandwidth_gb",
            type=float,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}memory-read-bandwidth-gb",
            dest="sim_memory_read_bandwidth_gb",
            type=float,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}memory-write-bandwidth-gb",
            dest="sim_memory_write_bandwidth_gb",
            type=float,
            default=None,
        )

        parser.add_argument(
            f"--{prefix}predictor-name",
            dest="sim_predictor_name",
            type=str,
            default=None,
            choices=["aiconfigurator"],
        )
        parser.add_argument(
            f"--{prefix}database-path", dest="sim_database_path", type=str, default=None
        )
        parser.add_argument(
            f"--{prefix}device-name", dest="sim_device_name", type=str, default=None
        )
        parser.add_argument(
            f"--{prefix}prefill-scale-factor",
            dest="sim_prefill_scale_factor",
            type=float,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}decode-scale-factor",
            dest="sim_decode_scale_factor",
            type=float,
            default=None,
        )

        parser.add_argument(
            f"--{prefix}tp-size", dest="sim_tp_size", type=int, default=None
        )
        parser.add_argument(
            f"--{prefix}ep-size", dest="sim_ep_size", type=int, default=None
        )
        parser.add_argument(
            f"--{prefix}data-type", dest="sim_data_type", type=str, default=None
        )
        parser.add_argument(
            f"--{prefix}kv-cache-data-type",
            dest="sim_kv_cache_data_type",
            type=str,
            default=None,
        )
        parser.add_argument(
            f"--{prefix}backend-name", dest="sim_backend_name", type=str, default=None
        )
        parser.add_argument(
            f"--{prefix}backend-version",
            dest="sim_backend_version",
            type=str,
            default=None,
        )

    @staticmethod
    def from_json(path: str) -> "SimulationArgs":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        platform = cfg.get("platform", {})
        accelerator = platform.get("accelerator", {})
        predictor = cfg.get("predictor", {})
        scheduler = cfg.get("scheduler", {})

        return SimulationArgs(
            config_path=path,
            platform=SimPlatformConfig(
                accelerator=AcceleratorConfig(name=accelerator.get("name", "H20")),
                disk_read_bandwidth_gb=platform.get("disk_read_bandwidth_gb", 4.0),
                disk_write_bandwidth_gb=platform.get("disk_write_bandwidth_gb", 4.0),
                memory_read_bandwidth_gb=platform.get("memory_read_bandwidth_gb", 64.0),
                memory_write_bandwidth_gb=platform.get(
                    "memory_write_bandwidth_gb", 64.0
                ),
            ),
            predictor=PredictorConfig(
                name=predictor.get("name", "aiconfigurator"),
                database_path=predictor.get("database_path"),
                device_name=predictor.get("device_name"),
                prefill_scale_factor=predictor.get("prefill_scale_factor", 1.0),
                decode_scale_factor=predictor.get("decode_scale_factor", 1.0),
            ),
            scheduler=SimSchedulerConfig(
                tp_size=scheduler.get("tp_size", 1),
                ep_size=scheduler.get("ep_size", 1),
                data_type=scheduler.get("data_type", "FP16"),
                kv_cache_data_type=scheduler.get("kv_cache_data_type", "FP16"),
                backend_name=scheduler.get("backend_name", "sglang"),
                backend_version=scheduler.get("backend_version"),
            ),
        )

    def to_dict(self, indent=2, ensure_ascii: bool = False) -> dict:
        data = dataclasses.asdict(self)
        data.pop("config_path", None)
        return data

    @classmethod
    def from_cli_args(cls, ns: argparse.Namespace) -> "SimulationArgs":
        # If config path is provided, load from JSON
        if getattr(ns, "sim_config_path", None) is not None:
            return SimulationArgs.from_json(ns.sim_config_path)

        args = SimulationArgs()

        # Platform config
        if getattr(ns, "sim_accelerator_name", None) is not None:
            args.platform.accelerator.name = ns.sim_accelerator_name
        _apply_cli_args(
            ns,
            args.platform,
            [
                ("sim_disk_read_bandwidth_gb", "disk_read_bandwidth_gb"),
                ("sim_disk_write_bandwidth_gb", "disk_write_bandwidth_gb"),
                ("sim_memory_read_bandwidth_gb", "memory_read_bandwidth_gb"),
                ("sim_memory_write_bandwidth_gb", "memory_write_bandwidth_gb"),
            ],
        )

        # Predictor config
        if getattr(ns, "sim_predictor_name", None) is not None:
            args.predictor.name = ns.sim_predictor_name
        _apply_cli_args(
            ns,
            args.predictor,
            [
                ("sim_database_path", "database_path"),
                ("sim_device_name", "device_name"),
                ("sim_prefill_scale_factor", "prefill_scale_factor"),
                ("sim_decode_scale_factor", "decode_scale_factor"),
            ],
        )

        # Scheduler config
        _apply_cli_args(
            ns,
            args.scheduler,
            [
                ("sim_tp_size", "tp_size"),
                ("sim_ep_size", "ep_size"),
                ("sim_data_type", "data_type"),
                ("sim_kv_cache_data_type", "kv_cache_data_type"),
                ("sim_backend_name", "backend_name"),
                ("sim_backend_version", "backend_version"),
            ],
        )

        return args
