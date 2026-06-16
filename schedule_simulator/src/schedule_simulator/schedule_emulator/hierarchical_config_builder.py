"""
Generates HierarchicalReplayConfig JSON from schedule_simulator's
SchedulerConfig + PlatformConfig + instance topology.
"""
import json
import os
import tempfile
from typing import Optional

from schedule_simulator._compat import ModelInfo, AcceleratorInfo, DataType

from schedule_simulator.schedule_emulator.types import (
    SchedulerConfig,
    PlatformConfig,
)
from schedule_simulator.schedule_emulator.utils import (
    calc_kv_cache_cell_elems,
    estimate_kv_cache_pool_capacity,
)


def build_hierarchical_config(
    scheduler_config: SchedulerConfig,
    platform_config: PlatformConfig,
    p_instance_ids: list[str],
    d_instance_ids: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    storage_pool_capacity_gb: float = 1.0,
    enable_p2p: bool = True,
    p2p_tier: str = "hbm",
    enable_lifecycle_tracking: bool = False,
) -> str:
    """
    Build a HierarchicalReplayConfig JSON file from schedule_simulator parameters.

    Returns the path to the generated JSON config file.
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="hierarchical_")
    os.makedirs(os.path.join(output_dir, "pool"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "infer"), exist_ok=True)

    trace_path = os.path.join(output_dir, "trace.jsonl")
    if not os.path.exists(trace_path):
        open(trace_path, "w").close()

    if isinstance(scheduler_config.model, str):
        model = ModelInfo.find_by_model_name(scheduler_config.model) if ModelInfo is not None else None
    else:
        model = scheduler_config.model

    if isinstance(platform_config.device, str):
        hw = AcceleratorInfo.find_by_hw_name(platform_config.device) if AcceleratorInfo is not None else None
    else:
        hw = platform_config.device

    if scheduler_config.data_type is None:
        if model is not None and DataType is not None:
            dt = DataType(DataType.alias().get(model.torch_dtype, "FP16"))
        else:
            dt = None
    else:
        dt = scheduler_config.data_type

    # Use scheduler_config override if set, otherwise derive from model
    if scheduler_config.kv_cache_space_per_token is not None:
        kv_bytes_per_token = scheduler_config.kv_cache_space_per_token
    elif model is not None and dt is not None:
        kv_bytes_per_token = (
            calc_kv_cache_cell_elems(model, scheduler_config.tp_size, scheduler_config.pp_size)
            * dt.bytes
        )
    else:
        kv_bytes_per_token = 320  # sensible default for request-level mode

    hbm_capacity_gb = platform_config.hbm_capacity_gb or (hw.hbm_capacity_gb if hw is not None else 80)
    dram_capacity_gb = platform_config.memory_capacity_gb or (hbm_capacity_gb * 2)

    write_mode_map = {
        "write_through": "write_through",
        "write_through_selective": "write_through_selective",
        "write_back": "cascading",
    }
    pool_write_mode = write_mode_map.get(
        scheduler_config.hicache_write_policy, "write_through"
    )

    all_infer_ids = list(p_instance_ids)
    if d_instance_ids:
        all_infer_ids.extend(d_instance_ids)

    tiers = [{"name": "hbm", "capacity": hbm_capacity_gb}]
    tier_flows = []
    if platform_config.memory_capacity_gb is not None or platform_config.memory_read_bandwidth_gb is not None:
        tiers.append({"name": "dram", "capacity": dram_capacity_gb})
        tier_flows.append({
            "from_tier": "hbm",
            "to_tier": "dram",
            "write_mode": write_mode_map.get(scheduler_config.hicache_write_policy, "write_through"),
            "access_propagation_enabled": False,
            "write_propagation_enabled": False,
            "selective_write_threshold": 2,
        })

    p2p_read_flows = []
    if enable_p2p and len(all_infer_ids) > 1:
        p2p_read_flows.append({
            "tier": p2p_tier,
            "peer_read_touch_enabled": True,
        })

    config = {
        "trace_file_path": trace_path,
        "output_result_path": output_dir,
        "infer_scheduling_strategy": "preserve_trace",
        "infer_active_windows_from_trace": False,
        "enable_lifecycle_tracking": enable_lifecycle_tracking,
        "infer_eviction_params": {
            "eviction_mode": 3,
            "eviction_batch_size_per_instance": 100,
        },
        "infer_clusters": [
            {
                "storage_pool_id": "pool_1",
                "engine_read_query_type": scheduler_config.hicache_read_query_type,
                "model": {
                    "block_size": scheduler_config.page_size if scheduler_config.page_size else 1,
                    "bytes_per_token": kv_bytes_per_token,
                    "eviction_policy_type": "lru",
                    "eviction_policy_params": {"sample_rate": 1.0},
                },
                "infer_ids": all_infer_ids,
                "ttl_config": {
                    "default_block_ttl_seconds": 0,
                    "refresh_on_read": True,
                },
                "tiers": tiers,
                "tier_flows": tier_flows,
                "storage_pool_flow": {
                    "write_mode": pool_write_mode,
                    "local_read_touch_enabled": False,
                    "shadow_write_touch_enabled": False,
                    "selective_write_threshold": 2,
                },
                "p2p_read_flows": p2p_read_flows,
            }
        ],
        "storage_pool": {
            "output_result_path": os.path.join(output_dir, "pool"),
            "storage_name": "l3",
            "capacity": storage_pool_capacity_gb,
            "eviction_params": {
                "eviction_mode": 3,
                "eviction_batch_size_per_instance": 100,
            },
            "pools": [
                {
                    "pool_id": "pool_1",
                    "model": {
                        "block_size": scheduler_config.page_size if scheduler_config.page_size else 1,
                        "bytes_per_token": kv_bytes_per_token,
                        "eviction_policy_type": "lru",
                        "eviction_policy_params": {"sample_rate": 1.0},
                    },
                    "ttl_config": {
                        "default_block_ttl_seconds": 0,
                        "refresh_on_read": True,
                    },
                }
            ],
        },
    }

    config_path = os.path.join(output_dir, "hierarchical_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path
