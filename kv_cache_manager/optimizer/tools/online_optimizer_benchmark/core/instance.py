"""InstanceGroup and Instance lifecycle helpers."""

import logging

from typing import Any

from .config import BenchmarkConfig

logger = logging.getLogger(__name__)


def setup_instance(client: Any, config: BenchmarkConfig):
    """Create InstanceGroup and register Instance for benchmark.

    ``client`` implements ``OptimizerClientBase``; both the gRPC and HTTP
    transports raise ``OptimizerClientError`` internally on non-OK response
    codes (tolerating ``DUPLICATE_ENTITY`` via ``allow_duplicate``), so no
    manual response-header parsing is needed here.
    """
    logger.info("Creating instance group: %s", config.instance_group)
    if config.block_bytes <= 0:
        logger.warning(
            "BENCH_BLOCK_BYTES 未设置（block 字节数=1 占位），容量近似无限，"
            "各容量档命中率将无区分度；若需真实容量曲线请设置 BENCH_BLOCK_BYTES")

    try:
        client.create_instance_group(
            name=config.instance_group,
            capacity_gb=config.capacity_gb_list,
            eviction_policy=config.eviction_policy,
            ttl_seconds=config.ttl_seconds,
            shared_group_quota=config.shared_group_quota,
            enable_theoretical_max_cache=config.enable_theoretical_max_cache,
        )
        logger.info("CreateInstanceGroup succeeded")
    except Exception as exc:
        logger.error("CreateInstanceGroup failed: %s", exc)
        raise

    logger.info("Registering instance: %s (group=%s, block_size=%d)",
                config.instance_id, config.instance_group, config.block_size)
    try:
        client.register_instance(
            instance_group=config.instance_group,
            instance_id=config.instance_id,
            block_size=config.block_size,
            block_bytes=config.block_bytes,
        )
        logger.info("RegisterInstance succeeded")
    except Exception as exc:
        logger.error("RegisterInstance failed: %s", exc)
        raise

    logger.info("Resetting stats for instance: %s", config.instance_id)
    try:
        client.reset_stats(config.instance_id)
    except Exception as exc:
        logger.warning("ResetStats failed: %s", exc)


def teardown_instance(client: Any, config: BenchmarkConfig):
    """Remove Instance and InstanceGroup (best effort)."""
    logger.info("Removing instance: %s", config.instance_id)
    try:
        client.remove_instance(config.instance_id)
    except Exception as exc:
        logger.warning("RemoveInstance failed: %s", exc)

    logger.info("Removing instance group: %s", config.instance_group)
    try:
        client.remove_instance_group(config.instance_group)
    except Exception as exc:
        logger.warning("RemoveInstanceGroup failed: %s", exc)
