"""Setup and teardown helpers for benchmark InstanceGroup and Instance."""

import logging

from .client import OptimizerClient
from .config import BenchmarkConfig

logger = logging.getLogger(__name__)


def setup_instance(client: OptimizerClient, config: BenchmarkConfig):
    """Create InstanceGroup and register Instance for benchmark.

    Idempotent: silently ignores DUPLICATE_ENTITY errors.
    """
    logger.info("Creating instance group: %s", config.instance_group)
    try:
        result = client.create_instance_group(
            name=config.instance_group,
            capacity_gb=config.capacity_gb,
            indexer_type=config.indexer_type,
            max_key_count=config.max_key_count,
        )
        _check_response(result, "CreateInstanceGroup")
    except Exception as exc:
        logger.warning("CreateInstanceGroup failed (may already exist): %s", exc)

    logger.info("Registering instance: %s (group=%s, block_size=%d)",
                config.instance_id, config.instance_group, config.block_size)
    try:
        result = client.register_instance(
            instance_group=config.instance_group,
            instance_id=config.instance_id,
            block_size=config.block_size,
        )
        _check_response(result, "RegisterInstance")
    except Exception as exc:
        logger.warning("RegisterInstance failed (may already exist): %s", exc)

    logger.info("Resetting stats for instance: %s", config.instance_id)
    try:
        client.reset_stats(config.instance_id)
    except Exception as exc:
        logger.warning("ResetStats failed: %s", exc)


def teardown_instance(client: OptimizerClient, config: BenchmarkConfig):
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


def _check_response(result: dict, operation: str):
    header = result.get("header", {})
    status = header.get("status", {})
    code = status.get("code", "UNSPECIFIED")
    if code not in ("OK", 1, "1", "DUPLICATE_ENTITY", 6, "6"):
        raise RuntimeError(f"{operation} failed: {status}")
    logger.info("%s succeeded (code=%s)", operation, code)
