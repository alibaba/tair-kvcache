#!/usr/bin/env python3
"""Configuration validator for KVCache instances.

Validates config keys against the allowed schema and checks
for common misconfiguration patterns.
"""

import re
from typing import Dict, List


ALLOWED_KEYS = {
    "instance_id", "max_memory_mb", "eviction_policy",
    "ttl_seconds", "replication_factor", "shard_count"
}

REQUIRED_KEYS = {"instance_id", "max_memory_mb"}


def validate_config(config: Dict[str, str]) -> List[str]:
    """Validate a KVCache configuration dictionary.

    Returns a list of warning/error messages. Empty list means valid config.

    Args:
        config: Dictionary of configuration key-value pairs.

    Example:
        >>> validate_config({"instance_id": "test", "max_memory_mb": "1024"})
        []
    """
    errors = []

    for key in REQUIRED_KEYS:
        if key not in config:
            errors.append(f"Missing required key: {key}")

    for key in config:
        if key not in ALLOWED_KEYS:
            errors.append(f"Unknown configuration key: {key}")

    instance_id = config.get("instance_id", "")
    if instance_id and not re.match(r'^[a-zA-Z0-9_-]{1,64}$', instance_id):
        errors.append(
            f"Invalid instance_id format: {instance_id!r}. "
            "Must match ^[a-zA-Z0-9_-]{1,64}$"
        )

    try:
        mem = int(config.get("max_memory_mb", "0"))
        if mem <= 0 or mem > 102400:
            errors.append(f"max_memory_mb must be between 1 and 102400, got {mem}")
    except ValueError:
        errors.append("max_memory_mb must be an integer")

    return errors


def sanitize_cache_key(key: str) -> str:
    """Remove potentially dangerous characters from a cache key."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', key)[:256]
