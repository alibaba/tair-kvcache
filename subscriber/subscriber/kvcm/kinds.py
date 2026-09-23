"""Shared bootstrap component-kind to KVCM-category contract."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

BUILTIN_ATTENTION_TYPE_CATEGORIES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "full_attention": "F",
        "mla_attention": "F",
        "sink_full_attention": "F",
        "mamba": "L",
        "sliding_window": "W",
        "sliding_window_mla": "W",
        "chunked_local_attention": "C",
        "encoder_only_attention": "E",
        "cross_attention": "X",
    }
)


def validate_extra_attention_types(extra_attention_types: object) -> None:
    """Ensure configured attention kinds extend, but never replace, the contract."""

    if not isinstance(extra_attention_types, dict) or any(
        not isinstance(kind, str) or not isinstance(category, str)
        for kind, category in extra_attention_types.items()
    ):
        raise ValueError(
            "extra_attention_types must be a mapping with string keys and values"
        )
    conflicts = sorted(
        set(extra_attention_types).intersection(BUILTIN_ATTENTION_TYPE_CATEGORIES)
    )
    if conflicts:
        raise ValueError(
            "extra_attention_types cannot override built-in attention types: "
            + ", ".join(conflicts)
        )


def effective_attention_type_categories(
    extra_attention_types: Mapping[str, str],
) -> dict[str, str]:
    """Return the canonical KVCM categories extended by explicit local mappings."""

    validate_extra_attention_types(extra_attention_types)
    return {**BUILTIN_ATTENTION_TYPE_CATEGORIES, **extra_attention_types}
