from __future__ import annotations

_U64_MASK = (1 << 64) - 1
_I64_MAX = (1 << 63) - 1


def to_signed_i64(value: int | bytes) -> int:
    """Map engine block hashes onto KVCM's signed int64 key domain.

    Integer and byte hashes preserve their low 64-bit representation. For
    vLLM this is exactly the representation selected by
    ``VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES`` and therefore stays stable whether
    the publisher sends the full digest or its legacy integer form.

    KVCM's metadata key is signed int64, so query-side integrations must use
    this same codec before calling KVCM with engine-native hashes.
    """

    if isinstance(value, bytes):
        if not value:
            raise ValueError("block key bytes cannot be empty")
        unsigned = int.from_bytes(value, "big", signed=False) & _U64_MASK
    elif isinstance(value, int) and not isinstance(value, bool):
        unsigned = value & _U64_MASK
    else:
        raise TypeError(f"unsupported block key type: {type(value).__name__}")
    return unsigned if unsigned <= _I64_MAX else unsigned - (1 << 64)
