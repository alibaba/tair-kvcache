from __future__ import annotations

import msgspec

from subscriber import logger
from subscriber.types import KVEventBatch


class KVEventBatchMsgpackHelper:
    """Decode KV event batch msgpack payloads with consistent logging."""

    def __init__(self) -> None:
        self._decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

    def decode(
        self,
        payload: bytes,
        *,
        step: str,
        tags: dict[str, object] | None = None,
    ) -> KVEventBatch | None:
        try:
            return self._decoder.decode(payload)
        except (msgspec.DecodeError, msgspec.ValidationError, TypeError) as exc:
            log_tags: dict[str, object] = {
                "error": exc.__class__.__name__,
                "message": str(exc),
            }
            if tags:
                log_tags = {**tags, **log_tags}
            logger.warning(
                "failed to decode kv event msgpack payload",
                step=step,
                tags=log_tags,
            )
            return None
