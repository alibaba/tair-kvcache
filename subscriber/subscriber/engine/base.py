from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, TypeVar

from subscriber.health.events import LivenessEvent
from subscriber.types import KVEventBatch

if TYPE_CHECKING:
    from subscriber.config import SubscriberConfig

_AdapterT = TypeVar("_AdapterT", bound="AbstractEngineAdapter")


class AbstractEngineAdapter(ABC):
    """Interface for engine-specific IO.

    Implementations isolate engine transport details from the subscriber core:
    KV event subscription, sequence-gap replay, liveness observation, and
    generation-local reset state all stay behind this interface.
    """

    _registry: dict[str, Callable[[SubscriberConfig], AbstractEngineAdapter]] = {}

    @classmethod
    def register(cls, engine_type: str) -> Callable[[type[_AdapterT]], type[_AdapterT]]:
        """Register an adapter class for lazy construction by engine type."""

        def decorator(adapter_cls: type[_AdapterT]) -> type[_AdapterT]:
            cls._registry[engine_type] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def create(
        cls, engine_type: str, config: SubscriberConfig
    ) -> AbstractEngineAdapter:
        """Create the adapter registered for ``engine_type``.

        Raises:
            KeyError: If no adapter has been registered for ``engine_type``.
        """

        if engine_type not in cls._registry:
            raise KeyError(
                f"Unknown engine_type {engine_type!r}. "
                f"Registered: {list(cls._registry)}"
            )
        return cls._registry[engine_type](config)

    @abstractmethod
    def subscribe_kv_events(self) -> AsyncGenerator[list[KVEventBatch], None]:
        """Yield realtime or replayed KV event batches in forwarding order.

        Each yield is either:
        - A single-element list: one real-time event batch.
        - A multi-element list: replayed batches when a sequence gap is detected.

        Implementations may aggregate one or more configured DP endpoints into
        this stream. Transport-specific replay and per-DP sequence tracking stay
        inside the adapter before yielding batches to the subscriber core.
        """
        ...

    @abstractmethod
    def watch_liveness(self) -> AsyncGenerator[LivenessEvent, None]:
        """Yield engine-agnostic liveness events for the health coordinator."""
        ...

    @abstractmethod
    def map_medium(self, medium: str | None) -> str:
        """Translate an engine-native medium string to the kvcm wire value.

        Returns the kvcm-side identifier (e.g. ``"hbm"``, ``"mem"``) for the
        given engine-specific medium.  Returns ``""`` when *medium* is ``None``
        or not recognized by this adapter.
        """
        ...

    @abstractmethod
    def supported_mediums(self) -> list[str]:
        """Return the kvcm wire medium values this adapter can produce."""
        ...

    @abstractmethod
    def storage_type(self) -> str:
        """Return the kvcm storage type identifier for this engine.

        For example, the vLLM adapter returns ``"ST_VLLM"``.
        """
        ...

    def reset_generation_state(self) -> None:
        """Reset adapter-local generation state after an engine restart.

        The health coordinator calls this before opening a new sendable epoch
        after recovery from DEAD. Adapters should clear sequence tracking and
        recreate any replay/session state that belongs to the old generation.
        """
        return None
