from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from subscriber.health.events import LivenessEvent
from subscriber.metrics import StageTimer
from subscriber.types import KVEventBatch

if TYPE_CHECKING:
    from subscriber.config import SubscriberConfig

_AdapterT = TypeVar("_AdapterT", bound="AbstractEngineAdapter")


@dataclass(frozen=True)
class EngineEventBatch:
    """Engine-agnostic carrier yielded by adapters to the subscriber core.

    Bundles the forwarded KV batches with the :class:`StageTimer` whose origin
    is the moment the adapter began handling this event. The adapter marks its
    own internal stages (e.g. ``decode`` for live events, ``replay_fetch`` for
    replayed events); the subscriber core marks the remaining pipeline stages on
    the same timer so a single span timeline covers receipt through kvcm send.
    """

    batches: list[KVEventBatch]
    timer: StageTimer
    on_delivery: Callable[[bool], Awaitable[None]] | None = None


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
    def subscribe_kv_events(self) -> AsyncGenerator[EngineEventBatch, None]:
        """Yield realtime or replayed KV event batches in forwarding order.

        Each yield is an :class:`EngineEventBatch` whose ``batches`` field is
        either:
        - A single-element list: one real-time event batch.
        - A multi-element list: replayed batches when a sequence gap is detected.

        The carried ``timer`` originates when the adapter begins handling the
        event and has the adapter's internal stage(s) already marked.

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

        For example, the vLLM adapter returns ``"ST_EVENT_REPORT"``.
        """
        ...

    @abstractmethod
    def location_spec_name(self, block_size: int) -> str:
        """Return the KVCM location-spec name for one engine cache block."""
        ...

    @abstractmethod
    def location_uri(self, host_ip_port: str, medium: str) -> str:
        """Build the engine-specific cache location URI reported to KVCM."""
        ...

    async def reset_generation_state(self) -> None:
        """Reset adapter-local generation state after an engine restart.

        The health coordinator calls this before opening a new sendable epoch
        after recovery from DEAD. Adapters should clear sequence tracking and
        recreate any replay/session state that belongs to the old generation.

        In-flight await contract
        ------------------------
        When this method runs, the adapter may still have awaits parked on the
        previous generation's sockets — for example a live ``recv_multipart``
        on the SUB socket, or a replay ``send``/``recv`` pair on the DEALER
        socket. If those awaits returned their results into the new epoch,
        stale seq/payload from the old engine instance would be yielded with
        the freshly reset sequence counter and forwarded to kvcm as if it
        belonged to the new generation.

        Implementations must therefore invalidate any in-flight result whose
        await spanned the reset. The vLLM adapter does this with a monotonic
        ``self._generation`` counter: every async helper snapshots the counter
        before each await and discards the result (returns ``None`` / skips
        the yield) if the counter has advanced by the time the await resumes.
        Closing the old sockets with ``linger=0`` inside this method provides
        a second line of defense by turning parked awaits into exceptions that
        the helpers already translate into ``None``.
        """
        return None
