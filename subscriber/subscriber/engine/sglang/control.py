"""SGLang validation for the shared DashLLM KV-event control plane."""

from __future__ import annotations

from subscriber.config import SubscriberConfig
from subscriber.engine.kv_event_control_client import DashllmKvEventControlClient
from subscriber.engine.metadata import (
    KvEventBootstrap,
    MetadataProtocolError,
)
from subscriber.engine.vllm.control import VllmControl
from subscriber.engine.worker_status_client import DashllmWorkerStatusClient
from subscriber.kvcm.kinds import validate_extra_attention_types

_SUPPORTED_COMPONENT_KINDS = frozenset({"full_attention", "sliding_window", "mamba"})
_SUPPORTED_CACHE_KEY_MODES = frozenset({"token", "bigram"})


class SglangControl(VllmControl):
    """Reuse TCP liveness and validate SGLang-specific bootstrap semantics."""

    def __init__(
        self,
        config: SubscriberConfig,
        status_client: DashllmWorkerStatusClient,
        kv_event_control_client: DashllmKvEventControlClient,
    ) -> None:
        super().__init__(config, status_client, kv_event_control_client)
        validate_extra_attention_types(config.extra_attention_types)
        self._supported_component_kinds = _SUPPORTED_COMPONENT_KINDS.union(
            config.extra_attention_types
        )

    @property
    def _engine_kind(self) -> str:
        return "sglang"

    async def fetch_kv_event_bootstrap(self) -> KvEventBootstrap:
        bootstrap = await super().fetch_kv_event_bootstrap()
        schema = bootstrap.sglang
        if schema is None:
            raise MetadataProtocolError("SGLang bootstrap is missing its schema")
        if schema.cache_key_mode not in _SUPPORTED_CACHE_KEY_MODES:
            raise MetadataProtocolError(
                "SGLang bootstrap has unsupported cache_key_mode "
                f"{schema.cache_key_mode!r}"
            )
        for component in bootstrap.components:
            if component.component_kind not in self._supported_component_kinds:
                raise MetadataProtocolError(
                    "SGLang bootstrap has unsupported component_kind "
                    f"{component.component_kind!r}"
                )
        return bootstrap
