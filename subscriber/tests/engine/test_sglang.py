from __future__ import annotations

from subscriber.config import SubscriberConfig
from subscriber.engine.base import AbstractEngineAdapter
from subscriber.engine.sglang import SGLangAdapter


def test_sglang_adapter_is_registered() -> None:
    assert "sglang" in AbstractEngineAdapter._registry
    assert AbstractEngineAdapter._registry["sglang"] is SGLangAdapter


def test_sglang_adapter_can_be_created_via_factory() -> None:
    config = SubscriberConfig()
    adapter = AbstractEngineAdapter.create("sglang", config)
    assert isinstance(adapter, SGLangAdapter)


def test_sglang_adapter_stub_methods_return_empty() -> None:
    config = SubscriberConfig()
    adapter = SGLangAdapter(config)

    assert adapter.map_medium("hbm") == ""
    assert adapter.supported_mediums() == []
    assert adapter.storage_type() == ""


async def test_sglang_adapter_subscribe_kv_events_is_empty_generator() -> None:
    config = SubscriberConfig()
    adapter = SGLangAdapter(config)

    events = []
    async for batch in adapter.subscribe_kv_events():
        events.append(batch)

    assert events == []


async def test_sglang_adapter_watch_liveness_is_empty_generator() -> None:
    config = SubscriberConfig()
    adapter = SGLangAdapter(config)

    events = []
    async for event in adapter.watch_liveness():
        events.append(event)

    assert events == []


async def test_sglang_adapter_reset_generation_state_is_noop() -> None:
    config = SubscriberConfig()
    adapter = SGLangAdapter(config)

    await adapter.reset_generation_state()


async def test_sglang_adapter_fetch_metadata_returns_none() -> None:
    config = SubscriberConfig()
    adapter = SGLangAdapter(config)

    result = await adapter.fetch_kv_cache_group_metadata()
    assert result is None


async def test_sglang_adapter_close_is_noop() -> None:
    config = SubscriberConfig()
    adapter = SGLangAdapter(config)

    await adapter.close()
