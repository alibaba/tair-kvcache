from __future__ import annotations

import dataclasses

import pytest

from subscriber.engine.metadata import (
    KvCacheDescriptor,
    MetadataFetchError,
    MetadataProtocolError,
    MetadataTemporarilyUnavailable,
    parse_kv_event_bootstrap,
)
from subscriber.proto import engine_service_rpc_pb2 as pb2
from subscriber.types import KvCacheGroupSpec


def _component(
    component_id: int,
    kind: str,
    *,
    block_size: int = 16,
    payload_size: int | None = 1024,
) -> pb2.CacheComponentPB:
    geometry = pb2.CacheGeometryPB(block_size_tokens=block_size)
    if payload_size is not None:
        geometry.group_payload_size_bytes.value = payload_size
    return pb2.CacheComponentPB(
        component_id=component_id,
        component_kind=kind,
        geometry=geometry,
    )


def _bootstrap(engine_kind: str = "vllm") -> pb2.KvEventBootstrapInfoPB:
    response = pb2.KvEventBootstrapInfoPB(
        protocol_version=1,
        engine_kind=engine_kind,
        err_code=pb2.KV_EVENT_BOOTSTRAP_OK,
    )
    response.event_transport.live_endpoint = "tcp://127.0.0.1:5557"
    response.event_transport.topic = "kv-events"
    response.event_transport.replay_supported = True
    response.event_transport.replay_endpoint = "tcp://127.0.0.1:5558"
    response.event_transport.serialization = "msgpack-v1"
    response.runtime_topology.data_parallel_size = 1
    response.runtime_topology.tensor_parallel_size = 2
    response.runtime_topology.pipeline_parallel_size = 1
    response.snapshot.supported = engine_kind == "vllm"
    response.snapshot.versioned = engine_kind == "vllm"
    if engine_kind == "vllm":
        response.vllm.event_schema_version = 2
        response.vllm.use_eagle_pop = True
        response.vllm.mamba_cache_mode = "none"
        response.vllm.hash_algorithm = "sha256"
        response.vllm.hash_version = "vllm-block-hash-v1"
    else:
        response.sglang.event_schema_version = 2
        response.sglang.cache_key_mode = "token"
        response.sglang.native_hash_algorithm = "sglang-radix-native-int64"
    return response


def test_parse_vllm_bootstrap_preserves_numeric_component_and_typed_settings() -> None:
    response = _bootstrap()
    component = response.components.add()
    component.CopyFrom(_component(0, "full_attention"))
    component.geometry.page_size_tokens.value = 16
    component.compatibility_settings.add(
        name="vllm.layer_names",
        string_list=pb2.StringListPB(values=["layer.0", "layer.1"]),
    )
    response.compatibility_settings.add(
        name="vllm.enable_prefix_caching", bool_value=True
    )

    bootstrap = parse_kv_event_bootstrap(response, expected_engine_kind="vllm")

    assert bootstrap.engine_kind == "vllm"
    assert bootstrap.event_transport.topic == "kv-events"
    assert bootstrap.components[0].component_id == 0
    assert bootstrap.components[0].geometry.group_payload_size_bytes == 1024
    assert bootstrap.components[0].compatibility_settings == (
        ("vllm.layer_names", ("layer.0", "layer.1")),
    )
    assert bootstrap.compatibility_settings == (("vllm.enable_prefix_caching", True),)
    assert bootstrap.to_kv_cache_descriptor().use_eagle_pop is True


def test_parse_sglang_bootstrap_keeps_payload_absent_for_kvcm_fallback() -> None:
    response = _bootstrap("sglang")
    response.components.add().CopyFrom(
        _component(2, "mamba", block_size=128, payload_size=None)
    )

    bootstrap = parse_kv_event_bootstrap(response, expected_engine_kind="sglang")

    assert bootstrap.components[0].component_id == 2
    assert bootstrap.components[0].geometry.group_payload_size_bytes is None
    descriptor = bootstrap.to_kv_cache_descriptor()
    assert descriptor.groups[0].group_idx == 2
    assert descriptor.groups[0].group_payload_size_bytes is None


def test_parse_bootstrap_allows_empty_transport_without_incremental_pipeline() -> None:
    response = _bootstrap()
    response.event_transport.Clear()
    response.components.add().CopyFrom(_component(0, "full_attention"))

    bootstrap = parse_kv_event_bootstrap(
        response,
        expected_engine_kind="vllm",
        require_incremental_transport=False,
    )

    assert bootstrap.event_transport.live_endpoint == ""
    assert bootstrap.event_transport.replay_supported is False


def test_parse_bootstrap_requires_event_transport_with_incremental_pipeline() -> None:
    response = _bootstrap()
    response.event_transport.Clear()
    response.components.add().CopyFrom(_component(0, "full_attention"))

    with pytest.raises(MetadataProtocolError, match="missing live_endpoint"):
        parse_kv_event_bootstrap(response, expected_engine_kind="vllm")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda response: setattr(response, "protocol_version", 99), "protocol"),
        (lambda response: setattr(response, "engine_kind", "sglang"), "engine"),
        (
            lambda response: setattr(response.event_transport, "serialization", "json"),
            "serialization",
        ),
    ],
)
def test_parse_bootstrap_rejects_incompatible_contract(mutate, match: str) -> None:
    response = _bootstrap()
    response.components.add().CopyFrom(_component(0, "full_attention"))
    mutate(response)

    with pytest.raises(MetadataProtocolError, match=match):
        parse_kv_event_bootstrap(response, expected_engine_kind="vllm")


def test_parse_bootstrap_rejects_duplicate_component_ids() -> None:
    response = _bootstrap()
    response.components.add().CopyFrom(_component(0, "full_attention"))
    response.components.add().CopyFrom(_component(0, "mamba"))

    with pytest.raises(MetadataProtocolError, match="duplicate component_id 0"):
        parse_kv_event_bootstrap(response, expected_engine_kind="vllm")


def test_parse_bootstrap_rejects_non_positive_block_size() -> None:
    response = _bootstrap()
    response.components.add().CopyFrom(_component(0, "full_attention", block_size=0))

    with pytest.raises(MetadataProtocolError, match="invalid block_size_tokens 0"):
        parse_kv_event_bootstrap(response, expected_engine_kind="vllm")


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ([pb2.TypedSettingPB(name="", string_value="value")], "empty setting name"),
        (
            [
                pb2.TypedSettingPB(name="duplicate", bool_value=True),
                pb2.TypedSettingPB(name="duplicate", bool_value=False),
            ],
            "duplicate setting name",
        ),
        ([pb2.TypedSettingPB(name="missing-value")], "has no typed value"),
    ],
)
def test_parse_bootstrap_rejects_invalid_typed_settings(
    settings: list[pb2.TypedSettingPB], match: str
) -> None:
    response = _bootstrap()
    response.components.add().CopyFrom(_component(0, "full_attention"))
    response.compatibility_settings.extend(settings)

    with pytest.raises(MetadataProtocolError, match=match):
        parse_kv_event_bootstrap(response, expected_engine_kind="vllm")


def test_bootstrap_log_json_contains_transport_geometry_and_engine_schema() -> None:
    response = _bootstrap()
    response.components.add().CopyFrom(_component(0, "full_attention"))
    bootstrap = parse_kv_event_bootstrap(response, expected_engine_kind="vllm")

    payload = bootstrap.to_log_json()

    assert '"live_endpoint": "tcp://127.0.0.1:5557"' in payload
    assert '"component_kind": "full_attention"' in payload
    assert '"event_schema_version": 2' in payload


def test_kv_cache_descriptor_holds_groups_tuple() -> None:
    spec = KvCacheGroupSpec(
        group_idx=0,
        kind="full_attention",
        block_size=16,
        sliding_window=None,
        group_payload_size_bytes=1024,
    )

    assert KvCacheDescriptor(groups=(spec,)).groups == (spec,)


def test_kv_cache_group_spec_can_mark_payload_size_unavailable() -> None:
    spec = KvCacheGroupSpec(
        group_idx=0,
        kind="full_attention",
        block_size=16,
        group_payload_size_bytes=None,
    )

    assert spec.group_payload_size_bytes is None


def test_kv_cache_descriptor_empty_groups_is_valid() -> None:
    metadata = KvCacheDescriptor(groups=())

    assert metadata.groups == ()


def test_kv_cache_descriptor_is_frozen() -> None:
    metadata = KvCacheDescriptor(groups=())

    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.groups = ()


def test_metadata_fetch_error_is_runtime_error() -> None:
    assert issubclass(MetadataFetchError, RuntimeError)


def test_metadata_error_subclasses_derive_from_base() -> None:
    assert issubclass(MetadataTemporarilyUnavailable, MetadataFetchError)
    assert issubclass(MetadataProtocolError, MetadataFetchError)


def test_metadata_error_subclasses_are_catchable_as_base() -> None:
    for exc_type in (MetadataTemporarilyUnavailable, MetadataProtocolError):
        with pytest.raises(MetadataFetchError):
            raise exc_type("boom")
