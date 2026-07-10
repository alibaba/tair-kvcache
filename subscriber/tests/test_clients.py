from __future__ import annotations

import asyncio
import threading
from enum import Enum
from unittest.mock import AsyncMock

import pytest

from subscriber.config import SubscriberConfig
from subscriber.kvcm.client import KvcmClient
from subscriber.types import AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch


class FakeSdkClient:
    def __init__(self, base_url: str = "") -> None:
        self.base_url = base_url
        self.started = False
        self.start = AsyncMock(side_effect=self._start)
        self.register_instance = AsyncMock(
            return_value={"header": {"status": {"code": "OK"}}}
        )
        self.report_event = AsyncMock(
            return_value={"header": {"status": {"code": "OK"}}}
        )
        self.close = AsyncMock()

    async def _start(self) -> None:
        self.started = True


_VLLM_MEDIUM_MAP = {"GPU": "hbm", "CPU": "mem"}


@pytest.fixture(autouse=True)
def _set_kvcm_vservice_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KVCM_VSERVICE_ID", "vs-test")


def _vllm_medium_mapper(medium: str | None) -> str:
    if medium is None:
        return ""
    return _VLLM_MEDIUM_MAP.get(medium, "")


def _client(config: SubscriberConfig | None = None) -> KvcmClient:
    return KvcmClient(
        config or SubscriberConfig(),
        medium_mapper=_vllm_medium_mapper,
        storage_type="ST_VLLM",
        supported_mediums=["hbm", "mem"],
        sdk_client_factory=lambda _url: FakeSdkClient(),
    )


def _make_kvcm(config: SubscriberConfig, fake_sdk: FakeSdkClient) -> KvcmClient:
    return KvcmClient(
        config,
        medium_mapper=_vllm_medium_mapper,
        storage_type="ST_VLLM",
        supported_mediums=["hbm", "mem"],
        sdk_client_factory=lambda _url: fake_sdk,
    )


def test_storage_type_maps_engine_type() -> None:
    assert _client(SubscriberConfig(engine_type="vllm"))._storage_type == "ST_VLLM"
    assert (
        KvcmClient(
            SubscriberConfig(engine_type="unknown"),
            medium_mapper=_vllm_medium_mapper,
            storage_type="ST_UNSPECIFIED",
            supported_mediums=["hbm", "mem"],
            sdk_client_factory=lambda _url: FakeSdkClient(),
        )._storage_type
        == "ST_UNSPECIFIED"
    )


def test_register_instance_request_uses_env_instance_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRUM_DEPLOYMENT_NAME", "deploy-a")
    request = _client()._register_instance_request()
    assert request["instance_id"] == "deploy-a"
    assert request["instance_group"] == "default"
    assert request["block_size"] == 1
    assert request["location_spec_infos"] == [{"name": "default", "size": 1}]
    assert request["location_spec_groups"] == [
        {"name": "default", "spec_names": ["default"]}
    ]
    assert request["model_deployment"] == {
        "model_name": "default",
        "dtype": "",
        "use_mla": False,
        "tp_size": 1,
        "dp_size": 1,
        "lora_name": "",
        "pp_size": 1,
        "extra": "",
        "user_data": "",
    }


def test_register_instance_request_uses_empty_instance_id_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPECTRUM_DEPLOYMENT_NAME", raising=False)
    assert _client()._register_instance_request()["instance_id"] == ""


def test_report_event_request_contains_common_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRUM_DEPLOYMENT_NAME", "deploy-a")
    client = _client(
        SubscriberConfig(kvcm_host_ip_port="10.0.0.8:9000", engine_type="vllm")
    )
    events = [{"event_type": "EVENT_HOST_DOWN", "host_down": {}}]

    request = client._report_event_request(events)

    assert request["instance_id"] == "deploy-a"
    assert request["host_ip_port"] == "10.0.0.8:9000"
    assert request["storage_type"] == "ST_VLLM"
    assert request["events"] == events


def test_report_events_for_batches_maps_all_supported_event_types() -> None:
    batch = KVEventBatch(
        ts=1.0,
        events=[
            BlockStored(
                block_hashes=[11, 12],
                parent_block_hash=None,
                token_ids=[1, 2],
                block_size=2,
                lora_id=None,
                medium="GPU",
                lora_name=None,
            ),
            BlockRemoved(block_hashes=[13], medium="CPU"),
            AllBlocksCleared(),
        ],
    )

    events = _client()._report_events_for_batches([batch])

    assert events == [
        {
            "event_type": "EVENT_BLOCK_ADD",
            "block_add": {
                "block_key": "11",
                "medium": "hbm",
                "specs": [],
            },
        },
        {
            "event_type": "EVENT_BLOCK_ADD",
            "block_add": {
                "block_key": "12",
                "medium": "hbm",
                "specs": [],
            },
        },
        {
            "event_type": "EVENT_BLOCK_DELETE",
            "block_delete": {"block_key": "13", "medium": "mem"},
        },
        {"event_type": "EVENT_HOST_DOWN", "host_down": {}},
    ]


def test_report_events_for_batches_maps_unknown_medium_to_empty_string() -> None:
    batch = KVEventBatch(
        ts=1.0,
        events=[BlockRemoved(block_hashes=[13], medium="DISK")],
    )
    assert _client()._report_events_for_batches([batch]) == [
        {
            "event_type": "EVENT_BLOCK_DELETE",
            "block_delete": {"block_key": "13", "medium": ""},
        }
    ]


def test_kvcm_event_types_are_string_enum_values() -> None:
    event_type = _client()._node_register_event()["event_type"]

    assert isinstance(event_type, Enum)
    assert event_type == "EVENT_NODE_REGISTER"


async def test_start_creates_sdk_registers_instance_reports_node_and_starts_heartbeat():
    fake_sdk = FakeSdkClient()
    client = _make_kvcm(
        SubscriberConfig(
            kvcm_host_ip_port="10.0.0.8:9000", kvcm_heartbeat_interval_s=60.0
        ),
        fake_sdk,
    )

    await client.start()
    await client.close()

    fake_sdk.start.assert_awaited_once()
    fake_sdk.register_instance.assert_called_once()
    register_request = fake_sdk.register_instance.call_args.args[0]
    assert register_request["block_size"] == 1

    fake_sdk.report_event.assert_called_once()
    node_request = fake_sdk.report_event.call_args.args[0]
    assert node_request["events"] == [
        {
            "event_type": "EVENT_NODE_REGISTER",
            "node_register": {"mediums": ["hbm", "mem"]},
        }
    ]
    fake_sdk.close.assert_awaited_once()


async def test_start_uses_spectrum_address_from_vservice_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVCM_VSERVICE_ID", "vs-a")
    created_clients: list[FakeSdkClient] = []

    def sdk_client_factory(base_url: str) -> FakeSdkClient:
        client = FakeSdkClient(base_url)
        created_clients.append(client)
        return client

    client = KvcmClient(
        SubscriberConfig(
            kvcm_host_ip_port="10.0.0.8:9000",
            kvcm_heartbeat_interval_s=60.0,
        ),
        medium_mapper=_vllm_medium_mapper,
        storage_type="ST_VLLM",
        supported_mediums=["hbm", "mem"],
        sdk_client_factory=sdk_client_factory,
    )

    await client.start()
    await client.close()

    assert [created.base_url for created in created_clients] == ["spectrum://vs-a"]


async def test_start_resolves_host_ip_port_once_and_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sdk = FakeSdkClient()
    config = SubscriberConfig(
        kvcm_host_ip_port="10.0.0.8:9000", kvcm_heartbeat_interval_s=60.0
    )
    main_thread = threading.current_thread()
    resolver_calls: list[tuple[str, str]] = []
    resolver_threads: list[threading.Thread] = []

    def resolve_host_ip_port(configured: str, health_url: str) -> str:
        resolver_calls.append((configured, health_url))
        resolver_threads.append(threading.current_thread())
        return "10.0.0.7:8123"

    monkeypatch.setattr(
        "subscriber.kvcm.client.resolve_host_ip_port",
        resolve_host_ip_port,
        raising=False,
    )
    client = _make_kvcm(config, fake_sdk)

    await client.start()
    fake_sdk.report_event.reset_mock()
    await client.send_batch(
        [KVEventBatch(ts=1.0, events=[AllBlocksCleared()])], epoch=7
    )
    await client._report_events([client._heartbeat_event()])
    await client.close()

    assert resolver_calls == [("10.0.0.8:9000", config.engine_health_url)]
    assert all(thread is not main_thread for thread in resolver_threads)
    assert [
        call.args[0]["host_ip_port"] for call in fake_sdk.report_event.call_args_list
    ] == ["10.0.0.7:8123", "10.0.0.7:8123"]


async def test_start_propagates_register_failure_and_does_not_start_heartbeat() -> None:
    fake_sdk = FakeSdkClient()
    fake_sdk.register_instance.side_effect = RuntimeError("register failed")
    client = _make_kvcm(SubscriberConfig(), fake_sdk)

    with pytest.raises(RuntimeError, match="register failed"):
        await client.start()

    fake_sdk.report_event.assert_not_called()
    fake_sdk.close.assert_not_called()


async def test_heartbeat_reports_periodically() -> None:
    fake_sdk = FakeSdkClient()
    client = _make_kvcm(
        SubscriberConfig(
            kvcm_host_ip_port="10.0.0.8:9000", kvcm_heartbeat_interval_s=0.01
        ),
        fake_sdk,
    )

    await client.start()
    await asyncio.sleep(0.03)
    await client.close()

    heartbeat_calls = [
        call.args[0]
        for call in fake_sdk.report_event.call_args_list
        if call.args[0]["events"]
        == [{"event_type": "EVENT_HEARTBEAT", "heartbeat": {"system_status": {}}}]
    ]
    assert heartbeat_calls
    fake_sdk.close.assert_awaited_once()


async def test_heartbeat_failure_logs_warning_and_continues(mocker) -> None:
    fake_sdk = FakeSdkClient()
    fake_sdk.report_event.side_effect = [
        {"header": {"status": {"code": "OK"}}},
        RuntimeError("heartbeat failed"),
        {"header": {"status": {"code": "OK"}}},
    ]
    warning = mocker.patch("subscriber.kvcm.client.logger.warning")
    client = _make_kvcm(
        SubscriberConfig(
            kvcm_host_ip_port="10.0.0.8:9000", kvcm_heartbeat_interval_s=0.01
        ),
        fake_sdk,
    )

    await client.start()
    await asyncio.sleep(0.03)
    await client.close()

    warning.assert_any_call(
        "kvcm heartbeat report failed",
        step="kvcm_heartbeat",
        exc_info=True,
    )


async def test_send_batch_reports_events_without_blocking_sdk_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sdk = FakeSdkClient()
    client = _make_kvcm(
        SubscriberConfig(
            kvcm_host_ip_port="10.0.0.8:9000", kvcm_heartbeat_interval_s=60.0
        ),
        fake_sdk,
    )
    main_thread = threading.current_thread()
    report_threads: list[threading.Thread] = []

    async def track_report(request: dict[str, object]) -> dict[str, object]:
        report_threads.append(threading.current_thread())
        return {"header": {"status": {"code": "OK"}}}

    monkeypatch.setattr(fake_sdk, "report_event", track_report)

    await client.start()
    await client.send_batch(
        [KVEventBatch(ts=1.0, events=[AllBlocksCleared()])], epoch=7
    )
    await client.close()

    assert report_threads
    assert report_threads == [main_thread, main_thread]


async def test_send_batch_reports_converted_events() -> None:
    fake_sdk = FakeSdkClient()
    client = _make_kvcm(
        SubscriberConfig(
            kvcm_host_ip_port="10.0.0.8:9000", kvcm_heartbeat_interval_s=60.0
        ),
        fake_sdk,
    )
    batch = KVEventBatch(
        ts=1.0,
        events=[BlockRemoved(block_hashes=[13], medium="CPU")],
    )

    await client.start()
    fake_sdk.report_event.reset_mock()
    await client.send_batch([batch], epoch=7)
    await client.close()

    fake_sdk.report_event.assert_called_once()
    request = fake_sdk.report_event.call_args.args[0]
    assert request["events"] == [
        {
            "event_type": "EVENT_BLOCK_DELETE",
            "block_delete": {"block_key": "13", "medium": "mem"},
        }
    ]


async def test_send_empty_batch_does_not_report_event() -> None:
    fake_sdk = FakeSdkClient()
    client = _make_kvcm(
        SubscriberConfig(
            kvcm_host_ip_port="10.0.0.8:9000", kvcm_heartbeat_interval_s=60.0
        ),
        fake_sdk,
    )

    await client.start()
    fake_sdk.report_event.reset_mock()
    await client.send_batch([], epoch=7)
    await client.close()

    fake_sdk.report_event.assert_not_called()


async def test_send_batch_logs_warning_for_item_results(mocker) -> None:
    fake_sdk = FakeSdkClient()
    fake_sdk.report_event.return_value = {
        "header": {"status": {"code": "OK"}},
        "item_results": ["OK", "INTERNAL_ERROR"],
    }
    warning = mocker.patch("subscriber.kvcm.client.logger.warning")
    client = _make_kvcm(
        SubscriberConfig(
            kvcm_host_ip_port="10.0.0.8:9000", kvcm_heartbeat_interval_s=60.0
        ),
        fake_sdk,
    )
    batch = KVEventBatch(
        ts=1.0,
        events=[BlockRemoved(block_hashes=[13], medium="CPU")],
    )

    await client.start()
    fake_sdk.report_event.reset_mock()
    await client.send_batch([batch], epoch=7)
    await client.close()

    warning.assert_called_once_with(
        "kvcm report_event returned partial item results",
        step="kvcm_send",
        tags={"epoch": 7, "item_results": ["OK", "INTERNAL_ERROR"]},
    )
