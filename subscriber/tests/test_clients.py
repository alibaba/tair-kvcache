from __future__ import annotations

import asyncio
import os
import threading
from enum import Enum
from unittest.mock import AsyncMock

import pytest

from subscriber.config import SubscriberConfig
from subscriber.kvcm.base import AbstractKvCacheManagerClient
from subscriber.kvcm.client import KvcmClient
from subscriber.types import AllBlocksCleared, BlockRemoved, BlockStored, KVEventBatch


class FakeSdkClient(AbstractKvCacheManagerClient):
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
        self.ready = True
        self.is_ready = AsyncMock(side_effect=self._is_ready)

    async def _start(self) -> None:
        self.started = True

    async def _is_ready(self) -> bool:
        return self.ready

    async def start(self) -> None:
        self.started = True

    async def is_ready(self) -> bool:
        return self.ready

    async def register_instance(
        self, data: dict[str, object], check_response: bool = True
    ) -> dict[str, object]:
        return {"header": {"status": {"code": "OK"}}}

    async def report_event(
        self, data: dict[str, object], check_response: bool = True
    ) -> dict[str, object]:
        return {"header": {"status": {"code": "OK"}}}

    async def close(self) -> None:
        pass


_VLLM_MEDIUM_MAP = {"GPU": "hbm", "CPU": "mem"}


@pytest.fixture(autouse=True)
def _set_kvcm_vservice_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPECTRUM_APPLICATION_SERVICE_ID", raising=False)
    monkeypatch.setenv("KVCM_VSERVICE_ID", "vs-test")

    async def resolve_host_ip_port() -> str:
        return "10.0.0.8:9000"

    monkeypatch.setattr(
        "subscriber.kvcm.client.resolve_host_ip_port", resolve_host_ip_port
    )


def _vllm_medium_mapper(medium: str | None) -> str:
    if medium is None:
        return ""
    return _VLLM_MEDIUM_MAP.get(medium, "")


def _client(config: SubscriberConfig | None = None) -> KvcmClient:
    client = KvcmClient(
        config or SubscriberConfig(),
        medium_mapper=_vllm_medium_mapper,
        storage_type="ST_EVENT_REPORT",
        supported_mediums=["hbm", "mem"],
        manager_client=FakeSdkClient(),
    )
    client._host_ip_port_value = "10.0.0.8:9000"
    return client


def _make_kvcm(config: SubscriberConfig, fake_sdk: FakeSdkClient) -> KvcmClient:
    return KvcmClient(
        config,
        medium_mapper=_vllm_medium_mapper,
        storage_type="ST_EVENT_REPORT",
        supported_mediums=["hbm", "mem"],
        manager_client=fake_sdk,
    )


def test_storage_type_maps_engine_type() -> None:
    assert (
        _client(SubscriberConfig(engine_type="vllm"))._storage_type == "ST_EVENT_REPORT"
    )
    assert (
        KvcmClient(
            SubscriberConfig(engine_type="unknown"),
            medium_mapper=_vllm_medium_mapper,
            storage_type="ST_UNSPECIFIED",
            supported_mediums=["hbm", "mem"],
            manager_client=FakeSdkClient(),
        )._storage_type
        == "ST_UNSPECIFIED"
    )


def test_register_instance_request_uses_env_instance_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRUM_DEPLOYMENT_NAME", "deploy-a")
    request = _client()._register_instance_request()
    assert request["instance_id"] == "deploy-a"
    assert request["instance_group"] == os.environ.get("KVCM_INSTANCE_GROUP", "")
    assert request["block_size"] == 1
    assert request["location_spec_infos"] == [{"name": "vllm_1", "size": 1}]
    assert request["location_spec_groups"] == [
        {"name": "default", "spec_names": ["vllm_1"]}
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


def test_block_specs_use_registered_location_spec_name() -> None:
    client = _client()
    register_request = client._register_instance_request()
    block_events = client._report_events_for_batches(
        [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockStored(
                        block_hashes=[11],
                        parent_block_hash=None,
                        token_ids=[1, 2],
                        block_size=2,
                        lora_id=None,
                        medium="GPU",
                        lora_name=None,
                    )
                ],
            )
        ]
    )

    registered_spec_names = {
        spec["name"] for spec in register_request["location_spec_infos"]
    }
    block_specs = block_events[0]["block_add"]["specs"]

    assert block_specs == [{"name": "vllm_1", "uri": "vllm://10.0.0.8:9000/hbm"}]
    assert {spec["name"] for spec in block_specs} <= registered_spec_names


def test_register_and_block_specs_use_configured_block_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DS_LLM_ENGINE_CONFIG", '{"block_size": 16}')
    client = _client()

    register_request = client._register_instance_request()
    block_events = client._report_events_for_batches(
        [
            KVEventBatch(
                ts=1.0,
                events=[
                    BlockStored(
                        block_hashes=[11],
                        parent_block_hash=None,
                        token_ids=[1, 2],
                        block_size=16,
                        lora_id=None,
                        medium="GPU",
                        lora_name=None,
                    )
                ],
            )
        ]
    )

    assert register_request["block_size"] == 16
    assert register_request["location_spec_infos"] == [{"name": "vllm_16", "size": 16}]
    assert register_request["location_spec_groups"] == [
        {"name": "default", "spec_names": ["vllm_16"]}
    ]
    assert block_events[0]["block_add"]["specs"] == [
        {"name": "vllm_16", "uri": "vllm://10.0.0.8:9000/hbm"}
    ]


@pytest.mark.parametrize(
    ("raw_config", "expected_block_size"),
    [
        ('{"block_size": 16}', 16),
        ("{invalid-json", 1),
        ("[]", 1),
        ("1", 1),
        ('{"block_size": true}', 1),
        ('{"block_size": 0}', 1),
    ],
)
def test_block_size_falls_back_for_invalid_engine_config_shapes(
    monkeypatch: pytest.MonkeyPatch, raw_config: str, expected_block_size: int
) -> None:
    monkeypatch.setenv("DS_LLM_ENGINE_CONFIG", raw_config)
    assert _client()._register_instance_request()["block_size"] == expected_block_size


def test_register_instance_request_uses_empty_instance_id_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SPECTRUM_DEPLOYMENT_NAME", raising=False)
    assert _client()._register_instance_request()["instance_id"] == ""


def test_register_instance_request_reads_parallelism_from_engine_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DS_LLM_ENGINE_CONFIG",
        '{"block_size": 64, "dtype": "bfloat16", '
        '"tensor_parallel_size": 2, "data_parallel_size": 4, '
        '"pipeline_parallel_size": 3}',
    )
    deployment = _client()._register_instance_request()["model_deployment"]
    assert deployment["tp_size"] == 2
    assert deployment["dp_size"] == 4
    assert deployment["pp_size"] == 3


def test_register_instance_request_falls_back_parallelism_for_invalid_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "DS_LLM_ENGINE_CONFIG",
        '{"tensor_parallel_size": true, "data_parallel_size": -1}',
    )
    deployment = _client()._register_instance_request()["model_deployment"]
    assert deployment["tp_size"] == 1
    assert deployment["dp_size"] == 1
    assert deployment["pp_size"] == 1


def test_report_event_request_contains_common_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPECTRUM_DEPLOYMENT_NAME", "deploy-a")
    client = _client(SubscriberConfig(engine_type="vllm"))
    events = [{"event_type": "EVENT_HOST_DOWN", "host_down": {}}]

    request = client._report_event_request(events)

    assert request["instance_id"] == "deploy-a"
    assert request["host_ip_port"] == "10.0.0.8:9000"
    assert request["storage_type"] == "ST_EVENT_REPORT"
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
                "specs": [{"name": "vllm_1", "uri": "vllm://10.0.0.8:9000/hbm"}],
            },
        },
        {
            "event_type": "EVENT_BLOCK_ADD",
            "block_add": {
                "block_key": "12",
                "medium": "hbm",
                "specs": [{"name": "vllm_1", "uri": "vllm://10.0.0.8:9000/hbm"}],
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
        SubscriberConfig(kvcm_heartbeat_interval_s=60.0),
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


async def test_start_uses_injected_manager_client() -> None:
    fake_sdk = FakeSdkClient()
    client = KvcmClient(
        SubscriberConfig(kvcm_heartbeat_interval_s=60.0),
        medium_mapper=_vllm_medium_mapper,
        storage_type="ST_EVENT_REPORT",
        supported_mediums=["hbm", "mem"],
        manager_client=fake_sdk,
    )

    await client.start()
    await client.close()

    fake_sdk.start.assert_awaited_once()
    fake_sdk.is_ready.assert_awaited_once()
    fake_sdk.register_instance.assert_awaited_once()
    fake_sdk.close.assert_awaited_once()


async def test_start_uses_spectrum_address_from_vservice_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVCM_VSERVICE_ID", "vs-a")
    created_clients: list[FakeSdkClient] = []

    def create_http_client(
        base_url: str, *, request_timeout_seconds: float
    ) -> FakeSdkClient:
        client = FakeSdkClient(base_url)
        created_clients.append(client)
        return client

    monkeypatch.setattr(
        "subscriber.kvcm.client.HttpKvCacheManagerClient", create_http_client
    )

    client = KvcmClient(
        SubscriberConfig(kvcm_heartbeat_interval_s=60.0),
        medium_mapper=_vllm_medium_mapper,
        storage_type="ST_EVENT_REPORT",
        supported_mediums=["hbm", "mem"],
    )

    await client.start()
    await client.close()

    assert [created.base_url for created in created_clients] == ["spectrum://vs-a:6382"]


async def test_start_resolves_host_ip_port_once_and_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sdk = FakeSdkClient()
    config = SubscriberConfig(kvcm_heartbeat_interval_s=60.0)
    resolver_calls: list[None] = []

    async def resolve_host_ip_port() -> str:
        resolver_calls.append(None)
        return "10.0.0.7:8123"

    monkeypatch.setattr(
        "subscriber.kvcm.client.resolve_host_ip_port",
        resolve_host_ip_port,
    )
    client = _make_kvcm(config, fake_sdk)

    await client.start()
    fake_sdk.report_event.reset_mock()
    await client.send_batch(
        [KVEventBatch(ts=1.0, events=[AllBlocksCleared()])], epoch=7
    )
    await client._report_events([client._heartbeat_event()])
    await client.close()

    assert resolver_calls == [None]
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


async def test_start_without_endpoint_drops_batches_then_registers_after_recovery(
    mocker,
) -> None:
    fake_sdk = FakeSdkClient()
    fake_sdk.ready = False
    registered = asyncio.Event()

    async def register_instance(_request: dict[str, object]) -> dict[str, object]:
        registered.set()
        return {"header": {"status": {"code": "OK"}}}

    fake_sdk.register_instance.side_effect = register_instance
    warning = mocker.patch("subscriber.main.logger.warning")
    client = _make_kvcm(SubscriberConfig(kvcm_heartbeat_interval_s=0.01), fake_sdk)

    await client.start()

    fake_sdk.register_instance.assert_not_awaited()
    fake_sdk.report_event.assert_not_awaited()

    from subscriber.main import send_kv_events

    queue = asyncio.Queue()
    batches = [KVEventBatch(ts=1.0, events=[AllBlocksCleared()])]
    await queue.put(mocker.Mock(batches=batches, epoch_snapshot=1))
    coordinator = mocker.Mock()
    coordinator.wait_ready_epoch = AsyncMock(return_value=1)
    coordinator.is_epoch_current.return_value = True
    sender = asyncio.create_task(send_kv_events(client, coordinator, queue))
    await queue.join()

    warning.assert_any_call(
        "kvcm has no available endpoint; starting in not-ready state",
        step="kvcm_register",
    )
    send_warning = next(
        call
        for call in warning.call_args_list
        if call.args == ("failed to send kv event batch to kvcm; dropping batch",)
    )
    assert "kvcm client is not ready" in send_warning.kwargs["tags"]["message"]

    fake_sdk.ready = True
    await asyncio.wait_for(registered.wait(), timeout=0.2)

    fake_sdk.report_event.reset_mock()
    await client.send_batch(batches, epoch=1)

    fake_sdk.report_event.assert_awaited_once()
    sender.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sender
    await client.close()


async def test_report_events_before_start_raises_clear_error() -> None:
    client = _client()

    with pytest.raises(RuntimeError, match="kvcm client has not been started"):
        await client._report_events([client._heartbeat_event()])


async def test_send_batch_before_start_raises_clear_error() -> None:
    client = _client()

    with pytest.raises(RuntimeError, match="kvcm client has not been started"):
        await client.send_batch(
            [KVEventBatch(ts=1.0, events=[AllBlocksCleared()])], epoch=1
        )


async def test_heartbeat_reports_periodically() -> None:
    fake_sdk = FakeSdkClient()
    client = _make_kvcm(
        SubscriberConfig(kvcm_heartbeat_interval_s=0.01),
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
        SubscriberConfig(kvcm_heartbeat_interval_s=0.01),
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


async def test_runtime_kvcm_failure_marks_not_ready_then_reregisters() -> None:
    fake_sdk = FakeSdkClient()
    reregistered = asyncio.Event()

    async def register_instance(_request: dict[str, object]) -> dict[str, object]:
        if fake_sdk.register_instance.await_count == 2:
            reregistered.set()
        return {"header": {"status": {"code": "OK"}}}

    fake_sdk.register_instance.side_effect = register_instance
    fake_sdk.report_event.side_effect = [
        {"header": {"status": {"code": "OK"}}},
        RuntimeError("kvcm unavailable"),
        {"header": {"status": {"code": "OK"}}},
        {"header": {"status": {"code": "OK"}}},
    ]
    client = _make_kvcm(SubscriberConfig(kvcm_heartbeat_interval_s=0.01), fake_sdk)
    batches = [KVEventBatch(ts=1.0, events=[AllBlocksCleared()])]

    await client.start()
    with pytest.raises(RuntimeError, match="kvcm unavailable"):
        await client.send_batch(batches, epoch=1)

    assert not client._registered
    await asyncio.wait_for(reregistered.wait(), timeout=0.2)
    assert client._registered

    await client.send_batch(batches, epoch=1)
    await client.close()


async def test_runtime_kvcm_failure_retries_registration_until_recovered() -> None:
    fake_sdk = FakeSdkClient()
    recovered = asyncio.Event()
    heartbeat_failed = False

    async def register_instance(_request: dict[str, object]) -> dict[str, object]:
        if fake_sdk.register_instance.await_count < 4:
            if fake_sdk.register_instance.await_count > 1:
                raise RuntimeError("kvcm registration unavailable")
        else:
            recovered.set()
        return {"header": {"status": {"code": "OK"}}}

    async def report_event(
        request: dict[str, object], **_kwargs: object
    ) -> dict[str, object]:
        nonlocal heartbeat_failed
        event_type = request["events"][0]["event_type"]
        if event_type == "EVENT_HEARTBEAT" and not heartbeat_failed:
            heartbeat_failed = True
            raise RuntimeError("kvcm unavailable")
        return {"header": {"status": {"code": "OK"}}}

    fake_sdk.register_instance.side_effect = register_instance
    fake_sdk.report_event.side_effect = report_event
    client = _make_kvcm(SubscriberConfig(kvcm_heartbeat_interval_s=0.01), fake_sdk)

    await client.start()
    await asyncio.wait_for(recovered.wait(), timeout=0.3)
    assert fake_sdk.register_instance.await_count == 4
    assert client._registered
    await client.close()


async def test_send_batch_reports_events_without_blocking_sdk_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sdk = FakeSdkClient()
    client = _make_kvcm(
        SubscriberConfig(kvcm_heartbeat_interval_s=60.0),
        fake_sdk,
    )
    main_thread = threading.current_thread()
    report_threads: list[threading.Thread] = []

    async def track_report(
        request: dict[str, object], *, check_response: bool = True
    ) -> dict[str, object]:
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
        SubscriberConfig(kvcm_heartbeat_interval_s=60.0),
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
        SubscriberConfig(kvcm_heartbeat_interval_s=60.0),
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
        SubscriberConfig(kvcm_heartbeat_interval_s=60.0),
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


async def test_send_batch_requests_manager_response_validation() -> None:
    fake_sdk = FakeSdkClient()
    client = _make_kvcm(
        SubscriberConfig(kvcm_heartbeat_interval_s=60.0),
        fake_sdk,
    )
    batch = KVEventBatch(
        ts=1.0,
        events=[BlockRemoved(block_hashes=[13], medium="CPU")],
    )

    await client.start()
    fake_sdk.report_event.reset_mock()
    fake_sdk.report_event.side_effect = RuntimeError(
        "KVCM /api/reportEvent failed: INTERNAL_ERROR "
        "ReportEvent partially failed; see item_results"
    )

    with pytest.raises(RuntimeError, match="INTERNAL_ERROR"):
        await client.send_batch([batch], epoch=7)

    await client.close()

    fake_sdk.report_event.assert_awaited_once()
    assert fake_sdk.report_event.call_args.kwargs == {"check_response": True}
