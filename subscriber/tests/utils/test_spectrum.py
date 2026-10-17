from __future__ import annotations

import importlib.util
import inspect

from subscriber.utils import network
from subscriber.utils.spectrum import SpectrumEndpoint, fetch_spectrum_endpoints


def test_spectrum_fetch_is_available_without_kvcm_dependency() -> None:
    assert importlib.util.find_spec("subscriber.utils.spectrum") is not None
    assert "subscriber.kvcm" not in inspect.getsource(network)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class FakeHttpClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, float]] = []

    async def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append((url, timeout))
        return FakeResponse(self.payload)


async def test_fetch_spectrum_endpoints_validates_response_once() -> None:
    http_client = FakeHttpClient(
        {
            "instances": [
                {"ip": "10.0.0.8", "port": "8000", "weight": 20},
                {"ip": "10.0.0.7", "port": 7000},
                {"ip": "10.0.0.9", "port": 0},
                {"ip": "10.0.0.10", "port": True},
            ]
        }
    )

    endpoints = await fetch_spectrum_endpoints(
        "engine-vs-a", http_client=http_client, refresh_timeout=0.5
    )

    assert http_client.calls == [
        (
            "http://127.0.0.1:8880/api/v1/discovery/virtual-services/"
            "engine-vs-a/instances",
            0.5,
        )
    ]
    assert endpoints == [
        SpectrumEndpoint(ip="10.0.0.7", port=7000),
        SpectrumEndpoint(ip="10.0.0.8", port=8000, weight=20),
    ]
