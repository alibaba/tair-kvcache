"""Idempotent KVCM instance registration for colocated DashTrace runtimes."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InstanceBootstrapConfig:
    base_url: str = ""
    registration: dict[str, Any] | None = None
    timeout_seconds: float = 300.0
    retry_interval_seconds: float = 1.0

    @classmethod
    def from_env(cls) -> "InstanceBootstrapConfig":
        raw = os.environ.get("DASHTRACE_KVCM_REGISTER_INSTANCE_JSON", "").strip()
        registration = json.loads(raw) if raw else None
        return cls(
            base_url=os.environ.get("DASHTRACE_KVCM_BASE_URL", "").rstrip("/"),
            registration=registration,
            timeout_seconds=float(
                os.environ.get("DASHTRACE_KVCM_REGISTER_TIMEOUT_SECONDS", "300")
            ),
            retry_interval_seconds=float(
                os.environ.get("DASHTRACE_KVCM_REGISTER_RETRY_SECONDS", "1")
            ),
        )

    @property
    def enabled(self) -> bool:
        return self.registration is not None

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("DASHTRACE_KVCM_BASE_URL must be an http(s) URL")
        if not isinstance(self.registration, dict):
            raise ValueError("DASHTRACE_KVCM_REGISTER_INSTANCE_JSON must be an object")
        if not self.registration.get("instance_id"):
            raise ValueError("registration instance_id must not be empty")
        if self.timeout_seconds <= 0 or self.retry_interval_seconds <= 0:
            raise ValueError("instance registration retry settings must be positive")


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1.0) as response:
        return json.loads(response.read())


def _status_code(response: dict[str, Any]) -> str:
    return str(response.get("header", {}).get("status", {}).get("code", ""))


def ensure_instance_registered(
    config: InstanceBootstrapConfig,
    ready: threading.Event,
    stopping: threading.Event,
) -> None:
    """Set ``ready`` only after the configured KVCM instance exists."""
    config.validate()
    if not config.enabled:
        ready.set()
        return

    assert config.registration is not None
    instance_id = str(config.registration["instance_id"])
    deadline = time.monotonic() + config.timeout_seconds
    while not stopping.is_set() and time.monotonic() < deadline:
        try:
            info = _post_json(
                f"{config.base_url}/api/getInstanceInfo",
                {
                    "trace_id": f"dashtrace-bootstrap-info-{os.getpid()}",
                    "instance_id": instance_id,
                },
            )
            if _status_code(info) == "OK":
                ready.set()
                print(f"DashTrace KVCM instance ready: {instance_id}", flush=True)
                return

            registration = dict(config.registration)
            registration["trace_id"] = f"dashtrace-bootstrap-register-{os.getpid()}"
            result = _post_json(
                f"{config.base_url}/api/registerInstance", registration
            )
            if _status_code(result) == "OK":
                ready.set()
                print(f"DashTrace KVCM instance registered: {instance_id}", flush=True)
                return
        except Exception as error:  # noqa: BLE001 - readiness retry is fail-closed
            print(f"DashTrace KVCM instance bootstrap retry: {error}", flush=True)
        stopping.wait(config.retry_interval_seconds)

    if not stopping.is_set():
        print(f"DashTrace KVCM instance bootstrap timed out: {instance_id}", flush=True)


def start_instance_bootstrap(
    ready: threading.Event, stopping: threading.Event
) -> threading.Thread | None:
    config = InstanceBootstrapConfig.from_env()
    config.validate()
    if not config.enabled:
        ready.set()
        return None
    thread = threading.Thread(
        target=ensure_instance_registered,
        args=(config, ready, stopping),
        name="dashtrace-kvcm-bootstrap",
        daemon=True,
    )
    thread.start()
    return thread
