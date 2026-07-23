from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from typing import Protocol

from .models import EngineUpdate, SourceNotReady
from .reporter import KvcmEventReporter


class TransactionalSource(Protocol):
    async def prepare(self) -> EngineUpdate: ...

    def commit(self, update: EngineUpdate) -> None: ...

    def abort(self, update: EngineUpdate) -> None: ...

    async def close(self) -> None: ...


class CacheEventSubscriberService:
    """Serialize source reads and KVCM ACKs with bounded backpressure."""

    def __init__(
        self,
        source: TransactionalSource,
        reporter: KvcmEventReporter,
        *,
        mediums: tuple[str, ...],
        poll_interval_s: float = 0.1,
        retry_interval_s: float = 1.0,
        heartbeat_interval_s: float = 10.0,
        source_failure_threshold: int = 3,
        health_probe: Callable[[], bool] | None = None,
        health_failure_threshold: int = 3,
    ) -> None:
        if poll_interval_s < 0 or retry_interval_s < 0 or heartbeat_interval_s <= 0:
            raise ValueError(
                "subscriber intervals must be non-negative and heartbeat positive"
            )
        if source_failure_threshold <= 0:
            raise ValueError("source_failure_threshold must be positive")
        if health_failure_threshold <= 0:
            raise ValueError("health_failure_threshold must be positive")
        self._source = source
        self._reporter = reporter
        self._mediums = mediums
        self._poll_interval_s = poll_interval_s
        self._retry_interval_s = retry_interval_s
        self._heartbeat_interval_s = heartbeat_interval_s
        self._source_failure_threshold = source_failure_threshold
        self._health_probe = health_probe
        self._health_failure_threshold = health_failure_threshold
        self._health_failures = 0
        self._external_healthy = health_probe is None
        self._source_healthy = False
        self._stopping = asyncio.Event()

    async def run(self) -> None:
        registered = False
        host_down_reported = False
        consecutive_source_failures = 0
        heartbeat: asyncio.Task[None] | None = None
        try:
            while not self._stopping.is_set():
                try:
                    update = await self._source.prepare()
                except asyncio.CancelledError:
                    raise
                except SourceNotReady:
                    # An empty, newly started vLLM publisher has no sequence-0
                    # batch yet. Keep the host unadvertised and retry without
                    # treating that state as an engine outage.
                    self._source_healthy = False
                    consecutive_source_failures = 0
                    await asyncio.sleep(self._retry_interval_s)
                    continue
                except Exception as error:
                    self._source_healthy = False
                    consecutive_source_failures += 1
                    logging.exception("cache-event source read failed; retrying")
                    if consecutive_source_failures >= self._source_failure_threshold:
                        if registered:
                            try:
                                await asyncio.to_thread(self._reporter.host_down)
                                host_down_reported = True
                            except Exception:
                                logging.exception(
                                    "KVCM HOST_DOWN report failed after source outage"
                                )
                        raise RuntimeError(
                            "cache-event source exceeded its failure threshold"
                        ) from error
                    await asyncio.sleep(self._retry_interval_s)
                    continue
                consecutive_source_failures = 0
                self._source_healthy = True
                if heartbeat is None and not update.full_snapshot:
                    self._source.abort(update)
                    raise RuntimeError(
                        "the first cache-event source update must be an "
                        "authoritative full snapshot"
                    )
                while not self._stopping.is_set():
                    if not registered:
                        try:
                            await asyncio.to_thread(
                                self._reporter.register_node, self._mediums
                            )
                            registered = True
                        except Exception:
                            logging.exception(
                                "KVCM node registration failed; retrying"
                            )
                            await asyncio.sleep(self._retry_interval_s)
                            continue
                    try:
                        if not update.empty:
                            await asyncio.to_thread(self._reporter.send, update)
                    except Exception:
                        logging.exception(
                            "KVCM cache-event report failed; "
                            "retrying without advancing source"
                        )
                        if heartbeat is None:
                            # Do not advertise a newly registered node until
                            # its first authoritative snapshot is visible.
                            try:
                                await asyncio.to_thread(self._reporter.host_down)
                            except Exception:
                                logging.exception(
                                    "KVCM HOST_DOWN failed after initial "
                                    "snapshot rejection"
                                )
                            registered = False
                        else:
                            await self._reregister_after_failure()
                        await asyncio.sleep(self._retry_interval_s)
                        continue
                    self._source.commit(update)
                    if heartbeat is None:
                        heartbeat = asyncio.create_task(
                            self._heartbeat_loop(), name="kvcm-heartbeat"
                        )
                    break
                if not self._stopping.is_set() and self._poll_interval_s > 0:
                    await asyncio.sleep(self._poll_interval_s)
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
            if registered and not host_down_reported:
                try:
                    await asyncio.to_thread(self._reporter.host_down)
                except Exception:
                    logging.exception(
                        "KVCM HOST_DOWN report failed during subscriber shutdown"
                    )
            await self._source.close()

    def stop(self) -> None:
        self._stopping.set()

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_interval_s)
            if self._health_probe is not None:
                try:
                    healthy = await asyncio.to_thread(self._health_probe)
                except Exception:
                    healthy = False
                    logging.exception("cache engine health probe failed")
                self._external_healthy = healthy
                if not healthy:
                    self._health_failures += 1
                    if self._health_failures >= self._health_failure_threshold:
                        logging.error(
                            "cache engine health probe exceeded failure threshold"
                        )
                        self.stop()
                    continue
                self._health_failures = 0
            if not self._source_healthy or not self._external_healthy:
                continue
            try:
                await asyncio.to_thread(self._reporter.heartbeat)
            except Exception:
                logging.exception("KVCM heartbeat failed")
                await self._reregister_after_failure()

    async def _reregister_after_failure(self) -> None:
        try:
            await asyncio.to_thread(self._reporter.register_node, self._mediums)
        except Exception:
            logging.exception("KVCM node re-registration failed")
