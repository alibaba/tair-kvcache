from __future__ import annotations

import argparse
import asyncio
import logging
import signal

import requests

from kv_cache_manager.py_connector.common.manager_client import KvCacheManagerClient

from .models import BlockRecord, LocationSpec
from .reporter import KvcmEventReporter
from .rtp_source import RtpCacheSource
from .service import CacheEventSubscriberService
from .vllm_source import VllmCacheSource


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report engine KV-cache state to KVCM")
    parser.add_argument("--engine", choices=("rtp", "vllm"), required=True)
    parser.add_argument("--manager-uri", required=True)
    parser.add_argument("--instance-group", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--host-ip-port", required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dtype", default="unknown")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--cache-group-count", type=int, default=1)
    parser.add_argument("--bytes-per-group", type=int, default=0)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--full-refresh-interval", type=float, default=300.0)
    parser.add_argument("--heartbeat-interval", type=float, default=10.0)
    parser.add_argument("--source-failure-threshold", type=int, default=3)
    parser.add_argument("--manager-request-timeout", type=float, default=1.0)
    parser.add_argument("--max-events-per-request", type=int, default=4096)
    parser.add_argument("--source-timeout", type=float, default=1.0)
    parser.add_argument("--engine-health-url")
    parser.add_argument("--engine-health-timeout", type=float, default=1.0)
    parser.add_argument("--engine-health-failure-threshold", type=int, default=3)
    parser.add_argument("--rtp-endpoints", help="comma-separated host:port endpoints")
    parser.add_argument("--vllm-pub-endpoint")
    parser.add_argument("--vllm-replay-endpoint")
    parser.add_argument("--vllm-topic", default="")
    return parser


def _check_engine_health(url: str, timeout_s: float) -> bool:
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    return True


def _parse_rtp_endpoints(value: str | None, dp_size: int) -> tuple[str, ...]:
    endpoints = tuple(item.strip() for item in (value or "").split(",") if item.strip())
    if len(endpoints) != dp_size:
        raise ValueError(
            "RTP requires exactly one endpoint per DP rank: "
            f"expected {dp_size}, got {len(endpoints)}"
        )
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("RTP endpoints must be unique")
    return endpoints


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.block_size <= 0:
        raise ValueError("block-size must be positive")
    if args.tp_size <= 0 or args.dp_size <= 0 or args.pp_size <= 0:
        raise ValueError("tp-size, dp-size, and pp-size must be positive")
    if args.cache_group_count <= 0:
        raise ValueError("cache-group-count must be positive")
    if args.bytes_per_group < 0:
        raise ValueError("bytes-per-group must be non-negative")
    spec_sizes = {
        f"group_{index}": args.bytes_per_group
        for index in range(args.cache_group_count)
    }

    def spec_factory(block: BlockRecord) -> list[LocationSpec]:
        groups = block.group_ids or (0,)
        unknown = [group for group in groups if f"group_{group}" not in spec_sizes]
        if unknown:
            raise ValueError(f"unregistered cache group ids: {unknown}")
        return [
            LocationSpec(
                f"group_{group}",
                f"event-report://{args.host_ip_port}/{block.medium}"
                f"?engine={args.engine}&group_id={group}",
            )
            for group in groups
        ]

    client = KvCacheManagerClient(
        args.manager_uri,
        instance_id=args.instance_id,
        auto_discover_leader=True,
        request_timeout_seconds=args.manager_request_timeout,
    )
    reporter = KvcmEventReporter(
        client,
        instance_id=args.instance_id,
        host_ip_port=args.host_ip_port,
        spec_factory=spec_factory,
        max_events_per_request=args.max_events_per_request,
    )

    if args.engine == "rtp":
        rtp_endpoints = _parse_rtp_endpoints(args.rtp_endpoints, args.dp_size)
        source = RtpCacheSource(
            rtp_endpoints,
            timeout_s=args.source_timeout,
            full_refresh_interval_s=args.full_refresh_interval,
            expected_block_size=args.block_size,
            cache_group_count=args.cache_group_count,
        )
        mediums = ("hbm",)
    else:
        if args.dp_size != 1:
            raise ValueError(
                "vLLM currently requires dp-size=1; multiple publishers "
                "must not silently subscribe to rank 0 only"
            )
        if not args.vllm_pub_endpoint or not args.vllm_replay_endpoint:
            raise ValueError("vLLM requires --vllm-pub-endpoint and --vllm-replay-endpoint")
        source = VllmCacheSource(
            args.vllm_pub_endpoint,
            args.vllm_replay_endpoint,
            topic=args.vllm_topic,
            replay_timeout_s=args.source_timeout,
            full_refresh_interval_s=args.full_refresh_interval,
            expected_block_size=args.block_size,
            cache_group_count=args.cache_group_count,
        )
        mediums = ("hbm", "mem")

    # Validate and construct the engine source before mutating KVCM. Invalid
    # endpoints, unsupported DP topology, or missing optional dependencies
    # must not leave behind a partially registered instance.
    reporter.register_instance(
        instance_group=args.instance_group,
        block_size=args.block_size,
        spec_sizes=spec_sizes,
        model_name=args.model_name,
        dtype=args.dtype,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        pp_size=args.pp_size,
    )

    service = CacheEventSubscriberService(
        source,
        reporter,
        mediums=mediums,
        poll_interval_s=args.poll_interval,
        retry_interval_s=args.poll_interval,
        heartbeat_interval_s=args.heartbeat_interval,
        source_failure_threshold=args.source_failure_threshold,
        health_probe=(
            (
                lambda: _check_engine_health(
                    args.engine_health_url, args.engine_health_timeout
                )
            )
            if args.engine_health_url
            else None
        ),
        health_failure_threshold=args.engine_health_failure_threshold,
    )

    async def run_service() -> None:
        loop = asyncio.get_running_loop()
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signal_number, service.stop)
        await service.run()

    try:
        asyncio.run(run_service())
    except KeyboardInterrupt:
        service.stop()
    finally:
        client.close()


if __name__ == "__main__":
    main()
