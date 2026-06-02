"""Detect the routing mode of a running ConfigServer.

Calls POST /serverCapability/getServerCapability to discover whether the
server is running in hash_bucket or instance_pin mode.

Usage:
    python3 -m kvcm_ops config_server server_capability
    python3 -m kvcm_ops config_server server_capability --url http://10.0.0.1:9101
    python3 -m kvcm_ops config_server server_capability --url http://10.0.0.1:9101 --json
"""

import argparse
import json
import sys

import requests

from ..kvcm.common.http_helper import http_post_text
from .response_util import check_response_header

DEFAULT_URL = "http://127.0.0.1:9101"

ROUTING_MODE_NAMES = {
    0: "UNSPECIFIED",
    1: "HASH_BUCKET",
    2: "INSTANCE_PIN",
    "ROUTING_MODE_UNSPECIFIED": "UNSPECIFIED",
    "ROUTING_MODE_HASH_BUCKET": "HASH_BUCKET",
    "ROUTING_MODE_INSTANCE_PIN": "INSTANCE_PIN",
}


def detect_mode(url: str, timeout: float, verbose: bool) -> dict:
    body = {"trace_id": "kvcm_ops/server_capability"}
    status, resp = http_post_text(
        url, "/serverCapability/getServerCapability", body,
        timeout=timeout, verbose=verbose,
    )
    if status != 200:
        raise SystemExit(f"[ERROR] POST /serverCapability/getServerCapability -> HTTP {status}: {resp[:512]}")
    try:
        result = json.loads(resp)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] server returned invalid JSON: {e}\nbody={resp[:512]}")
    check_response_header(result)
    return result


def normalize_mode(raw) -> str:
    if isinstance(raw, int):
        return ROUTING_MODE_NAMES.get(raw, f"UNKNOWN({raw})")
    if isinstance(raw, str):
        return ROUTING_MODE_NAMES.get(raw, raw)
    return str(raw)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python3 -m kvcm_ops config_server server_capability",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"ConfigServer HTTP base URL, default {DEFAULT_URL}")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds, default 5")
    p.add_argument("--verbose", "-v", action="store_true", help="print HTTP request details")
    p.add_argument("--json", action="store_true", help="print raw JSON response")
    args = p.parse_args()

    try:
        result = detect_mode(args.url, args.timeout, args.verbose)
    except requests.RequestException as e:
        raise SystemExit(f"[ERROR] failed to connect to {args.url}: {e}")

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    mode_raw = result.get("mode") or result.get("routing_mode", "UNKNOWN")
    mode = normalize_mode(mode_raw)
    version = result.get("serverVersion") or result.get("server_version", "N/A")
    print(f"routing_mode   = {mode}")
    print(f"server_version = {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
