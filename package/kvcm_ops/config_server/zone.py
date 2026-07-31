"""Unified zone lifecycle management (create / delete / list).

Uses the instance_pin HTTP endpoints:

    POST /instancePin/createZone            (body = {trace_id, zone_id})
    POST /instancePin/deleteZone            (body = {trace_id, zone_id})
    POST /instancePin/listZones             (body = {trace_id})

Usage:
    python3 -m kvcm_ops config_server create-zone --zone prod_zone_a
    python3 -m kvcm_ops config_server delete-zone --zone prod_zone_a [--yes]
    python3 -m kvcm_ops config_server list-zones
"""

import argparse
import json
import sys

import requests

from ..kvcm.common.http_helper import http_post_text
from .response_util import check_response_header_text

DEFAULT_URL = "http://127.0.0.1:9101"


def cmd_create(args: argparse.Namespace) -> int:
    if not args.zone:
        raise SystemExit("[ERROR] --zone is required for create-zone")

    body = {
        "trace_id": "kvcm_ops/zone/create",
        "zone_id": args.zone,
    }
    if args.dry_run:
        print(f"[dry-run] would POST /instancePin/createZone: {json.dumps(body)}")
        return 0
    try:
        status, resp = http_post_text(
            args.url, "/instancePin/createZone", body,
            timeout=args.timeout, verbose=args.verbose,
        )
    except requests.RequestException as e:
        raise SystemExit(f"[ERROR] POST /instancePin/createZone failed: {e}")
    if status != 200:
        raise SystemExit(f"[ERROR] POST /instancePin/createZone -> HTTP {status}: {resp}")
    check_response_header_text(resp)
    print(f"[OK] created zone {args.zone!r}: {resp}")

    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    if not args.zone:
        raise SystemExit("[ERROR] --zone is required for delete-zone")

    if not args.yes:
        try:
            confirm = input(
                f"Type the zone_id to confirm deletion ({args.zone!r}): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n[ABORT] confirmation cancelled")
        if confirm != args.zone:
            raise SystemExit(
                f"[ABORT] confirmation {confirm!r} does not match {args.zone!r}"
            )

    if args.dry_run:
        print(f"[dry-run] would delete zone={args.zone!r} on {args.url}")
        return 0

    body = {
        "trace_id": "kvcm_ops/zone/delete",
        "zone_id": args.zone,
    }
    try:
        status, resp = http_post_text(
            args.url, "/instancePin/deleteZone", body,
            timeout=args.timeout, verbose=args.verbose,
        )
    except requests.RequestException as e:
        raise SystemExit(f"[ERROR] POST /instancePin/deleteZone failed: {e}")
    if status != 200:
        raise SystemExit(f"[ERROR] POST /instancePin/deleteZone -> HTTP {status}: {resp}")
    check_response_header_text(resp)
    print(f"[OK] deleted zone {args.zone!r}: {resp}")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    req_body = {"trace_id": "kvcm_ops/zone/list"}
    try:
        status, resp = http_post_text(
            args.url, "/instancePin/listZones", req_body,
            timeout=args.timeout, verbose=args.verbose,
        )
    except requests.RequestException as e:
        raise SystemExit(f"[ERROR] POST /instancePin/listZones failed: {e}")
    if status != 200:
        raise SystemExit(f"[ERROR] POST /instancePin/listZones -> HTTP {status}: {resp}")
    check_response_header_text(resp)
    try:
        obj = json.loads(resp)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] server returned invalid JSON: {e}\nbody={resp[:512]}")
    zones = list(obj.get("zone_ids") or obj.get("zoneIds") or [])

    print(f"total: {len(zones)} zone(s)")
    for h in zones:
        print(f"  - {h}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=DEFAULT_URL,
                        help=f"ConfigServer HTTP base URL, default {DEFAULT_URL}")
    common.add_argument("--zone", default="",
                        help="target zone_id (required for create/delete)")
    common.add_argument("--timeout", type=float, default=5.0,
                        help="HTTP timeout in seconds, default 5")
    common.add_argument("--dry-run", action="store_true",
                        help="print what would be done without sending requests")
    common.add_argument("--verbose", "-v", action="store_true",
                        help="print HTTP request details")

    p = argparse.ArgumentParser(
        prog="python3 -m kvcm_ops config_server <zone-cmd>",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    sub = p.add_subparsers(dest="action", required=True)

    sp_create = sub.add_parser("create", parents=[common], help="create a new zone")
    sp_create.set_defaults(func=cmd_create)

    sp_delete = sub.add_parser("delete", parents=[common], help="delete an existing zone")
    sp_delete.add_argument("--yes", "-y", action="store_true",
                           help="skip the interactive confirmation prompt")
    sp_delete.set_defaults(func=cmd_delete)

    sp_list = sub.add_parser("list", parents=[common], help="list all zone_ids on the server")
    sp_list.set_defaults(func=cmd_list)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
