"""Unified hpnzone lifecycle management (create / delete / list).

Uses the instance_pin HTTP endpoints:

    POST /instancePin/createHpnzone            (body = {trace_id, hpnzone_id})
    POST /instancePin/deleteHpnzone            (body = {trace_id, hpnzone_id})
    POST /instancePin/listHpnzones             (body = {trace_id})

Usage:
    python3 -m kvcm_ops config_server create-hpnzone --hpnzone prod_zone_a
    python3 -m kvcm_ops config_server delete-hpnzone --hpnzone prod_zone_a [--yes]
    python3 -m kvcm_ops config_server list-hpnzones
"""

import argparse
import json
import sys

import requests

from ..kvcm.common.http_helper import http_post_text

DEFAULT_URL = "http://127.0.0.1:9101"


def _check_response_header(resp_text: str) -> None:
    """Raise SystemExit if the JSON response body contains a business error code."""
    try:
        obj = json.loads(resp_text)
    except (json.JSONDecodeError, TypeError):
        return
    header = obj.get("header") or {}
    st = header.get("status") or {}
    code = st.get("code")
    if code is None:
        code = "OK"
    code_str = code if isinstance(code, str) else str(code)
    if code_str not in ("OK", "1"):
        msg = st.get("message", "")
        raise SystemExit(f"[ERROR] server returned {code_str}: {msg}")


def cmd_create(args: argparse.Namespace) -> int:
    if not args.hpnzone:
        raise SystemExit("[ERROR] --hpnzone is required for create-hpnzone")

    body = {
        "trace_id": "kvcm_ops/hpnzone/create",
        "hpnzone_id": args.hpnzone,
    }
    if args.dry_run:
        print(f"[dry-run] would POST /instancePin/createHpnzone: {json.dumps(body)}")
        return 0
    try:
        status, resp = http_post_text(
            args.url, "/instancePin/createHpnzone", body,
            timeout=args.timeout, verbose=args.verbose,
        )
    except requests.RequestException as e:
        raise SystemExit(f"[ERROR] POST /instancePin/createHpnzone failed: {e}")
    if status != 200:
        raise SystemExit(f"[ERROR] POST /instancePin/createHpnzone -> HTTP {status}: {resp}")
    _check_response_header(resp)
    print(f"[OK] created hpnzone {args.hpnzone!r}: {resp}")

    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    if not args.hpnzone:
        raise SystemExit("[ERROR] --hpnzone is required for delete-hpnzone")

    if not args.yes:
        try:
            confirm = input(
                f"Type the hpnzone_id to confirm deletion ({args.hpnzone!r}): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n[ABORT] confirmation cancelled")
        if confirm != args.hpnzone:
            raise SystemExit(
                f"[ABORT] confirmation {confirm!r} does not match {args.hpnzone!r}"
            )

    if args.dry_run:
        print(f"[dry-run] would delete hpnzone={args.hpnzone!r} on {args.url}")
        return 0

    body = {
        "trace_id": "kvcm_ops/hpnzone/delete",
        "hpnzone_id": args.hpnzone,
    }
    try:
        status, resp = http_post_text(
            args.url, "/instancePin/deleteHpnzone", body,
            timeout=args.timeout, verbose=args.verbose,
        )
    except requests.RequestException as e:
        raise SystemExit(f"[ERROR] POST /instancePin/deleteHpnzone failed: {e}")
    if status != 200:
        raise SystemExit(f"[ERROR] POST /instancePin/deleteHpnzone -> HTTP {status}: {resp}")
    _check_response_header(resp)
    print(f"[OK] deleted hpnzone {args.hpnzone!r}: {resp}")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    req_body = {"trace_id": "kvcm_ops/hpnzone/list"}
    try:
        status, resp = http_post_text(
            args.url, "/instancePin/listHpnzones", req_body,
            timeout=args.timeout, verbose=args.verbose,
        )
    except requests.RequestException as e:
        raise SystemExit(f"[ERROR] POST /instancePin/listHpnzones failed: {e}")
    if status != 200:
        raise SystemExit(f"[ERROR] POST /instancePin/listHpnzones -> HTTP {status}: {resp}")
    _check_response_header(resp)
    try:
        obj = json.loads(resp)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] server returned invalid JSON: {e}\nbody={resp[:512]}")
    hpnzones = list(obj.get("hpnzone_ids") or obj.get("hpnzoneIds") or [])

    print(f"total: {len(hpnzones)} hpnzone(s)")
    for h in hpnzones:
        print(f"  - {h}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=argparse.SUPPRESS,
                        help=f"ConfigServer HTTP base URL, default {DEFAULT_URL}")
    common.add_argument("--hpnzone", default=argparse.SUPPRESS,
                        help="target hpnzone_id (required for create/delete)")
    common.add_argument("--timeout", type=float, default=argparse.SUPPRESS,
                        help="HTTP timeout in seconds, default 5")
    common.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                        help="print what would be done without sending requests")
    common.add_argument("--verbose", "-v", action="store_true", default=argparse.SUPPRESS,
                        help="print HTTP request details")

    p = argparse.ArgumentParser(
        prog="python3 -m kvcm_ops config_server <hpnzone-cmd>",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"ConfigServer HTTP base URL, default {DEFAULT_URL}")
    p.add_argument("--hpnzone", default="",
                   help="target hpnzone_id (required for create/delete)")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="HTTP timeout in seconds, default 5")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be done without sending requests")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="print HTTP request details")
    sub = p.add_subparsers(dest="action", required=True)

    sp_create = sub.add_parser("create", parents=[common], help="create a new hpnzone")
    sp_create.set_defaults(func=cmd_create)

    sp_delete = sub.add_parser("delete", parents=[common], help="delete an existing hpnzone")
    sp_delete.add_argument("--yes", "-y", action="store_true",
                           help="skip the interactive confirmation prompt")
    sp_delete.set_defaults(func=cmd_delete)

    sp_list = sub.add_parser("list", parents=[common], help="list all hpnzone_ids on the server")
    sp_list.set_defaults(func=cmd_list)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
