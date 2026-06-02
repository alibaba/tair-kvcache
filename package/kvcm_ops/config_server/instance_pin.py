"""Manage cells, group pins, and instance pins on a ConfigServer running in instance_pin mode.

In instance_pin mode, routing is an explicit mapping table rather than hash-based:
    hpnzone -> cells (service discovery URLs) -> instances (pinned to a cell)
    instance_group -> cell (group-level pin, lower priority than instance pin)

All operations go through the ConfigServer HTTP interface (default port 9101).
Every subcommand except the listing ones requires --hpnzone.

Subcommands — cell management:
    list-cells           List all cells registered in a hpnzone
    register-cell        Register a cell (service discovery URL) in a hpnzone
    unregister-cell      Unregister a cell (must have no instance pins or group pins)

Subcommands — group pin management:
    list-group-pins      List all group pins in a hpnzone
    register-group-pin   Pin an instance_group to a cell
    reassign-group-pin   Move a group pin to a different cell
    unregister-group-pin Remove a group pin

Subcommands — instance management:
    register-instance    Register an instance (optionally pin to a specific cell)
    resolve-instance     Resolve which cell an instance maps to
    reassign-instance    Move an instance pin to a different cell
    unregister-instance  Remove an instance pin

Examples:
    # List cells in hpnzone "prod_zone_a"
    python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a list-cells

    # Register a cell
    python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \\
        register-cell --cell-url url://kvcm-cell-0

    # Pin group "groupA" to a cell
    python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \\
        register-group-pin --group groupA --cell-url url://kvcm-cell-1

    # Register an instance (auto-resolve via group pin / fallback)
    python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \\
        register-instance --instance model-x:dep-007 --group groupA

    # Resolve an instance to see which cell it maps to
    python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \\
        resolve-instance --instance model-x:dep-007 --group groupA

    # Reassign an instance to a different cell
    python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a \\
        reassign-instance --instance model-x:dep-007 --cell-url url://kvcm-cell-2
"""

import argparse
import json
import sys

import requests

from ..kvcm.common.http_helper import http_post_text

DEFAULT_URL = "http://127.0.0.1:9101"

PIN_SOURCE_NAMES = {
    0: "UNSPECIFIED",
    1: "INSTANCE_PIN",
    2: "GROUP_PIN",
    3: "FALLBACK_NEW",
    "PIN_SOURCE_UNSPECIFIED": "UNSPECIFIED",
    "PIN_SOURCE_INSTANCE_PIN": "INSTANCE_PIN",
    "PIN_SOURCE_GROUP_PIN": "GROUP_PIN",
    "PIN_SOURCE_FALLBACK_NEW": "FALLBACK_NEW",
}


def require_hpnzone(args: argparse.Namespace) -> str:
    if not args.hpnzone:
        raise SystemExit("[ERROR] this subcommand requires the global --hpnzone HPNZONE_ID")
    return args.hpnzone


def _post(url: str, api: str, body: dict, timeout: float, verbose: bool) -> dict:
    """POST JSON and return parsed response; raise SystemExit on HTTP or JSON errors."""
    try:
        status, resp_text = http_post_text(
            url, api, body, timeout=timeout, verbose=verbose,
        )
    except requests.RequestException as e:
        raise SystemExit(f"[ERROR] POST {url}{api} failed: {e}")
    if status != 200:
        raise SystemExit(f"[ERROR] POST {url}{api} -> HTTP {status}: {resp_text[:512]}")
    try:
        obj = json.loads(resp_text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] server returned invalid JSON: {e}\nbody={resp_text[:512]}")
    header = obj.get("header") or {}
    st = header.get("status") or {}
    code = st.get("code")
    if code is None:
        code = "OK"
    code_str = code if isinstance(code, str) else str(code)
    if code_str not in ("OK", "1"):
        msg = st.get("message", "")
        raise SystemExit(f"[ERROR] server returned {code_str}: {msg}")
    return obj


def normalize_source(raw) -> str:
    if isinstance(raw, int):
        return PIN_SOURCE_NAMES.get(raw, f"UNKNOWN({raw})")
    if isinstance(raw, str):
        return PIN_SOURCE_NAMES.get(raw, raw)
    return str(raw)


# ===== cell subcommands =====

def cmd_list_cells(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {"trace_id": "kvcm_ops/instance_pin/list-cells", "hpnzone_id": hpnzone}
    result = _post(args.url, "/instancePin/listCells", body, args.timeout, args.verbose)
    cells = result.get("cells") or []
    print(f"hpnzone {hpnzone!r}: {len(cells)} cell(s)")
    for c in cells:
        cell_url = c.get("cellUrl") or c.get("cell_url", "")
        pin_count = c.get("instancePinCount") or c.get("instance_pin_count", 0)
        group_refs = c.get("groupPinBackRefs") or c.get("group_pin_back_refs") or []
        groups_str = ", ".join(group_refs) if group_refs else "(none)"
        print(f"  {cell_url}  instances={pin_count}  group_pins=[{groups_str}]")
    return 0


def cmd_register_cell(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {
        "trace_id": "kvcm_ops/instance_pin/register-cell",
        "hpnzone_id": hpnzone,
        "cell_url": args.cell_url,
    }
    if args.dry_run:
        print(f"[dry-run] would POST /instancePin/registerCell: {json.dumps(body)}")
        return 0
    _post(args.url, "/instancePin/registerCell", body, args.timeout, args.verbose)
    print(f"[OK] registered cell {args.cell_url!r} in hpnzone {hpnzone!r}")
    return 0


def cmd_unregister_cell(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {
        "trace_id": "kvcm_ops/instance_pin/unregister-cell",
        "hpnzone_id": hpnzone,
        "cell_url": args.cell_url,
    }
    if args.dry_run:
        print(f"[dry-run] would POST /instancePin/unregisterCell: {json.dumps(body)}")
        return 0
    _post(args.url, "/instancePin/unregisterCell", body, args.timeout, args.verbose)
    print(f"[OK] unregistered cell {args.cell_url!r} from hpnzone {hpnzone!r}")
    return 0


# ===== group pin subcommands =====

def cmd_list_group_pins(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {"trace_id": "kvcm_ops/instance_pin/list-group-pins", "hpnzone_id": hpnzone}
    result = _post(args.url, "/instancePin/listGroupPins", body, args.timeout, args.verbose)
    pins = result.get("groupPins") or result.get("group_pins") or []
    print(f"hpnzone {hpnzone!r}: {len(pins)} group pin(s)")
    for p in pins:
        group = p.get("instanceGroup") or p.get("instance_group", "")
        cell = p.get("cellUrl") or p.get("cell_url", "")
        print(f"  {group} -> {cell}")
    return 0


def cmd_register_group_pin(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {
        "trace_id": "kvcm_ops/instance_pin/register-group-pin",
        "hpnzone_id": hpnzone,
        "instance_group": args.group,
        "cell_url": args.cell_url,
    }
    if args.dry_run:
        print(f"[dry-run] would POST /instancePin/registerGroupPin: {json.dumps(body)}")
        return 0
    _post(args.url, "/instancePin/registerGroupPin", body, args.timeout, args.verbose)
    print(f"[OK] pinned group {args.group!r} -> {args.cell_url!r} in hpnzone {hpnzone!r}")
    return 0


def cmd_reassign_group_pin(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {
        "trace_id": "kvcm_ops/instance_pin/reassign-group-pin",
        "hpnzone_id": hpnzone,
        "instance_group": args.group,
        "new_cell_url": args.cell_url,
    }
    if args.dry_run:
        print(f"[dry-run] would POST /instancePin/reassignGroupPin: {json.dumps(body)}")
        return 0
    _post(args.url, "/instancePin/reassignGroupPin", body, args.timeout, args.verbose)
    print(f"[OK] reassigned group {args.group!r} -> {args.cell_url!r} in hpnzone {hpnzone!r}")
    return 0


def cmd_unregister_group_pin(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {
        "trace_id": "kvcm_ops/instance_pin/unregister-group-pin",
        "hpnzone_id": hpnzone,
        "instance_group": args.group,
    }
    if args.dry_run:
        print(f"[dry-run] would POST /instancePin/unregisterGroupPin: {json.dumps(body)}")
        return 0
    _post(args.url, "/instancePin/unregisterGroupPin", body, args.timeout, args.verbose)
    print(f"[OK] unpinned group {args.group!r} in hpnzone {hpnzone!r}")
    return 0


# ===== instance subcommands =====

def cmd_register_instance(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {
        "trace_id": "kvcm_ops/instance_pin/register-instance",
        "hpnzone_id": hpnzone,
        "instance_id": args.instance,
        "instance_group": args.group or "",
    }
    if args.pin_url:
        body["pin_url"] = args.pin_url
    if not body["instance_group"] and not args.pin_url:
        raise SystemExit("[ERROR] --group is required when --pin-url is not specified")
    if args.dry_run:
        print(f"[dry-run] would POST /instancePin/registerInstance: {json.dumps(body)}")
        return 0
    result = _post(args.url, "/instancePin/registerInstance", body, args.timeout, args.verbose)
    cell = result.get("cellUrl") or result.get("cell_url", "N/A")
    source = normalize_source(result.get("source", 0))
    epoch = result.get("mappingEpoch") or result.get("mapping_epoch", "N/A")
    print(f"[OK] instance {args.instance!r} -> {cell}")
    print(f"     source={source}  mapping_epoch={epoch}")
    return 0


def cmd_resolve_instance(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {
        "trace_id": "kvcm_ops/instance_pin/resolve-instance",
        "hpnzone_id": hpnzone,
        "instance_id": args.instance,
        "instance_group": args.group,
    }
    result = _post(args.url, "/instancePin/resolveInstance", body, args.timeout, args.verbose)
    cell = result.get("cellUrl") or result.get("cell_url", "N/A")
    source = normalize_source(result.get("source", 0))
    epoch = result.get("mappingEpoch") or result.get("mapping_epoch", "N/A")
    print(f"instance {args.instance!r} -> {cell}")
    print(f"  source={source}  mapping_epoch={epoch}")
    return 0


def cmd_reassign_instance(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {
        "trace_id": "kvcm_ops/instance_pin/reassign-instance",
        "hpnzone_id": hpnzone,
        "instance_id": args.instance,
        "new_cell_url": args.cell_url,
    }
    if args.dry_run:
        print(f"[dry-run] would POST /instancePin/reassignInstance: {json.dumps(body)}")
        return 0
    _post(args.url, "/instancePin/reassignInstance", body, args.timeout, args.verbose)
    print(f"[OK] reassigned instance {args.instance!r} -> {args.cell_url!r} in hpnzone {hpnzone!r}")
    return 0


def cmd_unregister_instance(args: argparse.Namespace) -> int:
    hpnzone = require_hpnzone(args)
    body = {
        "trace_id": "kvcm_ops/instance_pin/unregister-instance",
        "hpnzone_id": hpnzone,
        "instance_id": args.instance,
    }
    if args.dry_run:
        print(f"[dry-run] would POST /instancePin/unregisterInstance: {json.dumps(body)}")
        return 0
    _post(args.url, "/instancePin/unregisterInstance", body, args.timeout, args.verbose)
    print(f"[OK] unregistered instance {args.instance!r} from hpnzone {hpnzone!r}")
    return 0


# ===== parser =====

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--url", default=argparse.SUPPRESS,
                        help=f"ConfigServer HTTP base URL, default {DEFAULT_URL}")
    common.add_argument("--hpnzone", default=argparse.SUPPRESS,
                        help="target hpnzone_id; required by every subcommand")
    common.add_argument("--timeout", type=float, default=argparse.SUPPRESS,
                        help="HTTP timeout in seconds, default 5")
    common.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                        help="print the request body without sending")
    common.add_argument("--verbose", "-v", action="store_true", default=argparse.SUPPRESS,
                        help="print HTTP request details")

    p = argparse.ArgumentParser(
        prog="python3 -m kvcm_ops config_server instance_pin",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default=DEFAULT_URL,
                   help=f"ConfigServer HTTP base URL, default {DEFAULT_URL}")
    p.add_argument("--hpnzone", default="",
                   help="target hpnzone_id; required by every subcommand")
    p.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds, default 5")
    p.add_argument("--dry-run", action="store_true", help="print the request body without sending")
    p.add_argument("--verbose", "-v", action="store_true", help="print HTTP request details")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ----- cell -----
    sp = sub.add_parser("list-cells", parents=[common], help="list all cells in a hpnzone")
    sp.set_defaults(func=cmd_list_cells)

    sp = sub.add_parser("register-cell", parents=[common], help="register a cell (service discovery URL)")
    sp.add_argument("--cell-url", required=True, help="cell service discovery URL")
    sp.set_defaults(func=cmd_register_cell)

    sp = sub.add_parser("unregister-cell", parents=[common],
                        help="unregister a cell (fails if it still has instance pins or group pins)")
    sp.add_argument("--cell-url", required=True, help="cell service discovery URL")
    sp.set_defaults(func=cmd_unregister_cell)

    # ----- group pin -----
    sp = sub.add_parser("list-group-pins", parents=[common], help="list all group pins in a hpnzone")
    sp.set_defaults(func=cmd_list_group_pins)

    sp = sub.add_parser("register-group-pin", parents=[common], help="pin an instance_group to a cell")
    sp.add_argument("--group", required=True, help="instance_group name")
    sp.add_argument("--cell-url", required=True, help="target cell URL")
    sp.set_defaults(func=cmd_register_group_pin)

    sp = sub.add_parser("reassign-group-pin", parents=[common], help="move a group pin to a different cell")
    sp.add_argument("--group", required=True, help="instance_group name")
    sp.add_argument("--cell-url", required=True, help="new target cell URL")
    sp.set_defaults(func=cmd_reassign_group_pin)

    sp = sub.add_parser("unregister-group-pin", parents=[common], help="remove a group pin")
    sp.add_argument("--group", required=True, help="instance_group name")
    sp.set_defaults(func=cmd_unregister_group_pin)

    # ----- instance -----
    sp = sub.add_parser("register-instance", parents=[common],
                        help="register an instance; resolves to a cell via group pin or fallback")
    sp.add_argument("--instance", required=True, help="instance_id")
    sp.add_argument("--group", help="instance_group (required unless --pin-url is set)")
    sp.add_argument("--pin-url", help="explicit cell URL to pin to (bypasses group pin / fallback)")
    sp.set_defaults(func=cmd_register_instance)

    sp = sub.add_parser("resolve-instance", parents=[common], help="resolve which cell an instance maps to")
    sp.add_argument("--instance", required=True, help="instance_id")
    sp.add_argument("--group", required=True, help="instance_group (needed for group pin fallback)")
    sp.set_defaults(func=cmd_resolve_instance)

    sp = sub.add_parser("reassign-instance", parents=[common], help="move an instance pin to a different cell")
    sp.add_argument("--instance", required=True, help="instance_id")
    sp.add_argument("--cell-url", required=True, help="new target cell URL")
    sp.set_defaults(func=cmd_reassign_instance)

    sp = sub.add_parser("unregister-instance", parents=[common], help="remove an instance pin")
    sp.add_argument("--instance", required=True, help="instance_id")
    sp.set_defaults(func=cmd_unregister_instance)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
