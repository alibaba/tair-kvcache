"""ConfigServer ops sub-dispatcher.

Usage:
    python3 -m kvcm_ops config_server <subcommand> [args...]

Subcommands:
    hpnzone management:
        create-hpnzone      Create a new hpnzone
        delete-hpnzone      Delete an existing hpnzone
        list-hpnzones       List all hpnzone_ids on the server

    instance_pin mode:
        instance_pin         Manage cells, group pins, and instance pins

    common:
        server_capability    Detect server routing mode
"""

import argparse
import subprocess
import sys

COMMANDS = {
    "instance_pin": "kvcm_ops.config_server.instance_pin",
    "server_capability": "kvcm_ops.config_server.server_capability",
}

HPNZONE_COMMANDS = {
    "create-hpnzone": "create",
    "delete-hpnzone": "delete",
    "list-hpnzones": "list",
}


def main():
    parser = argparse.ArgumentParser(
        prog="python3 -m kvcm_ops config_server",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    all_cmds = list(HPNZONE_COMMANDS.keys()) + list(COMMANDS.keys())
    parser.add_argument(
        "command",
        nargs="?",
        help="subcommand name (see above)",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="arguments passed to the subcommand",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\nAvailable subcommands:")
        for name in all_cmds:
            print(f"  {name}")
        return 0

    if args.command in HPNZONE_COMMANDS:
        action = HPNZONE_COMMANDS[args.command]
        cmd = [sys.executable, "-m", "kvcm_ops.config_server.hpnzone", action] + args.args
        return subprocess.call(cmd)

    if args.command in COMMANDS:
        module = COMMANDS[args.command]
        cmd = [sys.executable, "-m", module] + args.args
        return subprocess.call(cmd)

    print(f"Unknown subcommand: {args.command}", file=sys.stderr)
    print("\nAvailable subcommands:")
    for name in all_cmds:
        print(f"  {name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
