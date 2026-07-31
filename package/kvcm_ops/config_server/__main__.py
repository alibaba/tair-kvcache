"""ConfigServer ops sub-dispatcher.

Usage:
    python3 -m kvcm_ops config_server <subcommand> [args...]

Subcommands:
    zone management:
        create-zone      Create a new zone
        delete-zone      Delete an existing zone
        list-zones       List all zone_ids on the server

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

ZONE_COMMANDS = {
    "create-zone": "create",
    "delete-zone": "delete",
    "list-zones": "list",
}


def main():
    parser = argparse.ArgumentParser(
        prog="python3 -m kvcm_ops config_server",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    all_cmds = list(ZONE_COMMANDS.keys()) + list(COMMANDS.keys())
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

    if args.command in ZONE_COMMANDS:
        action = ZONE_COMMANDS[args.command]
        cmd = [sys.executable, "-m", "kvcm_ops.config_server.zone", action] + args.args
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
