import argparse
import subprocess
import sys
import os
import importlib

COMMANDS = {
    "get_instance": "kvcm_ops.kvcm.instance.get_instance",
    "list_instance": "kvcm_ops.kvcm.instance.list_instance",
    "register_instance": "kvcm_ops.kvcm.instance.register_instance",
    "remove_instance": "kvcm_ops.kvcm.instance.remove_instance",

    "get_instance_group": "kvcm_ops.kvcm.instance_group.get_instance_group",
    "list_instance_group": "kvcm_ops.kvcm.instance_group.list_instance_group",
    "create_instance_group": "kvcm_ops.kvcm.instance_group.create_instance_group",
    "update_instance_group": "kvcm_ops.kvcm.instance_group.update_instance_group",
    "remove_instance_group": "kvcm_ops.kvcm.instance_group.remove_instance_group",

    "add_storage": "kvcm_ops.kvcm.storage.add_storage",
    "list_storage": "kvcm_ops.kvcm.storage.list_storage",
    "enable_storage": "kvcm_ops.kvcm.storage.enable_storage",
    "disable_storage": "kvcm_ops.kvcm.storage.disable_storage",
    "update_storage": "kvcm_ops.kvcm.storage.update_storage",
    "remove_storage": "kvcm_ops.kvcm.storage.remove_storage",

    "trace_key": "kvcm_ops.trace.trace_key",
    "trace_uri": "kvcm_ops.trace.trace_uri",

    "config_server": "kvcm_ops.config_server",
}

HELP_MODULE = [
    "kvcm_ops.kvcm.instance._help",
    "kvcm_ops.kvcm.instance_group._help",
    "kvcm_ops.kvcm.storage._help",
    "kvcm_ops.trace._help",
    "kvcm_ops.config_server._help",
]

def main():
    help_message_list = []
    for module_name in HELP_MODULE:
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            print(f"import failed: {module_name}, err={e}")
            continue
        help_message_list.append(getattr(module, "HELP_MESSAGE", None))
    help_message_str = "examples:" + '\n'.join(help_message_list)

    parser = argparse.ArgumentParser(
        prog="python3 -m kvcm_script",
        description="KVCM script entry",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=help_message_str,
    )
    parser.add_argument(
        "command",
        nargs="?",
        help='''subcommands:
  instance:
    get_instance
    list_instance
    register_instance
    remove_instance
  instance_group:
    get_instance_group
    list_instance_group
    create_instance_group
    update_instance_group
    remove_instance_group
  storage:
    add_storage
    list_storage
    enable_storage
    disable_storage
    update_storage
    remove_storage
  trace:
    trace_key
    trace_uri
  config_server:           (use: python3 -m kvcm_ops config_server <sub> ...)
    create-zone              create a new zone
    delete-zone              delete an existing zone
    list-zones               list all zones on the server
    instance_pin            (instance_pin mode)
    server_capability       (detect server routing mode)
        '''
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="arguments passed to the subcommand"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        print("\nAvailable commands:")
        for name in sorted(COMMANDS):
            print(f"  {name:<24} -> {COMMANDS[name]}")
        return 0

    if args.command not in COMMANDS:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        print("\nAvailable commands:")
        for name in sorted(COMMANDS):
            print(f"  {name}")
        return 1

    module = COMMANDS[args.command]
    cmd = [sys.executable, "-m", module] + args.args
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())