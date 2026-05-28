import argparse
from ..common.common_args import *

# for add_storage/update_storage


def gen_nfs_config_data(args):
    storage_spec = {
        "root_path": args.root_path,
        "key_count_per_file": args.key_count_per_file
    }
    return storage_spec


def gen_pace_config_data(args):
    storage_spec = {
        "domain": args.domain,
        "timeout": args.timeout,
        "service_discovery_url": args.service_discovery_url
    }
    return storage_spec


def gen_3fs_config_data(args):
    storage_spec = {
        "cluster_name": args.cluster_name,
        "mountpoint": args.mountpoint,
        "root_dir": args.root_dir,
        "key_count_per_file": args.key_count_per_file,
        "touch_file_when_create": args.touch_file_when_create
    }
    return storage_spec


def gen_vineyard_config_data(args):
    storage_spec = {
        "cluster_name": args.cluster_name,
    }
    if args.heartbeat_timeout_ms is not None:
        storage_spec["heartbeat_timeout_ms"] = args.heartbeat_timeout_ms
    if args.cleanup_grace_ms is not None:
        storage_spec["cleanup_grace_ms"] = args.cleanup_grace_ms
    if args.liveness_check_interval_ms is not None:
        storage_spec["liveness_check_interval_ms"] = args.liveness_check_interval_ms
    return storage_spec


def add_nfs_sub_parser(subparsers):
    parser_nfs = subparsers.add_parser('nfs', help='NFS storage options')
    parser_nfs.add_argument('--root_path', "-r", required=True, help='Root path for NFS, eg. /tmp/dir')
    parser_nfs.add_argument(
        '--key_count_per_file',
        "-k",
        required=True,
        type=positive_int,
        help='key count per file, eg. 64')
    return parser_nfs


def add_pace_sub_parser(subparsers):
    parser_pace = subparsers.add_parser('pace', help='PACE(tair memory pool) storage options')
    parser_pace.add_argument('--domain', "-d", required=True, help='pace domain')
    parser_pace.add_argument('--timeout', "-t", required=True, type=int, help='pace time out config')
    parser_pace.add_argument(
        '--service_discovery_url',
        default="",
        help=(
            'service discovery url, empty string means use --domain directly. '
            'Examples: vipserver://pace.meta.vipserver?timeout=10  |  '
            'spectrum://v-ad2d143d?cache_time=30&retry_time=3&timeout=5000  |  '
            'static://10.0.0.1:8080,10.0.0.2:8080'
        ),
    )
    return parser_pace


def add_3fs_sub_parser(subparsers):
    parser_3fs = subparsers.add_parser('3fs', help='3FS storage options')
    parser_3fs.add_argument('--cluster_name', "-c", required=True, help='cluster_name')
    parser_3fs.add_argument('--mountpoint', "-m", required=True, help='mountpoint, eg. /3fs/stage/3fs/')
    parser_3fs.add_argument(
        '--root_dir',
        "-r",
        required=True,
        help='mountpoint, real file path is mount_point/root_dir')
    parser_3fs.add_argument(
        '--key_count_per_file',
        "-k",
        required=True,
        type=positive_int,
        help='key count per file, eg. 64')
    parser_3fs.add_argument(
        "--not_touch_file_when_create",
        action="store_true",
        help="if set, not touch file when create")
    return parser_3fs


def add_vineyard_sub_parser(subparsers):
    parser_vineyard = subparsers.add_parser('vineyard', help='Vineyard (v6d) storage options')
    parser_vineyard.add_argument('--cluster_name', "-c", required=True, help='vineyard cluster name')
    parser_vineyard.add_argument(
        '--heartbeat_timeout_ms',
        type=int,
        default=None,
        help='heartbeat timeout in ms (server default: 30000)')
    parser_vineyard.add_argument(
        '--cleanup_grace_ms',
        type=int,
        default=None,
        help='cleanup grace period in ms (server default: 300000)')
    parser_vineyard.add_argument(
        '--liveness_check_interval_ms',
        type=int,
        default=None,
        help='liveness check interval in ms (server default: 5000)')
    return parser_vineyard


def add_or_update_main(method: str, handle_nfs, handle_pace, handle_3fs, handle_vineyard):
    common_parser = create_common_parser()
    parser = argparse.ArgumentParser(
        prog=f"python3 script.kvcm.storage.{method}",
        description=f"kvcm: {method}.",
        parents=[common_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--unique_name",
        "-u",
        type=str,
        required=True,
        help="global unique name"
    )

    subparsers = parser.add_subparsers(
        dest='storage_type',
        help='Supported storage types'
    )
    subparsers.required = True

    parser_nfs = add_nfs_sub_parser(subparsers)
    parser_pace = add_pace_sub_parser(subparsers)
    parser_3fs = add_3fs_sub_parser(subparsers)
    parser_vineyard = add_vineyard_sub_parser(subparsers)

    parser_nfs.set_defaults(func=handle_nfs)
    parser_pace.set_defaults(func=handle_pace)
    parser_3fs.set_defaults(func=handle_3fs)
    parser_vineyard.set_defaults(func=handle_vineyard)

    args = parser.parse_args()

    # 执行对应处理函数
    args.func(args)

# for remove/enable/disable


def parse_simple_args(method: str):
    common_parser = create_common_parser()
    parser = argparse.ArgumentParser(
        prog=f"python3 script.kvcm.storage.{method}",
        description=f"kvcm: {method}.",
        parents=[common_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--unique_name",
        "-u",
        type=str,
        required=True,
        help="storage unique name"
    )

    args = parser.parse_args()
    return args
