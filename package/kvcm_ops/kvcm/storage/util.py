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
        "vipserver_domain": args.vipserver_domain,
        "timeout": args.timeout,
        "enable_vipserver": args.enable_vipserver
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

def add_nfs_sub_parser(subparsers):
    parser_nfs = subparsers.add_parser('nfs', help='NFS storage options')
    parser_nfs.add_argument('--root_path', "-r", required=True, help='Root path for NFS, eg. /tmp/dir')
    parser_nfs.add_argument('--key_count_per_file', "-k", required=True, type=positive_int, help='key count per file, eg. 64')
    return parser_nfs

def add_pace_sub_parser(subparsers):
    parser_pace = subparsers.add_parser('pace', help='PACE(tair memory pool) storage options')
    parser_pace.add_argument('--domain', "-d", required=True, help='pace domain')
    parser_pace.add_argument('--vipserver_domain', "--vipserver_domain", required=True, help='pace vipserver domain')
    parser_pace.add_argument('--timeout', "-t", required=True, type=int, help='pace time out config')
    parser_pace.add_argument('--enable_vipserver', action="store_true", help='pace enable_vipserver')
    return parser_pace

def add_3fs_sub_parser(subparsers):
    parser_3fs = subparsers.add_parser('3fs', help='3FS storage options')
    parser_3fs.add_argument('--cluster_name', "-c", required=True, help='cluster_name')
    parser_3fs.add_argument('--mountpoint', "-m", required=True, help='mountpoint, eg. /3fs/stage/3fs/')
    parser_3fs.add_argument('--root_dir', "-r", required=True, help='mountpoint, real file path is mount_point/root_dir')
    parser_3fs.add_argument('--key_count_per_file', "-k", required=True, type=positive_int, help='key count per file, eg. 64')
    parser_3fs.add_argument("--not_touch_file_when_create", action="store_true", help="if set, not touch file when create")
    return parser_3fs

def add_or_update_main(method: str, handle_nfs, handle_pace, handle_3fs):
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

    parser_nfs.set_defaults(func=handle_nfs)
    parser_pace.set_defaults(func=handle_pace)
    parser_3fs.set_defaults(func=handle_3fs)

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