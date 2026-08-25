from .util import *
from ..common.http_helper import *
from ...util.json_helper import *

'''
curl -g -vvv -X POST http://localhost:56040/api/addStorage \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "default_trace_id",
    "storage": {
        "global_unique_name": "test_file_storage",
        "nfs": {
            "root_path": "/tmp/my_tmp_dir",
            "key_count_per_file": 8
        },
        "check_storage_available_when_open": true
    }
}'
'''


def create_add_storage_data(args, storage_type: str, storage_spec: dict):
    storage = {
        "global_unique_name": args.unique_name,
        storage_type: storage_spec,
        "check_storage_available_when_open": True
    }
    if storage_type == "event_report":
        storage["storage_type"] = args.event_report_storage_type
    elif storage_type == "tair_mem_pool":
        storage["storage_type"] = get_pace_storage_type(args.media_type)
    return {
        "trace_id": args.trace_id,
        "storage": storage
    }


def http_post_and_print(host: str, data: dict, verbose: bool):
    result = http_post(host, "/api/addStorage", data, verbose)
    pretty_print_json(result)


def handle_nfs(args):
    storage_spec = gen_nfs_config_data(args)
    data = create_add_storage_data(args, "nfs", storage_spec)
    http_post_and_print(args.host, data, args.verbose)


def handle_pace(args):
    storage_spec = gen_pace_config_data(args)
    data = create_add_storage_data(args, "tair_mem_pool", storage_spec)
    http_post_and_print(args.host, data, args.verbose)


def handle_3fs(args):
    storage_spec = gen_3fs_config_data(args)
    data = create_add_storage_data(args, "threefs", storage_spec)
    http_post_and_print(args.host, data, args.verbose)


def handle_event_report(args):
    storage_spec = gen_event_report_config_data(args)
    data = create_add_storage_data(args, "event_report", storage_spec)
    http_post_and_print(args.host, data, args.verbose)


def main():
    add_or_update_main("add_storage", handle_nfs, handle_pace, handle_3fs, handle_event_report)


if __name__ == "__main__":
    main()
