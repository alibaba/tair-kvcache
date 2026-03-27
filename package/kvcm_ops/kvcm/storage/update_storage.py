from .util import *
from ..common.http_helper import *
from ...util.json_helper import *

'''
curl -g -vvv -X POST http://localhost:56040/api/updateStorage \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "default_trace_id",
    "storage": {
        "global_unique_name": "test_file_storage",
        "nfs": {
            "root_dir": "/tmp/my_tmp_dir",
            "key_count_per_file": 8
        },
        "check_storage_available_when_open": true,
    },
    "force_update": false
}'
'''

def create_update_storage_data(args, storage_type: str, storage_spec: dict):
    return {
        "trace_id": args.trace_id,
        "storage" : {
            "global_unique_name" : args.unique_name,
            storage_type : storage_spec,
            "check_storage_available_when_open": True
        },
        "force_update": False
    }

def http_post_and_print(host: str, data: dict, verbose: bool):
    result = http_post(host, "/api/updateStorage", data, verbose)
    pretty_print_json(result)

def handle_nfs(args):
    storage_spec = gen_nfs_config_data(args)
    data = create_update_storage_data(args, "nfs", storage_spec)
    http_post_and_print(args.host, data, args.verbose)
    
def handle_pace(args):
    storage_spec = gen_pace_config_data(args)
    data = create_update_storage_data(args, "tair_mem_pool", storage_spec)
    http_post_and_print(args.host, data, args.verbose)

def handle_3fs(args):
    storage_spec = gen_3fs_config_data(args)
    data = create_update_storage_data(args, "threefs", storage_spec)
    http_post_and_print(args.host, data, args.verbose)

def main():
    add_or_update_main("update_storage", handle_nfs, handle_pace, handle_3fs)

if __name__ == "__main__":
    main()