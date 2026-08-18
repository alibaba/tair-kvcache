import copy

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

def create_update_storage_data(args, storage_type: str, storage_spec: dict, pace_storage_type=None):
    storage = {
        "global_unique_name": args.unique_name,
        storage_type: storage_spec,
        "check_storage_available_when_open": True
    }
    if storage_type == "tair_mem_pool":
        if pace_storage_type is None:
            pace_storage_type = get_pace_storage_type(args.media_type)
        storage["storage_type"] = pace_storage_type
    return {
        "trace_id": args.trace_id,
        "storage": storage,
        "force_update": False
    }

def get_existing_pace_identity(host: str, trace_id: str, unique_name: str, verbose: bool):
    result = http_post(host, "/api/listStorage", {"trace_id": trace_id}, verbose)
    status_code = result.get("header", {}).get("status", {}).get("code")
    if status_code not in ("OK", 1):
        raise RuntimeError(
            f"list storage failed while resolving existing PACE media type, status: {status_code}")

    for storage in result.get("storage", []):
        if storage.get("global_unique_name") != unique_name:
            continue

        storage_spec = storage.get("tair_mem_pool")
        if not isinstance(storage_spec, dict):
            raise RuntimeError(f"storage '{unique_name}' is not a PACE storage")

        media_type = storage_spec.get("media_type", 0)
        if type(media_type) is not int or media_type not in (0, 2, 5):
            raise RuntimeError(f"storage '{unique_name}' has invalid PACE media_type: {media_type}")

        storage_type = storage.get("storage_type", "ST_UNSPECIFIED")
        if storage_type in ("ST_UNSPECIFIED", "ST_TAIRMEMPOOL"):
            # Older Managers may omit the type. Preserve the legacy identity,
            # including the historical ST_TAIRMEMPOOL + media_type=5 form.
            return media_type, "ST_TAIRMEMPOOL"
        if storage_type == "ST_TAIRMEMPOOL_SSD":
            if media_type != 5:
                raise RuntimeError(
                    f"storage '{unique_name}' has ST_TAIRMEMPOOL_SSD but media_type is {media_type}, expected 5")
            return media_type, storage_type
        raise RuntimeError(f"storage '{unique_name}' has incompatible storage_type: {storage_type}")

    raise RuntimeError(f"PACE storage '{unique_name}' does not exist")


def http_post_and_print(host: str, data: dict, verbose: bool):
    result = http_post(host, "/api/updateStorage", data, verbose)
    pretty_print_json(result)

def handle_nfs(args):
    storage_spec = gen_nfs_config_data(args)
    data = create_update_storage_data(args, "nfs", storage_spec)
    http_post_and_print(args.host, data, args.verbose)
    
def handle_pace(args):
    media_type, pace_storage_type = get_existing_pace_identity(
        args.host, args.trace_id, args.unique_name, args.verbose)
    expected_storage_type = (
        "ST_TAIRMEMPOOL_SSD" if args.storage_type == "pace_ssd" else "ST_TAIRMEMPOOL")
    if pace_storage_type != expected_storage_type:
        expected_subcommand = "pace_ssd" if pace_storage_type == "ST_TAIRMEMPOOL_SSD" else "pace"
        raise RuntimeError(
            f"storage '{args.unique_name}' has storage_type {pace_storage_type}; "
            f"use the '{expected_subcommand}' subcommand to update it")

    resolved_args = args
    if args.media_type is None:
        resolved_args = copy.copy(args)
        resolved_args.media_type = media_type
    storage_spec = gen_pace_config_data(resolved_args)
    data = create_update_storage_data(
        resolved_args, "tair_mem_pool", storage_spec, pace_storage_type=pace_storage_type)
    http_post_and_print(args.host, data, args.verbose)

def handle_3fs(args):
    storage_spec = gen_3fs_config_data(args)
    data = create_update_storage_data(args, "threefs", storage_spec)
    http_post_and_print(args.host, data, args.verbose)

def handle_vineyard(args):
    storage_spec = gen_vineyard_config_data(args)
    data = create_update_storage_data(args, "vineyard", storage_spec)
    http_post_and_print(args.host, data, args.verbose)

def main():
    add_or_update_main("update_storage", handle_nfs, handle_pace, handle_3fs, handle_vineyard)

if __name__ == "__main__":
    main()
