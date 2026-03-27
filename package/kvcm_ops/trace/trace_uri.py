import argparse
import subprocess
import json
import datetime
from .util import *
from .trace_key import trace_key

def parse_args():
    parser = argparse.ArgumentParser(
        prog="python3 script.trace.trace_key",
        description="trace a uri create and delete.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=".",
        help="log path"
    )
    parser.add_argument(
        "--instance",
        "-i",
        type=str,
        default="",
        help="instance"
    )
    parser.add_argument(
        "--uri",
        "-u",
        type=str,
        required=True,
        help="uri"
    )
    parser.add_argument(
        "--max_lines",
        "-m",
        type=int,
        default=-1,
        help="max trace lines"
    )

    args = parser.parse_args()
    return args

def get_origin_false_index(lst, idx):
    count = 0
    for i, value in enumerate(lst):
        if value is False:
            if count == idx:
                return i
            count += 1
    raise IndexError(f"第 {idx} 个 False 不存在")

def get_keys_and_spec(all_start_write_access:list[dict], instance, uri) -> list[(str, int, str)]:
    results = []
    for access_json in all_start_write_access:
        real_instance = instance
        if instance:
            if access_json["request"]["instance_id"] != instance:
                continue
        else:
            real_instance = access_json["request"]["instance_id"]
        locations = access_json["response"]["locations"]
        real_idx = -1
        spec = None
        for idx, location in enumerate(locations):
            location_specs = location["location_specs"]
            for location_spec in location_specs:
                if location_spec["uri"] == uri:
                    real_idx = idx
                    spec = location_spec["name"]
                    break
            if spec:
                break
        assert(real_idx>=0)
        block_mask = access_json["response"]["block_mask"]
        if "offset" in block_mask:
            origin_key_idx = block_mask["offset"] + real_idx
        else:
            origin_key_idx = get_origin_false_index(block_mask["bool_masks"]["values"], real_idx)
        results.append((access_json["request_begin_time"], access_json["client_ip"], real_instance, int(access_json["request"]["block_keys"][origin_key_idx]), spec))
    return results


def trace_uri(path:str, instance:str, uri:str, max_lines:int):
    access_fils = list_logs(path, "access.log")
    all_start_write_access = system_grep_access_get_all(access_fils, f"StartWriteCache.*instance.*{uri}")
    key_spec_vec = get_keys_and_spec(all_start_write_access, instance, uri)
    for tm, ip, real_instance, key, spec in key_spec_vec:
        print(f"[{tm}][{ip}],instance[{real_instance}],key[{key}],spec[{spec}] history:")
        history = trace_key(path, real_instance, key, max_lines, False)
        for l in history:
            print(f"\t{l}")


def main():
    args = parse_args()
    max_lines = 2**31-1 if args.max_lines < 0 else args.max_lines
    trace_uri(args.path, args.instance, args.uri, max_lines)

if __name__ == "__main__":
    main()