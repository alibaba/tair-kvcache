import argparse
import subprocess
import json
import datetime
from .util import *

def parse_args():
    parser = argparse.ArgumentParser(
        prog="python3 script.trace.trace_key",
        description="trace a key create and delete.",
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
        "--need_uri",
        action="store_true",
        help="need uri"
    )
    parser.add_argument(
        "--instance",
        "-i",
        type=str,
        required=True,
        help="instance"
    )
    parser.add_argument(
        "--key",
        "-k",
        type=int,
        required=True,
        help="cache key"
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

def enumerate_line(files):
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield data
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON in {file_path} at line {line_num}: {e}")

def get_readable_time(timestamp_us):
    seconds = timestamp_us // 1_000_000
    microseconds = timestamp_us % 1_000_000
    dt = datetime.datetime.fromtimestamp(seconds).replace(microsecond=microseconds)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

def find_key_idx(lst, key, block_mask) -> int:
    idx = -1
    try:
        idx = lst.index(key)
    except ValueError:
        return -1
    if isinstance(block_mask, int):
        if block_mask > idx:
            return -1
    else:
        if block_mask[idx]:
            return -1
    return idx

def get_key_write_result(key_idx, write_keys_len:int, start_mask, finish_success_block):
    if isinstance(finish_success_block, int):
        if isinstance(start_mask, int):
            real_write_keys_len = write_keys_len - start_mask
            assert(real_write_keys_len >= finish_success_block)
            assert(key_idx >= start_mask)
            real_key_idx = key_idx - start_mask
            return "success" if finish_success_block > real_key_idx else "fail"
        else:
            real_key_idx = start_mask[:key_idx].count(False)
            return "success" if finish_success_block > real_key_idx else "fail"
    else:
        raise NotImplementedError("not implement now")

def get_access_uri(access_fils, write_session_id, key_idx, start_mask):
    access_json = system_grep_access_get_first(access_fils, f"StartWriteCache.*{write_session_id}")
    # tm = access_json["request_begin_time"]
    src_ip = access_json["client_ip"]
    if isinstance(start_mask, int):
        real_key_idx = key_idx - start_mask
    else:
        real_key_idx = start_mask[:key_idx].count(False)
    uri = access_json["response"]["locations"][real_key_idx]["location_specs"]
    # TODO: 如果response里失败了怎么办？
    return (src_ip, uri)

def trace_key(path:str, instance:str, key:int, max_lines:int, need_uri:bool):
    event_files = list_logs(path, "event_publisher.log")
    access_fils = list_logs(path, "access.log")
    history = []
    write_session_infos = dict()
    i = 1
    for event_data in enumerate_line(event_files):
        i += 1
        if event_data["source"] != instance:
            continue
        event_type = event_data["type"]
        if event_type == "StartWriteCache":
            idx = find_key_idx(event_data["keys"], key, event_data["block_mask"])
            if idx >= 0:
                raw_tm = event_data["trigger_time_us"]
                tm = get_readable_time(raw_tm)
                write_session_id = event_data["write_session_id"]
                access_uri_result = None
                block_mask = event_data["block_mask"]
                if need_uri:
                    access_uri_result = get_access_uri(access_fils, write_session_id, idx, block_mask)
                all_keys_len = len(event_data["keys"])
                info = (idx, all_keys_len, block_mask)
                write_session_infos[write_session_id] = info
                history.append(f"[{tm}][{raw_tm}][StartWriteCache][{write_session_id}][{idx},{all_keys_len}][{access_uri_result}]")
        elif write_session_infos and event_type == "FinishWriteCache":
            write_session_id = event_data["write_session_id"]
            info = write_session_infos.pop(write_session_id, None)
            if info:
                raw_tm = event_data["trigger_time_us"]
                tm = get_readable_time(raw_tm)
                write_session_id = event_data["write_session_id"]
                result = get_key_write_result(info[0], info[1], info[2], event_data["success_block"])
                history.append(f"[{tm}][{raw_tm}][FinishWriteCache][{write_session_id}][{result}]")
        elif event_type == "CacheReclaimSubmit":
            if key in event_data["block_keys"]:
                raw_tm = event_data["trigger_time_us"]
                tm = get_readable_time(raw_tm)
                history.append(f"[{tm}][{raw_tm}][CacheReclaimSubmit][delay:{event_data['delay_us']}us]")
        if i > max_lines:
            break
    return history

def main():
    args = parse_args()
    max_lines = 2**31-1 if args.max_lines < 0 else args.max_lines
    history = trace_key(args.path, args.instance, args.key, max_lines, args.need_uri)
    for l in history:
        print(l)

if __name__ == "__main__":
    main()