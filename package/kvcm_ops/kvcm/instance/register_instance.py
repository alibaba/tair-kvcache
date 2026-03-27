import argparse
from ..common.json_data import *
from ..common.http_helper import *
from ..common.common_args import *
from ...util.json_helper import *
from typing import List

'''
curl -g -vvv -X POST http://localhost:6492/api/registerInstance \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_213",
    "instance_group": "default",
    "instance_id": "instance1",
    "model_deployment": {
        "model_name": "test_model",
        "dtype": "FP8",
        "use_mla": false,
        "tp_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "extra": "extra_info",
        "user_data": "user_data"
    },
    "block_size": 128,
    "location_spec_infos": [
        {"name": "tp0", "size": 1024}
    ]
}'
'''

class LocationSpecInfo(JsonData):
    def __init__(self,
                 name: str,
                 size: int):
        self._name = name
        self._size = size
        self.check()
    
    def to_json_data(self) -> dict:
        return {
            "name" : self._name,
            "size" : self._size
        }
    
    def check(self) -> bool:
        if self._size <= 0:
            raise RuntimeError(f"LocationSpecInfo size {self._size} <= 0")
        return True
    
def split_location_spec_infos_value(value: str):
    location_spec_info_strs = split_strs(value, ';')
    result = []
    for location_spec_info_str in location_spec_info_strs:
        location_spec_info = split_strs(location_spec_info_str)
        if len(location_spec_info) != 2:
            raise argparse.ArgumentTypeError(f"Invalid location_spec_info_str format: expected 'name,size', got '{value}'")
        name, size = location_spec_info
        result.append(LocationSpecInfo(name, int(size)))
    return result

def location_spec_infos_to_json_data(location_spec_infos: List[LocationSpecInfo]):
    location_spec_infos_list = []
    for location_spec_info in location_spec_infos:
        location_spec_infos_list.append(location_spec_info.to_json_data())
    return location_spec_infos_list

def parse_args():
    common_parser = create_common_parser()
    parser = argparse.ArgumentParser(
        prog="python3 script.kvcm.instance.get_instance",
        description="kvcm: get_instance.",
        parents=[common_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--instance_group",
        "-g",
        type=str,
        required=True,
        help="instance_group"
    )

    parser.add_argument(
        "--instance_id",
        "-i",
        type=str,
        required=True,
        help="instance_id"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="test_model",
        help="model_name"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="FP8",
        help="model_dtype"
    )

    parser.add_argument(
        "--use_mla",
        action="store_true",
        help="model_use_mla"
    )

    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="model_tp_size"
    )

    parser.add_argument(
        "--dp_size",
        type=int,
        default=1,
        help="model_dp_size"
    )

    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
        help="model_pp_size"
    )

    parser.add_argument(
        "--extra",
        type=str,
        default="",
        help="model_extra"
    )

    parser.add_argument(
        "--user_data",
        type=str,
        default="",
        help="model_user_data"
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="block_size"
    )

    parser.add_argument(
        "--location_spec_infos",
        type=split_location_spec_infos_value,
        default=[LocationSpecInfo("tp0", 1024)],
        help="location spec infos"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = {
        "trace_id": args.trace_id,
        "instance_group": args.instance_group,
        "instance_id": args.instance_id,
        "model_deployment": {
            "model_name": args.model_name,
            "dtype": args.dtype,
            "use_mla": args.use_mla,
            "tp_size": args.tp_size,
            "dp_size": args.dp_size,
            "pp_size": args.pp_size,
            "extra": args.extra,
            "user_data": args.user_data
        },
        "block_size": args.block_size,
        "location_spec_infos": location_spec_infos_to_json_data(args.location_spec_infos)
    }
    result = http_post(args.host, "/api/registerInstance", data, args.verbose)
    pretty_print_json(result)

if __name__ == "__main__":
    main()