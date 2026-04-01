import argparse
from ..common.http_helper import *
from ..common.common_args import *
from ...util.json_helper import *

'''
curl -g -vvv -X POST http://localhost:56040/api/removeInstanceGroup \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "default_trace_id",
    "name": "default"
}'
'''

def parse_args():
    common_parser = create_common_parser()
    parser = argparse.ArgumentParser(
        prog="python3 script.kvcm.instance_group.remove_instance_group",
        description="kvcm: remove_instance_group.",
        parents=[common_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        required=True,
        help="group name"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = {
        "trace_id": args.trace_id,
        "name": args.name
    }
    result = http_post(args.host, "/api/removeInstanceGroup", data, args.verbose)
    pretty_print_json(result)

if __name__ == "__main__":
    main()