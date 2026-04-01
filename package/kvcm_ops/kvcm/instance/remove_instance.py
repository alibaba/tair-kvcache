import argparse
from ..common.http_helper import *
from ..common.common_args import *
from ...util.json_helper import *

'''
curl -g -vvv -X POST http://localhost:56040/api/listInstanceInfo \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id" : "default_trace_id",
    "instance_group" : "",
    "instance_id" : "test_instance_id"
}'
'''

def parse_args():
    common_parser = create_common_parser()
    parser = argparse.ArgumentParser(
        prog="python3 script.kvcm.instance.remove_instance",
        description="kvcm: remove_instance.",
        parents=[common_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--instance",
        "-i",
        type=str,
        required=True,
        help="instance_id"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = {
        "trace_id": args.trace_id,
        "instance_group": "",
        "instance_id": args.instance
    }
    result = http_post(args.host, "/api/removeInstance", data, args.verbose)
    pretty_print_json(result)

if __name__ == "__main__":
    main()