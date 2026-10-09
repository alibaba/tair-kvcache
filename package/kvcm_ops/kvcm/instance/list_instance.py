import argparse
from ..common.http_helper import http_post
from ..common.common_args import create_common_parser
from ...util.json_helper import pretty_print_json

'''
curl -g -vvv -X POST http://localhost:56040/api/listInstanceInfo \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "default_trace_id",
    "instance_group_name": "default"
}'
'''

def parse_args():
    common_parser = create_common_parser()
    parser = argparse.ArgumentParser(
        prog="python3 -m kvcm_ops list_instance",
        description="kvcm: list_instance.",
        parents=[common_parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="instance group name; omit to list instances from all groups"
    )

    args = parser.parse_args()
    return args


def list_all_instances(args):
    # ListInstanceInfo on existing servers requires a non-empty group name.
    groups = http_post(args.host, "/api/listInstanceGroup",
                       {"trace_id": args.trace_id}, args.verbose)
    if groups.get("header", {}).get("status", {}).get("code") != "OK":
        raise RuntimeError(f"listInstanceGroup failed, result:[{groups}]")

    instances = []
    for group in groups.get("instance_group", []):
        group_name = group["name"]
        data = {"trace_id": args.trace_id, "instance_group_name": group_name}
        result = http_post(args.host, "/api/listInstanceInfo", data, args.verbose)
        if result.get("header", {}).get("status", {}).get("code") != "OK":
            raise RuntimeError(
                f"listInstanceInfo failed for instance group {group_name!r}, result:[{result}]"
            )
        for instance in result.get("instance_info", []):
            instances.append({"instance_group_name": group_name, **instance})

    return {"header": {"status": {"code": "OK"}}, "instance_info": instances}


def main():
    args = parse_args()
    if args.name is None:
        result = list_all_instances(args)
    else:
        data = {
            "trace_id": args.trace_id,
            "instance_group_name": args.name
        }
        result = http_post(args.host, "/api/listInstanceInfo", data, args.verbose)
    pretty_print_json(result)

if __name__ == "__main__":
    main()
