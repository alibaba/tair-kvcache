from ..common.http_helper import *
from ...util.json_helper import *
from .util import parse_simple_args

'''
curl -g -vvv -X POST http://localhost:56040/api/removeStorage \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "default_trace_id",
    "storage_unique_name": "test_file_storage"
}'
'''

def main():
    args = parse_simple_args("remove_storage")
    data = {
        "trace_id": args.trace_id,
        "storage_unique_name": args.unique_name
    }
    result = http_post(args.host, "/api/removeStorage", data, args.verbose)
    pretty_print_json(result)

if __name__ == "__main__":
    main()