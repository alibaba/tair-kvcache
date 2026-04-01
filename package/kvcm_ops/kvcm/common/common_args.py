import argparse

def split_strs(value: str, sep = ','):
    if not value.strip():
        return []
    return [item.strip() for item in value.strip().split(sep)]

def is_list_of_str(obj) -> bool:
    return isinstance(obj, list) and all(isinstance(item, str) for item in obj)

def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} invalid int or <= 0")
    return ivalue

def create_common_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host",
        "-H",
        type=str,
        default="http://localhost:6492",
        help="kv cache manager host url"
    )
    parser.add_argument(
        "--trace_id",
        "-t",
        type=str,
        default="default_trace_id",
        help="custom trace_id"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="print verbose"
    )
    return parser