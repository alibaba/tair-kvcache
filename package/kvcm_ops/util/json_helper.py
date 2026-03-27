import json

def pretty_print_json(data : dict, with_common_prefix = True):
    if with_common_prefix:
        print("===========================================")
        print("respose:")
    pretty_str = json.dumps(data, indent=4)
    print(pretty_str)