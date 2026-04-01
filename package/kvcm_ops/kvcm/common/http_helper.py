import requests
import json

def http_post(host: str, api: str, data: dict, verbose = False) -> dict:
    host_url = host + api
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    def print_request(prepared):
        print("===========================================")
        print("request:")
        # 打印请求方法和 URL
        print(f'{prepared.method} {prepared.url}')
        # 打印请求头
        for key, value in prepared.headers.items():
            print(f'{key}: {value}')
        # 打印请求体（注意：json 参数会被自动序列化为字符串）
        print(prepared.body)

    req = requests.Request('POST', host_url, json=data, headers=headers)
    prepared = req.prepare()

    if verbose:
        print_request(prepared)

    session = requests.Session()
    response = session.send(prepared, timeout=30)

    if response.status_code != 200:
        print(f"error status code : {response.status_code}")
        raise RuntimeError(f"error status code : {response.status_code}")
    
    return json.loads(response.text)