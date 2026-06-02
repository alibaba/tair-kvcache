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

def http_get_text(host: str, api: str, timeout: float = 5.0, verbose: bool = False) -> str:
    """GET and return the raw response body text; non-200 raises RuntimeError (with body excerpt).

    Unlike http_post, this does not force JSON parsing; the caller decides how to handle the body.
    """
    host_url = host + api
    if verbose:
        print("===========================================")
        print(f"GET {host_url}")
    response = requests.get(host_url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"GET {host_url} -> HTTP {response.status_code}: {response.text[:512]}"
        )
    return response.text


def http_post_text(host: str, api: str, body, timeout: float = 5.0,
                   verbose: bool = False) -> tuple[int, str]:
    """POST and return (status_code, response_text); the caller decides success/failure.

    body can be a dict/list (will be json.dumps'd) or an already-serialized str.
    Unlike http_post, this does not force status==200 and does not parse JSON ——
    suitable when the server returns non-JSON text or when error bodies need to be inspected.
    """
    host_url = host + api
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if isinstance(body, (dict, list)):
        payload = json.dumps(body, ensure_ascii=False)
    else:
        payload = body
    if verbose:
        print("===========================================")
        print(f"POST {host_url}")
        print(payload)
    response = requests.post(
        host_url, data=payload.encode("utf-8"), headers=headers, timeout=timeout
    )
    return response.status_code, response.text