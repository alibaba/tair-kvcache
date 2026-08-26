# Tair KVCache Manager Python Client

`tair-kvcache-manager-client` is the lightweight HTTP client for Tair KVCache
Manager. It contains the Manager client and its service-discovery helpers only;
transfer SDKs, inference connectors, PyTorch, and native extensions are not
included.

## Installation

```bash
pip install tair-kvcache-manager-client
```

Python 3.9 or newer is required.

## Build

From the repository root:

```bash
bazelisk build //kv_cache_manager/manager_client:kv_cache_manager_manager_client_wheel.dist
```

## Usage

```python
from kv_cache_manager.manager_client import KvCacheManagerClient

client = KvCacheManagerClient("http://127.0.0.1:6382")
try:
    response = client.get_storage_configs_by_instance_group({
        "trace_id": "pace-bootstrap",
        "instance_group": "example-group",
    })
    storage_configs_json = response["storage_configs"]
finally:
    client.close()
```

`storage_configs` is a JSON-encoded array string using the same format returned
by `register_instance`. The method reuses the standard service-discovery,
leader retry, timeout, and business-status checks.

Static service discovery is also available through the Manager URI:

```python
client = KvCacheManagerClient(
    "static://10.0.0.1:6382,10.0.0.2:6382",
    auto_discover_leader=True,
)
try:
    response = client.get_cluster_info({"trace_id": "example"})
finally:
    client.close()
```
