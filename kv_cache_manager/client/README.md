## 目录结构
* include: 头文件
* src: 源文件

## KVMeta 客户端（embedding / 变长对象）

业务接入优先使用 `include/kv_meta_object_client.h` 中的 `KvMetaObjectClient`。它把 instance 注册、exact-key
metadata 事务、exact-size 数据搬运和失败回滚组合成一个同步接口；普通固定 block `TransferClient` 不适用于
KVMeta 对象。完整配置和状态机见
[KVMeta 变长对象存储设计](../../docs/design/kv_meta_object_storage.md) 与
[KVMeta 通用对象 API](../../docs/api/kv_meta_service.md)。

### Python client（RTP-LLM producer）

`//kv_cache_manager/client/pybind:kvcm_py_client_lib_wheel` 同时发布 native binding 和
`kv_cache_manager.client.KvMetaObjectClient`。高层 client 不依赖 torch；它按 tensor protocol 接受具有
`is_contiguous()`、`data_ptr()`、`numel()`、`element_size()` 和 `device.type` 的连续 CPU/CUDA tensor：

```python
from kv_cache_manager.client import KvMetaObjectClient, KvMetaObjectClientConfig

config = KvMetaObjectClientConfig(
    addresses=("127.0.0.1:6383",),
    instance_id="model-v1-embedding",
    instance_group="embedding-only-group",
    transfer_client_config=transfer_json,
    max_object_bytes=1024 * 1024 * 1024,
)

with KvMetaObjectClient(config) as client:
    client.save(["embedding", "position"], [embedding, position])
    client.load(["embedding", "position"], [embedding_out, position_out])
    client.remove(["embedding", "position"])
```

一次 Python 逻辑调用会先完整校验 keys、真实 byte size、buffer 和对象上限，再按服务端的 64 objects / 4 GiB
上限分批；不会自动重试或回滚 mutation。RTP 使用每个 receipt 新生成的 UUID key，并由自身 pending/release/GC
负责跨 batch 清理。CUDA producer 在调用 `save` 前仍须由框架侧同步对应 device stream。

需要自行编排控制面和数据面时，才直接使用下面的低层 `KvMetaClient`：

`include/kv_meta_client.h` 是独立于现有 `MetaClient` 的 exact-key 元数据客户端，随
`kv_cache_manager_client.so` 一起发布。配置中的地址必须指向服务端
`kvcm.kv_meta.rpc_port`，不能填写原 MetaService 端口。

```cpp
#include <kv_meta_client.h>

kv_cache_manager::KvMetaClientConfig config;
config.addresses = {"127.0.0.1:6383"};
config.instance_id = "model-v1-embedding";
auto client = kv_cache_manager::KvMetaClient::Create(config);

auto [ec, storage_config] =
    client->RegisterInstance("trace-register", "embedding-only-group", "");
auto [start_ec, write] =
    client->StartWrite("trace-put", {"key-a", "key-b"}, {1536, 4096}, 30);
// 仅写 write.key_mask=false 对应的 write.locations；每项使用自己的 value_size。
if (start_ec == kv_cache_manager::ER_OK && !write.locations.empty()) {
    // ...按紧凑 locations 完成 exact-size 数据写入...
    auto finish_ec = client->FinishWrite(
        "trace-finish", write.write_session_id, std::vector<bool>(write.locations.size(), true));
}
```

`StartWrite` 返回的 `locations` 仅对应 `key_mask=false` 的请求项。不同项可以有不同
`value_size`；该值必须作为数据面 IOV 的有效长度。一次会话采用整批失败语义，只要
`success_keys` 中存在 `false`，服务端就回滚整次会话；V1 不承诺多 key 对并发 `Get` 的线性化同时可见。
全部 key 已 committed 且尺寸一致时，`write_session_id` 和 `locations` 都为空，此时不能调用 `FinishWrite`。
一旦返回非空 session，无论数据面成功或失败都必须调用 `FinishWrite`，失败项用 `false` 回滚。

`KvMetaClient` 只提供 metadata/allocation API。自行编排时，应配套使用 `KvMetaTransferClient`，按返回 URI 和
`KvMetaValueLocation.value_size` 搬运；不要复用根据普通 KV cache 定长 spec 校验 buffer 的 `TransferClient`。
