## 目录结构
* include: 头文件
* src: 源文件

## KVMetaClient（embedding / 通用对象）

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
auto finish_ec = client->FinishWrite(
    "trace-finish", write.write_session_id, std::vector<bool>(write.locations.size(), true));
```

`StartWrite` 返回的 `locations` 仅对应 `key_mask=false` 的请求项。不同项可以有不同
`value_size`；该值必须作为数据面 IOV 的有效长度。一次会话采用整批失败语义，只要
`success_keys` 中存在 `false`，服务端就回滚整次会话；V1 不承诺多 key 对并发 `Get` 的线性化同时可见。

`KvMetaClient` 只提供元数据/allocation API。不要直接复用现有固定 block `TransferClient` 搬运这些对象：
它根据普通 KVCache instance 的定长 spec 初始化并校验 buffer size。embedding connector 应按返回 URI 和
`KvMetaValueLocation.value_size` 使用独立的动态长度 IO adapter。
