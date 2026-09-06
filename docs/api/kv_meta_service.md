# KVMeta 通用对象 API

KVMeta 是用于 embedding 等变长 opaque value 的独立 gRPC 服务。服务端需配置非零
`kvcm.kv_meta.rpc_port`；该端口与现有 MetaService 端口不同。完整数据一致性和部署约束见
[KVMeta 通用对象存储设计](../design/kv_meta_object_storage.md)。

## 请求契约

- `RegisterInstance`：`instance_group` 必须是 KVMeta 专用 group；注册幂等，但 schema 或 group 不一致会失败。
- `Get`：`keys` 不得为空或重复；`locations`、`hit_mask.values` 均与 keys 严格等长，miss 对应空 location。
- `PutStart`：`value_sizes` 与 keys 严格等长，每项必须大于 0。`key_mask=true` 表示相同尺寸的对象已存在或
  正在写；已有尺寸不同则整次请求返回 `SIZE_MISMATCH`，且不产生 allocation。`locations` 只包含 mask=false
  的项。
- `PutFinish`：`success_keys.values` 必须存在、非空，并与 `PutStartResponse.locations` 等长。任一 false 回滚整批。
- `Remove`：精确删除给定 keys 的 metadata 和物理数据；不存在的 key 幂等成功。
- `Trim`：支持 `TS_REMOVE_ALL_CACHE` 和 `TS_REMOVE_ALL_META`；暂不支持 `TS_TIMESTAMP`。

`PutStart` 成功后，即使数据写失败也必须调用 `PutFinish`。客户端崩溃时由 session timeout 清理 active
metadata 和 allocation；在 timeout 前其他 writer 会看到该 key 被 mask。

推荐直接使用 RPM 提供的 C++ `KvMetaClient`，它会处理多地址 failover、响应对齐校验、动态长度校验以及
异常 PutStart 的主动回滚。

`KvMetaClient` 不搬运数据。现有固定 block 的 `TransferClient` 不接受 KVMeta 的任意 value size；connector
必须按响应 URI 和 `ValueLocation.value_size` 使用独立的动态长度数据面实现。
