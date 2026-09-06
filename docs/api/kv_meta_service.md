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
- `Remove`：精确删除给定 keys 的 metadata 和物理数据；不存在的 key 幂等成功。若任一 key 仍有 active
  write session，则整次请求返回 `WRITE_IN_PROGRESS` 且不删除任何 key，避免使该 session 持有的 URI 失效。
- `Trim`：支持 `TS_REMOVE_ALL_CACHE` 和 `TS_REMOVE_ALL_META`；暂不支持 `TS_TIMESTAMP`。

`PutStart` 成功后，即使数据写失败也必须调用 `PutFinish`。客户端崩溃时由 session timeout 清理 active
metadata 和 allocation；在 timeout 前其他 writer 会看到该 key 被 mask。

推荐直接使用 RPM 提供的 C++/Python `KvMetaObjectClient` 完成元数据事务和 exact-size 数据搬运。它会在
任何 IO 前校验完整 batch，自动处理命中、miss 写入、提交和失败回滚；同一 batch 中的不同 value size 会逐
对象调用存储 SDK，并共享一次总超时预算。底层 `KvMetaClient` 处理多地址 failover、响应对齐校验、动态长度
校验以及异常 PutStart 响应的主动回滚。读请求和同配置注册允许在 transport 错误后切换地址；`PutStart`、`PutFinish`、
`Remove`、`Trim` 只对服务端明确返回的 not-leader/not-ready 做 failover，不会重试结果不确定的 transport
错误。此时调用方应按具体 mutation 查询或审计结果；key 级操作可通过 `Get` 确认。`PutStart` 后仍 miss 时
要等原 write timeout 过去再发起新写，未提交 allocation 由 session timeout 或 leader 恢复清理。C++ 接口
用 `ER_INVALID_GRPCSTATUS` 表示这一 transport 结果。

`KvMetaClient` 本身不搬运数据。需要自行编排事务时，可使用独立的 `KvMetaTransferClient` 按响应 URI 和
`ValueLocation.value_size` 搬运；现有固定 block 的 `TransferClient` 不接受 KVMeta 的任意 value size，行为
保持不变。
