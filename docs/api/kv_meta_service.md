# KVMeta 通用对象 API

## 1. API 定位

KVMeta 是面向 embedding 等变长 opaque value 的 exact-key 元数据与写事务 API。协议定义在
`kv_cache_manager/protocol/protobuf/kv_meta_service.proto`，完整 gRPC service 名为
`kv_cache_manager.proto.kv_meta.MetaService`。

KVMeta 只管理对象 key、真实字节数、storage location 和生命周期，不传输对象 bytes，也不保存 tensor shape、
dtype 等业务信息。推荐调用方使用 `KvMetaObjectClient`，由它组合本文 RPC 和 exact-size 数据面。

总体架构、隔离和 HA 设计见 [KVMeta 变长对象存储设计](../design/kv_meta_object_storage.md)。
本文描述当前已实现的 V1 wire contract；设计文档第 13 节的 capability、generation、cleanup ledger 和 object-set
均为后续演进方案，现有 client/server 不得假设这些字段或 RPC 已存在。

## 2. 调用前提

1. 服务端必须配置非零 `kvcm.kv_meta.rpc_port`；这是独立于既有 MetaService 的 gRPC 端口；
2. Instance Group 必须只用于 KVMeta，不能混入普通 KV cache instance；
3. 调用方先执行 `RegisterInstance`，并使用响应中的权威 `storage_configs` 初始化数据面；
4. 每次 RPC 都通过 `CommonResponseHeader.status` 判断业务结果，不能只看 gRPC transport status。

### 2.1 与原 `kv_meta_service.proto` 的关系

实现沿用仓库原有的 `kv_meta_service.proto`、package、service 名和 RPC 名，没有再建立第二套协议。变长对象能力以
protobuf 向后兼容的方式占用原消息中的新 field number：`ValueLocation.value_size=4`、
`GetResponse.hit_mask=3`、`PutStartRequest.value_sizes=5` 和 `PutFinishRequest.success_keys=5`，并补齐明确错误码；
既有字段编号没有改写。

这里的“wire format 可演进”不等于旧 client/server 可以混用新语义：缺少 `value_sizes` 或 `success_keys` 的旧请求
会被新版服务端拒绝；缺少 `hit_mask` 或 `value_size` 的旧响应也会被新版官方 client fail closed。部署 KVMeta V1
时，client library、生成的 proto stub 和 server 必须使用包含上述字段的匹配版本。既有 KV cache MetaService
协议不受影响。

## 3. 总体调用流程

### 3.1 写入

```text
RegisterInstance（每个 client 初始化时，幂等）
        |
        v
PutStart(keys, value_sizes, write_timeout_seconds)
        |
        +-- key_mask=true  -> 同尺寸 committed 对象，不写数据
        |
        +-- key_mask=false -> 按 compact locations 写入 exact bytes
                                  |
                                  v
                    PutFinish(session, success_keys)
```

`PutStart` 成功并返回非空 session 后，无论数据写成功还是失败，调用方都必须执行 `PutFinish`。客户端崩溃或
transport 结果不确定时，服务端 session timeout/leader recovery 负责清理不可读的 active 对象。

### 3.2 读取和删除

```text
Get(keys) -> 校验全部 hit/location/size -> 数据面 Load
Remove(keys) -> metadata Sync -> backend Delete
Trim(instance) -> 按策略清理整个 KVMeta instance
```

## 4. 通用契约

### 4.1 字符串与批次边界

| 项目 | 约束 |
|---|---:|
| 每次 keys 数量 | 1..64，且不能重复 |
| key | 1..512 bytes |
| instance id / group / session id | 1..512 bytes |
| `user_data` | 0..64 KiB |
| 单 value | 1 byte..1 GiB |
| 单批 value 总量 | 不超过 4 GiB |
| `write_timeout_seconds` | 1..1800 |

服务端和官方 client 都执行边界校验。超限请求在 allocation 或数据 I/O 前失败。

### 4.2 响应头

所有响应包含 `CommonResponseHeader`：

- `status.code`：KVMeta 业务错误码；
- `status.message`：简要结果和服务端错误上下文；
- `request_id`：服务端请求标识；
- `tracer_result`：仅在请求开启 span tracer 时返回。

非 `OK` 响应中的业务 payload 不应被使用。

## 5. RPC 契约

### 5.1 `RegisterInstance`

输入 `instance_group`、`instance_id` 和可选 `user_data`：

- group 必须已存在并且只包含 KVMeta instance；
- 相同 instance/group、KVMeta schema 和 `user_data` 的重复注册幂等；
- group、schema 或既有 instance 配置不一致时失败；
- 成功响应的 `storage_configs` 是后续 transfer client 的权威 backend 配置。

### 5.2 `GetInstanceInfo`

按公共 `instance_id` 查询注册信息。响应返回公共 instance id，不暴露 KVCM 内部 namespace 编码。

### 5.3 `Get`

- `query_type` 只接受 `QT_UNSPECIFIED` 或 `QT_BATCH_GET`；
- `metas` 在 V1 未实现，非空时返回 `UNSUPPORTED`；
- `locations` 和 `hit_mask.values` 始终与请求 keys 严格等长、同下标；
- miss 的 `hit_mask=false`，对应 `locations[i]` 为空；
- active 对象不可读，表现为 miss；
- hit location 必须包含恰好一个名为 `value` 的 URI，并携带真实 `value_size`。

V1 `Get` 不创建 server-side read lease，返回 location 后不会 pin 物理 allocation。调用方必须确保对应数据面 Load
结束前没有其他 client 执行同对象的 Remove/Trim/GC；需要读删并发保护的后续设计见总体文档第 13 节。

### 5.4 `PutStart`

`keys` 与 `value_sizes` 必须严格等长。成功响应有两种数组形状：

| 字段 | 对齐方式 |
|---|---|
| `key_mask.values` | 与请求 keys 等长；`true` 表示相同尺寸对象已 committed |
| `locations` | 只包含 `key_mask=false` 的 miss，按请求中的相对顺序紧凑排列 |

其他语义：

- 全部命中时 `write_session_id` 和 `locations` 均为空；
- 有任一 miss 时返回非空 session，且每个 location 的 `value_size`、URI `size` 与请求值完全相等；
- 同 key 的 committed 对象尺寸不同时，整批返回 `SIZE_MISMATCH`，不创建 allocation；
- 任一 key 仍 active 时，整批返回 `WRITE_IN_PROGRESS`，不把它误报为命中；
- 容量、storage type quota 或 active-session 数量不足时，不会返回可用 session；已产生的候选 allocation 或
  reservation 会在返回前进入补偿清理。

storage Create 抛出的标准或未知异常会转换为 `IO_ERROR`，不会穿透服务线程。服务端会对异常前已明确拿到的
singleton allocation 做一次补偿删除；异常调用本身若在 provider 端已经分配但没有返回 URI，则只能由 backend
orphan 清理发现，服务端不会猜测或重放该 Create。

### 5.5 `PutFinish`

- `write_session_id` 必须非空且属于相同 `instance_id`；
- `success_keys.values` 必须存在、非空，并与 `PutStartResponse.locations` 等长，而不是与原始 keys 等长；
- 全部为 `true` 时提交本 session 的全部新对象；
- 任一为 `false` 时回滚本 session 的全部新对象；
- 空 mask、长度错误或过期 session 不能被当作成功；
- 长度错误不会消费仍有效的 session，调用方可用正确 mask 重试；
- `locations` 字段为未来客户端分配模式保留；当前服务端分配模式以 session 保存的位置为准。

回滚和 session timeout 都先条件删除 active metadata 并完成 `Sync`，再对 allocation 发起一次物理 Delete。服务端
会捕获 storage provider 抛出的异常，避免异常终止 expiry worker 或服务进程。物理结果为错误或不确定时不会自动
重放 Delete：现有 URI 没有 allocation generation，重放可能误删已经复用该地址的后继对象。此时 metadata 已
不可见且不会恢复，接口/日志报告 backend orphan，由存储侧清理机制回收。

多 key commit/rollback 使用逐 key 条件更新和失败补偿，不承诺并发 `Get` 观察到同一瞬间的全有或全无。

### 5.6 `Remove`

- 精确删除给定 committed keys；不存在的 key 幂等成功；
- 任一 key 仍有 active write 时，整批返回 `WRITE_IN_PROGRESS`，不删除任何 key；
- 服务端先条件删除 metadata 并 `Sync`，随后调用 backend Delete；
- 若物理删除失败，接口返回错误，但已经删除的 metadata 不会重新暴露该 URI，也不会自动重放结果不确定的
  Delete；该 allocation 进入 backend orphan 清理范围。

### 5.7 `Trim`

| 策略 | V1 行为 |
|---|---|
| `TS_REMOVE_ALL_CACHE` | 删除 instance 的 KVMeta metadata 和可归属的物理对象 |
| `TS_REMOVE_ALL_META` | 只删除 metadata，保留物理对象；仅用于明确的修复场景 |
| `TS_TIMESTAMP` | 返回 `UNSUPPORTED` |
| `TS_UNSPECIFIED` | 返回 `INVALID_ARGUMENT` |

存在 active session 或正在提交/回滚的 session 时，Trim 整体返回 `WRITE_IN_PROGRESS`，不产生删除副作用。
`TS_REMOVE_ALL_CACHE` 与 Remove 使用相同的 metadata-first、单次物理 Delete 语义；若物理删除失败，Trim 返回错误，
已经持久化删除的 metadata 不会恢复，旧 URI 也不会在后续 Trim 中被重放。`TS_REMOVE_ALL_META` 则按定义只删除
metadata，调用方必须提前确认对应物理对象将由 backend/namespace 清理机制回收。

## 6. 错误码和调用方动作

| 错误码 | 含义 | 建议动作 |
|---|---|---|
| `OK` | 请求成功 | 使用响应 payload |
| `INVALID_ARGUMENT` | 字段、边界或数组形状非法 | 修正请求，不重试原请求 |
| `UNSUPPORTED` | V1 不支持该模式 | 改用受支持模式 |
| `DUPLICATE_ENTITY` | instance 已存在但注册配置不一致 | 对照既有 instance 配置，不覆盖重试 |
| `INSTANCE_NOT_EXIST` | 未注册或内部 schema 不匹配 | 检查注册和部署配置 |
| `SERVER_NOT_LEADER` / `SERVICE_NOT_READY` | endpoint 当前不能服务 | 官方 client 可切换下一地址 |
| `RESOURCE_EXHAUSTED` / `REACH_MAX_ENTITY_CAPACITY` | byte quota、session 或实体容量到限 | 释放对象或扩容后再试 |
| `WRITE_IN_PROGRESS` | 相同 key 或 instance 正在写/finalize | 等原 session 收敛，不并发覆盖 |
| `SESSION_NOT_FOUND` | session 过期、不存在或 instance 不匹配 | 查询最终状态，不把它当成功 |
| `SIZE_MISMATCH` | 已有对象或 location 尺寸不一致 | 使用新 key，或先确认并删除旧对象 |
| `NOT_FOUND` | 对象/instance 元数据不存在 | 按业务 miss 处理 |
| `IO_ERROR` | metadata/storage 操作失败或超时 | 根据操作幂等性判断，避免盲目重放 mutation |
| `INTERNAL_ERROR` / `UNKNOWN_ERROR` | 服务端不变量或未知错误 | 记录 trace/request id 并排查 |

## 7. Failover 与 transport error

多地址 client 在一个总 `call_timeout_ms` 预算内工作：

- `Get`、`GetInstanceInfo` 和同配置 `RegisterInstance` 遇到 transport error 可以尝试下一 endpoint；
- 所有 RPC 收到明确的 not-leader/not-ready 业务响应时可以 failover；
- `PutStart`、`PutFinish`、`Remove` 和 `Trim` 的 transport error 具有不确定结果，官方 client 不自动重放；
- C++ client 将这类结果映射为 `ER_INVALID_GRPCSTATUS`。

例如 `PutStart` transport error 后 `Get` 仍 miss，可能只是原 session 仍 active，并不代表可以立即用同 key
重写。调用方应等待原 write timeout 或通过审计确认结果。

## 8. 推荐客户端

### 8.1 `KvMetaObjectClient`

推荐业务使用它完成：

- 初始化时注册 instance 并创建数据面；
- 写入前校验整个 batch；
- `PutStart` 后只写 miss；
- 校验服务端 location、真实 URI 和 actual URI；
- 任何写入错误调用失败 mask 回滚；
- 读取时要求全部命中并在 I/O 前完成 location/size 校验。

数据面保护 caller-owned buffer，可能在名义 timeout 后等待不可取消 I/O 结束。调用方必须在同步方法返回前保持
buffer 和其 owner 存活。

该 drain 只保护本地 buffer 生命周期，不会续约服务端 V1 write session。所选 backend 必须保证在 write lease
到期前停止访问 remote allocation，或将 lease 配置为覆盖经过验证的最坏 I/O drain；否则 expiry 可能与越过
provider timeout 的旧 Put 竞争。可续约/fenced I/O 是设计文档第 13 节的 V2 能力，不是现有保证。

Python wheel 的推荐入口是：

```python
from kv_cache_manager.client import KvMetaObjectClient, KvMetaObjectClientConfig

client = KvMetaObjectClient(
    KvMetaObjectClientConfig(
        addresses=("127.0.0.1:6383",),
        instance_id="rtp-emb",
        instance_group="epd-emb-only",
        transfer_client_config=transfer_json,
    )
)
client.save(keys, contiguous_tensors)
client.load(keys, destination_tensors)
client.remove(keys)
client.close()
```

高层 Python client 在 native I/O 前校验整个逻辑调用，并自动按 64 objects / 4 GiB 拆成服务大小的 batch。
`save`/`remove` 不自动重试，`save` 也不自动删除已完成的前序 batch：通用 key 可能已经存在，而 transport error
可能具有不确定结果。异常 `KvMetaObjectClientError` 会给出 batch 位置、已确认完成数量及
`unknown_outcome`，由拥有 key 生命周期的上层决定审计或清理策略。

wheel 的 package 与 native extension 同时导出 `KV_META_OBJECT_API_VERSION=1`，extension 的值来自所链接 client
shared library 的版本查询。高层 client 在 native client 初始化前校验版本、必需类型、枚举成员和工厂方法；版本缺失
或不匹配直接失败，避免 Python wrapper、extension 与 client library 混装。
native mutation 若返回未知/畸形 code，也按 `unknown_outcome=True` 失败关闭，不能把它解释成可安全重试的明确拒绝。

### 8.2 底层客户端

需要自行编排事务时可以分别使用：

- `KvMetaClient`：只处理本文 metadata RPC；
- `KvMetaTransferClient`：只按 URI、`value_size` 和 caller buffer 搬运 exact bytes。

现有固定 block `TransferClient` 不接受 KVMeta 任意长度对象，不能互换。

Python 调用方可以依赖 Bazel target
`//kv_cache_manager/protocol/protobuf:kv_meta_service_py_proto` 使用生成的 gRPC stub，或使用
`//kv_cache_manager/client/pybind:kvcm_py_client_lib_wheel` 构建同时包含上述高层 client 与 native object API 的
wheel。
`kv-cache-manager-client` RPM 只发布 C++ headers 和 `kv_cache_manager_client.so`，不包含 Python wheel。
