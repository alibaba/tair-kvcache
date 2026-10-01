# KVMeta 通用对象 API

KVMeta 为 embedding 等变长对象提供独立的 exact-key 元数据协议。gRPC service 全名为 `kv_cache_manager.proto.kv_meta.MetaService`，与原 MetaService 共用 `kvcm.service.rpc_port`。

业务优先使用 `KvMetaObjectClient`，它已经组合注册、metadata 事务和 PACE I/O。只有需要自定义数据面时才直接调用以下 RPC。

## 前提

- 服务端设置 `kvcm.kv_meta.enabled=true`。
- Instance Group 配置 LRU reclaim policy、合法的 `used_percentage` 和 PACE storage candidate。
- 同一 `(instance_id, key)` 始终表示同一种内容和 size；key 应包含 tenant、模型版本、输入摘要和 tensor schema 版本。
- KVMeta 是 cache。miss、容量不足和 I/O 失败都应回退到重算。
- 每次同时检查 gRPC status 和 `response.header.status.code`。

## 调用顺序

### 1. RegisterInstance

启动时调用一次；重复调用必须与已有 instance schema 和 group 一致。响应中的 `storage_configs` 是初始化数据面客户端的权威配置。

### 2. Get

`keys` 与响应的 `hit_mask`、`locations` 一一对齐。miss 位置的 `ValueLocation` 为空；只有 hit 才允许读取数据面。

当前支持 `QT_UNSPECIFIED` 和 `QT_BATCH_GET`，不支持 `metas` 查询。

### 3. PutStart

`keys[i]` 与 `value_sizes[i]` 一一对应，不同对象可以使用不同 size。

成功响应包含：

- `key_mask[i] = true`：同 size 的对象已经提交，本次不用写。
- `key_mask[i] = false`：需要写入。
- `locations`：只包含 miss，顺序与请求中 miss 的相对顺序一致。
- `write_session_id`：存在 miss 时非空；全部 hit 时为空。

同 key 已有 `WRITING` 返回 `WRITE_IN_PROGRESS`；已有对象 size 不同返回 `SIZE_MISMATCH`。

`PutStart` 不预留 group 容量，不以 `used_percentage` 拒绝请求。并发写可短暂超过回收水位甚至 capacity，最终由 PACE allocation 和异步 GC 决定结果。

### 4. 写数据

客户端按每个 `ValueLocation.value_size` 把完整对象写入返回的 PACE URI。普通定长 `TransferClient` 不适用于该步骤；使用 `KvMetaTransferClient` 或 `KvMetaObjectClient`。

### 5. PutFinish

只要 `PutStart` 返回非空 session，无论数据面成功或失败都必须调用：

- `success_keys` 与紧凑的 `PutStartResponse.locations` 对齐。
- 全部为 `true` 时提交为 `SERVING`。
- 任一为 `false` 时整批回滚。

session 超时后服务端会自动清理。transport error 代表结果可能未知，应先 `Get` 对账，不能盲目重试 mutation。

## 删除

`Remove` 删除指定 key 的 metadata，并尽力释放 PACE allocation。不存在的 key 按成功处理；目标仍处于
`WRITING` 时返回 `WRITE_IN_PROGRESS`，不会释放数据面仍可能使用的 allocation。

`Trim` 不属于 embedding cache 的核心数据链路，所有 strategy 均返回 `UNSUPPORTED`。按 key 删除使用
`Remove`，容量回收由后台 GC 完成。

## 主要状态码

| 状态 | 含义 |
|---|---|
| `OK` | 成功 |
| `INVALID_ARGUMENT` | 参数、batch 或 size 非法 |
| `SERVICE_NOT_READY` | KVMeta/group/storage 配置不可用 |
| `SERVER_NOT_LEADER` | 当前节点不服务 KVMeta 写读请求 |
| `NOT_FOUND` | 对象或实例不存在 |
| `WRITE_IN_PROGRESS` | 相同 key 已有未结束写会话 |
| `SESSION_NOT_FOUND` | write session 不存在或已过期 |
| `SIZE_MISMATCH` | 已有对象 size 与请求不一致 |
| `RESOURCE_EXHAUSTED` | session 或后端资源不足 |
| `IO_ERROR` | 存储或超时错误 |

## 限制

- 单次最多 64 个 key。
- key、instance id、group 各最多 512 bytes。
- 单对象最多 1 GiB，单 batch 最多 4 GiB。
- write timeout 为 1 到 1800 秒。
- V1 只接受 PACE/TairMempool location。

具体容量统计、GC 和恢复机制见 [KVMeta EMB Cache 设计](../design/kv_meta_object_storage.md)。
