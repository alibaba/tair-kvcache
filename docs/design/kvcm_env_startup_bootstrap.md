# KVCM 环境变量启动与自动初始化设计

## 1. 说明

KVCM 的 EventReport 接入需要预先创建 L1.5/L2 Storage 和 Instance Group。当前发布包通过默认入口 `start_server.sh` 启动 KVCM，并在本机 Leader 上自动执行 `python3 -m kvcm_ops bootstrap`，根据环境变量创建或更新这些资源。

本文主要说明：

- 部署时需要配置哪些环境变量。
- 环境变量如何生成 Storage 和 Instance Group。
- 环境变量变化后，已有资源如何更新。

## 2. 启动与生效关系

```text
Docker ENTRYPOINT
  └─ start_server.sh
      ├─ 启动 kv_cache_manager_bin
      ├─ 等待本机 Admin HTTP 健康
      └─ python3 -m kvcm_ops bootstrap
          ├─ 解析环境变量
          ├─ 检查本机是否为 Leader
          ├─ 创建或更新 Storage
          ├─ 创建或更新 Instance Group
          └─ 回读并校验受管字段
```

单机和高可用部署都只访问本机 `http://127.0.0.1:6492`。Follower 不修改配置，也不查找或访问 Leader；Follower 晋升为 Leader 后执行相同的幂等 bootstrap。

## 3. 核心环境变量

| 环境变量 | 是否必填 | 说明 |
|---|---|---|
| `KVCM_ENABLE_SUBSCRIBER_EVENT_REPORT` | 是 | 布尔值，只接受 `true`/`false`；控制 L1.5 Storage |
| `KVCM_ENABLE_V6D_EVENT_REPORT` | 是 | 布尔值，只接受 `true`/`false`；控制 L2 Storage |
| `KVCM_INSTANCE_GROUP_NAME` | 开启任一上报时 | Instance Group 名称 |
| `KVCM_EVENT_REPORT_L1P5_STORAGE_NAME` | subscriber 开启时 | L1.5 Storage 名称 |
| `KVCM_EVENT_REPORT_L2_STORAGE_NAME` | V6D 开启时 | L2 Storage 名称 |
| `KVCM_META_STORAGE_TYPE` | 开启任一上报时 | Meta Storage 类型，支持 `redis`、`cached` |
| `KVCM_META_STORAGE_URI` | 开启任一上报时 | `redis://...` URI，包含 Redis 连接信息和统一 query 参数 |
| `KVCM_METADATA_BACKEND_MODE` | 否 | 1～4 的整数；设置时写入 `extra_info.metadata_backend_mode` |

两个上报开关都为 `false` 时，只启动 KVCM，不创建或更新 Storage/Instance Group。此时除两个开关外，其他 bootstrap 环境变量可以不提供。

## 4. Meta Storage 配置

### 4.1 Storage 类型与 URI 分开配置

原手工命令：

```text
--meta_storage_backend_config 'cached,redis://...'
```

对应环境变量：

```text
KVCM_META_STORAGE_TYPE=cached
KVCM_META_STORAGE_URI=redis://...
```

`KVCM_META_STORAGE_URI` 必须包含：

- Redis host。
- 1～65535 范围内的端口。
- 唯一且非空的 `cluster_name` query 参数。

为了兼容已有 `kvcm_registry_storage_uri` 和 `create_instance_group --meta_storage_backend_config`，密码中的原始 `#` 可以保持原写法。bootstrap 只在内部校验副本中处理 `#`，最终写入 Instance Group 的 URI 与环境变量原字符串完全一致。

### 4.2 Instance Group 参数

以下参数统一写在 `KVCM_META_STORAGE_URI` query 中；未提供时使用默认值：

| 参数 | 默认值 | 约束 | 写入位置 |
|---|---:|---|---|
| `max_instance_count` | 512 | 正整数 | Instance Group |
| `quota_capacity` | 2087740652912 | 正整数，单位为字节 | Group quota 及主 Storage quota |
| `max_key_count` | 1000000000 | 正整数 | MetaIndexer |
| `mutex_shard_num` | 131072 | 正整数且为 2 的幂 | MetaIndexer |
| `batch_key_size` | 1024 | 正整数 | MetaIndexer |
| `search_cache_capacity` | 10240 | 正整数，单位为 MB | 新建 Group 的 Meta Cache Policy |
| `search_cache_shard_bits` | 6 | 非负整数 | 新建 Group 的 Meta Cache Policy |

当前实现更新已有 Instance Group 时不覆盖已有 Meta Cache Policy，因此 `search_cache_capacity` 和 `search_cache_shard_bits` 只影响新建 Group。

### 4.3 EventReport Storage 参数

以下可选正整数参数也写在 URI query 中，并应用到受管的 L1.5/L2 Storage：

- `heartbeat_timeout_ms`
- `cleanup_grace_ms`
- `liveness_check_interval_ms`
- `snapshot_min_interval_ms`

同名 Storage 已存在时，只更新这些受管字段，保留 `event_report` 中其他字段。

### 4.4 Redis 及异步队列参数

以下参数由 bootstrap 做数值校验，并随原始 URI 传给 Meta Storage Backend：

- 正整数：`timeout_ms`、`async_queue_count`、`async_max_batch`、`async_wait_us`、`async_max_size`、`async_sync_timeout_ms`、`async_drain_ms`、`client_max_pool_size`、`sample_times`。
- 非负整数：`db`、`async_enqueue_timeout_ms`、`client_min_pool_size`、`num_shard_bits`。
- `async_queue_count` 最大为 2048。

`retry_count`、`persistent_type`、`cache_type` 等其他参数不由 bootstrap 解释，但会保留在 URI 中并传给 KVCM。

完整 URI 属于 secret。bootstrap 不主动打印用户名、密码、query 或完整 URI；成功日志只包含 Meta Storage 类型和 Redis host/port。

## 5. Subscriber 与 V6D 组合

| Subscriber | V6D | 创建的 Storage | 主 Storage / quota | EventReport candidates |
|---|---|---|---|---|
| 开 | 关 | L1.5 | L1.5 | L1.5 |
| 关 | 开 | L2 | L2 | L2 |
| 开 | 开 | L1.5、L2 | L1.5 | L1.5、L2 |
| 关 | 关 | 不创建 | 不配置 | 不配置 |

主 Storage 同时用于 `storage_candidates`、quota 类型和 `reclaim_strategy.storage_unique_name`。

`KVCM_METADATA_BACKEND_MODE` 与 subscriber/V6D 开关没有绑定关系。设置时写入或更新 `extra_info.metadata_backend_mode`；未设置时不管理该字段，并保留已有 `extra_info` 内容。

## 6. 完整配置示例

以下示例同时启用 subscriber 和 V6D。账号、密码、地址和 cluster name 均为示例值：

```json
{
  "KVCM_ENABLE_SUBSCRIBER_EVENT_REPORT": "true",
  "KVCM_ENABLE_V6D_EVENT_REPORT": "true",
  "KVCM_INSTANCE_GROUP_NAME": "example_group",
  "KVCM_EVENT_REPORT_L1P5_STORAGE_NAME": "example_l1p5",
  "KVCM_EVENT_REPORT_L2_STORAGE_NAME": "example_l2",
  "KVCM_META_STORAGE_TYPE": "cached",
  "KVCM_METADATA_BACKEND_MODE": "4",
  "KVCM_META_STORAGE_URI": "redis://default:example#password@redis.example.com:6379?timeout_ms=1000&retry_count=3&cluster_name=example_cluster&num_shard_bits=18&sample_times=1024&persistent_type=async_redis&cache_type=local&async_queue_count=16&async_max_batch=1024000&async_max_size=1024000&async_enqueue_timeout_ms=10&async_wait_us=1000000&max_instance_count=512&quota_capacity=2087740652912&max_key_count=1000000000&mutex_shard_num=131072&batch_key_size=1024"
}
```

生成结果：

- L1.5 Storage：`example_l1p5`。
- L2 Storage：`example_l2`。
- Instance Group：`example_group`。
- 主 Storage 和 quota：L1.5。
- EventReport candidates：L1.5、L2。
- Meta Storage：`cached`，底层 URI 为环境变量中的完整 Redis URI。
- `extra_info.metadata_backend_mode=4`。

## 7. 环境变量变化后的行为

bootstrap 不会先删除全部旧资源再重建，而是读取现有配置后按变化类型处理：

| 变化 | 行为 |
|---|---|
| URI、quota、MetaIndexer 参数变化 | 原地更新当前 Instance Group |
| Storage 参数变化但名称相同 | 原地更新对应 Storage |
| Storage 名称变化 | 创建新 Storage，切换 Group 引用，保留旧 Storage |
| Instance Group 名称变化 | 创建新 Group，保留旧 Group |
| `KVCM_METADATA_BACKEND_MODE` 修改 | 原地更新当前 Group 中的该字段，保留其他 `extra_info` |
| `KVCM_METADATA_BACKEND_MODE` 删除 | 不再管理该字段，保留 Group 中已有值 |
| 两个上报开关都改为 `false` | 跳过初始化，不主动删除已有 Storage 或 Group |

### 7.1 只修改 L2 Storage 名称

假设原配置为：

```text
L1.5 = storage_l1p5
L2   = storage_l2
Group candidates = [storage_l1p5, storage_l2]
```

只把 L2 名称改为 `storage_l2_new` 后：

```text
创建 storage_l2_new
保留 storage_l1p5
保留旧 storage_l2
Group candidates 更新为 [storage_l1p5, storage_l2_new]
```

L1.5 Storage 没有变化，因此不会重新创建或更新。

### 7.2 Instance Group 受管字段

更新已有 Group 时，bootstrap 覆盖：

- `storage_candidates`
- `event_report_storage_candidates`
- `max_instance_count`
- quota
- reclaim Storage
- `max_key_count`、`mutex_shard_num`、`batch_key_size`
- `meta_storage_backend_config`
- 设置了环境变量时的 `extra_info.metadata_backend_mode`

已有 `user_data`、quota group、Meta Cache Policy 和其他扩展字段保持不变。

Group 更新采用版本控制：读取当前版本，使用 `version + 1` 更新；遇到版本冲突时重新读取并重试。更新完成后重新读取 Storage 和 Group，校验所有受管字段。

旧 Storage 和旧 Group 不自动删除，因为它们可能仍被其他 Group、实例或人工配置引用。如需清理，由用户确认引用关系后手动删除。

## 8. 配置错误和重试

环境变量缺失、格式非法、Admin API 失败或回读不一致时：

- bootstrap 输出包含失败阶段、异常类型、异常消息和 Python 堆栈的 ERROR 日志。
- 首次失败后等待 5 秒重试，最多重试 2 次，即一次启动周期最多执行 3 次失败尝试。
- 重试耗尽后停止本次自动 bootstrap，但 KVCM 继续运行。
- 用户可以进入容器手动执行 `kvcm_ops` 命令。
- 容器重启后重新执行环境变量初始化并重新获得重试机会。

Follower 等待和执行中失去 Leader 身份不算配置失败，不消耗上述重试次数。

bootstrap Python 日志位于容器标准输出/标准错误；KVCM C++ 处理 Admin API 时产生的 `KVCM_LOG_xxx` 仍写入 `kv_cache_manager.log`。
