# KVCM 环境变量启动与自动初始化

## 1. 功能说明

KVCM 启动后会根据环境变量自动创建或更新 EventReport Storage 和 Instance Group，不再需要进入容器手动执行 `kvcm_ops` 命令。

单机和高可用部署都只访问本机 Admin API。只有 Leader 执行初始化；Follower 晋升为 Leader 后执行相同的初始化。

## 2. 环境变量

| 环境变量 | 是否必填 | 说明 |
|---|---|---|
| `KVCM_ENABLE_SUBSCRIBER_EVENT_REPORT` | 是 | 是否启用 L1.5 Storage，只接受 `true`/`false` |
| `KVCM_ENABLE_V6D_EVENT_REPORT` | 是 | 是否启用 L2 Storage，只接受 `true`/`false` |
| `KVCM_INSTANCE_GROUP_NAME` | 开启任一上报时 | Instance Group 名称 |
| `KVCM_META_STORAGE_BACKEND_CONFIG` | 开启任一上报时 | Meta Storage 配置，格式为 `type,uri`，type 支持 `redis`、`cached` |
| `KVCM_METADATA_BACKEND_MODE` | 否 | 1～4 的整数；设置时写入 `extra_info.metadata_backend_mode` |

两个上报开关都为 `false` 时，只启动 KVCM，不创建或更新 Storage 和 Instance Group。

EventReport Storage 名称由 Instance Group 名称自动生成：

- subscriber 开启时创建 `<instance_group_name>_event_report_l1p5`。
- V6D 开启时创建 `<instance_group_name>_event_report_l2`。
- 对应开关关闭时，不创建该类型的 Storage。

`KVCM_META_STORAGE_BACKEND_CONFIG` 与 `create_instance_group --meta_storage_backend_config` 使用相同的 `type,uri` 格式，当前推荐使用 `cached,redis://...`，URI 必须包含 `cluster_name`。Redis 密码可沿用已有 URI 写法，密码中的 `#` 无需转义。

## 3. URI 参数

以下参数统一配置在 `KVCM_META_STORAGE_BACKEND_CONFIG` 的 URI query 中，未配置时使用默认值：

| 参数 | 默认值 |
|---|---:|
| `max_instance_count` | 512 |
| `quota_capacity` | 2087740652912 |
| `max_key_count` | 1000000000 |
| `mutex_shard_num` | 131072 |
| `batch_key_size` | 1024 |

Redis timeout、shard、async queue 等参数也直接写在同一个 URI 中，并随 URI 传给 Meta Storage Backend。

## 4. Subscriber 与 V6D 组合

| Subscriber | V6D | 创建的 Storage | 主 Storage / quota | EventReport candidates |
|---|---|---|---|---|
| 开 | 关 | L1.5 | L1.5 | L1.5 |
| 关 | 开 | L2 | L2 | L2 |
| 开 | 开 | L1.5、L2 | L1.5 | L1.5、L2 |
| 关 | 关 | 不创建 | 不配置 | 不配置 |

`KVCM_METADATA_BACKEND_MODE` 与 subscriber、V6D 开关相互独立。配置时更新 `metadata_backend_mode`；未配置时删除 Instance Group 中已有的该字段，并保留 `extra_info` 中其他字段。

## 5. 配置示例

以下示例同时启用 subscriber 和 V6D，连接信息均为示例值：

```json
{
  "KVCM_ENABLE_SUBSCRIBER_EVENT_REPORT": "true",
  "KVCM_ENABLE_V6D_EVENT_REPORT": "true",
  "KVCM_INSTANCE_GROUP_NAME": "example_group",
  "KVCM_METADATA_BACKEND_MODE": "4",
  "KVCM_META_STORAGE_BACKEND_CONFIG": "cached,redis://default:example#password@redis.example.com:6379?timeout_ms=1000&retry_count=3&cluster_name=example_cluster&num_shard_bits=18&sample_times=1024&persistent_type=async_redis&cache_type=local&async_queue_count=16&async_max_batch=1024000&async_max_size=1024000&async_enqueue_timeout_ms=10&async_wait_us=1000000&max_instance_count=512&quota_capacity=2087740652912&max_key_count=1000000000&mutex_shard_num=131072&batch_key_size=1024"
}
```

该配置创建 `example_group_event_report_l1p5`、`example_group_event_report_l2` 两个 Storage 和 `example_group`。Instance Group 的主 Storage 和 quota 使用 L1.5，EventReport candidates 为 L1.5、L2。

## 6. 环境变量变化后的行为

KVCM 重启后，bootstrap 会读取已有配置并按变化类型处理，不会先删除全部资源再重新创建。

| 变化 | 行为 |
|---|---|
| Meta Storage、quota、MetaIndexer 参数变化 | 原地更新当前 Instance Group |
| Storage 配置变化但名称相同 | 原地更新对应 Storage |
| 开启 subscriber 或 V6D | 按自动命名规则创建对应 Storage，并更新 Group 引用 |
| 关闭 subscriber 或 V6D，且另一开关仍开启 | 从 Group candidates 中移除对应 Storage 引用，保留已有 Storage |
| Instance Group 名称变化 | 按新名称创建 Storage 和 Group，保留旧 Storage 和旧 Group |
| `KVCM_METADATA_BACKEND_MODE` 修改 | 更新当前 Group 的该字段，保留其他 `extra_info` |
| `KVCM_METADATA_BACKEND_MODE` 删除 | 删除当前 Group 中的该字段，保留其他 `extra_info` |
| 两个上报开关都改为 `false` | 跳过初始化，不删除已有 Storage 或 Group |

例如 Instance Group 名称从 `example_group` 改为 `example_group_new`，且两个上报开关都开启时：

- 创建 `example_group_new_event_report_l1p5`。
- 创建 `example_group_new_event_report_l2`。
- 创建 `example_group_new` 并引用两个新 Storage。
- 原有 Group 和 Storage 保留，后续由用户确认引用关系后手动清理。

## 7. 配置失败

环境变量错误或资源更新失败时，bootstrap 打印错误日志，等待 5 秒后重试，最多重试 2 次。重试仍失败时停止本次自动初始化，但 KVCM 继续运行，用户可以进入容器手动配置。

bootstrap 日志位于容器标准输出/标准错误；KVCM C++ 处理请求产生的日志仍写入 `kv_cache_manager.log`。
