# KVCM 环境变量自动初始化

## 1. 功能概览

KVCM 启动后，可以根据环境变量自动创建或更新 Storage 和 Instance Group，无需再进入容器手动执行 `kvcm_ops`。

自动初始化有四个入口变量：

| 环境变量 | 用途 |
|---|---|
| `KVCM_L1P5_STORAGE` | 配置 L1.5 EventReport Storage |
| `KVCM_L2P_STORAGE` | 配置 L2 EventReport Storage |
| `KVCM_PACE_STORAGE` | 配置 PACE Storage |
| `KVCM_INSTANCE_GROUP` | 配置公共 Instance Group |

三个 Storage 变量可以独立使用，也可以任意组合。只要配置了任意一个 Storage，就必须同时配置 `KVCM_INSTANCE_GROUP`。

如果三个 Storage 变量都没有配置，KVCM 正常启动，但不会访问 Admin API，也不会创建、更新或删除任何资源。

## 2. 环境变量配置

平台注入以下环境变量：

```json
{
  "KVCM_PACE_STORAGE": "{\"unique_name\":\"tairmempool_s_test\",\"domain\":\"empty\",\"timeout\":30,\"service_discovery_url\":\"spectrum://v-example?cache_time=30&retry_time=3&timeout=5000&port=12348\"}",
  "KVCM_L2P_STORAGE": "{\"unique_name\":\"vineyard_s_test\"}",
  "KVCM_INSTANCE_GROUP": "{\"name\":\"group_test\",\"quota_capacity\":7476679068870,\"reclaim_used_percentage\":0.8,\"metadata_backend_mode\":3,\"meta_storage_backend_config\":\"cached,redis://default:<password>@redis.example.com:6379?db=4&timeout_ms=1000&retry_count=3&cluster_name=group_test&num_shard_bits=15&sample_times=1024&persistent_type=async_redis&cache_type=local&async_queue_count=32&async_max_batch=1024000&async_max_size=1024000&async_enqueue_timeout_ms=10&async_wait_us=1000000&capacity=65536\"}"
}
```

以上环境变量等价于执行以下命令：

```bash
python3 -m kvcm_ops add_storage -u tairmempool_s_test pace \
  --domain empty \
  --timeout 30 \
  --service_discovery_url 'spectrum://v-example?cache_time=30&retry_time=3&timeout=5000&port=12348'

python3 -m kvcm_ops add_storage -u vineyard_s_test event_report_l2

python3 -m kvcm_ops create_instance_group \
  -n group_test \
  -s tairmempool_s_test \
  --user_data 'group_test' \
  --max_instance_count 512 \
  --quota_capacity 7476679068870 \
  --quota_configs ST_TAIRMEMPOOL,7476679068870 \
  --reclaim_policy POLICY_LRU \
  --reclaim_used_percentage 0.8 \
  --data_storage_strategy CPS_PREFER_TAIR_MEMPOOL \
  --max_key_count 1000000000 \
  --mutex_shard_num 131072 \
  --batch_key_size 1024 \
  --meta_storage_backend_config 'cached,redis://default:<password>@redis.example.com:6379?db=4&timeout_ms=1000&retry_count=3&cluster_name=group_test&num_shard_bits=15&sample_times=1024&persistent_type=async_redis&cache_type=local&async_queue_count=32&async_max_batch=1024000&async_max_size=1024000&async_enqueue_timeout_ms=10&async_wait_us=1000000&capacity=65536' \
  --search_cache_capacity 10240 \
  --search_cache_shard_bits 6 \
  --extra_info '{"metadata_backend_mode":3}' \
  --event_report_storage_candidates vineyard_s_test
```

## 3. 启动和主从行为

bootstrap 在 KVCM Admin 服务健康后执行，典型调用关系为：

```text
start_server.sh
  -> 启动 KVCM
  -> 执行 bootstrap
  -> 读取环境变量
  -> 对比 Registry 中的 Storage 和 Instance Group
  -> 创建或更新有变化的资源
```

高可用部署遵循以下规则：

- 每个 KVCM 只访问本机 Admin API。
- 只有 Leader 修改 Storage 和 Instance Group。
- Follower 不执行写操作，并持续等待；晋升为 Leader 后重新执行相同的 bootstrap。
- bootstrap 只负责当前节点，不会主动重启其他 KVCM 节点。
- 如果 MetaIndexer 配置变化且 Group 已经有 Instance，执行更新的 KVCM 会自动重启一次，以便按新配置恢复 MetaIndexer。

因此，多次启动、Leader 切换或环境变量未变化时，bootstrap 都是幂等的，不会重复创建相同资源。

## 4. 配置变化后的行为

线上修改环境变量后，KVCM 会重启。新进程启动时，bootstrap 会读取 Registry 中的已有配置并进行对比。

### Storage 变化

| 变化 | KVCM 行为 |
|---|---|
| Storage 不存在 | 创建 Storage |
| Storage 同名且配置一致 | 不处理 |
| Storage 同名但配置不同 | 更新 Storage；旧 backend 会被重新创建 |
| Storage 改名 | 创建新名称的 Storage，并更新 Group 引用；旧 Storage 保留 |
| 删除某个 Storage 环境变量 | 从 Group 中移除该 Storage 的引用；旧 Storage 保留 |
| 删除全部 Storage 环境变量 | 跳过 bootstrap，不修改或删除已有资源 |

### Instance Group 变化

| 变化 | KVCM 行为 |
|---|---|
| Group 不存在 | 创建 Group |
| Group 同名且配置一致 | 不处理 |
| Group 同名但配置不同 | 使用完整的新配置更新 Group，并保留 bootstrap 不管理的字段 |
| Group 改名 | 创建新 Group；旧 Group 和旧资源保留 |

### 是否需要再次重启

一般的 Storage、quota、reclaim 或 candidates 变化，只需要更新 Registry，不要求额外重启。

只有 MetaIndexer 相关配置变化时需要进一步判断：

| Group 状态 | KVCM 行为 |
|---|---|
| Group 已有 Instance | 先更新 Group，再重启一次；重启后按新配置重建 MetaIndexer |
| Group 没有 Instance | 只更新 Group，不重启；后续第一个 Instance 注册时直接使用新配置 |

MetaIndexer 相关字段包括：

- `meta_storage_backend_config`
- `max_key_count`
- `mutex_shard_num`
- `batch_key_size`
- `search_cache_capacity`
- `search_cache_shard_bits`

### 示例：从 L1.5 切换到 L2

第一次只配置：

`KVCM_L1P5_STORAGE`

```json
{
  "unique_name": "example_l1p5"
}
```

第二次在平台中删除 `KVCM_L1P5_STORAGE`，并新增：

`KVCM_L2P_STORAGE`

```json
{
  "unique_name": "example_l2"
}
```

重启后 KVCM 会：

1. 创建 `example_l2`。
2. 将同名 Group 的主 Storage、quota、reclaim Storage 和 EventReport candidates 更新为 L2。
3. 保留旧的 `example_l1p5`，但 Group 不再引用它。
4. 不执行第二次重启，因为 MetaIndexer 配置没有变化。

## 5. 字段说明和默认值

### L1.5 / L2 Storage

| 字段 | 必填 | 默认值 |
|---|---|---:|
| `unique_name` | 是 | 无 |
| `heartbeat_timeout_ms` | 否 | 30000 |
| `cleanup_grace_ms` | 否 | 300000 |
| `liveness_check_interval_ms` | 否 | 5000 |
| `snapshot_min_interval_ms` | 否 | 30000 |

### PACE Storage

| 字段 | 必填 | 默认值或限制 |
|---|---|---|
| `unique_name` | 是 | 无 |
| `domain` | 是 | 无 |
| `timeout` | 是 | 必须是正整数 |
| `service_discovery_url` | 否 | 空字符串 |
| `media_type` | 否 | 新建时为 0，仅支持 0、2 |

更新同名 PACE Storage 时，如果没有配置 `media_type`，会保留已有值；如果显式配置了不同的值，则直接报错。需要切换介质时应使用新的 `unique_name`。PACE SSD/media type 5 不在当前范围。

### Instance Group

| 字段 | 必填 | 默认值 |
|---|---|---:|
| `name` | 是 | 无 |
| `meta_storage_backend_config` | 是 | 无 |
| `user_data` | 否 | Group 名称 |
| `quota_capacity` | 否 | 1000000000 |
| `max_instance_count` | 否 | 512 |
| `reclaim_policy` | 否 | `POLICY_LRU` |
| `reclaim_used_percentage` | 否 | 0.8 |
| `max_key_count` | 否 | 1000000000 |
| `mutex_shard_num` | 否 | 131072 |
| `batch_key_size` | 否 | 1024 |
| `search_cache_capacity` | 否 | 10240 |
| `search_cache_shard_bits` | 否 | 6 |
| `metadata_backend_mode` | 否 | 不写入 |

`meta_storage_backend_config` 使用 `type,uri` 格式，type 支持 `redis` 和 `cached`，Redis URI 必须包含 `cluster_name`。URI query 会原样传递给 Meta Backend，Group 的数值字段应直接配置在 `KVCM_INSTANCE_GROUP` JSON 中。

## 6. 配置错误和重试

以下情况会被视为配置错误：

- JSON 格式错误、字段类型错误或包含未知字段。
- 配置了 Storage，但没有配置 `KVCM_INSTANCE_GROUP`。
- 多个 Storage 使用相同的 `unique_name`。
- 必填字段缺失或数值超出允许范围。
- 同名 PACE Storage 的 `media_type` 发生变化。

bootstrap 失败后会等待 5 秒并重试，最多重试 2 次。重试仍失败时停止自动初始化，但 KVCM 进程继续运行，可通过日志排查或手动配置。

bootstrap 日志输出到容器标准输出和标准错误，错误信息不会打印完整 Redis URI 或密码。
