# KVCM 环境变量自动初始化

KVCM 启动时，若目标 Instance Group 不存在，可从环境变量创建 Storage 和 Group。已有同名 Group 时，普通重启不会覆盖现有配置；需要按环境变量更新时，显式执行同步命令。

## 1. 先配置环境变量

在平台中设置下列环境变量。**每个变量的值都是一个 JSON 对象字符串**，不是 JSON 文件路径。

| 环境变量 | 作用 | 何时填写 |
|---|---|---|
| `KVCM_L1P5_STORAGE` | 创建 L1.5 EventReport Storage | 需要 L1.5 时 |
| `KVCM_L2P_STORAGE` | 创建 L2 EventReport Storage | 需要 L2 时 |
| `KVCM_PACE_STORAGE` | 创建 PACE Storage | 需要 PACE 时 |
| `KVCM_INSTANCE_GROUP` | 首次创建公共 Instance Group；显式同步时可更新 | 配置了任一 Storage 时必填 |

三个 Storage 变量可以单独设置，也可以组合。它们的 `unique_name` 必须各不相同。没有配置任何 Storage 时，Bootstrap 跳过资源初始化，不会创建、更新或删除已有资源。

**环境变量何时生效**

- **普通启动只做首次初始化**：同名 Group 不存在时，创建所需的 Storage 和 Group；同名 Group 已存在时，跳过整个初始化。比如手工用 `kvcm_ops update_instance_group` 修改 quota 后，普通重启会保留这个修改。
- **需要用环境变量更新已有 Group 时，显式同步一次**：在 Leader 容器中执行 `python3 -m kvcm_ops bootstrap --reconcile`。命令会比较并更新受管理字段，输出操作结果；如果提示需要重启 KVCM，再由执行者重启。
- **手工创建的同名 Group 也视为已初始化**：普通启动不会因为它并非由 bootstrap 创建，就自动接管或覆盖它。

以下组合只改变 Storage 类型；Group 的 quota、reclaim、`metadata_backend_mode` 和 Meta Redis 参数保持一致，数值需按实际部署调整。**`metadata_backend_mode` 不会自动默认为 `3`**；需要该值时必须显式填写。

### 组合示例：只配置 L2

以下是 shell 写法；在平台中配置时，将单引号**内**的 JSON 分别作为变量值：

```bash
export KVCM_L2P_STORAGE='{"unique_name":"vineyard_s_test"}'
export KVCM_INSTANCE_GROUP='{"name":"group_test","quota_capacity":7476679068870,"reclaim_used_percentage":0.8,"metadata_backend_mode":3,"meta_storage_backend_config":"cached,redis://default:<password>@redis.example.com:6379?db=4&cluster_name=group_test&capacity=65536"}'
```

将 Redis 地址和 `<password>` 替换为实际配置。`cluster_name` 是必填的 URI 参数；采用 `cached` 模式时，其余常用参数会按第 2 节的 URI 参数表自动补齐。主 Storage 优先级为 **PACE → L1.5 → L2**；已配置的 L1.5/L2 还会加入 EventReport candidates。

### 纯内存 Meta 示例：只配置 L2

```bash
export KVCM_L2P_STORAGE='{"unique_name":"vineyard_s_test"}'
export KVCM_INSTANCE_GROUP='{"name":"group_test","meta_storage_backend_config":"local,local://?capacity=65536&num_shard_bits=15&sample_times=1024"}'
```

`local` 不需要 Redis 地址或 `cluster_name`；也可简写为 `"meta_storage_backend_config":"local"`，此时使用 KVCM 后端默认值：`capacity=32768` MB、`num_shard_bits=10`、`sample_times=10`。纯内存 Meta 数据不会从旧 Redis 迁移，重启后也不会由 local 后端持久化恢复。

### 组合示例：PACE + L2

需要 PACE 存储并同时接收 L2 EventReport 时，可在平台中注入：

```json
{
  "KVCM_PACE_STORAGE": "{\"unique_name\":\"tairmempool_s_test\",\"domain\":\"empty\",\"timeout\":30,\"service_discovery_url\":\"spectrum://v-example?cache_time=30&retry_time=3&timeout=5000&port=12348\"}",
  "KVCM_L2P_STORAGE": "{\"unique_name\":\"vineyard_s_test\"}",
  "KVCM_INSTANCE_GROUP": "{\"name\":\"group_test\",\"quota_capacity\":7476679068870,\"reclaim_used_percentage\":0.8,\"metadata_backend_mode\":3,\"meta_storage_backend_config\":\"cached,redis://default:<password>@redis.example.com:6379?db=4&cluster_name=group_test&capacity=65536\"}"
}
```

### 组合示例：L1.5 + L2

同时接收 L1.5 和 L2 EventReport、且不使用 PACE 时，可配置：

```json
{
  "KVCM_L1P5_STORAGE": "{\"unique_name\":\"subscriber_s_test\"}",
  "KVCM_L2P_STORAGE": "{\"unique_name\":\"vineyard_s_test\"}",
  "KVCM_INSTANCE_GROUP": "{\"name\":\"group_test\",\"quota_capacity\":7476679068870,\"reclaim_used_percentage\":0.8,\"metadata_backend_mode\":3,\"meta_storage_backend_config\":\"cached,redis://default:<password>@redis.example.com:6379?db=4&cluster_name=group_test&capacity=65536\"}"
}
```

这组配置以 L1.5 为主 Storage，L1.5 和 L2 都加入 EventReport candidates。

### 完整组合示例：PACE + L1.5 + L2

同时配置全部四个环境变量时，可在平台中注入：

```json
{
  "KVCM_PACE_STORAGE": "{\"unique_name\":\"tairmempool_s_test\",\"domain\":\"empty\",\"timeout\":30,\"service_discovery_url\":\"spectrum://v-example?cache_time=30&retry_time=3&timeout=5000&port=12348\"}",
  "KVCM_L1P5_STORAGE": "{\"unique_name\":\"subscriber_s_test\"}",
  "KVCM_L2P_STORAGE": "{\"unique_name\":\"vineyard_s_test\"}",
  "KVCM_INSTANCE_GROUP": "{\"name\":\"group_test\",\"quota_capacity\":7476679068870,\"reclaim_used_percentage\":0.8,\"metadata_backend_mode\":3,\"meta_storage_backend_config\":\"cached,redis://default:<password>@redis.example.com:6379?db=4&cluster_name=group_test&capacity=65536\"}"
}
```

这组配置以 PACE 为主 Storage；L1.5 和 L2 都加入 EventReport candidates。将 Redis 地址、`<password>` 和 PACE 服务发现地址替换为实际配置。

## 2. JSON 字段与默认值

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
| `metadata_backend_mode` | 否 | 新建时不写入；更新已有 Group 时移除旧值 |

`meta_storage_backend_config` 支持 `redis,redis://...`、`cached,redis://...`、`local,local://?capacity=65536&num_shard_bits=15&sample_times=1024`，以及简写 `local`。Redis URI 必须包含 `cluster_name`；local URI 不需要 Redis 地址或参数。local 的 `capacity`、`num_shard_bits`、`sample_times` 可分别省略，省略时由 KVCM 后端使用默认值。显式填写时，`capacity` 和 `sample_times` 必须是正整数，`num_shard_bits` 必须是 0～19 的整数，URI 原样传给后端。Bootstrap 只为 `redis` 和 `cached` 补齐缺失的 URI 参数。Group 的数值字段应直接配置在 `KVCM_INSTANCE_GROUP` JSON 中。

| URI 参数 | Bootstrap 默认值 | 适用类型 |
|---|---:|---|
| `timeout_ms` | 1000 | `redis`、`cached` |
| `retry_count` | 3 | `redis`、`cached` |
| `num_shard_bits` | 15 | `cached` |
| `sample_times` | 1024 | `cached` |
| `persistent_type` | `async_redis` | `cached` |
| `cache_type` | `local` | `cached` |
| `async_queue_count` | 32 | `cached` 且使用 `async_redis` |
| `async_max_batch` | 204800 | `cached` 且使用 `async_redis` |
| `async_max_size` | 204800 | `cached` 且使用 `async_redis` |
| `async_enqueue_timeout_ms` | 10 | `cached` 且使用 `async_redis` |
| `async_wait_us` | 1000000 | `cached` 且使用 `async_redis` |

上面的 PACE + L2 示例只保留 `db`、`cluster_name` 和 `capacity` 等显式参数；其余参数由 Bootstrap 补齐。这些是 Bootstrap 补齐的默认值，不会修改 Meta Backend 本身的默认值。直接调用 `create_instance_group` 时，需要自行写出所需参数。未配置 `capacity` 时，本地 Meta 缓存使用后端默认的 32768 MB。

## 3. 普通启动：只初始化缺失的 Group

KVCM 的 Admin 服务健康后，启动脚本会自动运行 `python3 -m kvcm_ops bootstrap`。它按以下顺序处理：

启动脚本会从实际使用的服务配置文件、`-e` 启动参数和进程环境变量中确定 Admin HTTP 端口，并让健康检查与 bootstrap 使用同一端口。

1. **没有配置 Storage 环境变量**：不做资源初始化。
2. **同名 Group 已存在**：视为已初始化，直接跳过；不会检查或修改任何 Storage、Group。手工创建的 Group 也一样。
3. **同名 Group 不存在**：复用配置相同的同名 Storage，创建缺失的 Storage，再创建 Group。如果同名 Storage 的配置冲突，则报错，不覆盖它。

只有 Leader 写入资源。Follower 不写入，晋升为 Leader 后才执行上述检查。

**普通重启不会让新的环境变量覆盖已有 Group。** 因此，手工修改的 quota 等配置在重启后仍保留；已有部署升级后也继续使用 Registry 中的配置。启动过程不会因为 bootstrap 再自动重启 KVCM。

## 4. 需要环境变量覆盖已有配置时

修改环境变量后，在 **Leader 容器**中手工运行一次：

```bash
python3 -m kvcm_ops bootstrap --reconcile
```

如果手工执行时 Admin HTTP 端口不是默认的 `6492`，传入实际地址，例如 `python3 -m kvcm_ops bootstrap --reconcile --host http://127.0.0.1:6500`。

同步命令会比较 Registry 和环境变量，并在日志中记录创建或更新的资源：

- 环境变量指定的 Storage 或 Group 不存在时创建，配置有变化时更新，相同则跳过。Group 中不由 bootstrap 管理的字段会保留；环境变量中省略的字段按第 2 节所列规则处理。
- 改名会创建新资源，旧资源不会删除。删除部分 Storage 环境变量会移除 Group 对它们的引用，但仍保留旧 Storage。
- 删除全部 Storage 环境变量时不执行同步，不会修改已有资源。
- 从 Redis Meta 改为 local 时只更新 Group 配置，不迁移旧 Meta 数据；旧 Storage 资源仍按上述规则保留。

**何时需要重启？** 仅当 Group 已有 Instance，且本次同步改变了 MetaIndexer 配置时，命令会提示重启并返回退出码 `3`。执行者重启 KVCM 后，新配置才会用于重建 MetaIndexer。只改 Storage、quota、reclaim 或 candidates，无需因 bootstrap 再重启。

MetaIndexer 配置包括 `meta_storage_backend_config`、`max_key_count`、`mutex_shard_num`、`batch_key_size`、`search_cache_capacity` 和 `search_cache_shard_bits`。

## 5. 失败时如何处理

先检查环境变量是否有效：JSON 格式和字段类型是否正确、必填字段是否齐全、Storage 的 `unique_name` 是否重复。首次初始化时，已有同名 Storage 若配置冲突会报错；显式同步时，同名 PACE Storage 的 `media_type` 不能改变。

普通启动的 bootstrap 失败后每隔 5 秒重试，最多重试 2 次。仍失败时会停止自动初始化，**KVCM 服务继续运行**；可根据容器标准输出或标准错误中的 bootstrap 日志排查。手工执行 `--reconcile` 出错时会直接返回，不会自动重试。

日志不会打印完整 Redis URI 或密码。

## 附录：不使用 bootstrap 时的手工命令

按第 1 节配置环境变量时，无需执行以下命令。若选择手工创建，可参考 PACE + L2 的例子：先创建两个 Storage，再创建 Group。手工调用 `create_instance_group` 不会补齐第 2 节的 URI 默认参数，因此命令中要显式写出它们。

<details>
<summary>展开查看 PACE + L2 手工命令</summary>

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
  --meta_storage_backend_config 'cached,redis://default:<password>@redis.example.com:6379?db=4&cluster_name=group_test&capacity=65536&timeout_ms=1000&retry_count=3&num_shard_bits=15&sample_times=1024&persistent_type=async_redis&cache_type=local&async_queue_count=32&async_max_batch=204800&async_max_size=204800&async_enqueue_timeout_ms=10&async_wait_us=1000000' \
  --search_cache_capacity 10240 \
  --search_cache_shard_bits 6 \
  --extra_info '{"metadata_backend_mode":3}' \
  --event_report_storage_candidates vineyard_s_test
```

</details>
