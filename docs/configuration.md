# KVCacheManager Service Configuration Manual

## Commandline arguments

Argument list:

```TEXT
    -c, --config_file      config file path
    -l, --log_config_file  config file path for logger
    -e, --env              set env config values, e.g., -e a=1 -e b=2
    -d, --daemon           run worker as daemon
    -h, --help             display this help and exit
    -v, --version          version and build time
```

Example:

```TEXT
/home/admin/kv_cache_manager/bin/kv_cache_manager_bin \
    -c '/home/admin/kv_cache_manager/etc/default_server_config.conf' \
    -l '/home/admin/kv_cache_manager/etc/default_logger_config.conf' \
    -e 'kvcm.registry_storage.uri=redis://your_auth_token@redis-host:6379/?db=1&cluster_name=placeholder'
```

### Redis URI

Registry、Meta 和 Coordination Redis 后端共用以下 URI 格式：

```TEXT
redis://[auth_token@]host:port/?db=<non-negative-integer>[&param=value...]
```

- `auth_token` 会作为 Redis `AUTH` 命令的单个参数传入；`auth=...` query 参数不会用于认证。
- `db` 必须放在 query 参数中；未配置时使用 DB 0。`redis://host:port/1` 中的 `/1` 是 path，不能用于选择 DB。
- `db > 0` 依赖 Redis 服务端支持 `SELECT`。Redis Cluster 不支持多逻辑 DB，此时应使用 DB 0，并通过 `cluster_name` 和 key 前缀隔离，或使用独立 Redis 实例。

## KVCacheManager Server Config

KVCM server可识别的配置参数列表如下。可通过配置文件、启动参数--env、系统环境变量进行配置。
CacheReclaimer 异步删除相关参数的生命周期语义见
[CacheReclaimer 异步删除与过度逐出优化设计](design/cache_reclaimer_async_delete.md)。

```TEXT
## 参数配置方式有3种，相同配置按照从前往后的覆盖关系（后覆盖前）
## 1. 在配置文件中配置，如下示例
## 2. 通过启动参数配置，如：$BINARY_NAME --env kvcm.service.rpc_port=6381
## 3. 通过环境变量配置，set_env(kvcm.service.rpc_port, 6381)

# 指定系统Registry自身数据存储位置
# cluster_name必填；db必须通过query参数指定
# kvcm.registry_storage.uri=redis://your_auth_token@127.0.0.1:6379/?db=1&cluster_name=kvcm_cluster

# 指定协调后端服务的URI（用于多节点选主、节点信息存储等）。Redis选主key按cluster_name隔离。
# kvcm.coordination.uri=redis://your_auth_token@127.0.0.1:6379/?db=2&cluster_name=kvcm_app_0
# 旧配置 kvcm.distributed_lock.uri 仍可使用（向后兼容），但新配置优先

# 指定选主时当前节点使用的node_id（如果不指定会自动生成）
# kvcm.leader_elector.node_id=node_0

# 指定主节点租约时间。仅单节点的部署模式下建议配置一个很大的值。
# kvcm.leader_elector.lease_ms=600000
# 多节点的部署模式下建议配置一个较低的值
# kvcm.leader_elector.lease_ms=10000

# 指定选主逻辑后台循环间隔时间。仅单节点的部署模式下建议配置一个较大的值。
# kvcm.leader_elector.loop_interval_ms=10000
# 多节点的部署模式下建议配置一个较低的值。建议低于kvcm.leader_elector.lease_ms/10。
# kvcm.leader_elector.loop_interval_ms=100

# 额外指定日志级别，覆盖日志配置文件中的设置，方便进行动态调整
# 0: auto, 1: fatal, 2: error, 3: warn, 4: info, 5: debug
kvcm.logger.log_level=4

# 指定Metaservice主服务的RPC监听端口
kvcm.service.rpc_port=6381

# 指定Metaservice主服务的HTTP监听端口
kvcm.service.http_port=6382

# 指定管控接口的RPC监听端口，如果不指定则和主服务一致
kvcm.service.admin_rpc_port=6491

# 指定管控接口的RPC监听端口，如果不指定则和主服务一致
kvcm.service.admin_http_port=6492

# 指定对外服务IO服务线程数，默认为机器核数
kvcm.service.io_thread_num=2

# 是否开启DebugService服务，可以通过DebugService直接操作集群内部状态
kvcm.service.enable_debug_service=false

# 指定KVCache Manager初始配置JSON文件路径
kvcm.startup_config=package/etc/default_startup_config.json

# 删除、系统任务与分层迁移共享的后台线程池总大小，必须大于 1
kvcm.schedule_plan_executor_thread_count=8

# 分层迁移 Prepare/Copy/cleanup 最多同时占用的共享线程池 worker 数；必须满足
# 0 < migration_worker_budget < executor_thread_count
kvcm.schedule_plan_migration_worker_budget=3

# CacheReclaimer 删除 Future 在 delay 结束后可继续抵扣水位的最长时间；到期只关闭 credit，
# 不取消底层删除。默认 60000ms。
kvcm.cache_reclaimer.inflight_delete_timeout_ms=60000

# 单个 Instance Group × BaseStorageType 的未完成 Location 数和 bytes 上限。
kvcm.cache_reclaimer.pending_location_limit_per_group_type=100000
kvcm.cache_reclaimer.pending_bytes_limit_per_group_type=68719476736

# 进程级未完成删除请求数和 bytes 上限。Future 终态前始终占用配额，credit 到期不返还。
kvcm.cache_reclaimer.pending_delete_handler_limit=1024
kvcm.cache_reclaimer.pending_bytes_limit=274877906944

# 可选值有dummy，local，logging，kmonitor；若不配置，默认启用logging
kvcm.metrics.reporter_type

# 传递给metrics reporter的配置值
kvcm.metrics.reporter_config

# 若启用的metrics reporter有周期性report任务，指定该任务的唤醒间隔；默认20000
kvcm.metrics.report_interval_ms

# 是否开启Prometheus metrics端点（GET /metrics），默认true
kvcm.metrics.enable_prometheus=true

# Prometheus metrics名称前缀，默认kvcm
kvcm.metrics.prometheus_prefix=kvcm

# log event publisher的初始化配置值，暂未启用
kvcm.event.event_publishers_configs
```

### SchedulePlanExecutor 线程与迁移预算

`kvcm.schedule_plan_executor_thread_count` 是删除、系统任务和分层迁移共同使用的进程级线程池总大小；
`kvcm.schedule_plan_migration_worker_budget` 是该线程池内同时运行的 Migration Prepare、Copy 和 cleanup
任务上限，并不是额外创建的迁移线程数。随包提供的 `default_server_config.conf` 使用 `8/3`，未提供配置文件时
代码级默认值为 `2/1`。

两个参数必须满足：

```TEXT
executor_thread_count > 1
0 < migration_worker_budget < executor_thread_count
```

迁移任务可能在 Backend Create、Copy 等存储操作中长时间占用 worker。任务优先级只能决定尚未运行任务的
出队顺序，无法抢占已经运行的迁移任务，因此 Migration budget 必须小于线程池总大小，确保至少有一个 worker
不会被迁移任务占用，可以继续处理水位回收和系统任务。这里不是把线程池静态切分为两部分：没有迁移任务时，
回收和系统任务仍可使用全部 worker。

该约束是进程级 Executor 的启动条件，与启动时是否已经配置 migration strategy 无关；instance group 可在运行期
新增迁移策略，因此不能根据启动时的策略状态放宽校验。`2/1`、`8/3` 是合法配置，`1/1`、`8/8` 和 `8/0`
会导致 `ServerConfig::Check` 或 `CacheManager::Init` 失败。已有单线程自定义配置需要至少调整为 `2/1`。

同一个 `migration_config` 中，`strategies` 的 `(source_storage_name, target_storage_name)` 组合必须唯一。
相同 source 迁移到不同 target、不同 source 迁移到相同 target，以及 `hot -> warm -> cold` 级联均可配置；只有
完全相同的 source/target route 会被拒绝。重复 route 无法明确选择各自的 threshold、method、retention 和 Mark
timeout，因此不会使用配置数组顺序作为隐式优先级。

## CacheManager Initial Config

```TEXT
{
    "storage_config": {
        "type": "file", # 后端类型，可选值file,pace,pace_ssd,mooncake,hf3fs,vcns_hf3fs
        "global_unique_name": "nfs_01", # storage backend的名字，需要全局唯一
        "storage_spec": { # storage spec 需根据不同backend类型相应配置，TODO：具体每个type的spec配置文档
            "root_path": "/tmp/nfs/",
            "key_count_per_file": 8
        }
    },
    "instance_group": {
        "name": "default", # instance group的名字，需要全局唯一
        "storage_candidates": [ # storage backend列表，用于指定写入cache数据的候选storage，即写入目标只会从这个列表指定的backend中选择
            "nfs_01"
        ],
        "global_quota_group_name": "default_quota_group", # 暂未使用
        "max_instance_count": 100, # 与该group绑定的instance数量上限
        "quota": { # 该instance group的用量quota配置，该配置与下列行为相关：写入行为，数据回收（逐出）时机
            "capacity": 30000000000, # 属于该instance group的所有instance可使用的总byte size上限，超过该值后会停止分配存储后端
            "quota_config": [ # 分storage type的quota值，同样由各个instance的用量累加得到，超过该quota后停止往该storage type的后端写入
                {
                    "storage_type": "file",
                    "capacity": 10000000000
                },
                {
                    "storage_type": "hf3fs",
                    "capacity": 10000000000
                },
                {
                    "storage_type": "pace",
                    "capacity": 10000000000
                },
                {
                    "storage_type": "pace_ssd",
                    "capacity": 10000000000
                }
            ]
        },
        "cache_config": {
            "reclaim_strategy": {
                "reclaim_policy": 1, # 控制reclaim策略，目前只支持1：LRU
                "trigger_strategy": {
                    "used_percentage": 0.8 # 控制数据用量水位，当用量达到或超过quota * percentage时将触发回收（逐出）
                },
                "delay_before_delete_ms": 1000 # 控制从提交删除请求到实际执行删除动作的间隔，类似于租约概念
            },
            # cache_prefer_strategy与storage candidates一起控制storage backend选择策略，可选值如下：
            # enum class CachePreferStrategy {
            # CPS_UNSPECIFIED = 0,
            # CPS_ALWAYS_3FS = 1, 只选择HF3FS类型的backend，否则写入报错
            # CPS_PREFER_3FS = 2, 优先选择HF3FS类型的backend，允许fallback到其他type
            # CPS_ALWAYS_MOONCAKE = 3,
            # CPS_PREFER_MOONCAKE = 4,
            # CPS_ALWAYS_TAIR_MEMPOOL = 5,
            # CPS_PREFER_TAIR_MEMPOOL = 6,
            # CPS_ALWAYS_VCNS_3FS = 7,
            # CPS_PREFER_VCNS_3FS = 8,
            # CPS_ALWAYS_TAIR_MEMPOOL_SSD = 9,
            # CPS_PREFER_TAIR_MEMPOOL_SSD = 10,
            # };
            "cache_prefer_strategy": 2,
            "meta_indexer_config": {
                "max_key_count": 1000000, # 单个meta indexer的key数量上限，同样影响reclaimer的逐出水位计算
                "mutex_shard_num": 16,
                "batch_key_size": 16,
                "meta_storage_backend_config": { # 控制meta indexer的storage backend，可选local本地文件或者redis
                    # Redis示例：
                    # "storage_type": "redis",
                    # "storage_uri": "redis://your_auth_token@redis-host:6379/?db=3&client_max_pool_size=16"
                    "storage_type": "local",
                    "storage_uri": ""
                },
                "meta_cache_policy_config": { # 控制 meta indexer数据cache的配置
                    "type": "LRU",
                    "capacity": 10000,
                    "cache_shard_bits": 0,
                    "high_pri_pool_ratio": 0.0
                }
            }
        },
        "user_data": "{\"description\": \"Default instance group for KV Cache Manager\"}",
        "version": 1
    }
}
```

TairMempool DRAM 使用 `pace`（proto `ST_TAIRMEMPOOL`），LocalSSD 使用
`pace_ssd`（proto `ST_TAIRMEMPOOL_SSD`，同时要求 `media_type=5`）。两类 storage
仍使用 `pace://` 数据面 URI，但 quota、类型水位和迁移触发用量分别统计。旧的
`ST_TAIRMEMPOOL + media_type=5` 配置仍可读取，迁移到新类型后才能获得独立 SSD 水位。

启用独立 SSD 类型时还需遵守以下配置约束：

- 使用 `kvcm_ops add_storage ... pace_ssd` 创建一个新的、全局唯一的 Storage；不要把已有
  `pace` Storage 原地更新为 `pace_ssd`。Storage 类型会写入 CacheLocation 和用量账本，
  服务端会拒绝同名 Storage 的类型变更。
- Instance Group 的 `quota.quota_config` 必须同时为 `pace` 和 `pace_ssd` 配置正容量。
  类型水位只遍历这里显式出现的类型；缺少 `pace_ssd` 时不会形成 SSD 类型水位和容量上限，
  缺少迁移源 `pace` 时迁移规则不会触发。
- SSD 仅作为迁移目标时，不要求放入 `storage_candidates`；迁移规则按
  `target_storage_name` 精确选择它。若需要把普通新写入直接落到 SSD，则必须把 SSD Storage
  放入 `storage_candidates`，并使用 `CPS_ALWAYS_TAIR_MEMPOOL_SSD` 或
  `CPS_PREFER_TAIR_MEMPOOL_SSD`。
- `pace` 和 `pace_ssd` 的数据面 URI scheme 都是 `pace://`。`pace_ssd` 只用于配置、计量
  和选择，不能生成 `pace_ssd://` URI。

启用前应先升级 Manager 和 Client，再创建 `pace_ssd` Storage。一旦 registry 或
usage 中写入新类型，不支持直接回滚到不识别 `ST_TAIRMEMPOOL_SSD` 的旧版本。

## Logger Config

```TEXT
alog.rootLogger=INFO, rootAppender
alog.max_msg_len=20480

alog.appender.rootAppender=FileAppender
alog.appender.rootAppender.fileName=logs/kv_cache_manager.log
alog.appender.rootAppender.layout=PatternLayout
alog.appender.rootAppender.layout.LogPattern=[%%d] [%%l] [%%p:%%t] %%m
alog.appender.rootAppender.async_flush=false
alog.appender.rootAppender.flush=true
alog.appender.rootAppender.compress=false

alog.logger.access=INFO, accessAppender
inherit.access=false
alog.appender.accessAppender=FileAppender
alog.appender.accessAppender.fileName=logs/access.log
alog.appender.accessAppender.layout=PatternLayout
alog.appender.accessAppender.layout.LogPattern=%%m
alog.appender.accessAppender.async_flush=true
alog.appender.accessAppender.flush_threshold=100
alog.appender.accessAppender.flush_interval=100
alog.appender.accessAppender.compress=false
alog.appender.accessAppender.max_file_size=256
alog.appender.accessAppender.log_keep_count=20

alog.logger.metrics=INFO, metricsAppender
inherit.metrics=false
alog.appender.metricsAppender=FileAppender
alog.appender.metricsAppender.fileName=logs/metrics.log
alog.appender.metricsAppender.layout=PatternLayout
alog.appender.metricsAppender.layout.LogPattern=%%m
alog.appender.metricsAppender.async_flush=true
alog.appender.metricsAppender.flush_threshold=10240
alog.appender.metricsAppender.flush_interval=100
alog.appender.metricsAppender.compress=false
alog.appender.metricsAppender.max_file_size=512
alog.appender.metricsAppender.log_keep_count=10

alog.logger.publisher=INFO, publisherAppender
inherit.publisher=false
alog.appender.publisherAppender=FileAppender
alog.appender.publisherAppender.fileName=logs/event_publisher.log
alog.appender.publisherAppender.layout=PatternLayout
alog.appender.publisherAppender.layout.LogPattern=%%m
alog.appender.publisherAppender.async_flush=true
alog.appender.publisherAppender.flush_threshold=10240
alog.appender.publisherAppender.flush_interval=100
alog.appender.publisherAppender.compress=false
alog.appender.publisherAppender.max_file_size=256
alog.appender.publisherAppender.log_keep_count=20

alog.logger.console=INFO, consoleAppender
inherit.console=false
alog.appender.consoleAppender=ConsoleAppender
alog.appender.consoleAppender.layout=PatternLayout
alog.appender.consoleAppender.layout.LogPattern=[%%d] [%%l] [%%p:%%t] %%m
```
