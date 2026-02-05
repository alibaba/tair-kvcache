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
    -e 'kvcm.registry_storage.uri=redis://:foo@bar:6379?cluster_name=placeholder'
```

## KVCacheManager Server Config

KVCM server可识别的配置参数列表如下。可通过配置文件、启动参数--env、系统环境变量进行配置：

```TEXT
## 参数配置方式有3种，相同配置按照从前往后的覆盖关系（后覆盖前）
## 1. 在配置文件中配置，如下示例
## 2. 通过启动参数配置，如：$BINARY_NAME --env kvcm.service.rpc_port=6381
## 3. 通过环境变量配置，set_env(kvcm.service.rpc_port, 6381)

# 指定系统Registry自身数据存储位置
# kvcm.registry_storage.uri=redis://127.0.0.1:6379?auth=123456

# 指定分布式锁服务的URI（用于多节点选主等）
# kvcm.distributed_lock.uri=redis://127.0.0.1:6379?auth=123456

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

# 可选值有dummy，local，logging，kmonitor；若不配置，默认启用logging
kvcm.metrics.reporter_type

# 传递给metrics reporter的配置值
kvcm.metrics.reporter_config

# 若启用的metrics reporter有周期性report任务，指定该任务的唤醒间隔；默认20000
kvcm.metrics.report_interval_ms

# log event publisher的初始化配置值，暂未启用
kvcm.event.event_publishers_configs
```

## CacheManager Initial Config

```TEXT
{
    "storage_config": {
        "type": "file", # 后端类型，可选值file,pace,mooncake,hf3fs,vcns_hf3fs
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
            # };
            "cache_prefer_strategy": 2,
            "meta_indexer_config": {
                "max_key_count": 1000000, # 单个meta indexer的key数量上限，同样影响reclaimer的逐出水位计算
                "mutex_shard_num": 16,
                "batch_key_size": 16,
                "meta_storage_backend_config": { # 控制meta indexer的storage backend，可选local本地文件或者redis
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
