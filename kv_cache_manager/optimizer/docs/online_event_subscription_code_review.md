# Online Optimizer 订阅 KVCM 事件代码审查指南

本文按运行时顺序梳理 Online Optimizer 从启动、发现 KVCM、同步配置、订阅事件、执行 `TraceQuery`，直到产生监控指标的完整代码链路。

本文用于逐步审查当前实现，不代表最终设计结论。每一节列出入口文件、调用关系、当前行为和待确认项。

## 1. 链路总览

```text
启动脚本和 JSON / 环境变量
  -> OnlineOptimizerServer::Init
  -> 创建 Registry / Manager / Metrics / Subscriber
  -> KvcmEventSubscriber supervisor 线程
  -> ServiceDiscovery 获取 KVCM seed
  -> MetaService.GetClusterInfo 获取 Leader
  -> OptimizerEventStreamService.GetConfiguration
  -> OptimizerServiceImpl::ApplyKvcmConfiguration
  -> 创建缺失的 Optimizer Group / Instance
  -> OptimizerEventStreamService.SubscribeEvents
  -> KvcmEventSubscriber::ProcessEvent
  -> OptimizerServiceImpl::ExecuteTraceQuery
  -> OnlineOptimizerManager::TraceQuery
  -> LiteHit / CacheIndexer
  -> 累计查询统计和 MRC 窗口
  -> OptimizerMetricsReporter::ReportInterval
  -> MetricsRegistry / KMonitor
  -> Optimizer HTTP /metrics
```

需要先明确两个边界：

1. HTTP、gRPC 和 `KvcmEventSubscriber` 都是接入适配器。Subscriber 将配置和事件分别交给 `OptimizerServiceImpl::ApplyKvcmConfiguration`、`OptimizerServiceImpl::ExecuteTraceQuery`，自身不直接操作 `OnlineOptimizerManager`。
2. Optimizer 是 KVCM 事件流的 gRPC 客户端。它主动连接 KVCM 的 Meta gRPC 端口，不需要为 KVCM 新增 Optimizer 入站端口。

## 2. 进程、端口和线程

### 2.1 端口

| 端口 | 方向 | 用途 |
|---|---|---|
| Optimizer `rpc_port` | 外部客户端 -> Optimizer | Group / Instance 管理和主动 `TraceQuery` |
| Optimizer `http_port` | 外部客户端 -> Optimizer | HTTP 管理接口和 `/metrics` |
| KVCM `meta_rpc_port` | Optimizer -> KVCM | `GetClusterInfo`、`GetConfiguration`、`SubscribeEvents` |

KVCM 事件不会经过 Optimizer 的 `rpc_port` 或 `http_port`。

### 2.2 新增线程

| 线程 | 创建位置 | 职责 |
|---|---|---|
| supervisor 线程 | `KvcmEventSubscriber::Start` | 服务发现、Leader 查询、配置同步 |
| stream 线程 | `KvcmEventSubscriber::UpdateWorker` | 读取当前 Leader 的事件并同步执行 `ProcessEvent` |

Optimizer 侧没有额外事件队列。stream 线程直接执行 `TraceQuery`；KVCM 侧 subscriber queue 是网络读取变慢时的有界缓冲。

## 3. 步骤一：启动进程

### 代码入口

- [`package/online_optimizer/start_optimizer_server.sh`](../../../package/online_optimizer/start_optimizer_server.sh)
- [`service/online_optimizer_server_main.cc`](../service/online_optimizer_server_main.cc)
- [`package/online_optimizer/default_optimizer_config.json`](../../../package/online_optimizer/default_optimizer_config.json)

打包脚本执行：

```bash
online_optimizer_server_main -c default_optimizer_config.json
```

调用顺序：

```text
解析 -c 配置文件
  -> 解析 -e key=value 覆盖项
  -> OnlineOptimizerServer::Init
  -> OnlineOptimizerServer::Start
  -> OnlineOptimizerServer::WaitForShutdown
```

当前行为：

- `-c` 是必填参数，必须先读取一份 JSON 配置。
- `-e`、系统环境变量和下划线形式环境变量可以覆盖 JSON 字段。
- 打包默认 JSON 当前直接启用订阅，并配置 static localhost 地址。

审查清单：

- [ ] 是否继续要求 `-c` 必填。
- [ ] 打包默认配置是否应该启用 KVCM 订阅。
- [ ] static localhost 是否只适合本地测试。
- [ ] Spectrum 部署由哪个环境变量写入实际服务发现 URL。

## 4. 步骤二：解析订阅配置

### 代码入口

- [`service/online_optimizer_server_config.h`](../service/online_optimizer_server_config.h)
- [`service/online_optimizer_server_config.cc`](../service/online_optimizer_server_config.cc)

订阅配置对象是 `KvcmEventSubscriptionConfig`，字段为：

```text
enable
service_discovery_url
consumer_id
discovery_refresh_interval_ms
```

解析顺序：

```text
OnlineOptimizerServerConfig::FromRapidValue
  -> KvcmEventSubscriptionConfig::FromRapidValue
  -> OnlineOptimizerServerConfig::OverrideFromEnviron
  -> KvcmEventSubscriptionConfig::Validate
```

当前行为：

- `enable=false` 时不要求服务发现地址。
- `enable=true` 时要求 discovery URL、consumer ID 非空，刷新周期大于 0。
- `consumer_id` 只用于 KVCM 日志，不是带 offset 的消费组。

审查清单：

- [ ] 当前四个字段是否都是运行必需配置。
- [ ] `consumer_id` 是否需要包含 Optimizer 副本标识。
- [x] keepalive 和退避保持为代码常量，不新增配置项。
- [ ] 订阅配置错误是否应该阻止整个 Optimizer 启动。

## 5. 步骤三：构造 Server 对象图

### 代码入口

- [`service/online_optimizer_server.cc`](../service/online_optimizer_server.cc)
- [`config/optimizer_registry_manager.cc`](../config/optimizer_registry_manager.cc)
- [`manager/online_runtime/online_optimizer_manager.cc`](../manager/online_runtime/online_optimizer_manager.cc)

`OnlineOptimizerServer::Init()` 的创建顺序：

```text
OptimizerRegistryManager
  -> OnlineOptimizerManager
  -> OnlineOptimizerManager::Recover
  -> MetricsRegistry
  -> OptimizerMetricsReporter
  -> OptimizerServiceImpl
  -> KvcmEventSubscriber（仅 enable=true）
```

| 对象 | 职责 |
|---|---|
| `OptimizerRegistryManager` | 保存和恢复 Optimizer Group / Instance 配置 |
| `OnlineOptimizerManager` | 保存每个 Instance 的运行状态并执行 `TraceQuery` |
| `OptimizerMetricsReporter` | 把 Manager 状态转换成 Prometheus / KMonitor 指标 |
| `OptimizerServiceImpl` | HTTP、gRPC 和 KVCM 事件共用的 TraceQuery 处理路径，以及 Optimizer API 实现 |
| `KvcmEventSubscriber` | 发现 KVCM、同步配置并消费事件 |

当前行为：

- 正常启动先执行 `Recover()`，再启动订阅。
- `Recover()` 恢复配置和运行时对象，不恢复 LRU 内容或累计指标。
- 首次恢复失败时，Server 启动后异步重试。

审查清单：

- [ ] 恢复失败时是否允许先启动 Subscriber 并消费事件。
- [ ] 自动同步的 Group / Instance 是否应该持久化。
- [ ] KVCM 是否是自动配置的唯一数据源，还是允许人工配置并存。
- [ ] 已持久化配置与最新 KVCM 配置冲突时采用哪一方。

## 6. 步骤四：启动端口和后台线程

### 代码入口

- [`service/online_optimizer_server.cc`](../service/online_optimizer_server.cc)
- [`service/grpc/optimizer_service_grpc.cc`](../service/grpc/optimizer_service_grpc.cc)
- [`service/http/optimizer_service_http.cc`](../service/http/optimizer_service_http.cc)

`OnlineOptimizerServer::Start()` 当前顺序：

```text
InitRpcServer
  -> InitHttpServer
  -> running = true
  -> KvcmEventSubscriber::Start
  -> RecoveryRetryLoop（按需）
  -> Metrics LoopThread（按需）
```

审查清单：

- [ ] Subscriber 启动失败是否应该让整个 Optimizer 启动失败。
- [ ] 只运行监控链路时，Optimizer 自己的 gRPC API 是否仍必须开启。
- [ ] HTTP 启动失败后，已经启动的 gRPC server 是否正确清理。

## 7. 步骤五：服务发现和 Leader 查询

### 代码入口

- [`service/event_subscriber/kvcm_event_subscriber.cc`](../service/event_subscriber/kvcm_event_subscriber.cc)
- [`common/service_discovery_factory.cc`](../../common/service_discovery_factory.cc)
- [`common/service_discovery.h`](../../common/service_discovery.h)

初始化：

```text
KvcmEventSubscriber::Init
  -> ServiceDiscoveryFactory::CreateServiceDiscovery
```

支持 `static://`、`vipserver://` 和 `spectrum://`。

supervisor 线程循环：

```text
KvcmEventSubscriber::SupervisorLoop
  -> RefreshLeader
  -> ServiceDiscovery::Refresh
  -> DiscoverLeader
  -> 等待 discovery_refresh_interval_ms
```

Leader 查询：

```text
ServiceDiscovery::GetAllEndpoints
  -> 逐个健康 seed
  -> MetaService.GetClusterInfo（1 秒 deadline）
  -> leader.host + leader.meta_rpc_port
```

当前行为：

- 服务发现结果只是 seed，不是最终事件流地址。
- Optimizer 只连接 KVCM 返回的当前 Leader。
- 多个 seed 按顺序尝试。
- Optimizer 或 KVCM 任意一方先启动都可以，失败后会继续刷新。

审查清单：

- [ ] Spectrum 返回的端口是否就是 KVCM Meta gRPC 端口。
- [ ] 是否需要通过 Spectrum URL 的 `port` 参数覆盖端口。
- [ ] 逐个 seed、每个最多等待 1 秒是否可以接受。
- [ ] Leader host 是否需要处理 IPv6 格式。

## 8. 步骤六：拉取 KVCM 配置快照

### 代码入口

- Optimizer：[`KvcmEventSubscriber::SyncConfiguration`](../service/event_subscriber/kvcm_event_subscriber.cc)
- 配置应用：[`OptimizerServiceImpl::ApplyKvcmConfiguration`](../service/optimizer_service_impl.cc)
- KVCM：[`OptimizerEventServiceGRpc::GetConfiguration`](../../service/grpc_service/optimizer_event_service_grpc.cc)
- 协议：[`optimizer_service.proto`](../../protocol/protobuf/optimizer_service.proto)

Optimizer 调用 `OptimizerEventStreamService.GetConfiguration`，KVCM 返回：

```text
KvcmInstanceGroupConfiguration
  - name
  - capacity_bytes

KvcmInstanceConfiguration
  - instance_group_name
  - instance_id
  - block_size
  - location_spec_infos
  - location_spec_groups
```

当前行为：

- 启动及每次服务发现刷新都会拉完整配置快照。
- 未知 `instance_id` 会立即唤醒一次配置刷新。
- 配置同步只新增缺失项，不更新、不删除已有配置。
- 只有配置同步成功才会启动或切换 stream；同步失败时保留当前 Leader 的旧 stream。
- Group、registry 或受支持的 full-only Instance 应用失败时，整个配置同步返回失败。
- 暂不支持的 multi-group Instance 会被明确记录并跳过，不阻止其他 Instance 开始消费。

审查清单：

- [x] 首次连接和 Leader 切换必须等待配置同步成功。
- [ ] 是否需要区分 Leader 刷新周期和完整配置刷新周期。
- [ ] quota、block size、spec 变化后是否要求热更新。
- [ ] KVCM 删除 Instance 后是否自动删除 Optimizer Instance。
- [ ] 未知 Instance 的首条事件允许丢弃，还是等待配置后重试。

## 9. 步骤七：映射 Group 和 Instance

网络请求和响应状态检查位于 `KvcmEventSubscriber::SyncConfiguration()`；映射和注册逻辑位于
`OptimizerServiceImpl::ApplyKvcmConfiguration()`。

### Group 映射

```text
KVCM group name        -> Optimizer group name
KVCM capacity_bytes    -> 一个 capacity_gb 容量点
eviction_policy        -> lru
enable_prefix_hash     -> true
theoretical max        -> true
ttl_seconds            -> 24 hours
```

### Instance 映射

```text
KVCM instance_id       -> Optimizer instance_id
KVCM block_size        -> Optimizer block_size
location_spec_infos    -> 全量原样转换
location_spec_groups   -> 全量保留 group name 和 spec names
linear_step            -> 0
OptimizerStateInfo     -> 接入层不填写，由 OnlineOptimizerManager 判断
```

`KvcmEventSubscriber` 调用：

```text
OptimizerServiceImpl::ApplyKvcmConfiguration
```

ServiceImpl 随后调用：

```text
OnlineOptimizerManager::CreateInstanceGroup
OnlineOptimizerManager::RegisterInstance
```

当前行为：

- 自动注册明确按 full-attention、full-only 处理。
- ServiceImpl 不判断哪个 spec group 是 full，将 KVCM 返回的所有 group 交给 Manager。
- Manager 的直接注册接口对无显式状态的 full-only Instance 只在唯一 group 时采用该 group；多 group 返回
  `EC_BADARGS`。
- KVCM 自动接入暂不推断 linear state；多 group 的语义不明确时标记为 unsupported 并跳过。
- Subscriber 收到已知 unsupported Instance 的事件时直接丢弃，不触发配置刷新。
- KVCM Group quota 被每个 Instance 分别作为完整容量模拟，不是共享 quota。
- KVCM 自动创建的 Group 固定使用 24 小时 TTL，限制 LiteHit 保留的历史工作集。
- Group 已存在时不检查 capacity、prefix hash、theoretical flag。
- Instance 已存在时不检查 block size 和 specs。

审查清单：

- [x] 自动 Group 默认开启 `enable_prefix_hash=true`。
- [x] 自动 Group 默认开启 `enable_theoretical_max_cache=true`。
- [x] 不根据 group 名称猜测 full group，也不使用全部 specs fallback。
- [ ] 支持多 group 时，由协议显式提供 `full_location_spec_group_name`。
- [ ] GLM-5.2 是否可以按 `linear_step=0` 的 full-only 模型处理。
- [ ] 已存在但配置不一致的 Group / Instance 如何迁移。

## 10. 步骤八：创建运行时 InstanceState

### 代码入口

- [`manager/online_runtime/online_optimizer_manager.cc`](../manager/online_runtime/online_optimizer_manager.cc)
- [`config/optimizer_registry_manager.cc`](../config/optimizer_registry_manager.cc)
- [`liteHit/lite_hit.cc`](../liteHit/lite_hit.cc)

`OnlineOptimizerManager::RegisterInstance()` 执行：

```text
校验 Instance 和 Group
  -> 无显式 full group 且仅有一个 group 时，RegisterInstance 直接补齐 full-only 状态
  -> 持久化 OptimizerInstanceInfo（启用 registry 时）
  -> RegisterInstanceInternal
  -> 计算 size_full
  -> capacity_gb 换算为 capacity blocks
  -> 创建 InstanceState
  -> linear_step == 0：创建 LiteHit
  -> linear_step > 0：创建 CacheIndexer
  -> 写入 instances_[instance_id]
```

当前行为：

- `instance_id` 是运行时隔离边界。
- 每个 InstanceState 有独立 mutex。
- 自动注册只会进入 LiteHit full-only 路径。
- registry 只持久化配置，不持久化 LRU 状态和统计值。

审查清单：

- [ ] 每个 Instance 独立模拟完整 Group quota 是否符合口径。
- [ ] Optimizer 重启后从空 LRU 状态重新预热是否可以接受。
- [ ] registry 写入失败时是否阻止订阅 stream。
- [ ] 自动配置和人工注册同名 Instance 时采用什么行为。

## 11. 步骤九：建立长连接

### 代码入口

- Optimizer：[`KvcmEventSubscriber::UpdateWorker / EndpointLoop`](../service/event_subscriber/kvcm_event_subscriber.cc)
- KVCM：[`OptimizerEventServiceGRpc::SubscribeEvents`](../../service/grpc_service/optimizer_event_service_grpc.cc)
- 协议：[`OptimizerEventStreamService`](../../protocol/protobuf/optimizer_service.proto)

调用链：

```text
UpdateWorker(leader_endpoint)
  -> 创建 EndpointWorker
  -> 创建 stream 线程
  -> grpc::CreateChannel
  -> SubscribeEvents(consumer_id)
  -> reader->Read(event)
  -> ProcessEvent(event, kvcm_ip)
```

当前行为：

- 同时只维护当前 Leader 的一条 stream。
- Leader 地址不变时不重建 worker。
- 不做 query type 筛选，全部事件进入 `ProcessEvent`。
- stream 线程同步处理事件，不分发到线程池。
- 断流后按 500 ms 起步、最大 30 秒的指数退避重连，并添加最多 ±20% jitter。
- stream channel 使用 6 分钟 keepalive 和 20 秒 keepalive timeout。
- 协议没有 offset、ack 和 replay，断线期间可能丢事件。

审查清单：

- [x] 增加保守客户端 keepalive：6 分钟，timeout 20 秒。
- [x] 固定重连改为指数退避并添加 jitter。
- [x] Leader 切换和停止能够立即唤醒退避等待。
- [ ] 单 stream 线程是否满足预计事件 QPS。
- [ ] 是否接受无重放、暂不监控丢失的语义。

## 12. 步骤十：事件进入 TraceQuery

### 代码入口

- [`KvcmEventSubscriber::ProcessEvent`](../service/event_subscriber/kvcm_event_subscriber.cc)
- [`OptimizerServiceImpl::ExecuteTraceQuery`](../service/optimizer_service_impl.cc)
- [`OnlineOptimizerManager::TraceQuery`](../manager/online_runtime/online_optimizer_manager.cc)

两类入口共用同一个业务处理函数：

```text
Optimizer HTTP / gRPC
  -> OptimizerServiceImpl::TraceQuery
      -> OptimizerServiceImpl::ExecuteTraceQuery

KVCM stream
  -> KvcmEventSubscriber::ProcessEvent
      -> OptimizerServiceImpl::ExecuteTraceQuery

ExecuteTraceQuery
  -> protobuf block_keys 转为 std::vector<int64_t>
  -> location_spec_names 保留在 TraceQueryRequest 中
  -> input_token_len == 0 且 token_ids 非空时使用 token_ids.size()
  -> manager->TraceQuery(instance_id, block_keys, input_token_len, timestamp_ns, result)
  -> 填充 TraceQueryResponse

KvcmEventSubscriber::ProcessEvent
  -> 基于 TraceQueryResponse 填充现有 OptimizerServiceMetricsCollector
  -> 调用现有 ReportPerQuery 上报指标
```

错误处理：

```text
EC_INSTANCE_NOT_EXIST
  -> 记录丢弃日志
  -> RequestConfigurationRefresh

其他错误
  -> 记录丢弃日志
```

当前行为：

- Subscriber 和 HTTP/gRPC handler 共用 `ExecuteTraceQuery` 的输入转换与 Manager 调用。
- Subscriber 不经过面向 RPC 的 `OptimizerServiceImpl::TraceQuery` handler。
- Subscriber 不构造 `RequestContext` 或 `OptimizerCallGuard`。
- Subscriber 根据 `ExecuteTraceQuery` 返回的现有 `TraceQueryResponse` 填充 Collector，再由现有 `ReportPerQuery`
  上报；Stream 的 `client_ip` 来自实际 gRPC 对端 KVCM IP。
- Manager 聚合状态同样保留，因此 Stream 同时进入单次 `query_*` 和周期累计 `trace_*`。
- 每条 Stream Event 作为一次逻辑服务请求计入 service QPS、error QPS 和本地处理耗时，但不产生 RPC access log。
- Stream 的处理耗时从 `ProcessEvent` 调用 `ExecuteTraceQuery` 前开始，到调用返回为止，不包含等待 Stream 消息的时间。

审查清单：

- [x] 订阅事件产生与 HTTP/gRPC 相同的 per-query business metrics。
- [x] 订阅事件通过现有 Collector 和 `ReportPerQuery` 复用 HTTP/gRPC 的指标上报实现。
- [ ] 是否需要保存每条 `TraceQueryResult`。
- [x] 输入转换和 Manager 调用已通过 `ExecuteTraceQuery` 统一，不伪造 RPC `RequestContext`。
- [ ] `input_token_len==0` 回退比例是否需要监控。

## 13. 步骤十一：LiteHit、理论命中率和 MRC

`TraceQuery` 的 full-only 路径：

```text
校验 input_token_len / timestamp_ns
  -> timestamp_ns == 0 时使用到达时间
  -> 查找 InstanceState
  -> input_token_len == 0 时按完整 block 推算
  -> 加 InstanceState mutex
  -> NormalizeRequest / prefix hash
  -> LiteHit::ProcessRequest
  -> 生成 RequestFact.hit_curve
  -> MrcWindow::Record
  -> 投影各容量命中率
  -> 按需计算理论无限容量命中
  -> 更新累计统计
```

时间语义：

- producer timestamp 非零时，TTL 使用生产时间。
- 时间戳为零时回退到 Optimizer 到达时间。
- LiteHit 和 legacy TTL indexer 对乱序时间做单调保护。
- 纯 LRU 只依赖事件顺序。

理论结果只有 `enable_theoretical_max_cache=true` 时才计算。

MRC 当前定义：

```text
窗口目标命中量 = 窗口理论无限容量最大可命中 block 数 × target_hit_rate_percent

MRC = 最近一个上报窗口内，保留上述目标命中量所需的最小 LRU 容量，单位 byte
```

`target_hit_rate_percent` 是相对于理论最大可命中量的比例，不是绝对请求命中率。例如窗口理论最大
命中率为 68.6% 时，`target_hit_rate_percent=95` 对应的绝对命中率目标约为
`68.6% × 95% = 65.17%`，而不是 95%。当前固定输出 60%、80%、90%、95%、99%、99.5% 六个相对目标。

MRC 窗口使用 required blocks 的稀疏差分点，避免大 reuse distance 直接扩张出同等长度的数组。

审查清单：

- [x] MRC 只在 theoretical 统计开启时累计。
- [x] 输出理论无限容量命中 block 数 60%、80%、90%、95%、99%、99.5% 的容量点。
- [x] 使用稀疏差分点，避免大 reuse distance 导致大内存。
- [x] MRC 使用 full location spec group 的 `size_full` 转换成 byte。
- [ ] 乱序时间采用单调 clamp 是否符合 trace 语义。
- [ ] input length 回退是否会系统性高估命中率。

## 14. 步骤十二：指标上报和暴露

### 代码入口

- [`OnlineOptimizerServer::Start / DoStop`](../service/online_optimizer_server.cc)
- [`OptimizerMetricsReporter::ReportInterval`](../metrics/optimizer_metrics_reporter.cc)
- [`OptimizerMetricsReporter::ReportPerQuery`](../metrics/optimizer_metrics_reporter.cc)
- [`OptimizerServiceHttp::RegisterPrometheusEndpoint`](../service/http/optimizer_service_http.cc)

每个 `metrics_report_interval_ms`：

```text
OnlineOptimizerManager::ListInstances
  -> 读取累计查询数和累计命中率

OnlineOptimizerManager::TakeMrcMetrics
  -> 在同一个 MrcWindow 快照中计算六个目标比例的 MRC 容量点
  -> 清空该窗口

OptimizerMetricsReporter::ReportInterval
  -> MetricsRegistry
  -> KMonitor

PrometheusExporter::Expose
  -> HTTP GET /metrics
```

每次 HTTP/gRPC handler 或 Stream Event 处理都会通过现有 `ReportPerQuery` 上报指标。对于
`TraceQuery`，它产生 `query_hit_rate`、`query_hit_count`、`query_total_blocks`、`query_max_hit_count`、
`query_max_hit_rate` 和 `query_capacity_efficiency`；同时产生 `service.qps`、`service.query_rt_us` 和
`service.error_qps`。Stream 按每条 Event 计一次，而不是按长连接建连计一次。

当前窗口语义：

| 指标 | 当前口径 |
|---|---|
| `trace_query_total` | 启动以来累计 |
| `trace_query_hit_rate` | 启动以来累计 |
| `trace_query_max_hit_rate` | 启动以来累计 |
| `mrc{target_hit_rate_percent=...}` | 最近一个上报窗口中，相对理论最大可命中量的六个容量需求点 |

单次实时理论命中率继续由 Collector 上报为 `query_max_hit_rate`；MRC 只负责容量曲线。

审查清单：

- [x] 单次理论命中率由 Collector 上报，不在 MRC 内重复计算。
- [x] 六个 MRC 容量点在同一次持锁快照中取出。
- [x] 保留累计 `trace_query_max_hit_rate`。
- [x] 无事件窗口上报 0。
- [x] Prometheus 和 KMonitor 保持相同标签和语义。

## 15. 步骤十三：停止和切主

### 代码入口

- [`OnlineOptimizerServer::DoStop`](../service/online_optimizer_server.cc)
- [`KvcmEventSubscriber::Stop / StopWorker`](../service/event_subscriber/kvcm_event_subscriber.cc)

停止顺序：

```text
running = false
  -> KvcmEventSubscriber::Stop
      -> 唤醒并 join supervisor
      -> TryCancel 当前 stream
      -> join stream worker
  -> 停止 Optimizer gRPC / HTTP
  -> join recovery 线程并停止 metrics LoopThread
  -> Shutdown KMonitor
```

切主顺序：

```text
发现新 Leader
  -> SyncConfiguration
  -> 停止旧 worker 和 stream
  -> 创建新 Leader stream
```

审查清单：

- [x] 配置同步失败时禁止切换 stream，并保留旧 Leader 的 stream。
- [x] keepalive / 退避等待能被 `StopWorker` 立即打断。
- [x] metrics 周期任务复用 `LoopThread`，停止时通过条件变量立即唤醒。
- [ ] supervisor 正在执行 unary RPC 时的停止延迟是否可以接受。

## 16. 建议审查顺序

1. [x] 自动 Group 同时开启 prefix hash 和 theoretical max。
2. [ ] 明确 GLM-5.2 是否可以按 full-only 建模。
3. [x] 配置同步失败不得开始消费新 Leader 事件。
4. [x] 单次实时理论命中率由 Collector 上报，MRC 独立输出容量曲线。
5. [x] MRC 改为稀疏结构，控制内存。
6. [x] stream channel 增加保守 keepalive、指数退避和 jitter。
7. [x] Subscriber 复用现有 Collector 和 Reporter 上报 per-query metrics。
8. [ ] 决定默认配置是否关闭订阅。
9. [ ] 明确配置同步是否只新增，还是处理修改和删除。

## 17. 文件职责索引

| 文件 | 职责 |
|---|---|
| `service/online_optimizer_server_main.cc` | 进程参数和信号入口 |
| `service/online_optimizer_server_config.h/cc` | Server 和订阅配置 |
| `service/online_optimizer_server.h/cc` | 对象、端口和线程生命周期 |
| `service/event_subscriber/kvcm_event_subscriber.h/cc` | 服务发现、Leader、配置拉取、stream、事件入口及订阅错误处理 |
| `service/optimizer_service_impl.h/cc` | HTTP、gRPC 和 Subscriber 共用的配置应用、TraceQuery 处理路径及 API 实现 |
| `config/optimizer_registry_manager.h/cc` | Group / Instance 配置及持久化 |
| `manager/online_runtime/online_optimizer_manager.h/cc` | InstanceState、TraceQuery 和累计统计 |
| `metrics/mrc_window.h/cc` | MRC 容量曲线累计、快照和清空 |
| `liteHit/lite_hit.h/cc` | full-attention 容量无关 LRU replay |
| `index/online/cache_indexer.h` | legacy indexer 和 producer timestamp 入口 |
| `index/online/ttl_cache_indexer_wrapper.h/cc` | legacy TTL 时间处理 |
| `metrics/optimizer_metrics_collector.h/cc` | 单次请求和查询结果采集 |
| `metrics/optimizer_metrics_reporter.h/cc` | per-query 和 interval metrics 上报 |
| `service/http/optimizer_service_http.h/cc` | HTTP API 和 Prometheus `/metrics` |
| `protocol/protobuf/optimizer_service.proto` | Optimizer API、配置快照和事件流协议 |
| `service/grpc_service/optimizer_event_service_grpc.h/cc` | KVCM 侧配置快照和事件 stream |
