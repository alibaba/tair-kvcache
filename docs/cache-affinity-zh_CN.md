# 缓存亲和性配置与使用

本页对应 `kvcm_affinity_merge` 当前实现。先读[整体设计](design-cache-affinity-v1.md)了解方案；调用链、默认值、后端能力和完整缺口见[实现参考](cache-affinity-implementation.md)。[English](cache-affinity-en_US.md)

## 1. 启用服务端策略

在服务端配置中添加：

```properties
kvcm.affinity.enabled=true
kvcm.affinity.strategy_file=/path/to/affinity.json
```

总开关默认 false。开启且不指定文件时使用 `Server::Init` 内置策略；指定文件加载失败时只记警告，不另行安装内置策略。策略文件在启动时读取，修改后需要重启加载，没有自动热加载。

以下是完整策略示例；数值为显式配置，不是所有字段的默认值：

```json
{
  "type": "local_replica",
  "enabled_aspects": {"write": true, "read": true, "eviction": true},
  "write": {
    "ops": {
      "filter": {"metric": "free_bytes", "min": 1073741824},
      "prefer_local": {"on_miss": "passthrough"},
      "sample": {"n": 5, "seed": "trace_id"},
      "sort": [{"metric": "load_ratio", "weight": -1}],
      "limit": 2
    }
  },
  "read": {
    "on_miss": {
      "enabled": true,
      "replication_hot_threshold": 3,
      "heat_half_life_ms": 60000,
      "caller_capacity_threshold": 0.90,
      "caller_capacity_buffer": 0.05,
      "suppression_window_ms": 60000,
      "max_replication_bytes": 67108864,
      "min_benefit_ratio": 1.0,
      "prefix_bonus": 0.0
    }
  },
  "eviction": {
    "ops": [{"op": "node_water_level", "threshold": 0.85, "low": 0.70}]
  },
  "replica_limits": {
    "max_replicas_per_key": 3,
    "max_instance_bytes": 4294967296,
    "min_retained_replicas": 1
  }
}
```

关键语义：

- `type` 必填；可用 `{"strategy":{...}}` 包装完整策略。写配置位于 `write.ops` 对象，淘汰配置位于 `eviction.ops` 数组。
- 写流水线固定为 `filter → prefer_local → sample → sort → limit`，每段可省略。负 sort weight 表示该指标越小越优先；`limit` 是候选数量，不是副本数量。
- `prefer_local.on_miss` 可为 `passthrough` 或 `abort`。普通写的 Abort 仍会降级后端自由分配，所以流水线过滤不是普通写的强制放置限制。
- 读取先在选定 storage 内选择已有副本；上例在热度、大小、收益、容量及抑制条件通过后才产生复制提示。指标未知时容量门槛放行。
- 上例容量门槛为 `0.90 - 0.05 = 0.85`。裸 `{"type":"local_replica"}` 的类默认门槛为 `0.85 - 0.05 = 0.80`，且不含写流水线。
- `max_replicas_per_key` 统计 Location 条目；`max_instance_bytes` 含原始副本、全部 spec 和 WRITING 预留。它们作用于普通及复制 StartWrite。
- `min_retained_replicas` 用于节点压力回收，按 spec 检查剩余 SERVING Location 数；默认保留 1，设 0 关闭保护。旧字段 `critical` 已弃用，兼容接受但在解析时警告并忽略；节点回收只使用 `threshold` 与 `low`。
- 最低副本保护不用于普通配额回收、显式删除或 GC；这些路径仍可删除最后一份副本。Noop/总开关关闭或关闭 eviction 时不触发节点压力回收。

策略整体按 instance、instance_group、process 优先级选择，不按字段合并。实例/实例组通过其 `affinity_strategy_json` 配置；不完整但解析成功的覆盖策略会替换低层配置。省略字段使用该策略的默认值；已知字段校验规则见下文。

**已有 instance 的 `RegisterInstance` 是幂等注册，不提供策略更新**：省略 `affinity_strategy_json`（空串）保留已有策略；显式提供与已存 JSON 不同的策略返回 `EC_DUPLICATE_ENTITY`，错误字段包含 `affinity_strategy_json`。相同 JSON 仅调整空白或对象键顺序仍按幂等重试接受，原存储值不变。空串不能清除已有覆盖；原来没有覆盖的实例也不能通过重复注册追加覆盖。组策略可通过 `UpdateInstanceGroup` 更新完整配置，需要 `current_version` 匹配且新 `version` 递增；有效的 instance 覆盖仍会遮挡组策略。配置写入 OK 也不代表策略 JSON 的全部字段经过严格校验。

选用 `local_replica` 时，解析器校验已知开关的布尔类型、数值范围和水位关系，写流水线解析错误也会使整个策略无效。无效的 instance/group 策略向下一层回退；进程策略加载失败则保留旧策略。未知字段仍兼容忽略；具体范围见[默认值与校验规则](cache-affinity-implementation.md#configuration)。

节点压力回收还需要在元数据 backend URI 中配置 `reclaim_indexer_type=node_lru`；仅打开 affinity 不会创建节点索引。未配置时节点采样返回 `EC_NOENT`，不能期待节点回收生效。

非 local 的单个持久化 meta 后端启用 `node_lru` 后，启动时会 O(N) 扫描 Location，重建节点成员关系并对账用量；扫描失败则启动失败，不保留重启前的精确 LRU 顺序。其他后端组合的用量仍依赖周期快照，见[恢复边界](cache-affinity-implementation.md#eviction)。

## 2. 配置客户端执行复制

以下字段追加到现有完整的 ManagerClient 配置，不是可独立初始化客户端的完整配置：

```json
{
  "auto_replicate": true,
  "replication_workers": 2,
  "replication_max_buffer_bytes": 268435456,
  "replication_max_pending_bytes": 268435456,
  "replication_node_bytes_per_second": 0,
  "replication_max_age_ms": 30000,
  "caller_node_refresh_seconds": 0
}
```

自动执行要求 HYBRID ManagerClient 同时有 MetaClient 和 TransferClient，并走可返回 hints 的 MatchLocation。`auto_replicate` 默认 false；仅打开服务端开关不会自动搬运数据。

SDK 默认每次读取 PACE 本地身份；设正刷新间隔会允许复用旧身份。启用 mempool 编译支持时，DRAM 和 SSD-only 配置均可选择 PACE provider，仍要求本机恰好有一个非零节点身份；NFS 使用独立 provider。

无复用 buffer 的任务先调用服务端 ReplicateCache，仅“不支持”回退客户端 Load/Save。显式复用接口支持按 spec 名称传 buffer 与共享 owner；异步返回值表示入队，结果回调表示复制结果。释放回调不表示复制成功。

队列过期时间不是运行中 RPC/传输的硬截止时间；资源预算和目标限速只在一个 SDK 进程内共享。全局服务端开关与 `auto_replicate` 均不会禁止调用显式复制接口。

即使走服务端 Copy，自动 hint 的单任务估算大小也受 `replication_max_buffer_bytes` 的入队检查。`Shutdown` 会等待 worker，队列中已有任务仍可能继续处理，不能视为立即取消所有复制。

多 spec 复制需检查源/目标集合：读结果的 `location_spec_names` 过滤不会裁剪 hint；复制申请未传 spec group，目标按实例全部 spec 分配。源只保存较小 spec group、未覆盖实例全部 spec 时可能失败回滚；多 storage 下，目标 storage 若与源不同，服务端 Copy 也会拒绝。具体边界见[读与复制提示](cache-affinity-implementation.md#read)、[复制执行](cache-affinity-implementation.md#replication)。

## 3. 可选的超节点拓扑

为 SDK 和服务端设置 `KVCM_NODE_TOPOLOGY_FILE=/path/to/topology.json`，文件内容例如：

```json
{"nodes": {"101": "rack-a", "102": "rack-a", "201": "rack-b"}}
```

键须与实际 `LocationSpec.node_id` / caller 一致：mempool 是数字 node id 字符串，NFS 是 IP。建议原子替换文件。拓扑映射按需每 5 秒重读，连续失败后旧数据 30 秒过期；这不表示策略 JSON 文件也会自动刷新。

服务端文件映射优先于 caller 自报的 supernode；同超节点候选还要有新鲜的节点指标。仅提供拓扑文件不会创建指标，也不保证同超节点选路始终有效。

## 4. 检查实际效果

1. 用 StartWrite/Finish 完成普通写，检查 GetCacheMeta 中各 spec 的实际 node id 和 SERVING 状态；不能只看请求中的 preferred nodes。
2. 从远端 caller 查询，检查正确的 spec 名称、hint target 和能力位。热度统计元数据远端命中，不等待数据 Load 成功；查询轮询也会增加热度。默认阈值为 3，仍受衰减、抑制、容量及成本门槛约束；前缀加权仅 PrefixMatch 使用实际位置。
3. 通过 SDK 结果回调/统计确认复制结果，再用 GetCacheMeta 的 `replica_locations` 检查目标节点完整的 SERVING 组件；兼容字段 `locations` 仅含每 key 第一个 Location。批量 RPC 顶层 OK、`server_copy_succeeded`（包含 already-exists）均不等于实际新建副本数。
4. 验证回收时同时核对物理空间、删除逐项结果与元数据。后端部分删除失败会保留完整 DELETING Location，全部 URI 确认已删除或不存在后才清理元数据。服务端 `kvcm.cache_gc.enabled` 默认 true，仅对保证安全重发的后端扫描重试；PACE 地址会复用且缺分配代次校验，未知结果后保留记录与预算，不自动重发。关闭 GC 后不再扫描重试。NFS 指标是合成容量，删除接口仍是占位实现，不能用 NFS 集成测试证明 PACE 真实空间释放。

若已收到提示却没有本地副本，按[复制结果与失败处理](cache-affinity-implementation.md#replication-results)依次确认任务是否被接受、是否执行成功、目标是否发布；发布应答失败时尤其不能只凭返回码认定目标不存在。

测试入口：[真实 Manager 集成](../integration_test/affinity/affinity_replication_test.py)、[SDK 复制测试](../kv_cache_manager/client/test/replication_executor_test.cc)、[策略测试](../kv_cache_manager/affinity/test/)。完整限制及后续开发优先级见[后端与待办](cache-affinity-implementation.md#status)。

当前生产候选限关闭透明 SSD 分层的 DRAM 或显式 SSD；透明分层删除仍有独立阻断，真实 PACE 复制、压力回收和重启仍须实机验收。
