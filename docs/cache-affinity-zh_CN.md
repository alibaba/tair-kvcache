# 缓存亲和性配置与使用

本页对应 `kvcm_affinity_merge` 当前实现。整体调用链、默认值、后端能力和已知缺口见[整体设计](design-cache-affinity-v1.md)。[English](cache-affinity-en_US.md)

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
- **`min_retained_replicas` 已有底层保护函数，但当前 ReclaimByNode 未传入此参数，不能保证节点压力回收保留一个副本。** `critical` 虽可解析，当前也没有独立执行逻辑。
- Noop/总开关关闭时，`GetReplicaLimits` 的默认保留数仍为 1，`ReclaimByLRU` 可能继续应用该保护；关闭 affinity 不等于恢复全部旧回收行为。

策略整体按 instance、instance_group、process 优先级选择，不按字段合并。实例/实例组通过其 `affinity_strategy_json` 配置；不完整但解析成功的覆盖策略会替换低层配置。解析器对部分错误字段使用默认值，加载成功不等于全部字段有效。

**已有 instance 不能靠再次 RegisterInstance 更新亲和性策略**：该分支不比较或覆盖 `affinity_strategy_json`，可能返回 OK 但继续使用旧值。组策略可通过 `UpdateInstanceGroup` 更新完整配置，需要 `current_version` 匹配且新 `version` 递增；有效的 instance 覆盖仍会遮挡组策略。配置写入 OK 也不代表策略 JSON 的全部字段经过严格校验。

节点压力回收还需要在元数据 backend URI 中配置 `reclaim_indexer_type=node_lru`；仅打开 affinity 不会创建节点索引。未配置时节点采样返回 `EC_NOENT`，不能期待节点回收生效。

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

SDK 默认每次读取 PACE 本地身份；设正刷新间隔会允许复用旧身份。当前仅支持 mempool 内存类型和 NFS 的 provider 工厂选择，SSD-only 配置尚未接入自动选择。

无复用 buffer 的任务先调用服务端 ReplicateCache，仅“不支持”回退客户端 Load/Save。显式复用接口支持按 spec 名称传 buffer 与共享 owner；异步返回值表示入队，结果回调表示复制结果。释放回调不表示复制成功。

队列过期时间不是运行中 RPC/传输的硬截止时间；资源预算和目标限速只在一个 SDK 进程内共享。全局服务端开关与 `auto_replicate` 均不会禁止调用显式复制接口。

即使走服务端 Copy，自动 hint 的单任务估算大小也受 `replication_max_buffer_bytes` 的入队检查。`Shutdown` 会等待 worker，队列中已有任务仍可能继续处理，不能视为立即取消所有复制。

多 spec 复制需检查源/目标集合：读结果的 `location_spec_names` 过滤不会裁剪 hint；复制申请未传 spec group，目标按实例全部 spec 分配。源只保存较小 spec group、未覆盖实例全部 spec 时可能失败回滚；多 storage 下，目标 storage 若与源不同，服务端 Copy 也会拒绝。具体边界见[整体设计第 5、6 节](design-cache-affinity-v1.md)。

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
4. 验证回收时同时核对物理空间与元数据。NFS 指标是合成容量，删除接口仍是占位实现，不能用 NFS 集成测试证明 PACE 真实空间释放。

测试入口：[真实 Manager 集成](../integration_test/affinity/affinity_replication_test.py)、[SDK 复制测试](../kv_cache_manager/client/test/replication_executor_test.cc)、[策略测试](../kv_cache_manager/affinity/test/)。完整限制及后续开发优先级见[整体设计](design-cache-affinity-v1.md)。
