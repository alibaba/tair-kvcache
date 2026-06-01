# Cache Affinity v1 实现说明

**前序文档**：
- `cache-affinity-v1-zh_CN.md` —— v1 完整设计规范（策略框架 + 数据模型 + 算法决策全集）
- `cache-affinity-zh_CN.md` / `cache-affinity-en_US.md` —— v0 写时 affinity

**对应分支**：`feat/cache-affinity-supernode`（46 files, +1501 / -527）

---

## 目录

**第一部分 范围与核心设计**
- §1 做了什么 / 没做什么
- §2 策略框架
- §3 数据模型变更
- §4 三条路径接入

**第二部分 具体修改**
- §5 文件变更索引
- §6 侵入边界

**第三部分 待做与扩展**
- §7 待完成项
- §8 扩展指南

---

# 第一部分：范围与核心设计

---

## 1. 做了什么 / 没做什么

### 1.1 做了什么

在 KVCM 中建立节点级亲和性框架，打通写/读/淘汰三条路径的完整信号管道。

本次改动分三层：

| 层 | 内容 |
|---|---|
| 框架层 | `AffinityStrategy` 抽象接口（3 个对称入口）、`StrategyFactory`（JSON 解析 + memoize）、3 层优先级链（instance > instance_group > process）、全局 kill-switch |
| 管道层 | `caller_node_ip` / `caller_supernode_id` 从 proto 到策略的全链路透传；`ReplicationHint` 从策略到 response 的全链路回传；`node_id` 从 backend 到 `LocationSpec` 的持久化 |
| 接入层 | 写路径（`GenWriteLocation` → `ResolveWrite`）、读路径（`SelectAndMergeForMatch` → `ResolveRead`）、淘汰路径（`TryReclaimOnGroup` → `ResolveEviction` → per-node `FilterLocID`） |

其中**节点级淘汰**是本次关键改造点。改造前 `CacheReclaimer` 只能按整 key 淘汰，无法区分同一 key 的不同副本分布在哪些节点。本次把淘汰粒度下沉到 `LocationSpec.node_id`：`ResolveEviction` 输出超水位节点集合，`FilterLocID` 新增 per-node 匹配路径，只淘汰超载节点上的副本。没有节点级淘汰，本地副本只增不减，多副本方案无法闭环。

### 1.2 没做什么

**不包含算法层面的严格设计。** `LocalReplicaAffinityStrategy` 是跑通链路的最小实现——写路径复用 v0 五段流水线，读路径本地优先和复制触发只做最朴素判定。

算法的设计需要长期迭代，不同业务场景（集群规模、热度分布、存储拓扑）下的最优策略不同，需要结合线上数据逐步调优。后续方向：

- 写 admission：prefix-aware 热度判定（父 block 热 → 走本地），替代当前的"有 caller_node_ip 就 prefer local"
- 读 on_miss：结合本地水位和全局副本分布的复制触发条件，替代当前的纯频率阈值
- 淘汰：跨节点公平性、副本冗余度感知（最后一份不淘汰），替代当前的纯水位比较

**超节点亲和性是优先级最高的待做项。** `caller_supernode_id` 管道在本 PR 已完整铺通（proto → RequestContext → AffinityResolveContext → StrategyContext），策略层尚未消费。超节点维度的价值：同机房/同交换机节点间带宽远优于跨机房，引入"同节点 > 同超节点 > 远端"层次化局部性后，即使 caller 本地没有副本也能从同超节点读取，降低跨域流量。建议作为算法迭代第一个 PR。

**本 PR 的核心价值：后续无论算法怎么改，都不需要动框架层和管道层。** 新算法只需实现 `AffinityStrategy` 子类。

---

## 2. 策略框架

### 2.1 AffinityStrategy 接口

一级行为只有 3 个，接口固定不再变化：

```
AffinityStrategy (抽象接口)
├── ResolveWrite()    → WriteDecision {status, hints}
├── ResolveRead()     → ReadDecision  {picked_specs, side_effects}
└── ResolveEviction() → unordered_set<node_id>
```

设计要点：

| 点 | 选择 | 原因 |
|---|---|---|
| 接口数量 | 3 个一级入口 | 写/读/淘汰覆盖缓存生命周期全部决策点 |
| toggle | 算法内部 `Params.enable_*` | 关闭的行为由子类 short-circuit 成 no-op，调用方不需 if-else |
| 上下文 | `StrategyContext`（通用信号）+ `AffinityResolveContext`（管道上下文） | 策略只看通用信号，不依赖管道细节 |
| 策略选择 | 3 层优先级链 instance > instance_group > process | 支持多租户不同粒度策略覆盖 |
| 策略创建 | `StrategyFactory` 从 JSON 解析，按 JSON 文本 memoize | 相同 JSON 共享解析结果，热更新换 JSON 即可 |
| 复制触发 | 读一级的 on_miss 子项，不独立为一级 | `ReplicationHint` 作为 `ReadSideEffect` 子类经 `ReadDecision.side_effects` 透传 |

### 2.2 两个内置策略

**LocalReplicaAffinityStrategy**（v1 默认，需显式配置启用）：

基于 caller 节点局部性 + read-miss 反应式复制 + 节点水位 LRU。3 个一级 method 内部分别调用 private helper（`RunWritePipeline` / `PickLocalSpec` + `ShouldEmitReplicationHint` / `ShouldEvictByNodeWaterLevel`），算法细节不出现在通用接口中。

**NoopAffinityStrategy**（兜底 / kill-switch）：

3 个 method 直接返回 no-op 决策。配置 `{"type": "noop"}` 后 KVCM 行为等价 v0。

### 2.3 应急配置

| 场景 | 配置（热加载） |
|---|---|
| 整体关闭 | `{"type": "noop"}` |
| 仅关复制 | `{"read": {"on_miss": {"enabled": false}}}` |
| 仅关读亲和 | `{"enabled_aspects": {"read": false}}` |
| 仅关节点淘汰 | `{"enabled_aspects": {"eviction": false}}` |

总开关 `KVCM_AFFINITY_ENABLED=0` 等价全局 noop。

### 2.4 文件结构

```
affinity/
├── affinity_strategy.h           # 抽象接口 + ReadSideEffect + ReadDecision + WriteDecision
├── noop_strategy.h               # 兜底空实现
├── local_replica_strategy.{h,cc} # v1 默认策略，含 ReplicationHint
├── cache_affinity_manager.{h,cc} # 策略持有 + metrics 快照 + 决策分发
├── frequency_sketch.{h,cc}       # per-(caller, key) 频率计数
├── strategy_factory.{h,cc}       # JSON → 策略实例
├── node_metrics.h                # 节点指标数据结构
└── pipeline/
    ├── candidate_pipeline.{h,cc} # 写路径 5 段流水线
    ├── filter_cond.{h,cc}        # 过滤条件
    └── metric_catalog.{h,cc}     # 指标名注册表
```

---

## 3. 数据模型变更

### 3.1 LocationSpec.node_id

```proto
message LocationSpec {
    string name = 1;
    string uri = 2;
    string node_id = 3;  // 新增：spec 落在哪个物理节点
}
```

node_id 放在 LocationSpec 而非 CacheLocation 顶层，因为 strict=false 写入时同一 location 的 spec 可能跨节点。

### 3.2 LocationDescriptor

替代原来的 `pair<ErrorCode, DataStorageUri>`：

```cpp
struct LocationDescriptor {
    ErrorCode ec = EC_OK;
    DataStorageUri uri;
    std::string node_id;  // 空 = backend 未上报
};
```

URI hostname 是集群名，物理节点藏在 query 参数里。让 backend 显式回传 node_id，manager 不解析 URI。

### 3.3 Proto 新增字段

| 消息 | 字段 | 用途 |
|---|---|---|
| `GetCacheLocationRequest` | `caller_node_ip = 9`, `caller_supernode_id = 10` | 调用方自报位置 |
| `StartWriteCacheRequest` | `caller_node_ip = 7`, `caller_supernode_id = 9`, `is_replication = 8` | 同上 + 复制写标志 |
| `GetCacheLocationResponse` | `repeated ReplicationHint hints = 3` | 服务端下发复制提示 |

所有新字段 additive，老客户端字段为空时退化为未启用 affinity。

### 3.4 storage_key 随机后缀

复制写绕过全局去重后，同一 `(instance, spec_name, block_key)` 可写多份。storage key 末尾追加 8 字符随机后缀避免覆盖。碰撞概率 ~10⁻⁹。

---

## 4. 三条路径接入

### 4.1 写路径

```
proto(caller_node_ip, caller_supernode_id)
  → MetaServiceImpl::StartWriteCache → RequestContext
    → CacheManager::GenWriteLocation → 构建 AffinityResolveContext
      → CreateInSingleBatch / CreateBySpec
        → CacheAffinityManager::ResolveWrite
          → strategy.ResolveWrite → WriteDecision.hints
            → backend.CreateWithHints(hints, strict)
        → LocationDescriptor.node_id → LocationSpec.node_id（持久化）
```

`is_replication=true` 时：跳过全局去重（`ExistsForWrite`），仅检查 caller 节点是否已有副本（`existsOnCallerNode`），`strict=true` 传给 backend。

### 4.2 读路径

```
proto(caller_node_ip, caller_supernode_id)
  → MetaServiceImpl::GetCacheLocation → RequestContext
    → CacheManager::GetCacheLocationByQueryType
      → 构建 AffinityResolveContext（含 caller_node_ip + caller_supernode_id）
      → MetaSearcher::SelectAndMergeForMatch
        → strategy.ResolveRead → ReadDecision {picked_specs, side_effects}
          - PickLocalSpec: spec.node_id == caller_node_ip → 本地
          - ShouldEmitReplicationHint: 远端命中频率超阈值 → ReplicationHint
      → side_effects 透传到 response.hints
```

亲和性在 merge 步生效，`SelectForMatch` 保留 tier 选择语义不看 caller，避免副本数量稀释 type 权重。

`FrequencySketch` 管理 per-(caller, key) 远端命中计数，sketch 由 `CacheAffinityManager` 持有，策略重解析不丢失状态。不持久化，重启有分钟级 warm 期。

### 4.3 淘汰路径

```
CacheReclaimer::TryReclaimOnGroup
  → CacheAffinityManager::ResolveEviction
    → strategy: 遍历 all_nodes，比较 load_ratio vs 阈值
    → exceeded_node_ids
  → IsTriggerReclaiming（general ∨ per-type ∨ per-node）
  → ReclaimByLRU / LFU / TTL → FilterLocID
    → 3 条匹配路径:
       1. general 水位超限 → 任意 loc
       2. per-type 超限 → 匹配 storage type
       3. per-node 超限 → 匹配 spec.node_id ∈ exceeded_node_ids（新增）
    → ReportEvictedBytes（hysteresis 累计）
```

改造前 `FilterLocID` 只按整 key 淘汰。现在 path 3 配合 `LocationSpec.node_id` 实现节点级定向淘汰。

---

# 第二部分：具体修改

---

## 5. 文件变更索引

| 模块 | 文件 | 变更 |
|---|---|---|
| **策略框架** | `affinity/affinity_strategy.h` | 新增：抽象接口 + ReadSideEffect + 决策结构体 |
| | `affinity/noop_strategy.h` | 新增：空实现兜底 |
| | `affinity/local_replica_strategy.{h,cc}` | 新增：v1 默认策略，含 ReplicationHint |
| | `affinity/strategy_factory.{h,cc}` | 新增：JSON → 策略实例 |
| | `affinity/frequency_sketch.{h,cc}` | 新增：per-(caller, key) 频率计数 |
| | `affinity/cache_affinity_manager.{h,cc}` | 重构：3 入口 + metrics pull loop + kill-switch |
| | `affinity/pipeline/` | 重命名：`strategy.*` → `candidate_pipeline.*` 等 |
| **数据模型** | `meta/cache_location.h` | `LocationSpec` 加 `node_id` |
| | `data_storage/data_storage_backend.h` | `LocationDescriptor` + `SnapshotPerNodeMetrics` |
| | `data_storage/data_storage_manager.{h,cc}` | `Create` 返回 `LocationDescriptor` |
| **写路径** | `manager/cache_manager.{h,cc}` | `GenWriteLocation` 构建 resolve_ctx；`CreateInSingleBatch/CreateBySpec` 调 `ResolveWrite`；`FilterWriteCache` 加 `existsOnCallerNode` |
| **读路径** | `manager/meta_searcher.{h,cc}` | `SelectAndMergeForMatch` 接入 `ResolveRead` + side_effects 透传 |
| | `manager/cache_manager.{h,cc}` | `GetCacheLocationByQueryType` 构建 resolve_ctx（修复 caller_node_ip 缺失）；`GetCacheLocation` 加 `out_hints` |
| **淘汰路径** | `manager/cache_reclaimer.{h,cc}` | `FilterLocID` 加 per-node 匹配；`IsTriggerReclaiming` 加 `exceeded_node_ids` |
| **Service** | `service/meta_service_impl.cc` | 注入 caller 信号 + is_replication；hints 回填 response |
| | `service/server.{h,cc}` | 创建 `CacheAffinityManager`，加载策略文件 |
| | `service/server_config.{h,cc}` | `kvcm.affinity.enabled` / `strategy_file` |
| **Proto** | `protocol/protobuf/meta_service.proto` | `LocationSpec.node_id`、`ReplicationHint`、caller 信号、`is_replication`、`response.hints` |
| **配置** | `config/registry_manager.{h,cc}` | `GetGroupAffinityStrategyJson` |
| **测试** | `manager/test/` | CacheManagerTest（3 个 affinity 场景）、MetaSearcherTest |
| | `affinity/test/` | 策略单元测试（7 target） |
| | `common/test/` | RequestContext getter/setter |

---

## 6. 侵入边界

策略框架对主路径的侵入集中在 3 个调用点：

| 调用点 | 文件 | 函数 |
|---|---|---|
| write | `cache_manager.cc` | `CreateInSingleBatch` / `CreateBySpec` |
| read | `meta_searcher.cc` | `SelectAndMergeForMatch` |
| eviction | `cache_reclaimer.cc` | `FilterLocID` |

每个调用点的模式一致：取策略 → 调对应 Resolve → 用返回值。策略内部 toggle 关闭时返回 no-op 决策，调用方无感知。

不侵入的代码：`SelectForMatch`（tier 选择）、`BatchAddLocation`（元数据写入）、recovery 路径、client SDK 主路径。

---

# 第三部分：待做与扩展

---

## 7. 待完成项

### 7.1 算法层（最关键）

| 决策点 | 当前（临时） | 需要的设计 |
|---|---|---|
| 写 admission | 有 caller_node_ip 就 prefer local | prefix-aware：父 block 热度 + 本地水位联合判定 |
| 读 on_miss | 纯频率阈值 | 结合本地水位、全局副本分布、key 大小的综合判定 |
| 淘汰 | 纯水位比较 | 跨节点公平性、副本冗余度保护、冷热分层 |
| **超节点亲和性** | **仅透传，策略不消费** | **同节点 > 同超节点 > 远端，优先级最高** |

算法设计是长期迭代过程。建议以独立 PR 逐点推进，每个 PR 只改策略子类，不动框架。

### 7.2 外部依赖

Backend 接口已定义，当前全部走默认空实现。`CreateWithHints` 和 `SnapshotPerNodeMetrics` 需要 mempool 侧实现。**在此之前 node_id 始终为空，整条链路虽通但不生效。**

### 7.3 工程完善

| 项 | 说明 |
|---|---|
| 端到端集成测试 | 当前用 file backend 验证管道正确性。缺基于 mempool 的端到端覆盖 |
| 内部 metrics | 决策路径缺少 prometheus gauge/counter，上线后排查困难 |

---

## 8. 扩展指南

### 新增策略算法

1. 创建 `AffinityStrategy` 子类，实现 `ResolveWrite` / `ResolveRead` / `ResolveEviction`
2. 在 `StrategyFactory` 中注册 `type` → 构造函数映射
3. 配置 JSON `{"type": "your_strategy", ...}`

不需要改 `CacheAffinityManager`、`CacheManager`、`MetaSearcher` 或任何管道代码。

### 新增信号（如 `caller_supernode_id` 的铺设方式）

7 处机械性改动：Proto 字段 → `RequestContext` getter/setter → `AffinityResolveContext` 字段 → `StrategyContext` 字段 → `BuildStrategyContext` 拷贝 → Service 层注入 → `GetCacheLocationByQueryType` / `GenWriteLocation` 设到 resolve_ctx。
