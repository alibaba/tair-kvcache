# Cache Affinity / 缓存亲和性管理

> 本文说明写入流水线及当前副本生命周期控制。已包含读取亲和性、热点复制和节点淘汰，
> 使用 `caller.node_id`（mempool 为当前 PACE 数字 node id 的字符串形式），并已接入指标自动采样。
> 当前协议、默认开关和完成状态以 [完整设计说明](design-cache-affinity-v1.md) 为准。

KVCacheManager 提供一个可选的亲和性层，用来影响**写入时 block →
storage 节点的放置**。主要场景是**推理与存储混部**：同一台物理机
既跑推理 worker 又跑一个 storage 节点,把 KV cache 直接写到本机
storage 节点上就能省掉网络带宽。

决策由一段 **5 段固定流水线**驱动:每次写入按
`filter → prefer_local → sample → sort → limit` 的顺序求值
(顺序固定,配置不可改),每段都是可选的,不配置即跳过。
策略以 JSON 形式描述,在加载时一次性解析。

策略可以分别配置在三个层级,每次写入按 **instance > instance_group >
process** 的优先级选取最先命中的非空策略;任何一层未配置都自动落到
下一层。所有层都未配置时(默认状态),亲和性层是静默 no-op —— 所有
现有写路径保持原有行为。

## 写路径

```
StartWriteCacheRequest{caller.node_id, ...}                          (proto)
    │
    ▼
MetaServiceImpl::StartWriteCache
    │  request_context->set_caller_node_id(...)                       (透传)
    ▼
CacheManager::StartWriteCache  →  CreateBySpec / CreateInSingleBatch
    │  AffinityResolveContext{caller_node, instance_id, ...} → ResolveWrite(...)
    │      ├── instance_info.affinity_strategy_json    → 注入 ResolveContext (instance 层)
    │      ├── registry_manager_->GetInstanceGroup(...).affinity_strategy_json
    │      │                                          → 注入 ResolveContext (instance_group 层)
    │      ├── affinity_manager_ == nullptr            → 空 hints (老路径)
    │      ├── 三层都未配置                             → 空 hints
    │      ├── 策略返回 Abort                          → 普通写降级；严格复制写失败
    │      └── 策略返回节点列表                         → hints.preferred_node_ids
    ▼
DataStorageManager::Create(... , hints, strict, cb)                  (manager API)
    │  strict=true  → 后端必须只在 hints.preferred_node_ids 上分配;分配不到的 key 直接报错
    │  strict=false → hints 仅作建议,找不到偏好节点时回退到任意节点
    ▼
DataStorageBackend::CreateWithHints(... , hints, strict, ...)        (backend API)
    └── 默认实现:普通写转发老 Create();strict 写返回不支持
```

## 三层优先级链

策略按以下优先级选取**第一个解析成功的非空 JSON**,后续层级被忽略:

| 优先级 | 来源 | 持久化位置 | 配置入口 |
|---|---|---|---|
| 1 (最高) | instance 级 | `InstanceInfo.affinity_strategy_json`,随 `RegisterInstance` 落到 registry | `RegisterInstanceRequest.affinity_strategy_json`(admin / meta proto,field 8) |
| 2 | instance_group 级 | `InstanceGroup.affinity_strategy_json`,随实例组配置写入 registry | `InstanceGroup.affinity_strategy_json`(admin proto,field 9) |
| 3 (最低) | process 级 | 进程内存 | `LoadProcessStrategyFromJsonFile/String(...)` |

要点:

- **任一层为空串视为"该层未配置"**,自动 fall through 到下一层。
- **解析失败的 override** 等价于"该层未配置",落到下一层(不会让请求失败)。
- **持久化**:instance 与 instance_group 级 JSON 都通过 registry_manager 落盘,重启后由 `DoRecoverOnce` 在 `RegisterInstance` 时回放,无需重新下发。
- **解析缓存**:`CacheAffinityManager::ParseOrCacheLocked` 以原始 JSON 文本为 key 把已解析的 Strategy memoize 起来,相同 JSON 的多个 instance / instance_group 共享一份已解析的 Strategy。

## 启用方式

下面这些到位以后,对应层级才会真正参与决策;任何一层缺失都会自动退到下一层:

| 步骤 | 内容 |
|---|---|
| 1. 构造 `CacheAffinityManager` 并传给 `CacheManager` 构造函数 | 第 3 个可选参数;`nullptr` = 整个亲和性层关闭(所有层都失效) |
| 2. 加载 process 级策略 JSON(可选) | `LoadProcessStrategyFromJsonFile(path)` 或 `LoadProcessStrategyFromJsonString(json)`;不调用就只剩 instance / instance_group 级生效 |
| 3. 配置 instance_group 级策略 JSON(可选) | 在创建/更新 `InstanceGroup` 时填 `affinity_strategy_json` |
| 4. 配置 instance 级策略 JSON(可选) | 在 `RegisterInstanceRequest.affinity_strategy_json` 里下发 |
| 5. 上报节点指标 | 每个节点调一次 `UpsertNodeMetrics(...)`；服务端周期拉取后端容量快照，也可接入自定义数据源 |
| 6. 客户端在请求里带上 `caller.node_id` | `StartWriteCacheRequest` 新增的字段;老客户端不填,`prefer_local` 直接按"本机不在候选里"处理 |

## 执行顺序

策略的 5 段是**固定顺序、不可重排**:

```
filter  →  prefer_local  →  sample  →  sort  →  limit
```

每段都是 **可选** 的,缺省即跳过该步。每段拿到的"输入候选"都是上一段
的输出;任何一段决定 abort 时(目前只有 `prefer_local.on_miss=
"abort"` 会 abort),整个策略立即返回 abort,后续段不再执行。

固定顺序的设计动机:

| 段 | 在这一位的原因 |
|---|---|
| `filter` | 先把硬约束不满足的节点剔掉,后续步骤都建立在合法集合上 |
| `prefer_local` | 在数据集已被合法化之后再判定本机命中;本机若被 filter 干掉就视同未命中 |
| `sample` | 缩小候选规模供后续 sort 使用,避免在大集合上做无用排序 |
| `sort` | 在已经过滤+采样的小集合内排序;sort 不能放在 filter 之前——会浪费排序工作 |
| `limit` | 永远是最后一步:截前 N,与排序结果对齐 |

如果你需要"先排序再取前 K"的语义,把 `sort` 和 `limit` 都填上即可;
旧 schema 的 `top_k(k, child)` 直接平移成 `sort: [...] + limit: k`。

## 策略文件

顶层是一个对象,最多 5 个 slot 字段;每个字段都是可选的。顶层可以
裸写,也可以用 `{ "strategy": { ... } }` 包一层。

**示例 1:基本三段(filter + sort + limit)**

```json
{
  "strategy": {
    "filter": {
      "and": [
        { "metric": "free_bytes", "min": 1073741824 },
        { "metric": "load_ratio", "max": 0.8 }
      ]
    },
    "sort":  [ { "metric": "load_ratio", "weight": -1 } ],
    "limit": 3
  }
}
```

含义:剔除"剩余空间 < 1 GiB 或 load > 0.8"的节点;剩下的按 `load_ratio`
**升序**(weight=−1)排列;只取前 3 个。

**示例 2:加 `prefer_local`**

```json
{
  "strategy": {
    "filter":       { "metric": "free_bytes", "min": 1073741824 },
    "prefer_local": { "on_miss": "passthrough" },
    "sort":         [ { "metric": "load_ratio", "weight": -1 } ],
    "limit":        3
  }
}
```

含义:先按容量过滤;如果 caller 同机节点在剩下的候选里,**只返回本
机**;否则按 load 升序取前 3。`on_miss: "passthrough"` 表示"本机不
在候选里就把上一步的结果整段透传到下一段",等价于把 `prefer_local`
当成"本机能用就强偏好,否则不影响后续"。

**示例 3:加 `sample`**

```json
{
  "strategy": {
    "filter": { "metric": "load_ratio", "max": 0.8 },
    "sample": {
      "n": 5,
      "node_pattern": "^gpu-.*$",
      "seed": "trace_id"
    },
    "sort":  [ { "metric": "load_ratio", "weight": -1 } ],
    "limit": 2
  }
}
```

含义:先过滤掉 `load > 0.8` 的节点;从 `node_name` 匹配 `^gpu-.*$`
的子集里**按 trace_id 哈希采样 5 个**(同一个 trace 的多次重试每次
看到的采样集合一致);这 5 个再按 load 升序取前 2 个。

## 5 段语义

| Slot | 必填字段 | 可选字段 | 行为 |
|---|---|---|---|
| `filter` | 一棵 `Cond` 表达式(见下) | — | 剔除不满足条件的候选;候选**没有指标**时叶子默认评估为 `true`(permissive) |
| `prefer_local` | — | `on_miss: "passthrough" \| "abort"`(默认 `passthrough`) | 候选含本机(`node_id == caller.node_id`)→ 只返回本机;不含 → 由 `on_miss` 决定:`passthrough` 把输入原样传给下一段,`abort` 整段策略 abort |
| `sample` | `n: int (>= 1)` | `node_pattern: regex`、`seed: "random" \| "trace_id"`(默认 `random`) | 在(可选 `node_pattern` 命中的)子集里随机抽最多 `n` 个;`seed=trace_id` → 同一 trace 多次调用结果一致;输出顺序未定义 |
| `sort` | `[ { metric, weight }, ... ]` 非空数组 | — | score = Σ(metric_value × weight);按 score **降序稳定排列**。负权重 = 升序。指标缺失 → 该项贡献 0 |
| `limit` | `int (>= 1)` | — | 截到前 `n` 个 |

### `filter` 的 Cond 语法

`filter` 接受的是一棵递归表达式树,根和每个内部节点都是一个对象,按
唯一一个 dispatch key(`and / or / metric / node_name`)区分类型:

```text
Cond ::=
  | { "and":       [Cond, Cond, ...] }                                       // 复合
  | { "or":        [Cond, Cond, ...] }                                       // 复合
  | { "metric":    "<name>", "min"?: <num>, "max"?: <num> }                  // 叶子
  | { "node_name": { "include"?: [<regex>...], "exclude"?: [<regex>...] } }  // 叶子
```

边界规则(解析时直接报错,不会让请求带病通过):

- `and / or` 数组不能为空;单元素合法(等价于子项)。
- `metric` 至少要有 `min` / `max` 之一;`name` 必须是已注册指标。
- `node_name` 至少要有 `include` / `exclude` 之一。
- 候选缺指标 → 叶子返 `true`(AND/OR 一致语义,permissive;保证一个
  指标系统宕机不会瞬间把所有候选过滤光)。

### `sort` 用负权重表达升序

`sort` 总是按"线性组合分"**降序**排列。如果你要的是"低优先"(如低
load、低 latency),把 `weight` 设为负数即可:

```json
"sort": [
  { "metric": "load_ratio", "weight": -1 },
  { "metric": "rx_mbps",    "weight": -0.5 }
]
```

含义:在 score = `-load_ratio - 0.5 × rx_mbps` 上降序,等同于按
`load_ratio` 升序为主、`rx_mbps` 升序为辅。

## NodeMetrics

`NodeMetrics` 是 filter / sort / sample 唯一读取的数据结构。当前版本字段:

| 字段 | 谁使用 |
|---|---|
| `node_id` | `prefer_local`(与 `caller.node_id` 比对);也是 `WriteHints.preferred_node_ids` 写出的值 |
| `node_name` | `filter` 里的 `node_name` 叶子,以及 `sample.node_pattern`。当作稳定的业务标签用,不要等同于 IP |
| `free_bytes` | `filter` / `sort` 中名为 `free_bytes` 的指标 |
| `load_ratio` | `filter` / `sort` 中名为 `load_ratio` 的指标 |
| `rx_mbps` | `filter` / `sort` 中名为 `rx_mbps` 的指标 |
| `tx_mbps` | `filter` / `sort` 中名为 `tx_mbps` 的指标 |
| `updated_at_us` | 由亲和性管理器校验新旧采样并按 TTL 过滤；缓存快照不会续期 |

`caller.node_id` 与后端返回的 `node_id` 必须使用同一套标识：mempool 使用当前 PACE 数字 node id 的字符串形式，NFS 使用本机身份。PACE node id 变化表示新的数据状态代际，旧副本不得继续被判为 caller 本地副本。`total_bytes` 用于容量滞回，`supernode_id` 用于同超节点偏好。`rx_mbps` / `tx_mbps` 仍需外部观测来源提供。

客户端参数 `caller_node_refresh_seconds` 默认是 `0`，表示每次请求都重新查询后端身份，避免 PACE node id 换代后继续使用旧 caller。显式设置正数可以降低查询频率，但会引入最长为该配置值的旧身份窗口。

> 已注册指标只有上表中的 `free_bytes / load_ratio / rx_mbps / tx_mbps`
> 四件套。`filter.metric` / `sort.metric` 名不在这张表里,解析时直接
> 报错。新增指标需要同时改 `NodeMetrics` 字段和 `affinity/pipeline/metric_catalog.cc`
> 的 `Extract` 表。

## 与 `SelectLocationPolicy` 的关系

| | `SelectLocationPolicy`(已有) | `CacheAffinityManager`(本特性) |
|---|---|---|
| 决策 | 选一个**后端**(NFS / 3FS / Mooncake / TairMempool / …) | 在选定后端内部选**存储节点** |
| 输出 | 后端的 `unique_name` | 传给该后端的 `WriteHints.preferred_node_ids` |
| 生命周期 | 按 `InstanceGroup` 配置 | 三层(instance / instance_group / process),前两者随 registry 持久化,process 级可热加载 |
| 顺序 | 先跑 | 在后端选定后跑 |

两层不冲突 —— 亲和性层不会反向去重新选后端。

## DataStorageManager / Backend 接口

亲和性写路径上有两层接口需要透传 hints。它们的形状是对偶的:上层
(`CacheManager`)调 manager,manager 转发给 backend。

```cpp
// kv_cache_manager/data_storage/data_storage_manager.h
class DataStorageManager {
public:
    // 老接口:不带 hints;内部以 strict=false 转发。
    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(
        RequestContext *request_context, const std::string &unique_name,
        const std::vector<std::string> &keys, size_t size_per_key,
        std::function<void()> cb);

    // 亲和性接口:hints + strict 是一对独立参数。
    std::vector<LocationDescriptor> Create(
        RequestContext *request_context, const std::string &unique_name,
        const std::vector<std::string> &keys, size_t size_per_key,
        const WriteHints &hints,
        bool strict,
        std::function<void()> cb);
};

// kv_cache_manager/data_storage/data_storage_backend.h
class DataStorageBackend {
public:
    virtual std::vector<std::pair<ErrorCode, DataStorageUri>> Create(
        const std::vector<std::string> &keys, size_t size_per_key,
        const std::string &trace_id, std::function<void()> cb) = 0;       // 老接口

    virtual std::vector<LocationDescriptor> CreateWithHints(
        const std::vector<std::string> &keys, size_t size_per_key,
        const WriteHints &hints,
        bool strict,
        const std::string &trace_id, std::function<void()> cb);            // 新接口

    virtual bool SupportsAffinity() const { return false; }
};
```

### `hints` 与 `strict` 的职责划分

两个参数刻意分开:

| 参数 | 含义 | 谁来填 |
|---|---|---|
| `WriteHints.preferred_node_ids` | **偏好哪些节点**(按优先级) | 亲和性层 (`CacheAffinityManager::ResolveWrite`) 或上层手动构造 |
| `bool strict` | **能不能放弃这些偏好** | `CacheManager` 对复制写传 `true`，普通写传 `false` |

语义对照:

| `hints.preferred_node_ids` | `strict` | 后端行为 |
|---|---|---|
| 空 | `false` | 后端按自己的策略放置 |
| 空 | `true` | 拒绝分配，不能回退到任意节点 |
| 非空 | `false` | 优先在 preferred 节点上分配;不可用时**允许回退到其他节点**，仍可能因容量或后端错误失败 |
| 非空 | `true` | **只能**在 preferred 节点上分配;放不下的 key 在结果里以非 `EC_OK` 返回,调用方自行决定是否重试或降级 |

> 历史注解:`strict` 之前是 `WriteHints` 的一个字段,现已提到接口
> 顶层。两个参数语义独立 —— hints 描述"想去哪儿",strict 描述"能
> 不能不去",分开传可以避免后端只 override `CreateWithHints` 但忘记
> 看结构体里那个布尔。

### 后端兼容与严格放置

默认 `CreateWithHints` 在普通写时转发到旧 `Create`，严格写返回 `EC_UNIMPLEMENTED`。
`DataStorageManager` 在分配前拒绝空的严格目标或不支持亲和性的后端。
NFS 与内源 mempool 已实现亲和性放置，并返回包含实际 `node_id` 的 `LocationDescriptor`。
复制写只允许目标为 caller 本机；即使配置了同超节点回退，也不能把远端副本发布成本地副本。

## 退化与失败语义

| 条件 | 结果 |
|---|---|
| 三层都未加载策略 | `ResolveWrite` 返回 kOk + 空 hints;后端用自己的放置逻辑 |
| 高优先级层 JSON 解析失败 | 视为"该层未配置",自动落到下一层;不会让请求失败 |
| caller.node_id 为空 | `prefer_local` 把"本机命中"判为 false,按 `on_miss` 走(默认 passthrough) |
| 候选 NodeMetrics 缺失 | `filter` 叶子默认 true(permissive);`sort` 中该指标贡献 0;`prefer_local` 仍按 node_id 比对 caller.node_id |
| `prefer_local{on_miss:"abort"}` 找不到本机 | 策略返回 abort；普通写记录日志并降级为空 hints；复制写因严格目标为空而失败 |
| process 级 JSON 格式错(含未注册指标名、`and:[]` 等) | `LoadProcessStrategyFromJson*` 返回 `false`;已有 process 级策略(如果有的话)保持不变;instance / instance_group 级别不受影响 |
| `node_name.include / exclude` 里有非法正则 | 同上 —— process 级加载失败不会留下半截状态;override 级别则视为该层解析失败、落到下一层 |

### 删除完成反馈

节点压力回收只在异步删除返回终态后计入释放容量。每个实际删除成功的 URI 按 spec 所属节点累计字节；重复 URI、已不存在、失败、超时和仅删除元数据均不产生释放量。部分成功保留成功部分的反馈，迟到的成功也可反馈。容量采样时间必须早于物理删除开始时间，否则以新采样为准，防止重复扣减。无时间戳的采样不接受异步删除估算，等待后端容量刷新。

### 副本预算与最低保留数

`local_replica.replica_limits` 可配置 `max_replicas_per_key`、`max_instance_bytes`（两者为 0 时不限）和 `min_retained_replicas`（默认 1）。例如 `"replica_limits":{"max_replicas_per_key":3,"max_instance_bytes":10737418240,"min_retained_replicas":1}`。

写入预算在元数据 RMW 时执行：每 key 上限保守统计所有 Location，包括 WRITING/DELETING 和拆分 spec 的 Location；实例容量包含原始副本及正在写入的副本，复用持久化容量统计。这样无需靠可能丢失的 hint 预留计数恢复预算。超限回滚新分配空间并返回容量错误；释放的元数据容量可再次使用。该预算约束 StartWriteCache，外部 ReportEvent 与独立迁移写入仍按各自配额管理。

最低保留数仅用于节点压力淘汰，按每个 spec 的 SERVING 副本重新校验；WRITING、DELETING 和已消失的副本不计入。并发删除在元数据分片锁内扣减候选副本，保留数大于 1 同样有效。普通 TTL/显式删除不受此限制，仍可清空过期数据。

### SDK 复制资源控制

客户端配置增加 `replication_max_buffer_bytes`（进程内已保留用户缓冲区与复制缓冲区的总预算，默认 256 MiB）、`replication_max_pending_bytes`（每执行器队列预计字节数，默认 256 MiB）、`replication_node_bytes_per_second`（同一进程向每个目标节点复制的字节速率，0 不限）与 `replication_max_age_ms`（入队后最长存活时间，默认 30 秒）。同一进程中的客户端应配置一致的进程预算。

源 spec 总大小在分配缓冲区前检查；用户缓冲区在入队时预留预算，完成或丢弃时释放。多个实例共享轮转准入和目标节点限速，单实例保持 FIFO，复用缓冲区任务也排队。限速允许一次块传输的突发，之后按该块大小占用传输时间；它限制后台复制，不影响前台读写。进程间的全局带宽由部署侧配额控制。

### 复制结果观测

`ManagerClient::GetReplicationStats()` 返回当前客户端的累计提交/准入、成功、失败、跳过、过期、队列/预算丢弃、重复提示计数，以及成功复制字节、执行耗时、排队耗时、活动数和队列字节。只有 `FinishWrite` 成功确认后才增加成功次数和字节；已在本地跳过与失败分开统计。`InitParams.replication_metrics_callback` 可接入业务指标系统，完成任务后在执行器锁外调用；导出异常不会终止工作线程。仅发生准入丢弃时可周期读取快照。以上为 SDK 侧实际结果，服务端 hint/StartWrite 计数不能替代它们。

### 分批升级

新 SDK 在 CallerNode 中声明 `replication_capabilities=1`，服务端确认后才传送多 spec 提示。旧客户端（字段缺省/0）保留正常读选路与单 spec 复制。新 SDK 遇到未确认能力的旧服务端会忽略多 spec 提示，不影响正常读取；不支持的提示不会占用服务端的复制抑制窗口。

### 热度衰减和收益准入

`read.on_miss.heat_half_life_ms` 默认 60000，远端命中热度每个半衰期减半，长期未访问的 key 不再保持高热度。设为 0 可保留累计计数；更改半衰期会重新累积证据。Instance/调用节点隔离不变。

可选 `max_replication_bytes` 限制整个 block 的复制字节，`min_benefit_ratio` 限制“衰减热度 × 每次可避免的远端字节 / 复制总字节”。二者默认 0（关闭）；启用后无法解析 size 的 spec 保守拒绝复制。已本地命中的 spec 仍产生复制成本，但不算远端收益。`prefix_bonus` 默认 0，前缀查询中的收益乘以 `1 + prefix_bonus / (position + 1)`；批量和滑窗查询不加权。这是可调的收益估计，并非在线预测模型。

### 超节点拓扑和选路

部署时为 KVCM 和 SDK 设置 `KVCM_NODE_TOPOLOGY_FILE`，文件内容例如 `{"nodes":{"41":"rack1","42":"rack1"}}`，节点标识必须与存储返回的 node_id 一致。mempool 的 PACE node id 在重注册后可能变化，拓扑控制面需要同步替换成新 ID。以原子替换方式更新文件；进程每 5 秒刷新。读取失败或 JSON 非法时短期保留上一次映射，30 秒后降级为未知拓扑。空 nodes 可主动清空。SDK 据此填充 CallerNode.supernode_id，服务端也会根据映射补全调用方和 NodeMetrics；后端直接上报的拓扑仍可作为来源。

读优先本机，其次同 supernode，最后沿用远端候选顺序。写流水线使用 `"prefer_local":{"same_supernode":true,"on_miss":"abort"}` 可在没有本机候选时尝试同 supernode；默认 false 保留原先行为。严格复制写仍必须落到调用节点，不会因同 supernode 回退而发布错误的本地副本。

### 按 spec 名称复用缓冲区

`ManagerClient::ReplicateWithBuffers(hint, buffers)` 接受 `ClientReplicationBuffer{spec_name, data, size, memory_type, owner}`，返回是否入队。每个缓冲区必须有非空共享 owner；调用方保持内容不可变，直到 SDK 释放 owner。按 spec 名称匹配源和目标，输入顺序无关，支持 CPU/GPU（具体传输后端须支持该类型）。重复名称、未知名称、无 owner、空地址或源 size 不匹配在分配目标前拒绝。已有旧单缓冲区接口保持可用。

允许只提供部分 spec：已有缓冲区直接写目标，缺失项从提示中的 URI 读取。共享 owner 保活到所有传输与发布结束；失败、超时、队列拒绝同样释放。任何 spec 失败都撤销整个写会话，只有所有 spec 写成功才整体发布。缓冲区按实际字节计入复制预算；无效输入由 `dropped_invalid` 指标统计。
