# Cache Affinity / 缓存亲和性管理

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
StartWriteCacheRequest{caller_node_ip, ...}                          (proto)
    │
    ▼
MetaServiceImpl::StartWriteCache
    │  request_context->set_caller_node_ip(...)                       (透传)
    ▼
CacheManager::StartWriteCache  →  CreateBySpec / CreateInSingleBatch
    │  ResolveAffinityHints(request_context, instance_info, block_count, bytes_per_block)
    │      ├── instance_info.affinity_strategy_json    → 注入 ResolveContext (instance 层)
    │      ├── registry_manager_->GetInstanceGroup(...).affinity_strategy_json
    │      │                                          → 注入 ResolveContext (instance_group 层)
    │      ├── affinity_manager_ == nullptr            → 空 hints (老路径)
    │      ├── 三层都未配置                             → 空 hints
    │      ├── 策略返回 Abort                          → 日志 + 空 hints (v1)
    │      └── 策略返回节点列表                         → hints.preferred_node_ids
    ▼
DataStorageManager::Create(... , hints, strict, cb)                  (manager API)
    │  strict=true  → 后端必须只在 hints.preferred_node_ids 上分配;分配不到的 key 直接报错
    │  strict=false → hints 仅作建议,找不到偏好节点时回退到任意节点
    ▼
DataStorageBackend::CreateWithHints(... , hints, strict, ...)        (backend API)
    └── 默认实现:忽略 hints 与 strict,转发到老 Create()
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
| 5. 上报节点指标 | 每个节点调一次 `UpsertNodeMetrics(...)`;v1 没有自动数据源,需要从你的心跳/registry 接进来 |
| 6. 客户端在请求里带上 `caller_node_ip` | `StartWriteCacheRequest` 新增的字段;老客户端不填,`prefer_local` 直接按"本机不在候选里"处理 |

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
| `prefer_local` | — | `on_miss: "passthrough" \| "abort"`(默认 `passthrough`) | 候选含本机(`node_id == caller_node_ip`)→ 只返回本机;不含 → 由 `on_miss` 决定:`passthrough` 把输入原样传给下一段,`abort` 整段策略 abort |
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
| `node_id` | `prefer_local`(与 `caller_node_ip` 比对);也是 `WriteHints.preferred_node_ids` 写出的值 |
| `node_name` | `filter` 里的 `node_name` 叶子,以及 `sample.node_pattern`。当作稳定的业务标签用,不要等同于 IP |
| `free_bytes` | `filter` / `sort` 中名为 `free_bytes` 的指标 |
| `load_ratio` | `filter` / `sort` 中名为 `load_ratio` 的指标 |
| `rx_mbps` | `filter` / `sort` 中名为 `rx_mbps` 的指标 |
| `tx_mbps` | `filter` / `sort` 中名为 `tx_mbps` 的指标 |
| `updated_at_us` | filter / sort 不读 —— 调用方负责在 `UpsertNodeMetrics` 之前丢掉过期条目 |

v1 的混部假设下,`node_id` 就是跑推理 worker 的同一台机器,所以
`caller_node_ip` 与 `node_id` 用同一种标识(IP / hostname)。

> 已注册指标只有上表中的 `free_bytes / load_ratio / rx_mbps / tx_mbps`
> 四件套。`filter.metric` / `sort.metric` 名不在这张表里,解析时直接
> 报错。新增指标需要同时改 `NodeMetrics` 字段和 `metric_registry.cc`
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
    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(
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

    virtual std::vector<std::pair<ErrorCode, DataStorageUri>> CreateWithHints(
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
| `WriteHints.preferred_node_ids` | **偏好哪些节点**(按优先级) | 亲和性层 (`CacheAffinityManager::Resolve`) 或上层手动构造 |
| `bool strict` | **能不能放弃这些偏好** | 调用方(v1 由 `CacheManager` 传 `false`,未来可由 strategy / 配置驱动) |

语义对照:

| `hints.preferred_node_ids` | `strict` | 后端行为 |
|---|---|---|
| 空 | 任意 | `strict` 被忽略;后端按自己的策略放置 |
| 非空 | `false` | 优先在 preferred 节点上分配;不可用时**回退到任意节点**,写不会失败 |
| 非空 | `true` | **只能**在 preferred 节点上分配;放不下的 key 在结果里以非 `EC_OK` 返回,调用方自行决定是否重试或降级 |

> 历史注解:`strict` 之前是 `WriteHints` 的一个字段,现已提到接口
> 顶层。两个参数语义独立 —— hints 描述"想去哪儿",strict 描述"能
> 不能不去",分开传可以避免后端只 override `CreateWithHints` 但忘记
> 看结构体里那个布尔。

### 默认实现 = 兼容降级

默认的 `CreateWithHints` 忽略 `hints` 与 `strict`,直接转发到老
`Create`。能把 key 路由到指定节点的后端可以 override
`CreateWithHints` 与 `SupportsAffinity()`。v1 阶段所有后端都走默认实
现 —— 亲和性的端到端链路都通了、可验证,但暂时还没有任何 backend 真
正消费 hints。

`strict=true` 在这种"全默认"的形态下等价于 `strict=false`(hints 都
不看,自然也不会"严格遵守 hints")。等到第一个 backend 真正实现
`CreateWithHints` 时,`strict=true` 才会变成可观察的行为差异。

## 退化与失败语义

| 条件 | 结果 |
|---|---|
| 三层都未加载策略 | `Resolve` 返回 `EC_OK` + 空 hints;后端用自己的放置逻辑 |
| 高优先级层 JSON 解析失败 | 视为"该层未配置",自动落到下一层;不会让请求失败 |
| caller IP 为空 | `prefer_local` 把"本机命中"判为 false,按 `on_miss` 走(默认 passthrough) |
| 候选 NodeMetrics 缺失 | `filter` 叶子默认 true(permissive);`sort` 中该指标贡献 0;`prefer_local` 仍按 node_id 比对 caller IP |
| `prefer_local{on_miss:"abort"}` 找不到本机 | strategy abort 向上传,`Resolve` 返回 `EC_ERROR`。v1 在 `CacheManager::ResolveAffinityHints` 里降级为日志 + 空 hints(写继续走老路径)。如果想升级成硬错误,去掉那段降级即可 |
| process 级 JSON 格式错(含未注册指标名、`and:[]` 等) | `LoadProcessStrategyFromJson*` 返回 `false`;已有 process 级策略(如果有的话)保持不变;instance / instance_group 级别不受影响 |
| `node_name.include / exclude` 里有非法正则 | 同上 —— process 级加载失败不会留下半截状态;override 级别则视为该层解析失败、落到下一层 |
