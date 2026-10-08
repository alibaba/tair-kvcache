# Cache Affinity

> This page describes the write-placement pipeline and replica lifecycle controls. The implementation also
> includes local reads, replication and node eviction, uses the current numeric PACE node ID as a string in `caller.node_id`,
> and pulls backend metrics automatically. See [the current design](design-cache-affinity-v1.md).

KVCacheManager has an optional affinity layer that influences **block →
storage-node placement** during writes. Its primary use case is the
**inference + storage co-location** topology: the same physical machine
runs both the inference worker and a storage node, so writing the KV
cache to the local storage node spends zero network bandwidth.

The decision is driven by a **fixed 5-stage pipeline**, evaluated on
every write:

```
filter  →  prefer_local  →  sample  →  sort  →  limit
```

The order is hard-coded — configuration cannot reorder the stages —
and every stage is optional (omitted = skipped). The strategy is
described as JSON and parsed once at load time.

A strategy can be configured at three different tiers; on every write
the affinity layer picks the **first non-empty strategy** in priority
order **instance > instance_group > process**, and any tier that is
unset transparently falls through to the next. When no tier has a
strategy configured (the default) the affinity layer is a silent no-op
— every existing write path keeps its old behavior.

## Write Path

```
StartWriteCacheRequest{caller.node_id, ...}                          (proto)
    │
    ▼
MetaServiceImpl::StartWriteCache
    │  request_context->set_caller_node_id(...)                       (transport)
    ▼
CacheManager::StartWriteCache  →  CreateBySpec / CreateInSingleBatch
    │  AffinityResolveContext{caller_node, instance_id, ...} → ResolveWrite(...)
    │      ├── instance_info.affinity_strategy_json    → injected into ResolveContext (instance tier)
    │      ├── registry_manager_->GetInstanceGroup(...).affinity_strategy_json
    │      │                                          → injected into ResolveContext (instance_group tier)
    │      ├── affinity_manager_ == nullptr            → empty hints (legacy path)
    │      ├── all three tiers empty                   → empty hints
    │      ├── strategy returns Abort                  → ordinary write fallback; strict replication fails
    │      └── strategy returns nodes                  → hints.preferred_node_ids
    ▼
DataStorageManager::Create(... , hints, strict, cb)                  (manager API)
    │  strict=true  → backend MUST allocate only on hints.preferred_node_ids;
    │                 keys it cannot place there are returned with an error
    │  strict=false → hints are advisory; backend may fall back to any node
    ▼
DataStorageBackend::CreateWithHints(... , hints, strict, ...)        (backend API)
    └── default impl: forward ordinary writes; reject unsupported strict placement
```

## Three-Tier Priority Chain

On every write, the affinity layer picks the **first JSON that parses
successfully** in priority order; later tiers are ignored:

| Priority | Tier | Persisted on | How to set |
|---|---|---|---|
| 1 (highest) | instance | `InstanceInfo.affinity_strategy_json`, written through registry on `RegisterInstance` | `RegisterInstanceRequest.affinity_strategy_json` (admin / meta proto, field 8) |
| 2 | instance_group | `InstanceGroup.affinity_strategy_json`, written through registry alongside the group config | `InstanceGroup.affinity_strategy_json` (admin proto, field 9) |
| 3 (lowest) | process | in-process memory | `LoadProcessStrategyFromJsonFile/String(...)` |

Notes:

- **An empty string at any tier means "this tier is unset"**, and the
  layer falls through to the next.
- **A parse failure on an override** is treated the same as "tier
  unset" — it falls through to the next tier rather than aborting the
  write.
- **Persistence**: the instance and instance_group JSON are persisted
  via `registry_manager`, so they survive restarts and are replayed by
  `DoRecoverOnce` re-issuing `RegisterInstance` — no client-side
  re-push required.
- **Parse caching**: `CacheAffinityManager::ParseOrCacheLocked`
  memoizes the parsed `Strategy` by the raw JSON text, so multiple
  instances / groups that share the same JSON share a single parsed
  `Strategy`.

## Enabling

For a tier to participate in the decision, the corresponding piece
below must be in place; any missing tier simply falls through:

| Step | What |
|---|---|
| 1. Build a `CacheAffinityManager` and pass it to `CacheManager` ctor | Third (optional) ctor argument; `nullptr` = the affinity layer is disabled altogether (no tier participates) |
| 2. Load a process-level strategy JSON (optional) | `LoadProcessStrategyFromJsonFile(path)` or `LoadProcessStrategyFromJsonString(json)`; if you skip this, only the instance / instance_group tiers can take effect |
| 3. Configure the instance_group strategy JSON (optional) | Set `affinity_strategy_json` when creating / updating an `InstanceGroup` |
| 4. Configure the instance strategy JSON (optional) | Send via `RegisterInstanceRequest.affinity_strategy_json` |
| 5. Populate node metrics | The server periodically pulls backend capacity snapshots; `UpsertNodeMetrics(...)` also accepts custom sources |
| 6. Have the client send `caller.node_id` | New field on `StartWriteCacheRequest`; old clients leave it empty and `prefer_local` simply treats "local node not in candidates" as a miss |

## Pipeline Order

The five stages run in a **fixed, non-reorderable** sequence:

```
filter  →  prefer_local  →  sample  →  sort  →  limit
```

Every stage is **optional** — an absent slot is skipped. The "input
candidates" of each stage are the previous stage's output. If any stage
decides to abort (today only `prefer_local.on_miss = "abort"` does
this) the whole strategy returns abort and the rest of the pipeline is
not executed.

Why this particular order:

| Stage | Reason |
|---|---|
| `filter` | Drop nodes that fail hard constraints first, so every later stage operates on a legal subset |
| `prefer_local` | Decide local-host preference *after* the candidate set has been legalized; if the local node was filtered out, that's effectively the same as "no local node" |
| `sample` | Trim the candidate set before sort, so we don't sort more nodes than we need |
| `sort` | Rank a small, already-filtered set; sort before filter would waste sort work |
| `limit` | Always last — truncate to the top-N according to whatever order sort produced |

If you want "sort then take top-K" semantics, set both `sort` and
`limit`; the old schema's `top_k(k, child)` translates one-to-one to
`sort: [...] + limit: k`.

## Strategy File

The top level is one object with up to five optional slot fields. The
top level can be bare or wrapped in `{ "strategy": { ... } }`.

**Example 1: basic three-stage (filter + sort + limit)**

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

Reads as: drop nodes with "free space < 1 GiB or load > 0.8"; rank the
remainder by `load_ratio` **ascending** (`weight=-1`); keep the first
three.

**Example 2: add `prefer_local`**

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

Reads as: filter by free space first; if the caller's local node is in
the survivors, **return only the local node**; otherwise pass the
survivors through and rank them by ascending load. `on_miss:
"passthrough"` means "if the local node isn't in the candidates, hand
the previous stage's output to the next stage unchanged" — the
intuition is "prefer local strongly, but only when it's available."

**Example 3: add `sample`**

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

Reads as: drop nodes with `load > 0.8`; from the subset whose
`node_name` matches `^gpu-.*$`, **sample 5 by trace_id hash** (so
multiple retries of the same trace see the same sample); rank those 5
by ascending load and keep the first 2.

## Slot Reference

| Slot | Required | Optional | Behavior |
|---|---|---|---|
| `filter` | a `Cond` expression (see below) | — | Drop candidates that fail the condition. A leaf evaluates to `true` (permissive) when the candidate has no metrics — see "missing-metric semantics" |
| `prefer_local` | — | `on_miss: "passthrough" \| "abort"` (default `passthrough`) | If a candidate has `node_id == caller.node_id`, **return only that one**; otherwise behave per `on_miss`: `passthrough` hands input through unchanged, `abort` aborts the whole strategy |
| `sample` | `n: int (>= 1)` | `node_pattern: regex`, `seed: "random" \| "trace_id"` (default `random`) | Sample at most `n` from the (optionally `node_pattern`-matched) subset; `seed=trace_id` makes repeated calls with the same trace deterministic; output ordering is not specified |
| `sort` | `[ { metric, weight }, ... ]` non-empty | — | score = Σ(metric_value × weight); **stable-sort descending** by score. Negative weight = ascending. Missing metric contributes 0 to that term |
| `limit` | `int (>= 1)` | — | Truncate to the first `n` |

### `filter`: the `Cond` grammar

`filter` accepts a recursive expression tree. Every node is an object
that dispatches on exactly one of `and / or / metric / node_name`:

```text
Cond ::=
  | { "and":       [Cond, Cond, ...] }                                       // composite
  | { "or":        [Cond, Cond, ...] }                                       // composite
  | { "metric":    "<name>", "min"?: <num>, "max"?: <num> }                  // leaf
  | { "node_name": { "include"?: [<regex>...], "exclude"?: [<regex>...] } }  // leaf
```

Edge rules (rejected at parse time, so a bad config can't slip
through):

- `and / or` arrays must be non-empty; a single-element array is
  legal (equivalent to its only child).
- A `metric` leaf must specify at least one of `min` / `max`; the
  metric name must be registered.
- A `node_name` leaf must specify at least one of `include` /
  `exclude`.
- Missing-metric semantics: a candidate with no metrics evaluates to
  `true` at every leaf (consistent under AND/OR; permissive — keeps
  one degraded metric channel from quietly filtering out the whole
  candidate set).

### `sort`: negative weight for ascending

`sort` always sorts the linear-combination score **descending**. To get
"prefer low" (low load, low latency, etc.), use a negative weight:

```json
"sort": [
  { "metric": "load_ratio", "weight": -1 },
  { "metric": "rx_mbps",    "weight": -0.5 }
]
```

Reads as: descending sort on score = `-load_ratio - 0.5 × rx_mbps`,
which is equivalent to "ascending by `load_ratio` first, then
ascending by `rx_mbps` as a tiebreaker."

## Node Metrics

`NodeMetrics` is the only thing filter / sort / sample read from. The
current set of fields:

| Field | Used by |
|---|---|
| `node_id` | `prefer_local` (matched against `caller.node_id`); also the value emitted in `WriteHints.preferred_node_ids` |
| `node_name` | `filter`'s `node_name` leaf, and `sample.node_pattern`. Treat it as a stable business label, not an IP |
| `free_bytes` | `filter` / `sort` term named `free_bytes` |
| `load_ratio` | `filter` / `sort` term named `load_ratio` |
| `rx_mbps` | `filter` / `sort` term named `rx_mbps` |
| `tx_mbps` | `filter` / `sort` term named `tx_mbps` |
| `updated_at_us` | The affinity manager validates sample order and TTL; cached snapshots do not renew freshness |

`caller.node_id` must use the same identity as the backend result: the current numeric PACE node ID as a string for mempool, and the local host identity for NFS. A PACE node ID change represents a new data-state incarnation, so replicas tagged with the old ID must no longer count as caller-local. `total_bytes` supports capacity hysteresis; `supernode_id` supports nearby placement. `rx_mbps` / `tx_mbps` still require an external telemetry source.

The client option `caller_node_refresh_seconds` defaults to `0`, which resolves the backend identity on every request so a PACE node incarnation change is observed immediately. A positive value reduces lookup frequency but permits an old caller identity to remain cached for up to that interval.

> The only registered metrics are the four above (`free_bytes /
> load_ratio / rx_mbps / tx_mbps`). A `filter.metric` or `sort.metric`
> name not in this list is a parse error. To add a new metric you must
> extend both `NodeMetrics` and the `Extract` table in
> `affinity/pipeline/metric_catalog.cc`.

## How It Coexists With `SelectLocationPolicy`

| | `SelectLocationPolicy` (existing) | `CacheAffinityManager` (this) |
|---|---|---|
| Decision | Pick a **backend** (NFS / 3FS / Mooncake / TairMempool / …) | Within a chosen backend, pick **storage nodes** |
| Output | A backend's `unique_name` | `WriteHints.preferred_node_ids` passed to that backend |
| Lifetime | Per-`InstanceGroup` config | Three tiers (instance / instance_group / process). The first two are persisted through registry; the process tier is hot-reloadable in memory |
| Order | Runs first | Runs after the backend is selected |

The two layers don't conflict — the affinity layer never reaches up to
re-pick the backend.

## DataStorageManager / Backend Interface

The affinity write path threads hints through two layers — the upper
`CacheManager` calls into the manager, and the manager forwards into
the backend. Their shapes mirror each other:

```cpp
// kv_cache_manager/data_storage/data_storage_manager.h
class DataStorageManager {
public:
    // Legacy overload: no hints. Internally forwards with strict=false.
    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(
        RequestContext *request_context, const std::string &unique_name,
        const std::vector<std::string> &keys, size_t size_per_key,
        std::function<void()> cb);

    // Affinity-aware overload: hints + strict are independent params.
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
        const std::string &trace_id, std::function<void()> cb) = 0;       // legacy

    virtual std::vector<LocationDescriptor> CreateWithHints(
        const std::vector<std::string> &keys, size_t size_per_key,
        const WriteHints &hints,
        bool strict,
        const std::string &trace_id, std::function<void()> cb);            // affinity-aware

    virtual bool SupportsAffinity() const { return false; }
};
```

### Roles of `hints` and `strict`

The two parameters are deliberately split:

| Param | Meaning | Who fills it |
|---|---|---|
| `WriteHints.preferred_node_ids` | **Which nodes to prefer** (priority order) | The affinity layer (`CacheAffinityManager::ResolveWrite`) or a hand-built struct from a higher caller |
| `bool strict` | **Whether the backend may abandon those preferences** | `CacheManager` passes `true` for replication and `false` for ordinary writes |

Combined semantics:

| `hints.preferred_node_ids` | `strict` | Backend behavior |
|---|---|---|
| empty | `false` | Backend uses its own placement |
| empty | `true` | Allocation is rejected; arbitrary-node fallback is forbidden |
| non-empty | `false` | Try preferred nodes first; **fall back** to any other node when none are usable. Write does not fail |
| non-empty | `true` | **Only** preferred nodes are usable. Keys that cannot be placed there come back with a non-`EC_OK` status, and the caller decides whether to retry or degrade |

> Historical note: `strict` used to live on `WriteHints` and is now
> lifted to a top-level parameter. Splitting them keeps "where to go"
> (the struct) orthogonal to "must I obey" (the flag) — and makes it
> harder for a backend to override `CreateWithHints` and forget the
> boolean buried inside the struct.

### Backend compatibility and strict placement

The default `CreateWithHints` forwards ordinary writes to legacy `Create` and returns `EC_UNIMPLEMENTED` for strict writes. `DataStorageManager` rejects empty strict targets and backends without affinity support before allocation. NFS and the internal mempool backend implement affinity placement and return actual `node_id` values in `LocationDescriptor`. Replication targets must match the caller node; same-supernode fallback cannot publish a remote replica as local.

## Failure Modes

| Condition | What happens |
|---|---|
| No strategy loaded at any tier | `ResolveWrite` returns kOk + empty hints; backend uses its own placement |
| A higher-priority tier's JSON fails to parse | Treated as "tier unset" and falls through to the next tier; the write does not fail |
| Caller node ID empty | `prefer_local` treats "local node in candidates" as false and applies `on_miss` (default `passthrough`) |
| Candidate has no `NodeMetrics` | `filter` leaves default to `true` (permissive); the term contributes 0 in `sort`; `prefer_local` still matches `node_id` against `caller.node_id` |
| `prefer_local{on_miss:"abort"}` finds no local node | The strategy aborts. Ordinary writes log and fall back to empty hints; replication fails because strict targets are empty |
| Process-level JSON malformed (unregistered metric, `and:[]`, etc.) | `LoadProcessStrategyFromJson*` returns `false`; existing process-level strategy (if any) is unchanged; instance / instance_group tiers are not affected |
| Regex compile error in `node_name.include / exclude` | Same as above for process-level loads — no partial state. For an override (instance / instance_group) it is treated as a parse failure for that tier and falls through |

## Replica lifecycle controls

`local_replica.replica_limits` accepts `max_replicas_per_key`, `max_instance_bytes` (0 means unlimited), and `min_retained_replicas` (default 1). Admission checks metadata reservations atomically; the byte budget includes original copies and WRITING locations. The per-key count conservatively includes all locations. Node-pressure deletion retains the requested number of SERVING copies of each spec; explicit and TTL deletion are unaffected. Physical deletion credit counts only successful backend URI deletions and is fenced against newer capacity samples.

`read.on_miss.heat_half_life_ms` defaults to 60000. Optional `max_replication_bytes`, `min_benefit_ratio`, and `prefix_bonus` gate copying by size, decayed heat, avoided remote bytes and prefix position. Their default is 0 (disabled). Unknown sizes are rejected when cost admission is enabled.

Set `KVCM_NODE_TOPOLOGY_FILE` in manager and SDK processes to a JSON file such as `{"nodes":{"41":"rack1","42":"rack1"}}`. Node keys must match backend node IDs. Since a mempool Provider can receive a new PACE node ID after re-registration, the topology control plane must replace the old key with the new one. Replace the file atomically. It refreshes every 5 seconds; failed refreshes expire the old mapping after 30 seconds. Reads prefer local, then same-supernode, then existing remote order. Writes can enable `prefer_local.same_supernode`; strict replication still requires the caller node.

Client options `replication_max_buffer_bytes` and `replication_max_pending_bytes` default to 256 MiB; `replication_node_bytes_per_second=0` disables pacing and `replication_max_age_ms=30000` bounds queue lifetime. Executors share in-process memory admission, instance rotation and target-node pacing. Use consistent process budgets across clients; cross-process quotas remain a deployment concern.

The caller advertises `replication_capabilities=1` for complete named sources, and the response confirms the intersection. Legacy clients continue reading and receive only single-spec hints. `ReplicateWithBuffers` accepts named CPU/GPU buffers with shared ownership; missing specs are loaded, and all specs publish together. `GetReplicationStats()` and `InitParams.replication_metrics_callback` expose acknowledged successes, failures, skipped/expired work, queue/budget/invalid drops, bytes, latency and queue state.
