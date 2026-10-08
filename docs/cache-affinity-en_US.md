# Cache affinity configuration and usage

This guide describes the current `kvcm_affinity_merge` implementation. See the [implementation design](design-cache-affinity-v1.md) for call flows, defaults, backend support and known gaps. [中文](cache-affinity-zh_CN.md)

## 1. Enable the server strategy

Add these settings to the server configuration:

```properties
kvcm.affinity.enabled=true
kvcm.affinity.strategy_file=/path/to/affinity.json
```

The global switch defaults to false. When enabled without a file, `Server::Init` installs its built-in strategy. A specified file that fails to load only produces a warning; it does not install the built-in fallback. The strategy file is read at startup and requires a restart to reload; there is no automatic file watcher.

This is a complete example strategy. Its explicit values are not all defaults:

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

Key semantics:

- `type` is required. A `{"strategy":{...}}` wrapper is also accepted. Write stages belong in the `write.ops` object; eviction operators belong in the `eviction.ops` array.
- Write stages run in the fixed order `filter → prefer_local → sample → sort → limit`. All stages are optional. A negative sort weight prefers smaller metric values. `limit` bounds preferred candidates, not replica count.
- `prefer_local.on_miss` accepts `passthrough` or `abort`. Ordinary writes fall back to unrestricted backend placement even on a strategy Abort; these filters are not strict placement constraints for ordinary writes.
- Reads first select existing components within the selected storage. The example emits replication hints only after heat, size, benefit, capacity and suppression checks. Unknown node capacity is permissive.
- The example's capacity cutoff is `0.90 - 0.05 = 0.85`. A bare `{"type":"local_replica"}` uses class defaults of `0.85 - 0.05 = 0.80` and has no write pipeline.
- `max_replicas_per_key` counts Location entries. `max_instance_bytes` includes original replicas, all specs and WRITING reservations. Both limits apply to ordinary and replication StartWrite calls.
- **The retention helper exists, but ReclaimByNode currently does not pass `min_retained_replicas`; node-pressure eviction does not guarantee retaining one replica.** The parsed `critical` threshold also has no separate execution branch.
- Under Noop or the disabled global switch, `GetReplicaLimits` still defaults to one retained replica, which ReclaimByLRU can apply. Disabling affinity does not restore every previous reclamation behavior.

Strategies are selected as whole objects in instance, instance_group, process order, without field merging. Instance and group settings use `affinity_strategy_json`. A partial but accepted override replaces the lower-level strategy. Some malformed fields fall back to defaults, so successful loading does not prove every field was applied.

Node-pressure eviction also requires `reclaim_indexer_type=node_lru` in the metadata backend URI. Enabling affinity does not create this index automatically; without it, node sampling returns `EC_NOENT` and no node-pressure deletion is submitted.

## 2. Enable client replication execution

Add these fields to an otherwise complete ManagerClient configuration; this fragment cannot initialize a client by itself:

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

Automatic execution requires a HYBRID ManagerClient with both MetaClient and TransferClient, using the MatchLocation path that returns hints. `auto_replicate` defaults to false; enabling the server alone does not move data.

The SDK queries the current PACE local identity on every request by default. A positive refresh interval permits cached identities. The provider factory currently handles the mempool memory type and NFS, but not automatic selection for SSD-only configurations.

Tasks without reusable buffers first call server ReplicateCache and fall back to client Load/Save only for an unsupported operation. Explicit buffer reuse accepts named specs with shared owners. An asynchronous boolean return indicates queue admission; the result callback reports the replication outcome. A release callback does not indicate success.

Task expiry does not forcibly cancel an executing RPC or transfer. Shared memory budgets and target pacing apply within one SDK process. Neither the server affinity switch nor `auto_replicate` prohibits explicit replication APIs.

## 3. Optional supernode topology

Set `KVCM_NODE_TOPOLOGY_FILE=/path/to/topology.json` in SDK and server processes, for example:

```json
{"nodes": {"101": "rack-a", "102": "rack-a", "201": "rack-b"}}
```

Keys must match actual `LocationSpec.node_id` and caller IDs: numeric node ID strings for mempool, IP addresses for NFS. Replace the file atomically. Topology is reread on demand at five-second intervals; stale mappings expire after 30 seconds of unsuccessful reads. This refresh behavior does not apply to the strategy JSON file.

## 4. Verify the resulting behavior

1. Complete an ordinary StartWrite/Finish flow and inspect actual spec node IDs and SERVING states with GetCacheMeta. Preferred-node input alone does not prove placement.
2. Read from a remote caller and check spec names, hint target and capabilities. The default heat threshold is three, but decay, suppression, capacity and cost gates can affect emission.
3. Check SDK result callbacks/statistics, then confirm complete SERVING components on the target with GetCacheMeta. A batch RPC's top-level OK does not imply every item succeeded.
4. Check both physical space and metadata when validating eviction. NFS reports synthetic capacity and has a placeholder Delete implementation, so NFS integration tests do not establish real PACE space reclamation.

Test entry points: [real Manager integration](../integration_test/affinity/affinity_replication_test.py), [SDK replication tests](../kv_cache_manager/client/test/replication_executor_test.cc), [strategy tests](../kv_cache_manager/affinity/test/). See the [implementation design](design-cache-affinity-v1.md) for remaining limitations and development priorities.
