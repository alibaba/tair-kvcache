# P2P Read 设计

P2P read 用于在 `hierarchical_replay_main` 中模拟同一个推理集群内，不同推理实例之间的本地缓存复用。它位于 engine-local read 和 storage pool read 之间，不改变普通 `OptimizerManager` 的 instance 隔离语义。

核心路径：

```text
engine local read -> P2P read -> storage pool read
```

命中来源拆分为：

```text
Hit = LocalHit + PeerHit + RemoteHit
```

- `LocalHit`：当前推理实例本地命中。
- `PeerHit`：同一个 `infer_cluster` 内其他推理实例命中。
- `RemoteHit`：storage pool 命中。

## 适用范围

- 只在同一个 `infer_clusters[]` 内匹配，不跨推理集群。
- 只读其他推理实例，不读当前实例自己。
- 只读配置指定的同名 tier，例如 `tier=dram` 只查 peer 的 `dram`。
- P2P 不表达 `hbm -> dram` 这类跨层 peer read。
- P2P 只影响 read path；trace write 仍按 engine 和 storage pool 原有写入策略执行。

## 配置

P2P 配在 `infer_clusters[].p2p_read_flows`：

```json
{
  "infer_clusters": [
    {
      "infer_ids": ["infer_a", "infer_b", "infer_c"],
      "tiers": [
        {"name": "hbm", "capacity": 1200},
        {"name": "dram", "capacity": 1600}
      ],
      "p2p_read_flows": [
        {
          "tier": "dram",
          "peer_read_touch_enabled": true
        }
      ]
    }
  ]
}
```

字段：

| 字段 | 说明 |
|---|---|
| `tier` | peer source tier。必须存在于当前 `infer_cluster.tiers[]` 中 |
| `peer_read_touch_enabled` | P2P 命中后是否刷新 peer 对应 tier 中命中 block 的读访问时间 |

不保留旧兼容字段。配置缺少 `tier`、tier 不存在、同一个 tier 重复配置，都应初始化失败。

## Read 流程

配置 P2P 后，一个 get 按本地 tier 顺序分阶段执行：

```text
local tier[0]
  -> P2P read for tier[0], if configured
local tier[1]
  -> P2P read for tier[1], if configured
...
storage pool
```

例如本地层级是 `hbm -> dram`，只配置 `tier=dram`，读顺序为：

```text
local hbm -> local dram -> peer dram -> storage pool
```

整个 read path 维护四组 mask：

```text
local_hit_mask
peer_hit_mask
remote_hit_mask
satisfied_mask = local_hit_mask | peer_hit_mask | remote_hit_mask
```

local / peer / remote hit 可以是不连续的。最终统计只统计 `satisfied_mask` 的连续前缀范围。

## P2P Tracker

P2P 命中判定直接查 hash table tracker，不再真实查询 peer engine 的 radix tree。tracker 维护：

```text
(cluster_id, tier_name, block_key) -> infer_ids holding this block
```

tracker 是 P2P presence index。回放过程中 engine tier flow 是它的唯一更新来源：

| 事件 | tracker 更新 |
|---|---|
| `ENTER_TIER` | 加入 `(tier, block_key, infer_id)` |
| `LEAVE_TIER` | 删除 `(tier, block_key, infer_id)` |
| `FINAL_EVICT` | 从该 infer 的所有 tracked tier 中删除对应 block |
| `READ_TOUCH` | 不改变位置，不更新 tracker |
| `WRITE_TOUCH` | 不改变位置，不更新 tracker |

`ENTER_TIER` 只表示 block 进入某个 tier。promote、fill、copy 都可以产生 `ENTER_TIER`，但它们不是外部真实写入，不增加 `write_touch_count`，也不触发 selective write 阈值。

## Peer 选择

每个 P2P stage 只选择一台 peer，不把一次 P2P read 拆给多台 peer 拼接。

选择过程：

1. 从当前 `infer_cluster.infer_ids[]` 中排除当前 infer。
2. 根据 `satisfied_mask` 生成 ordered missing sequence。
3. 在配置的 `tier` 上查 tracker，选择从 ordered missing sequence 开头连续覆盖最长的 peer。
4. 并列时按 `infer_ids[]` 顺序确定。
5. 如果没有 peer 持有 ordered missing sequence 的第一个 block，本次 P2P stage miss。

示例：

```text
keys:        A B C D E F
local hit:   Y N Y N N Y
missing seq:   B   D E
```

如果 `tier=dram`：

```text
peer_1 has: B D       -> match_len = 2
peer_2 has: B D E     -> match_len = 3
peer_3 has: D E       -> match_len = 0
```

本次选择 `peer_2`，命中 `B/D/E`。已经由 local 命中的 `A/C/F` 不要求 peer 持有。

## Touch 与 Fill

peer 侧：

- P2P 命中来自 tracker。
- `peer_read_touch_enabled=true` 时，对选中的 peer 执行指定 tier 的 read touch。
- peer 侧不做 promote，不因为 P2P 被读取而回填到 peer 高层。

target engine 侧：

- P2P 命中的 block 会像 storage pool 命中一样回填当前 engine。
- 回填是 read-triggered fill/promote，不是外部 write。
- 回填可以产生 `ENTER_TIER(reason=PROMOTE)` 和容量驱逐。
- 回填不会产生 `WRITE_TOUCH`，不会增加 `write_touch_count`，不会触发 selective write 阈值。
- 如果回填导致 engine 最后一层发生 eviction，且 `storage_pool_flow.write_mode=cascading`，该 eviction 仍按既有规则写入 storage pool。

## Storage Pool 关系

P2P 完成后，storage pool 仍接收完整 keys。传给 storage pool 的非 remote mask 是：

```text
non_storage_hit_mask = local_hit_mask | peer_hit_mask
```

storage pool 只补仍未满足的 block。`query_type=prefix_match` 时，local / peer 已满足的 block 可以让 prefix 继续向后推进，但不会重复计成 remote hit；`query_type=batch_get` 时逐 block 独立查询。

`storage_pool_flow.local_read_touch_enabled` 只控制 storage pool 中已有副本的 read touch，不代表 P2P touch。

## 统计口径

最终 combined 统计按 token 计算：

```text
HitTokens = LocalHitTokens + PeerHitTokens + RemoteHitTokens
```

统计方式：

```text
final_prefix_len = longest consecutive true in satisfied_mask

LocalHitBlocks  = count(local_hit_mask  within final_prefix_len)
PeerHitBlocks   = count(peer_hit_mask   within final_prefix_len)
RemoteHitBlocks = count(remote_hit_mask within final_prefix_len)
```

示例：

```text
keys:   A B C D
local:  Y N Y N
peer:     Y
```

P2P 补上 `B` 后，最终连续前缀是 `A/B/C`：

```text
LocalHit  = A, C
PeerHit   = B
RemoteHit = none
HitBlocks = 3
```

`AccWriteBlocks` 只统计 trace 外部 write。P2P read、storage pool read touch、read-triggered fill、层间 copy、层间 promote 和 eviction 下沉都不算新的外部 write。

## 实现边界

- tracker 使用 hash table，不使用 radix tree。
- P2P 命中判定不再二次查询 peer engine index。
- peer touch 是独立副作用，只在 `peer_read_touch_enabled=true` 时执行。
- target fill 复用 storage pool hit 后的 engine fill/promote 语义。
- P2P 不改变 write-through、cascading、write-through-selective 的写入触发规则。
