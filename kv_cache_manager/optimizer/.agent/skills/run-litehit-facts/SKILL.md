# LiteHit Facts Skill

当分析对象是 full-attention prefix cache，且满足以下任一条件时，使用这个 skill：

- 容量假设会反复变化（容量事后投影，不重放 trace）；
- 需要同一份 trace 一次回放扫多个 block size；
- 需要精确 LRU prefix 命中率（非采样、非近似）；
- 需要在线实时命中率（gRPC/HTTP 服务）。

## 需要确认的输入

- trace 原生 block 粒度（config `block_size`，默认 256 token/block）。
- 各 instance 的分析粒度（instance `block_size`，必须是 trace 粒度的整数倍，只允许变粗）。
- key 形态：trace 里是前缀链式 key 还是逐块 raw hash（后者在 Instance Group 上开 `enable_prefix_hash`）。
- 路由：按 trace `instance_id`、`override_instance_id` 汇聚成一个服务视角，还是 `fanout_all_instances` 广播扫粒度（与 override 互斥）。
- location spec size（决定 `block_bytes`，容量 GB → block 换算的锚点）。
- 要查询的容量列表（GB，可含重复和 0，负数 = 无限；离线不需要预先给定）。

## 离线两步流程

```bash
# 第一步：回放产 facts（config 不含容量）
bazel run //kv_cache_manager/optimizer:lite_hit_main -- /path/to/lite_hit_config.json
# -> <output_result_path>/litehit_facts.csv（原子发布，fail-fast 全有或全无）

# 第二步：任意容量事后投影，可反复跑
bazel run //kv_cache_manager/optimizer:lite_hit_facts_query_main -- \
  /path/to/litehit_facts.csv /path/to/result.jsonl 10 50 100 -1
```

多 block size 扫描：配置 N 个不同 `block_size` 的 instance + `fanout_all_instances: true`，一次回放各 lane 独立产 facts；query 的 summary 按 instance 分组（每 instance 一行 + 总计一行）。

## 在线服务

```bash
bazel run //kv_cache_manager/optimizer:online_optimizer_server_main -- /path/to/server_config.json
```

引擎侧用 `client/`（Python gRPC/HTTP SDK）：建组（容量档、`enable_prefix_hash`、`enable_theoretical_max_cache`）→ 注册实例 → 逐请求 TraceQuery → `ListInstances` 看累计。压测用 `tools/online_optimizer_benchmark`。

## 校验

- trace 按 `timestamp_ns` 排序；`get/request` 带 `input_len`。
- facts 行数 = 有效读请求数 ×（fanout 时的 lane 数）。
- query summary 的累计命中率 = `Σ hit_tokens / Σ input_tokens`（token 口径，尾部 token 在分母）。
- 理论上界（无限容量投影）≥ 一切有限容量结果。

## 回复内容

报告：

- facts CSV 路径与行数
- 各容量（和各 instance/block size）的累计 token 命中率
- 理论上界命中率
- 未完成的校验项

语义细节：[../../../liteHit/README.md](../../../liteHit/README.md)。
