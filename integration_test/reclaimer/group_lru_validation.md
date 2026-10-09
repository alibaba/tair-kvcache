# Group LRU 持续流量验证

验证日期：2026-09-14。本文描述本次测试条件下的实际结果，不作为生产吞吐或清空时限承诺。

测试代码：[group_lru_traffic_test.py](group_lru_traffic_test.py)。策略与预算规则见 [Group LRU 设计](../../docs/design/cache_reclaimer_group_lru.md)。

## 1. 结论

- 在本次持续回收压力下，停止读写的小 Instance 和大 Instance 都能清到 0；无水位压力时，不会仅因 Instance 停止访问就清空它。
- 新旧部署切换时，1:1 采删比稳定复现了新 Instance 数据被提前删除；相同场景使用 5:1、10:1 后，新 Instance 数据全部保留。多个 Instance 持续活跃、写入量相差超过 10 倍时，稳态和突发流量下的热点也全部保留。
- 当前仍是采样近似 LRU，不是严格的全量 Group LRU。部署切换场景中，保护新 Instance 后仍存在旧 Instance 内部的顺序偏差；具体统计口径和不利结果保留在第 4 节。

## 2. 测试条件与范围

### 2.1 验证代码

验证代码为 `c98f9630` 基线加上与本文同一提交交付的采样比例保护、Local 采样任务调整、持续流量测试和年龄指标改动，不能把结果归于基线提交本身。最新回归所用的 17 个相关源码和 BUILD 文件已与本次提交内容逐一核对 SHA-256，一致。不同测试批次的范围见第 2.3 节。

本次关注的行为是：

- Group LRU 使用独立的 `group_lru_min_sampling_ratio`，默认 10。采样基准为 `max(S_cfg, B_cfg * ratio)`，理论删除预算仍是 `B_cfg * N`，不放大容量比例和固定策略的预算。
- 按去重、Location 资格过滤后的实际候选数进一步约束删除量；非空候选保留至少一个删除名额，避免最后几个 key 因取整长期无法清空。
- 纯 Local、以及恢复完成且采样源为 Local 的 cached 后端，同一 Instance 不再拆成多个采样任务；不同 Instance 仍可并行。Redis、恢复中的 cached 后端保留任务拆分。
- Group LRU 上报最近一个 accepted 删除请求的 LRU 年龄和 Location 创建年龄。

### 2.2 环境与参数

| 项目 | 本次设置 |
|---|---|
| 运行环境 | 隔离 Linux 测试环境，不访问生产服务 |
| 构建选项 | `--define=ENABLE_MOONCAKE=false`，未构建 Mooncake |
| 元数据后端 | Local，64 个分片，`sample_times=64`，每 Instance 容量上限 1024 MiB |
| block 统计大小 | 每个 block 1024 bytes |
| 逐出策略 | `instance_reclaim_budget_policy=GROUP_LRU`，LRU 选择 |
| 共享采样 / 删除基准 | `key_sampling_size_total=100`、`del_batch_size=100` |
| 采样子任务基准 | `key_sampling_size_per_task=100`；本次 Local 路径每 Instance 只提交一个采样任务 |
| Group 保护参数 | `group_lru_max_sampling_size=65536`、`group_lru_max_delete_requests_per_round=128` |
| 采样比例 | `group_lru_min_sampling_ratio=1 / 5 / 10`；除对照场景外均为 10 |
| GC / 删除延迟 | 关闭 GC，`delay_before_delete_ms=0` |
| 流量方式 | 每个活跃 Instance 独立线程和 HTTP session，每轮完成读写后间隔 20 ms |

采样基准、删除基准、采样比例和 Group 保护参数使用 `kvcm.cache_reclaimer.*` 前缀；逐出策略和删除延迟属于 Group 的 `cache_config.reclaim_strategy` 配置。`1:1`、`5:1`、`10:1` 指配置的采样与删除预算比例，不是最终实际删除量的保证。`ratio=1` 使用同一份采样优化后的二进制，只调整参数，不是历史版本的性能对照。

测试经过真实 HTTP `StartWriteCache` / `FinishWriteCache` / `GetCacheLocation`，以及 MetaIndexer、Reclaimer、Executor 和异步删除链路。NFS 后端只提供有效 Location URI，不写入真实 KV payload。因此这是十万级元数据持续流量与回收行为验证，不是数据存储带宽或生产吞吐压测。

场景中的水位目标以等价 key 数表示，实际按每 key 1024 bytes 设置 Group bytes 触发阈值。结束流量后先提高阈值、等待在途删除收敛，再逐 key 查询 Location；以非空 URI 判断存活，避免核验读改变回收期间的 LRU。每个场景均核对最终消失的 key 数与提交删除的 block 数一致。停压后的剩余量可能略高于原水位目标，不据此宣称水位已经完全收敛。

### 2.3 结果批次

为区分规模和代码状态，下面保留两批扩大规模测试的来源，不将它们合称为最新代码上的一次全量重跑：

- 采样比例与持续流量验证：7 个场景变体，各重复 3 次，共 21 组。压力场景约 30 秒，无压力对照 2 秒。
- 大 Instance 清零与年龄指标验证：新增 2 个场景变体，各重复 3 次，共 6 组；观测到旧 Instance 清零后停压。
- 最新代码的完整回归：14 个测试目标全部重新执行，其中持续流量目标使用默认 12,000 基准 key，覆盖全部 9 个场景变体。不是上述 21 组十万级测试的重跑。

## 3. 场景与结果

### 3.1 多个 Instance 持续活跃

三个 Instance 初始分别有 60,000 / 40,000 / 20,000 个 key，每轮分别写入 128 / 32 / 8 个新 key，并各自持续读取 64 个热点 key。Group 水位目标为 96,000 个 key。突发场景每隔 8 轮将当轮写入量增至 4 倍。

每轮写入配置的最大差距为 16 倍；受请求耗时影响，实测最快与最慢 Instance 在同一测试窗口内新增 key 数相差 12.96～13.98 倍。这是写入 key 数量的差距，不是 HTTP 请求数或初始容量相差 16 倍。

| 场景 | 三次运行的热点结果 | 快照新旧倒置比例 |
|---|---|---:|
| 稳态流量 | 每次均保留全部 192 个热点 key | 0.059%～0.102% |
| 突发流量 | 每次均保留全部 192 个热点 key | 0.113%～0.357% |

用例自动检查热点不丢失，并要求快照倒置比例低于 5%；表中更低的数值是本次观测结果，不是通用上界。快照指标定义见第 4 节。

各 Instance 最终容量不要求均分。例如一次稳态运行中，快 / 中 / 慢 Instance 分别保留 71,721 / 19,564 / 4,727 个 key，符合按访问时间选择、允许不同容量的目标。

### 3.2 旧大 Instance 停止访问，新小 Instance 持续写入

旧 Instance 初始有 120,000 个 key，新 Instance 初始有 12,000 个 key。旧 Instance 不再读写，新 Instance 每轮写入 32 个 key，持续约 30 秒；Group 水位目标为 120,000 个 key。

| 采删比 | 第 1 次新 Instance 删除数 | 第 2 次 | 第 3 次 | 结果 |
|---|---:|---:|---:|---|
| 1:1 | 6,007 | 6,230 | 5,808 | 稳定复现新数据被提前删除 |
| 5:1 | 0 | 0 | 0 | 本场景中新 Instance 的已有及新增数据全部保留 |
| 10:1 | 0 | 0 | 0 | 本场景中新 Instance 的已有及新增数据全部保留 |

5:1、10:1 的运行结束时，旧 Instance 仍有约 7 万个 key，因此比较期间并未耗尽旧数据。该场景验证的是旧数据仍充足时保护新 Instance，不以清空旧 Instance 为结束条件。

### 3.3 停止访问的 Instance 是否能清到 0

以下场景均使用 10:1 采删比，停止访问的 Instance 在压力阶段不发生业务读取或写入，水位目标均为 120,000 个 key。

| 场景 | 初始状态与持续流量 | 三次运行的停止访问 Instance 最终 key 数 | 测试结束条件 |
|---|---|---|---|
| 大 Instance 活跃，小 Instance 停止访问 | 大 Instance 120,000、小 Instance 12,000；大 Instance 每轮写 128 个 | 0 / 0 / 0 | 持续流量约 30 秒后核验 |
| 旧大 Instance 停止访问，新 Instance 从空开始 | 先写满旧 Instance 的 120,000 个 key，再注册空新 Instance；新 Instance 每轮写 256 个 | 0 / 0 / 0 | 观测到清零后停压，流量时长 20.041 / 20.421 / 20.084 秒 |
| 旧大 Instance 停止访问，新 Instance 已有少量数据 | 旧 Instance 120,000、新 Instance 12,000；新 Instance 每轮写 256 个 | 0 / 0 / 0 | 观测到清零后停压，流量时长 14.973 / 14.983 / 15.027 秒 |

后两行持续写入直到观测到旧 Instance 用量为 0，再停压并逐 key 核对 URI。用量 recorder 每 5 秒刷新，测试区分“指标尚未发布”和“值为 0”；表中时长包含观测延迟，不是精确的物理删除完成时间，更不是生产清空时限。

大 Instance 清零的 6 组测试中，新 Instance 累计新增 797,952 个 key，最终各保留约 120,000 个 key。新 Instance 自身也会回收历史写入：测试不能证明每个新 key 都等旧 Instance 完全清空后才删除。

### 3.4 无回收压力时不自动清空

不活跃 Instance 初始有 12,000 个 key，活跃 Instance 初始有 120,000 个 key，只有后者继续写入。设置水位不触发回收，运行 2 秒，重复 3 次。

三次均无删除提交，不活跃 Instance 均保留全部 12,000 个 key。停止流量不是清空信号；Group LRU 仍由水位触发。

## 4. 如何理解排序偏差

测试记录客户端成功写入或命中读取后的最后已知业务访问时间。停压后，将所有 key 分成“已删除”和“仍保留”两组，计算：

```text
快照新旧倒置比例 = 已删除 key 比仍保留 key 更新的配对数 / 所有“已删除 × 仍保留”配对数
```

时间相同不计倒置；分母为 0 时记为 0。同一批 key 可能共享客户端时间，且观察时点是最终快照，不是每次删除提交时的服务端快照。因此该值不是误删 key 的比例、命中率，也不是逐请求的严格 LRU 违例率。

| 部署切换场景的采删比 | 三次运行的快照倒置比例 | 新 Instance 是否被删除 |
|---|---:|---|
| 1:1 | 9.109%～9.552% | 是 |
| 5:1 | 13.141%～15.238% | 否 |
| 10:1 | 10.081%～12.132% | 否 |

5:1、10:1 虽然保护了新 Instance，整体快照指标却没有优于 1:1。这些运行中已删除 key 全部来自先写入的旧 Instance，倒置发生在旧 Instance 内部；但表中分母仍是整个 Group 的配对数，不能把它解释成仅在旧 Instance 内部计算的倒置比例。

从 [Local 采样实现](../../kv_cache_manager/meta/meta_local_backend.cc) 和 [LRU 采样游标](../../kv_cache_manager/common/cache/lru_cache.cc) 看，游标随采样前进，不会因候选未进入 Top B 而回退，因此可能在后续采样中读到旧 Instance 内稍新的 key。这是与代码一致的机制解释，不是对每次倒置的完整归因。

提高采样比例改善了本次测试中的新旧 Instance 选择，但没有消除 Instance 内部排序偏差。不能只依据新 Instance 没被删除，就宣称已经实现严格全局 LRU。

## 5. 回归与指标验证

最新代码的 14 个完整回归目标全部通过，无跳过或禁用用例：

| 范围 | 实际结果 |
|---|---|
| CacheReclaimer | 175 个 C++ 用例通过 |
| BackendManager / MetaIndexer | 52 / 21 个 C++ 用例通过 |
| Local / Redis / AsyncRedis / Dummy backend | 51 / 16 / 41 / 18 个 C++ 用例通过 |
| ServerConfig | 9 个 C++ 用例通过 |
| LocalMetricsReporter / PrometheusExporter | 13 / 29 个 C++ 用例通过 |
| reclaiming / location_pruning / multi_location | 11 / 6 / 3 个 Python 测试方法通过 |
| 持续流量测试，默认规模 | 5 个 Python 测试方法、9 个场景变体通过 |

合计 425 个 C++ 用例、25 个 Python 测试方法。Redis / AsyncRedis 在这里属于单元测试覆盖，不代表已经开展对应后端的持续流量压测。

采样优化阶段的 47 个 Reclaimer、5 个 BackendManager、1 个 ServerConfig 专项用例均打乱顺序重复 20 次通过。最新代码另将 4 个年龄相关用例打乱顺序重复 20 次，80 次执行均通过。相关 C++ 文件通过仓库 `.clang-format` 配置下的 clang-format 13.0.1 检查。

两批扩大规模测试共 27 组，累计写入 6,054,928 个 key 的元数据，最终核对删除 2,912,303 个 block。所有场景的 `group_lru_partial_plan_count`、`group_lru_deadline_count` 均为 0；这也意味着这批持续流量结果不覆盖 deadline 频繁截断时的 LRU 精度。

6 组大 Instance 清零测试及最新默认规模回归验证了 `reclaim_batch_lru_age_{min,max,avg}_us`、`reclaim_batch_create_age_{min,max,avg}_us` 在管理接口和 Prometheus 中可读，数值在现有文本输出的舍入精度范围内一致，并满足 `min ≤ avg ≤ max`。这些 gauge 描述最近一个 accepted 请求，LRU 年龄按最终待删 block 统计，创建年龄按最终待删 Location 统计；不代表整轮平均年龄或物理删除已完成。单测另覆盖无效时间、过滤掉的候选、拒绝请求和水位提前恢复等边界。

## 6. 复现方法

在具备项目构建依赖的隔离 Linux 环境中，从仓库根目录运行。测试自行启动 KVCM 测试进程，使用生成的测试配置，不需要连接已有服务。扩大规模运行时避免与其他测试或负载并行。

日常功能回归，默认基准 key 数 12,000、普通流量时长 4 秒：

```bash
bazelisk test \
  --define=ENABLE_MOONCAKE=false \
  --jobs=4 --local_ram_resources=10000 --local_test_jobs=1 \
  --nocache_test_results --test_output=summary --test_timeout=1200 \
  //integration_test/reclaimer:group_lru_traffic_test
```

十万级验证，普通压力场景持续约 30 秒，各重复 3 次：

```bash
bazelisk test \
  --define=ENABLE_MOONCAKE=false \
  --jobs=4 --local_ram_resources=10000 --local_test_jobs=1 \
  --nocache_test_results --test_output=summary --test_timeout=1200 \
  --test_env=KVCM_GROUP_LRU_TRAFFIC_KEYS=120000 \
  --test_env=KVCM_GROUP_LRU_TRAFFIC_SECONDS=30 \
  --runs_per_test=3 \
  //integration_test/reclaimer:group_lru_traffic_test
```

当前目标包含全部 9 个场景变体，上述命令会运行 27 组；第 2.3 节已说明本次结果来自两批测试，不声称在最新代码上重新执行过这一整条扩大规模命令。

仅复现大 Instance 清零的两个变体，可在上述命令中增加：

```text
--test_arg=GroupLruTrafficTest.test_traffic_inactive_large_instance_drains_to_zero
```

`KVCM_GROUP_LRU_TRAFFIC_KEYS` 是基准数，不是所有场景的 Group 初始总量；具体构成见第 3 节。旧大 Instance 清零场景在观测到清零时结束，最多运行 `max(60, SECONDS * 4)` 秒；冷小 Instance 场景至少运行 6 秒；无压力场景固定 2 秒。

每个场景在 Bazel 测试日志中输出一行 `GROUP_LRU_TRAFFIC_RESULT=<JSON>`，包含各 Instance 写入、删除、剩余量，热点保留、快照倒置比例、回收指标和请求延迟。可按此前缀提取结果做对照；原始运行日志不随本文入库。

## 7. 已知限制

- 这里只验证了少量 Instance、Local 元数据、统一 block 统计大小和受控流量。没有覆盖生产规模 Instance 数、真实 Redis / PACE 持续流量、真实 KV payload 或高并发存储删除。
- 多活跃测试覆盖约 13～14 倍实测写入量差距、固定热点及周期突发；不代表已经覆盖任意更大倍率、热点迁移或长期混合负载。
- 流量带有固定循环间隔，属于闭环测试。输出的 HTTP QPS 和延迟用于描述本次负载，不能作为服务吞吐上限或性能提升证明。
- 清零依赖持续回收压力、候选可删除及后端正常推进。不承诺无压力清空、固定时间清空，也不保证所有新 key 都晚于所有旧 key 删除。
- 默认采样比 10、采样总量上限 65536 和请求上限 128 是当前可调配置，不据此宣称已经最优。后续优化采样覆盖时，应同时验证旧 Instance 内部顺序与不可删除候选下的回收进展。
