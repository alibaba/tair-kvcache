# Optimizer 任务 Skills

这些 skill 是任务级工作流。先选一个 skill，再按 skill 要求阅读更深层参考。

| Skill | 使用场景 |
|---|---|
| [run-single-replay](run-single-replay/SKILL.md) | 跑一个 optimizer config、查看命中率、画单次回放图 |
| [run-multi-infer-replay](run-multi-infer-replay/SKILL.md) | 按 pod/infer 独立 cache 回放并聚合 |
| [run-hierarchical-replay](run-hierarchical-replay/SKILL.md) | 模拟 engine-local tier、storage pool、P2P、active window、cache drop |
| [run-pareto-analysis](run-pareto-analysis/SKILL.md) | 跑容量 Pareto，找 95%/99% 理论命中对应容量 |
| [prepare-trace](prepare-trace/SKILL.md) | 转换或校验外部 trace 为 optimizer JSONL |
| [integrate-optimizer-api](integrate-optimizer-api/SKILL.md) | 通过 C++ 或 Python API 接入 Optimizer |
| [add-optimizer-feature](add-optimizer-feature/SKILL.md) | 新增标准 optimizer 能力 |
| [add-eviction-policy](add-eviction-policy/SKILL.md) | 新增驱逐策略 |

总入口：[../handbook/README.md](../handbook/README.md)。
