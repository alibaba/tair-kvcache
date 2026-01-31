# KVCacheManager Optimizer 使用指南

## 概述

KVCacheManager Optimizer 是一个独立的缓存优化分析模块，通过回放 trace 数据来模拟缓存读写操作，评估不同驱逐策略和配置对缓存命中率的影响。

**核心功能**：
- 支持多种驱逐策略（LRU、RandomLRU、LeafAwareLRU）
- 支持多种 trace 格式（Publisher Log、Qwen Bailian）
- 提供详细的缓存命中率统计和分析
- 支持多种可视化分析工具

**应用场景**：
- 在生产环境部署前评估不同驱逐策略的效果
- 分析缓存访问模式，为驱逐策略参数提供优化建议
- 预测不同容量配置下的缓存命中率
- 通过可视化工具理解缓存行为和性能瓶颈

## 编译

```bash
bazel build //kv_cache_manager/optimizer:optimizer_main
```

## 配置文件

配置文件示例（参考 `kv_cache_manager/optimizer/optimizer_startup_config_load.json`）：

```json
{
    "trace_file_path": "/path/to/trace_file", // trace文件路径
    "trace_type": "qwen_bailian", // trace类型，目前包含2种
    "output_result_path": "/path/to/output/result/", //输出文件路径
    "eviction_params": {
        "eviction_mode": 1, //驱逐模式，目前包含3种
        "eviction_batch_size_per_instance": 100 //轮询驱逐批大小
    },
    "instance_groups": [
        {
            "group_name": "instance_group_01",
            "quota_capacity": 12000, //实例组容量配额
            "used_percentage": 1.0,  // quota可供使用百分比
            "hierarchical_eviction_enabled": false, //开启多层存储，目前不可用
            "storages": [ // 各层存储配置
                {
                    "unique_name": "pace_00",
                    "storage_type": "pace",
                    "band_width_mbps": 20000,
                    "priority": 0,
                    "capacity": 100000
                }
            ],
            "instances": [
                {
                    "instance_id": "instance",
                    "block_size": 16,
                    "eviction_policy_type": "random_lru", //该实例使用的驱逐策略
                    "eviction_policy_params": { //驱逐策略相关参数
                        "sample_rate": 0.1
                    }
                }
            ]
        }
    ]
}
```

**可选配置项**：
- `trace_type`: publisher_log, qwen_bailian
- `eviction_mode`: 1=GROUP_ROUGH, 2=INSTANCE_ROUGH, 3=INSTANCE_PRECISE
- `eviction_policy_type`: lru、random_lru、leaf_aware_lru

## 基本使用
基本示例见 [Optimizer README](../kv_cache_manager/optimizer/README.md#示例)
### 运行 Optimizer

```bash
bazel run //kv_cache_manager/optimizer:optimizer_main -- /path/to/config.json
```

### 可视化分析

**命中率随时间变化图表**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:optimizer_run -- \
    -c /path/to/config.json --draw-chart
```

**Radix Tree 可视化**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:visualize_tree -- \
    -c /path/to/config.json
```

**Trade-off 曲线分析**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff_analysis_run_by_instances -- \
    -c /path/to/config.json
```

**多策略对比分析**：
```bash
bazel run //kv_cache_manager/optimizer/analysis/script:tradeoff_analysis_run_by_policies -- \
    -c /path/to/config.json --eviction-policies lru random_lru leaf_aware_lru
```

## 扩展开发

### 添加新的驱逐策略

1. 在 `kv_cache_manager/optimizer/eviction_policy/` 创建新策略文件，继承 `EvictionPolicy` 基类
2. 在 `kv_cache_manager/optimizer/config/types.h` 添加新的策略类型枚举值
3. 在 `kv_cache_manager/optimizer/eviction_policy/policy_factory.cc` 添加新的策略创建逻辑

### 添加新的 Trace 转换器

1. 在 `kv_cache_manager/optimizer/trace_converter/` 创建新转换器文件，继承 `BaseConverter` 基类
2. 在 `kv_cache_manager/optimizer/config/types.h` 添加新的 trace 类型枚举值
3. 在 `kv_cache_manager/optimizer/trace_converter/converter_factory.cc` 添加新的转换器创建逻辑

### 添加新的分析指标

1. 在 `kv_cache_manager/optimizer/analysis/result_structure.h` 添加新的统计字段
2. 在 `kv_cache_manager/optimizer/analysis/result_analysis.cc` 添加新的分析逻辑
3. 实现新的导出函数来输出自定义指标

## 详细文档

- [Optimizer README](../kv_cache_manager/optimizer/README.md)
- [Optimizer 架构文档](../kv_cache_manager/optimizer/docs/optimizer_architecture.md)