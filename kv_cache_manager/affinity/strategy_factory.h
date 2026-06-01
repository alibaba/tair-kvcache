#pragma once

// affinity v1 §17.4: strategy_factory 解析 JSON 配置并返回 AffinityStrategy。
//
// 接受的 JSON 形态：
//   {"type": "noop"}                                       -> NoopAffinityStrategy
//   {"type": "local_replica", "enabled_aspects": {...},
//     "write"/"read"/"eviction": {...}}                    -> LocalReplicaAffinityStrategy
//   空字符串 / 解析失败                                       -> nullptr (调用方 fall through)

#include <memory>
#include <string>

#include "kv_cache_manager/affinity/affinity_strategy.h"

namespace kv_cache_manager {

class FrequencySketch;

class StrategyFactory {
public:
    // 解析顶层 JSON。失败时返回 nullptr 并填 error_msg（非空时）。
    // sketch：manager 持有的频率反馈机制层，构造期注入需要它的算法
    //         （如 LocalReplica）；不需要的算法忽略。可空。
    static std::shared_ptr<AffinityStrategy>
    ParseJsonString(const std::string &json, FrequencySketch *sketch = nullptr, std::string *error_msg = nullptr);
};

} // namespace kv_cache_manager
