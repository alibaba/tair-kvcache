#pragma once

// affinity v1 §2.5: NoopAffinityStrategy —— "框架不破坏 v0" 的可执行兜底。
// 3 个一级行为 toggle 全 false ⇒ 写 / 读 / 淘汰三条路径都 short-circuit；
// 3 个 method 即使被错误调用也返回安全 default 值。配置 `{"type":"noop"}`
// 后 KVCM 写时不亲和 / 读不偏本地 / 不发任何 side-effect / 不按节点淘汰，
// 与 v0 兼容行为一致。

#include "kv_cache_manager/affinity/affinity_strategy.h"

namespace kv_cache_manager {

class NoopAffinityStrategy : public AffinityStrategy {
public:
    std::string Name() const override { return "noop"; }

    WriteDecision ResolveWrite(const std::vector<std::string> & /*candidates*/,
                               const StrategyContext & /*ctx*/) const override {
        return WriteDecision{}; // kOk + 空 hints
    }

    ReadDecision ResolveRead(const ReadRequest & /*req*/, const StrategyContext & /*ctx*/) const override {
        return ReadDecision{}; // 空 picked_specs + 空 hints
    }

    std::unordered_set<std::string> ResolveEviction(const StrategyContext & /*ctx*/) const override {
        return {};
    }
};

} // namespace kv_cache_manager
