#pragma once

// affinity v1 §2.4: LocalReplicaAffinityStrategy —— v1 默认策略（多副本方案）。
//
// 算法本质：基于 caller 节点局部性 + read-miss 反应式复制 + 节点水位 LRU
//   - 写时：把数据导向 caller 节点（preferred_node_ids[0]=caller）
//   - 读时：优先选 caller 本地的 spec
//   - 读未命中本地：通过 sketch 累积远端命中次数，超阈值产出 ReplicationHint
//   - 节点超水位：定向淘汰该节点上的副本
//
// 实现 3 个一级行为 method：ResolveWrite / ResolveRead / ResolveEviction。算法特有的子决策
// （写流水线、本地 spec 选择、复制触发、节点水位匹配）全部封装在 private
// helper 内部，不污染通用接口。

#include <cstdint>
#include <memory>
#include <string>

#include "kv_cache_manager/affinity/affinity_strategy.h"
#include "kv_cache_manager/affinity/pipeline/candidate_pipeline.h" // v0 CandidatePipeline 作为 write 流水线实现
#include "kv_cache_manager/common/affinity_types.h"

namespace kv_cache_manager {

class FrequencySketch;
class HintSuppressor;

// LocalReplica 算法的读副作用包装：把纯数据 ReplicationHint 挂上多态基类
// ReadSideEffect，透传到 ReadDecision.side_effects；调用方 downcast 后切片回数据基类。
struct ReplicationHintSideEffect : public ReadSideEffect, public ReplicationHint {};

class LocalReplicaAffinityStrategy : public AffinityStrategy {
public:
    struct Params {
        // === 3 个一级 toggle ===
        bool enable_write = true;
        bool enable_read = true;
        bool enable_eviction = true;

        // === 写一级 (write.ops)：v0 CandidatePipeline 5 段流水线作为实现 ===
        // 空 = 不计算 preferred_node_ids（backend 自由放置）
        std::shared_ptr<CandidatePipeline> write_pipeline;

        // === 读一级 on_miss 子项 (read.on_miss)：复制触发参数 ===
        bool enable_on_miss = true;              // 子开关：关闭后 ResolveRead 不再产出 hints
        uint32_t replication_hot_threshold = 3;  // 远端命中次数阈值
        double caller_capacity_threshold = 0.85; // caller 节点 load_ratio 上限
        double caller_capacity_buffer = 0.05;    // 缓冲带

        // 频率反馈机制层，构造期由 affinity_manager 注入（manager 持有，
        // 重解析策略后状态不丢）。nullptr ⇒ 算法保守不产复制提示。
        FrequencySketch *sketch = nullptr;

        // suppressor == nullptr ⇒ 不做时间抑制；window == 0 ⇒ 同上。
        HintSuppressor *suppressor = nullptr;
        uint32_t suppression_window_ms = 60000;

        // === 淘汰一级 (eviction.ops)：节点水位 ===
        double node_water_threshold = 0.85;
        double node_water_critical = 0.95;
        double node_water_low = 0.70;
    };

    LocalReplicaAffinityStrategy() = default;
    explicit LocalReplicaAffinityStrategy(Params params) : params_(std::move(params)) {}

    std::string Name() const override { return "local_replica"; }

    WriteDecision ResolveWrite(const std::vector<std::string> &candidates, const StrategyContext &ctx) const override;

    ReadDecision ResolveRead(const ReadRequest &req, const StrategyContext &ctx) const override;

    std::unordered_set<std::string> ResolveEviction(const StrategyContext &ctx) const override;

    const Params &params() const { return params_; }

private:
    // === 算法特有的私有 helper（曾经的接口字眼现在收进这里）===

    // 写一级实现细节：跑 v0 5 段流水线，返回 hints + abort 状态
    WriteDecision RunWritePipeline(const std::vector<std::string> &candidates, const StrategyContext &ctx) const;

    // 读一级实现细节 1：spec name 多候选时选 caller 本地的
    const LocationSpec *PickLocalSpec(const std::vector<const LocationSpec *> &candidates,
                                      const StrategyContext &ctx) const;

    // 读一级实现细节 2（on_miss 路径）：判定是否触发复制 + 喂 sketch + 产 hint
    bool ShouldEmitReplicationHint(int64_t block_key,
                                   bool has_local_in_picked,
                                   const CacheLocation *winner_tier,
                                   const StrategyContext &ctx) const;

    Params params_;
};

} // namespace kv_cache_manager
