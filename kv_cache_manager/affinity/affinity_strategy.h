#pragma once

// affinity v1 §2.2: AffinityStrategy 抽象接口。
//
// 设计承诺：
//   1. 接口只暴露 3 个一级行为入口（write / read / eviction），完全对称。
//      接口契约稳定，未来不会再扩 method。
//   2. 接口命名完全通用，不带任何具体算法的概念（不出现 Replication /
//      NodeWaterLevel 等字眼）。算法特有逻辑全部封装到子类内部 private 方法。
//   3. 每个一级行为有独立 toggle，可热加载关闭。
//   4. 策略可替换：默认 LocalReplica（多副本方案）+ 兜底 Noop；新增算法
//      只需新建 AffinityStrategy 子类。
//
// 调用范式：
//   auto strategy = affinity_manager->GetStrategy(instance_id, group_name);
//   return strategy->Resolve<Write|Read|Eviction>(args, ctx);
// 一级 toggle 不在接口暴露：关闭的行为由对应 Resolve* method 内部 short-circuit
// 成 no-op 决策（write 返回空 hints、read 返回空 picks、eviction 返回 false）。

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/affinity/node_metrics.h"
#include "kv_cache_manager/common/affinity_types.h"
#include "kv_cache_manager/data_storage/write_hints.h"
#include "kv_cache_manager/meta/cache_location.h"

namespace kv_cache_manager {

// 3 个一级行为共用的上下文信号（只含通用信号，不含任何算法特有机制）。
struct StrategyContext {
    CallerNode caller_node;
    std::string instance_id;
    std::string instance_group_name;
    std::string trace_id;
    // 查 NodeMetrics 的回调；nullptr 时算法应视为缺失指标（permissive）。
    std::function<const NodeMetrics *(const std::string &)> get_node_metrics;

    // 淘汰一级：全量节点快照（manager 填入）
    std::vector<NodeMetrics> all_nodes;
    // 淘汰一级：每节点已淘汰字节快照（manager 从 hysteresis 状态填入）
    std::unordered_map<std::string, int64_t> evicted_bytes;
};

// 读一级副作用的通用基类。具体副作用类型（如 LocalReplica 的 ReplicationHint）
// 由各算法自己定义并 downcast 取回；接口本身不假设存在任何具体类型。
struct ReadSideEffect {
    virtual ~ReadSideEffect() = default;
};

// === 读一级 IO 结构 ===

struct ReadRequest {
    int64_t block_key = 0;
    // per-spec 候选已按 spec name 聚合（由 meta_searcher 的 merge 步预处理）
    std::map<std::string, std::vector<const LocationSpec *>> spec_candidates;
    // SelectForMatch 已选出的 winner tier（决定 backend type）
    const CacheLocation *winner_tier = nullptr;
};

struct ReadDecision {
    // 每个 spec name 选中的 winner spec；缺失或为空 = 调用方走默认
    std::map<std::string, const LocationSpec *> picked_specs;
    // 算法产出的通用副作用（如复制提示 / prefetch 建议等）；不产副作用的算法留空。
    // 调用方按需 downcast 取回具体类型。
    std::vector<std::unique_ptr<ReadSideEffect>> side_effects;
};

// 写一级决策结果（仿 v0 CandidatePipeline::ApplyResult）：
//   kOk    ⇒ hints 为最终偏好（可空 = 无偏好，backend 自由放置）
//   kAbort ⇒ 策略主动中止（如 prefer_local on_miss=abort 找不到本地候选），
//            调用方据此降级（v1 退化为无 hint 写）。
enum class AffinityStatus {
    kOk,
    kAbort
};
struct WriteDecision {
    AffinityStatus status = AffinityStatus::kOk;
    WriteHints hints;
};

class AffinityStrategy {
public:
    virtual ~AffinityStrategy() = default;

    // 策略名，用于诊断 / metric 维度。
    virtual std::string Name() const = 0;

    // === 3 个一级行为入口（完全对称，均值返回）===
    // 一级 toggle 是算法内部细节（如 LocalReplica 的 Params.enable_*），不暴露到
    // 接口；关闭的行为由对应 method 内部 short-circuit 成 no-op 决策。

    // 写一级：把 caller_node 等信号转换为 WriteHints.preferred_node_ids。
    virtual WriteDecision ResolveWrite(const std::vector<std::string> &candidates,
                                       const StrategyContext &ctx) const = 0;

    // 读一级：per-key 一次性输入 spec 候选 + winner tier，返回 spec 选择
    // 结果 + 任何算法产出的 side-effects（如复制提示）。
    // 调用方（meta_searcher）按 picked_specs 构造 merged CacheLocation；
    // 把 side_effects 累加到 wrapper 透传给 service 层。
    virtual ReadDecision ResolveRead(const ReadRequest &req, const StrategyContext &ctx) const = 0;

    // 淘汰一级：返回亲和层认为应触发节点级淘汰的节点 ID 集合。
    // 算法从 ctx.all_nodes + ctx.evicted_bytes 做 hysteresis 判定；空 = 无需节点级淘汰。
    // 返回 unordered_set 供 FilterLocID O(1) 查找。
    virtual std::unordered_set<std::string> ResolveEviction(const StrategyContext &ctx) const = 0;
};

} // namespace kv_cache_manager
