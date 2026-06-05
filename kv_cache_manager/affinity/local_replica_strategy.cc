#include "kv_cache_manager/affinity/local_replica_strategy.h"

#include "kv_cache_manager/affinity/frequency_sketch.h"
#include "kv_cache_manager/affinity/hint_suppressor.h"

namespace kv_cache_manager {

WriteDecision LocalReplicaAffinityStrategy::ResolveWrite(const std::vector<std::string> &candidates,
                                                         const StrategyContext &ctx) const {
    if (!params_.enable_write) {
        return WriteDecision{}; // 写一级关闭 ⇒ kOk + 无偏好
    }
    return RunWritePipeline(candidates, ctx);
}

ReadDecision LocalReplicaAffinityStrategy::ResolveRead(const ReadRequest &req, const StrategyContext &ctx) const {
    ReadDecision dec;
    if (!params_.enable_read) {
        return dec; // 读一级关闭 ⇒ 空 picked，调用方退化为首到
    }

    // ==== Step 1: 对每个 spec name 选 caller 本地优先的 winner ====
    bool any_local = false;
    for (auto &kv : req.spec_candidates) {
        const auto &spec_name = kv.first;
        const auto &cands = kv.second;
        const LocationSpec *picked = PickLocalSpec(cands, ctx);
        dec.picked_specs[spec_name] = picked;
        if (picked != nullptr && !ctx.caller_node.node_id.empty() && picked->node_id() == ctx.caller_node.node_id) {
            any_local = true;
        }
    }

    // ==== Step 2: on_miss 路径 —— 决定是否产出 ReplicationHint ====
    // 这里包含机制层（sketch.Observe）+ 策略层（4 gate 判定）。
    // 算法内部封装：调用方（meta_searcher / cache_manager）完全不知道复制怎么算的。
    if (!params_.enable_on_miss || ctx.caller_node.node_id.empty()) {
        return dec;
    }
    // TODO: 当前用 any_local（任意 spec 在本地即跳过），对多 spec 场景存在
    // 部分副本永远无法补全的问题：spec A 在本地 → any_local=true → spec B 的
    // 远端命中不被 sketch 记录 → 永远不触发 hint。修为 all_local 需同时：
    //   1. ReplicationHint 支持 per-spec 粒度（当前 proto 只有 block 级）
    //   2. SDK 侧只复制缺失 spec（避免重复创建已有 spec）
    //   3. existsOnCallerNode 按 spec 粒度去重
    // 暂保持 any_local 语义；单 spec 场景无此问题。
    if (any_local) {
        return dec;
    }
    if (req.winner_tier == nullptr) {
        return dec;
    }
    // 喂 sketch（机制层，永远 active；只在远端命中时累加）
    if (params_.sketch != nullptr) {
        params_.sketch->Observe(ctx.caller_node.node_id, req.block_key);
    }
    if (ShouldEmitReplicationHint(req.block_key, /*has_local=*/false, req.winner_tier, ctx)) {
        // TODO: 当前只取 winner_tier 的第一个非空 URI 作为触发信号。
        // 服务端同时兼容两种 SDK 复制模式：
        //   1. 立即复制：SDK 读完数据后内存中已有全部 spec，直接发起
        //      StartWriteCache(is_replication=true)，source_uri 仅作标识/校验。
        //   2. 延迟复制：SDK 排队异步处理 hint，需从 source_uri 重新读取数据。
        // 当前 source_uri 是单个 spec 的 URI。对模式 1 无影响（数据已在内存）；
        // 对模式 2 的多 spec 场景，只能读到第一个 spec 的数据，其余 spec 缺失。
        // 若需完整支持模式 2 + 多 spec，proto 应改为 repeated SourceSpec
        // （每个 spec name 一个 URI）。
        std::string source_uri;
        for (const auto &spec : req.winner_tier->location_specs()) {
            if (!spec.uri().empty()) {
                source_uri = spec.uri();
                break;
            }
        }
        if (!source_uri.empty()) {
            const bool allow = params_.suppressor == nullptr
                                   ? true
                                   : params_.suppressor->TryEmit(
                                         req.block_key, ctx.caller_node.node_id, params_.suppression_window_ms);
            if (allow) {
                auto h = std::make_unique<ReplicationHintSideEffect>();
                h->block_key = req.block_key;
                h->source_uri = std::move(source_uri);
                h->target_node_id = ctx.caller_node.node_id;
                dec.side_effects.push_back(std::move(h));
            }
        }
    }
    return dec;
}

std::unordered_set<std::string> LocalReplicaAffinityStrategy::ResolveEviction(const StrategyContext &ctx) const {
    if (!params_.enable_eviction) {
        return {};
    }

    const double high = params_.node_water_threshold;
    const double low = params_.node_water_low;

    std::unordered_set<std::string> result;
    for (const auto &node : ctx.all_nodes) {
        if (node.node_id.empty()) {
            continue;
        }
        double estimated_load = node.load_ratio;
        auto it = ctx.evicted_bytes.find(node.node_id);
        if (it != ctx.evicted_bytes.end() && it->second > 0) {
            double total = node.free_bytes / std::max(1.0 - node.load_ratio, 0.01);
            estimated_load -= static_cast<double>(it->second) / total;
        }
        if (estimated_load <= low) {
            continue;
        }
        if (node.load_ratio > high || it != ctx.evicted_bytes.end()) {
            result.insert(node.node_id);
        }
    }
    return result;
}

// ============================================================================
// 私有 helper
// ============================================================================

WriteDecision LocalReplicaAffinityStrategy::RunWritePipeline(const std::vector<std::string> &candidates,
                                                             const StrategyContext &ctx) const {
    WriteDecision dec;
    if (!params_.write_pipeline) {
        return dec; // 未配 write.ops ⇒ kOk + 无偏好（backend 自由放置）
    }
    auto result = params_.write_pipeline->Apply(candidates, ctx.get_node_metrics, ctx.caller_node, ctx.trace_id);
    if (result.status == CandidatePipeline::Status::kAbort) {
        dec.status = AffinityStatus::kAbort;
        return dec;
    }
    dec.hints.preferred_node_ids = std::move(result.nodes);
    return dec;
}

const LocationSpec *LocalReplicaAffinityStrategy::PickLocalSpec(const std::vector<const LocationSpec *> &candidates,
                                                                const StrategyContext &ctx) const {
    if (candidates.empty()) {
        return nullptr;
    }
    if (ctx.caller_node.node_id.empty()) {
        return candidates.front(); // 空 caller ⇒ 退化为首到
    }
    for (const LocationSpec *s : candidates) {
        if (s != nullptr && s->node_id() == ctx.caller_node.node_id) {
            return s;
        }
    }
    return candidates.front();
}

bool LocalReplicaAffinityStrategy::ShouldEmitReplicationHint(int64_t block_key,
                                                             bool has_local_in_picked,
                                                             const CacheLocation * /*winner_tier*/,
                                                             const StrategyContext &ctx) const {
    // gate 1: caller_node.node_id 非空（ResolveRead 已检过，这里 belt-and-suspender）
    if (ctx.caller_node.node_id.empty()) {
        return false;
    }
    // gate 2: 没有本地 spec（ResolveRead 已检过）
    if (has_local_in_picked) {
        return false;
    }
    // gate 3: 频率超阈值
    if (params_.sketch == nullptr) {
        return false; // 没 sketch 拿不到计数，保守不发
    }
    uint32_t cnt = params_.sketch->RemoteCount(ctx.caller_node.node_id, block_key);
    if (cnt < params_.replication_hot_threshold) {
        return false;
    }
    // gate 4: caller 节点容量允许
    if (ctx.get_node_metrics) {
        const NodeMetrics *m = ctx.get_node_metrics(ctx.caller_node.node_id);
        if (m != nullptr) {
            const double thr = params_.caller_capacity_threshold - params_.caller_capacity_buffer;
            if (m->load_ratio > thr) {
                return false;
            }
        }
        // metrics 缺失视为 permissive（§5.2）
    }
    return true;
}

} // namespace kv_cache_manager
