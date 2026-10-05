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
    bool all_local = !req.spec_candidates.empty();
    std::vector<ReplicationSourceSpec> sources;
    bool complete_sources = true;
    for (auto &kv : req.spec_candidates) {
        const auto &spec_name = kv.first;
        const auto &cands = kv.second;
        const LocationSpec *picked = PickLocalSpec(cands, ctx);
        dec.picked_specs[spec_name] = picked;
        all_local = all_local && picked != nullptr && !ctx.caller_node.node_id.empty() &&
                    picked->node_id() == ctx.caller_node.node_id;
        if (picked == nullptr || picked->uri().empty()) {
            complete_sources = false;
        } else {
            sources.push_back({spec_name, picked->uri()});
        }
    }

    // ==== Step 2: on_miss 路径 —— 决定是否产出 ReplicationHint ====
    if (!params_.enable_on_miss || ctx.caller_node.node_id.empty()) {
        return dec;
    }
    if (all_local || !complete_sources || sources.empty()) {
        return dec;
    }
    // Gate before frequency/suppression updates so an old client cannot consume
    // the emission window of a newly upgraded client on the same node.
    if (sources.size() > 1 && !(ctx.caller_node.replication_capabilities & kReplicationNamedSpecs)) return dec;
    if (req.winner_tier == nullptr) {
        return dec;
    }
    // 喂 sketch（机制层，永远 active；只在远端命中时累加）
    if (params_.sketch != nullptr) {
        params_.sketch->Observe(ctx.caller_node.node_id, req.block_key, ctx.instance_id);
    }
    if (ShouldEmitReplicationHint(req.block_key, /*has_local=*/false, req.winner_tier, ctx)) {
        const bool allow = params_.suppressor == nullptr ||
                           params_.suppressor->TryEmit(req.block_key, ctx.caller_node.node_id,
                                                      params_.suppression_window_ms, ctx.instance_id);
        if (allow) {
            auto h = std::make_unique<ReplicationHintSideEffect>();
            h->block_key = req.block_key;
            // Legacy SDKs can consume only single-spec hints. Multi-spec clients
            // must use the named sources, including already-local components.
            if (sources.size() == 1) {
                h->source_uri = sources.front().uri;
            }
            h->source_specs = std::move(sources);
            h->target_node_id = ctx.caller_node.node_id;
            dec.side_effects.push_back(std::move(h));
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
            const double total = node.total_bytes > 0 ? static_cast<double>(node.total_bytes) :
                                node.free_bytes / std::max(1.0 - node.load_ratio, 0.01);
            if (total > 0) {
                estimated_load -= static_cast<double>(it->second) / total;
            }
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
    // gate 2: 还没有完整的本地副本（ResolveRead 已检过）
    if (has_local_in_picked) {
        return false;
    }
    // gate 3: 频率超阈值
    if (params_.sketch == nullptr) {
        return false; // 没 sketch 拿不到计数，保守不发
    }
    uint32_t cnt = params_.sketch->RemoteCount(ctx.caller_node.node_id, block_key, ctx.instance_id);
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
