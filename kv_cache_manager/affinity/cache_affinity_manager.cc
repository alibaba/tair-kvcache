#include "kv_cache_manager/affinity/cache_affinity_manager.h"

#include <fstream>
#include <sstream>
#include <utility>

namespace kv_cache_manager {

namespace {
void SetError(std::string *out, std::string msg) {
    if (out != nullptr) {
        *out = std::move(msg);
    }
}
} // namespace

bool CacheAffinityManager::LoadProcessStrategyFromJsonString(const std::string &json, std::string *error_msg) {
    auto parsed = Strategy::ParseJsonString(json, error_msg);
    if (!parsed) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mux_);
    process_strategy_ = std::move(parsed);
    return true;
}

bool CacheAffinityManager::LoadProcessStrategyFromJsonFile(const std::string &path, std::string *error_msg) {
    std::ifstream f(path);
    if (!f.is_open()) {
        SetError(error_msg, "failed to open strategy file: " + path);
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return LoadProcessStrategyFromJsonString(ss.str(), error_msg);
}

void CacheAffinityManager::UpsertNodeMetrics(const NodeMetrics &metrics) {
    std::lock_guard<std::mutex> lock(mux_);
    nodes_[metrics.node_id] = metrics;
}

void CacheAffinityManager::RemoveNode(const std::string &node_id) {
    std::lock_guard<std::mutex> lock(mux_);
    nodes_.erase(node_id);
}

std::vector<NodeMetrics> CacheAffinityManager::SnapshotNodes() const {
    std::lock_guard<std::mutex> lock(mux_);
    std::vector<NodeMetrics> out;
    out.reserve(nodes_.size());
    for (const auto &kv : nodes_) {
        out.push_back(kv.second);
    }
    return out;
}

std::shared_ptr<Strategy> CacheAffinityManager::ParseOrCacheLocked(const std::string &json) const {
    auto it = parsed_strategy_cache_.find(json);
    if (it != parsed_strategy_cache_.end()) {
        return it->second;
    }
    auto parsed = Strategy::ParseJsonString(json, nullptr);
    if (!parsed) {
        return nullptr;
    }
    std::shared_ptr<Strategy> shared = std::move(parsed);
    parsed_strategy_cache_.emplace(json, shared);
    return shared;
}

ErrorCode CacheAffinityManager::Resolve(const ResolveContext &ctx,
                                        const std::vector<std::string> &candidates,
                                        WriteHints &out_hints) const {
    out_hints = WriteHints{};

    std::lock_guard<std::mutex> lock(mux_);

    // Priority chain: instance > instance_group > process. The first override
    // JSON whose parse succeeds wins; parse failures fall through to the next
    // tier.
    std::shared_ptr<Strategy> chosen;

    if (!ctx.instance_strategy_json.empty()) {
        chosen = ParseOrCacheLocked(ctx.instance_strategy_json);
    }
    if (!chosen && !ctx.instance_group_strategy_json.empty()) {
        chosen = ParseOrCacheLocked(ctx.instance_group_strategy_json);
    }
    if (!chosen) {
        chosen = process_strategy_;
    }
    if (!chosen) {
        // No strategy configured at any tier: degrade silently.
        return EC_OK;
    }

    // Build a candidate-only metrics view via in-place lookup; nodes_ is
    // protected by mux_, which is held for the entire Resolve call.
    auto find_metrics = [this](const std::string &id) -> const NodeMetrics * {
        auto it = nodes_.find(id);
        return it != nodes_.end() ? &it->second : nullptr;
    };

    auto result = chosen->Apply(candidates, find_metrics, ctx.caller_node_ip, ctx.trace_id);
    if (result.status == Strategy::Status::kAbort) {
        return EC_ERROR;
    }
    out_hints.preferred_node_ids = std::move(result.nodes);
    return EC_OK;
}

} // namespace kv_cache_manager
