#include "kv_cache_manager/affinity/cache_affinity_manager.h"

#include <chrono>
#include <fstream>
#include <sstream>
#include <utility>

#include "kv_cache_manager/affinity/local_replica_strategy.h"
#include "kv_cache_manager/affinity/noop_strategy.h"
#include "kv_cache_manager/affinity/strategy_factory.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"

namespace kv_cache_manager {

CacheAffinityManager::~CacheAffinityManager() { StopMetricsPullLoop(); }

namespace {
void SetError(std::string *out, std::string msg) {
    if (out != nullptr) {
        *out = std::move(msg);
    }
}
} // namespace

bool CacheAffinityManager::LoadProcessStrategyFromJsonString(const std::string &json, std::string *error_msg) {
    std::string aff_err;
    auto aff = StrategyFactory::ParseJsonString(json, &sketch_, &suppressor_, &aff_err);
    if (!aff) {
        if (error_msg != nullptr) {
            *error_msg = aff_err.empty() ? std::string("strategy json parse failed") : aff_err;
        }
        return false;
    }
    std::lock_guard<std::mutex> lock(mux_);
    process_affinity_strategy_ = std::move(aff);
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

std::shared_ptr<AffinityStrategy> CacheAffinityManager::ParseOrCacheAffinityLocked(const std::string &json) const {
    auto it = affinity_strategy_cache_.find(json);
    if (it != affinity_strategy_cache_.end()) {
        return it->second;
    }
    auto parsed = StrategyFactory::ParseJsonString(json, &sketch_, &suppressor_, nullptr);
    if (!parsed) {
        return nullptr;
    }
    affinity_strategy_cache_.emplace(json, parsed);
    return parsed;
}

std::shared_ptr<AffinityStrategy>
CacheAffinityManager::GetStrategy(const std::string &instance_strategy_json,
                                  const std::string &instance_group_strategy_json) const {
    // Global kill-switch: bypass all JSON configs and return Noop.
    static const auto kNoop = std::make_shared<NoopAffinityStrategy>();
    if (globally_disabled_.load(std::memory_order_relaxed)) {
        return kNoop;
    }
    std::lock_guard<std::mutex> lock(mux_);
    // Priority chain: instance > instance_group > process. The first override
    // JSON whose parse succeeds wins; parse failures fall through to the next
    // tier.
    std::shared_ptr<AffinityStrategy> chosen;
    if (!instance_strategy_json.empty()) {
        chosen = ParseOrCacheAffinityLocked(instance_strategy_json);
    }
    if (!chosen && !instance_group_strategy_json.empty()) {
        chosen = ParseOrCacheAffinityLocked(instance_group_strategy_json);
    }
    if (!chosen) {
        chosen = process_affinity_strategy_;
    }
    if (!chosen) {
        // No strategy configured at any tier: degrade silently.
        chosen = kNoop;
    }
    return chosen;
}

StrategyContext CacheAffinityManager::BuildStrategyContext(const AffinityResolveContext &ctx) const {
    StrategyContext sctx;
    sctx.caller_node_id = ctx.caller_node_id;
    sctx.caller_supernode_id = ctx.caller_supernode_id;
    sctx.instance_id = ctx.instance_id;
    sctx.instance_group_name = ctx.instance_group_name;
    sctx.trace_id = ctx.trace_id;
    sctx.get_node_metrics = MakeNodeMetricsAccessor();
    return sctx;
}

WriteDecision CacheAffinityManager::ResolveWrite(const AffinityResolveContext &ctx) {
    auto nodes = SnapshotNodes();
    if (nodes.empty()) {
        return WriteDecision{AffinityStatus::kOk, {}};
    }
    std::vector<std::string> candidates;
    candidates.reserve(nodes.size());
    for (const auto &n : nodes) {
        candidates.push_back(n.node_id);
    }
    auto strategy = GetStrategy(ctx.instance_strategy_json, ctx.group_strategy_json);
    auto sctx = BuildStrategyContext(ctx);
    return strategy->ResolveWrite(candidates, sctx);
}

ReadDecision CacheAffinityManager::ResolveRead(const ReadRequest &req, const AffinityResolveContext &ctx) {
    auto strategy = GetStrategy(ctx.instance_strategy_json, ctx.group_strategy_json);
    auto sctx = BuildStrategyContext(ctx);
    return strategy->ResolveRead(req, sctx);
}

std::unordered_set<std::string> CacheAffinityManager::ResolveEviction(const AffinityResolveContext &ctx) {
    if (globally_disabled_.load(std::memory_order_relaxed)) {
        return {};
    }

    auto strategy = GetStrategy(ctx.instance_strategy_json, ctx.group_strategy_json);

    StrategyContext sctx;
    {
        std::lock_guard<std::mutex> lock(mux_);

        // Reset evicted-bytes accumulator when NodeMetrics refreshes
        int64_t max_updated_at = 0;
        for (const auto &kv : nodes_) {
            if (kv.second.updated_at_us > max_updated_at) {
                max_updated_at = kv.second.updated_at_us;
            }
        }
        if (max_updated_at > node_metrics_last_reset_us_) {
            node_evicted_bytes_.clear();
            node_metrics_last_reset_us_ = max_updated_at;
        }

        sctx.all_nodes.reserve(nodes_.size());
        for (const auto &kv : nodes_) {
            sctx.all_nodes.push_back(kv.second);
        }
        sctx.evicted_bytes = node_evicted_bytes_;
    }

    return strategy->ResolveEviction(sctx);
}

void CacheAffinityManager::ReportEvictedBytes(const std::string &node_id, int64_t bytes) {
    std::lock_guard<std::mutex> lock(mux_);
    node_evicted_bytes_[node_id] += bytes;
}

std::function<const NodeMetrics *(const std::string &)> CacheAffinityManager::MakeNodeMetricsAccessor() const {
    auto snapshot = std::make_shared<std::vector<NodeMetrics>>(SnapshotNodes());
    return [snapshot](const std::string &id) -> const NodeMetrics * {
        for (const auto &n : *snapshot) {
            if (n.node_id == id) {
                return &n;
            }
        }
        return nullptr;
    };
}

void CacheAffinityManager::StartMetricsPullLoop(std::shared_ptr<DataStorageManager> dsm, uint32_t interval_seconds) {
    if (metrics_thread_.joinable()) {
        return; // idempotent
    }
    metrics_dsm_ = std::move(dsm);
    metrics_interval_seconds_ = interval_seconds == 0 ? 1 : interval_seconds;
    metrics_stop_.store(false);
    metrics_thread_ = std::thread([this]() {
        while (!metrics_stop_.load()) {
            if (metrics_dsm_) {
                auto backends = metrics_dsm_->GetAvailableStorages();
                for (const auto &backend : backends) {
                    if (!backend) {
                        continue;
                    }
                    auto snap = backend->SnapshotPerNodeMetrics();
                    for (auto &m : snap) {
                        if (m.node_id.empty()) {
                            continue;
                        }
                        UpsertNodeMetrics(m);
                    }
                }
            }
            std::unique_lock<std::mutex> lock(metrics_cv_mu_);
            metrics_cv_.wait_for(
                lock, std::chrono::seconds(metrics_interval_seconds_), [this]() { return metrics_stop_.load(); });
        }
        KVCM_LOG_INFO("affinity metrics pull loop exited");
    });
    KVCM_LOG_INFO("affinity metrics pull loop started (interval=%us)", metrics_interval_seconds_);
}

void CacheAffinityManager::StopMetricsPullLoop() {
    if (!metrics_thread_.joinable()) {
        return;
    }
    metrics_stop_.store(true);
    metrics_cv_.notify_all();
    metrics_thread_.join();
    metrics_dsm_.reset();
}

} // namespace kv_cache_manager
