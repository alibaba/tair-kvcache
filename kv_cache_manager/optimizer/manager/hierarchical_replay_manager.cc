#include "kv_cache_manager/optimizer/manager/hierarchical_replay_manager.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/trace_loader/standard_trace_loader.h"
#include "kv_cache_manager/optimizer/trace_loader/trace_util.h"

namespace kv_cache_manager {
namespace {

std::unordered_map<std::string, OptInstanceConfig> CollectInstances(const OptimizerConfig &config) {
    std::unordered_map<std::string, OptInstanceConfig> instances;
    for (const auto &group : config.instance_groups()) {
        for (auto instance : group.instances()) {
            instance.set_instance_group_name(group.group_name());
            instances.emplace(instance.instance_id(), instance);
        }
    }
    return instances;
}

struct StoragePoolInfo {
    size_t block_size = 0;
    int64_t bytes_per_token = 0;
};

std::unordered_map<std::string, StoragePoolInfo> CollectStoragePools(const HierarchicalStoragePoolConfig &config) {
    std::unordered_map<std::string, StoragePoolInfo> pools;
    for (const auto &pool : config.pools()) {
        pools.emplace(pool.pool_id(),
                      StoragePoolInfo{static_cast<size_t>(pool.model().block_size()), pool.model().bytes_per_token()});
    }
    return pools;
}

size_t PositiveBlockSizeOrZero(const OptInstanceConfig &instance) {
    return instance.block_size() > 0 ? static_cast<size_t>(instance.block_size()) : 0;
}

bool ValidateEngineInstanceIsolation(const OptimizerConfig &config) {
    for (const auto &group : config.instance_groups()) {
        if (group.instances().size() != 1) {
            KVCM_LOG_ERROR("Hierarchical replay engine_config group '%s' has %zu instances; each engine instance must "
                           "use a dedicated group so L1/L2 capacities are independent.",
                           group.group_name().c_str(),
                           group.instances().size());
            return false;
        }
    }
    return true;
}

void MarkIndices(const std::vector<size_t> &indices, std::vector<bool> *mask) {
    if (mask == nullptr) {
        return;
    }
    for (const size_t idx : indices) {
        if (idx < mask->size()) {
            (*mask)[idx] = true;
        }
    }
}

std::vector<size_t> IndicesFromMask(const std::vector<bool> &mask) {
    std::vector<size_t> indices;
    for (size_t idx = 0; idx < mask.size(); ++idx) {
        if (mask[idx]) {
            indices.push_back(idx);
        }
    }
    return indices;
}

} // namespace

HierarchicalReplayManager::HierarchicalReplayManager(const HierarchicalReplayConfig &config) : config_(config) {}

std::string HierarchicalReplayManager::TierGlobalTracker::ScopeKey(const std::string &cluster_id,
                                                                   const std::string &tier) {
    return cluster_id + "\x1f" + tier;
}

void HierarchicalReplayManager::TierGlobalTracker::Add(const std::string &cluster_id,
                                                       const std::string &tier,
                                                       int64_t key,
                                                       const std::string &infer_id) {
    holders_[ScopeKey(cluster_id, tier)][key].insert(infer_id);
}

void HierarchicalReplayManager::TierGlobalTracker::Remove(const std::string &cluster_id,
                                                          const std::string &tier,
                                                          int64_t key,
                                                          const std::string &infer_id) {
    auto scope_it = holders_.find(ScopeKey(cluster_id, tier));
    if (scope_it == holders_.end()) {
        return;
    }
    auto key_it = scope_it->second.find(key);
    if (key_it == scope_it->second.end()) {
        return;
    }
    key_it->second.erase(infer_id);
    if (key_it->second.empty()) {
        scope_it->second.erase(key_it);
    }
}

void HierarchicalReplayManager::TierGlobalTracker::RemoveFromAllTiers(const std::string &cluster_id,
                                                                      int64_t key,
                                                                      const std::string &infer_id) {
    const std::string prefix = cluster_id + "\x1f";
    for (auto &scope : holders_) {
        if (scope.first.rfind(prefix, 0) != 0) {
            continue;
        }
        auto key_it = scope.second.find(key);
        if (key_it == scope.second.end()) {
            continue;
        }
        key_it->second.erase(infer_id);
        if (key_it->second.empty()) {
            scope.second.erase(key_it);
        }
    }
}

bool HierarchicalReplayManager::TierGlobalTracker::Contains(const std::string &cluster_id,
                                                            const std::string &tier,
                                                            int64_t key,
                                                            const std::string &infer_id) const {
    auto scope_it = holders_.find(ScopeKey(cluster_id, tier));
    if (scope_it == holders_.end()) {
        return false;
    }
    auto key_it = scope_it->second.find(key);
    return key_it != scope_it->second.end() && key_it->second.count(infer_id) > 0;
}

bool HierarchicalReplayManager::Init() {
    if (!ValidateAndBuildMappings()) {
        return false;
    }
    write_delay_ns_ = config_.trace_replay_config().write_delay_ns();
    if (write_delay_ns_ <= 0) {
        throw std::runtime_error("trace_replay.write_delay_ns must be positive");
    }
    pending_writes_ = {};
    next_pending_write_sequence_ = 0;

    const bool enable_lifecycle_tracking = config_.enable_lifecycle_tracking();
    engine_manager_ = std::make_unique<OptimizerManager>(
        config_.engine_config(), enable_lifecycle_tracking, false, HitRatePerspective::ENGINE_LOCAL);
    if (!engine_manager_->Init()) {
        KVCM_LOG_ERROR("Hierarchical replay failed to initialize engine manager.");
        return false;
    }

    storage_pool_manager_ = std::make_unique<HashStoragePoolManager>(config_.storage_pool(), enable_lifecycle_tracking);
    if (!storage_pool_manager_->Init()) {
        KVCM_LOG_ERROR("Hierarchical replay failed to initialize storage pool manager.");
        return false;
    }
    return true;
}

bool HierarchicalReplayManager::ValidateAndBuildMappings() {
    engine_to_storage_pool_.clear();
    engine_to_cluster_.clear();
    cluster_infer_ids_.clear();
    cluster_p2p_read_flows_.clear();
    engine_read_query_type_.clear();
    engine_storage_pool_flow_.clear();
    engine_block_size_.clear();
    sorted_engine_instance_ids_.clear();

    const auto engine_instances = CollectInstances(config_.engine_config());
    const auto storage_pools = CollectStoragePools(config_.storage_pool());
    if (engine_instances.empty()) {
        KVCM_LOG_ERROR("Hierarchical replay engine_config has no instances.");
        return false;
    }
    if (storage_pools.empty()) {
        KVCM_LOG_ERROR("Hierarchical replay storage_pool has no pools.");
        return false;
    }
    if (!ValidateEngineInstanceIsolation(config_.engine_config())) {
        return false;
    }

    if (!config_.infer_clusters().empty()) {
        for (size_t cluster_idx = 0; cluster_idx < config_.infer_clusters().size(); ++cluster_idx) {
            const auto &cluster = config_.infer_clusters()[cluster_idx];
            const std::string cluster_id = "infer_cluster_" + std::to_string(cluster_idx);
            cluster_infer_ids_[cluster_id] = cluster.infer_ids();
            cluster_p2p_read_flows_[cluster_id] = cluster.p2p_read_flows();
            for (const auto &infer_id : cluster.infer_ids()) {
                if (!engine_to_cluster_.emplace(infer_id, cluster_id).second) {
                    KVCM_LOG_ERROR("engine instance appears in more than one infer cluster: %s", infer_id.c_str());
                    return false;
                }
            }
        }
    } else {
        std::unordered_map<std::string, std::string> storage_pool_clusters;
        for (const auto &mapping : config_.engine_to_storage_pool()) {
            const auto [cluster_it, inserted] = storage_pool_clusters.emplace(
                mapping.storage_pool_id(), "manual_cluster_" + std::to_string(storage_pool_clusters.size()));
            const std::string &cluster_id = cluster_it->second;
            cluster_infer_ids_[cluster_id].push_back(mapping.engine_instance_id());
            engine_to_cluster_[mapping.engine_instance_id()] = cluster_id;
            if (inserted) {
                cluster_p2p_read_flows_[cluster_id] = {};
            }
        }
    }

    for (const auto &mapping : config_.engine_to_storage_pool()) {
        const auto &engine_instance_id = mapping.engine_instance_id();
        const auto &storage_pool_id = mapping.storage_pool_id();
        auto engine_it = engine_instances.find(engine_instance_id);
        if (engine_it == engine_instances.end()) {
            KVCM_LOG_ERROR("engine_to_storage_pool references unknown engine instance: %s", engine_instance_id.c_str());
            return false;
        }
        auto storage_pool_it = storage_pools.find(storage_pool_id);
        if (storage_pool_it == storage_pools.end()) {
            KVCM_LOG_ERROR("engine_to_storage_pool references unknown storage pool: %s", storage_pool_id.c_str());
            return false;
        }
        if (engine_to_storage_pool_.find(engine_instance_id) != engine_to_storage_pool_.end()) {
            KVCM_LOG_ERROR("engine instance is mapped more than once: %s", engine_instance_id.c_str());
            return false;
        }
        if (!IsSupportedQueryType(mapping.engine_read_query_type())) {
            KVCM_LOG_ERROR("engine_to_storage_pool has invalid engine_read_query_type for engine=%s: %s",
                           engine_instance_id.c_str(),
                           mapping.engine_read_query_type().c_str());
            return false;
        }

        const size_t engine_block_size = PositiveBlockSizeOrZero(engine_it->second);
        const size_t storage_pool_block_size = storage_pool_it->second.block_size;
        if (engine_block_size == 0 || storage_pool_block_size == 0 || engine_block_size != storage_pool_block_size) {
            KVCM_LOG_ERROR("engine/storage_pool block_size mismatch for engine=%s storage_pool=%s",
                           engine_instance_id.c_str(),
                           storage_pool_id.c_str());
            return false;
        }
        if (engine_it->second.bytes_per_token() != storage_pool_it->second.bytes_per_token) {
            KVCM_LOG_ERROR("engine/storage_pool bytes_per_token mismatch for engine=%s storage_pool=%s",
                           engine_instance_id.c_str(),
                           storage_pool_id.c_str());
            return false;
        }

        engine_to_storage_pool_[engine_instance_id] = storage_pool_id;
        engine_read_query_type_[engine_instance_id] = mapping.engine_read_query_type();
        engine_storage_pool_flow_[engine_instance_id] = mapping.storage_pool_flow();
        engine_block_size_[engine_instance_id] = engine_block_size;
        sorted_engine_instance_ids_.push_back(engine_instance_id);
    }

    if (engine_to_storage_pool_.size() != engine_instances.size()) {
        KVCM_LOG_ERROR("Every engine instance must have exactly one engine_to_storage_pool mapping.");
        return false;
    }
    std::sort(sorted_engine_instance_ids_.begin(), sorted_engine_instance_ids_.end());
    return true;
}

const std::string &HierarchicalReplayManager::StoragePoolForEngine(const std::string &engine_instance_id) const {
    auto it = engine_to_storage_pool_.find(engine_instance_id);
    if (it == engine_to_storage_pool_.end()) {
        throw std::runtime_error("No storage pool mapping for engine instance: " + engine_instance_id);
    }
    return it->second;
}

const std::string &
HierarchicalReplayManager::EngineReadQueryTypeForEngine(const std::string &engine_instance_id) const {
    auto it = engine_read_query_type_.find(engine_instance_id);
    if (it == engine_read_query_type_.end()) {
        throw std::runtime_error("No engine read query type for engine instance: " + engine_instance_id);
    }
    return it->second;
}

const StoragePoolFlowConfig &
HierarchicalReplayManager::StoragePoolFlowForEngine(const std::string &engine_instance_id) const {
    auto it = engine_storage_pool_flow_.find(engine_instance_id);
    if (it == engine_storage_pool_flow_.end()) {
        throw std::runtime_error("No storage pool flow for engine instance: " + engine_instance_id);
    }
    return it->second;
}

const std::string &HierarchicalReplayManager::ClusterForEngine(const std::string &engine_instance_id) const {
    auto it = engine_to_cluster_.find(engine_instance_id);
    if (it == engine_to_cluster_.end()) {
        throw std::runtime_error("No infer cluster mapping for engine instance: " + engine_instance_id);
    }
    return it->second;
}

const std::vector<std::string> &HierarchicalReplayManager::InferIdsForCluster(const std::string &cluster_id) const {
    auto it = cluster_infer_ids_.find(cluster_id);
    if (it == cluster_infer_ids_.end()) {
        throw std::runtime_error("No infer ids for cluster: " + cluster_id);
    }
    return it->second;
}

const std::vector<P2PReadFlowConfig> &
HierarchicalReplayManager::P2PReadFlowsForCluster(const std::string &cluster_id) const {
    auto it = cluster_p2p_read_flows_.find(cluster_id);
    if (it == cluster_p2p_read_flows_.end()) {
        throw std::runtime_error("No P2P read flow mapping for cluster: " + cluster_id);
    }
    return it->second;
}

void HierarchicalReplayManager::ApplyEngineTierEvents(const std::vector<TierFlowKeyEvent> &events) {
    for (const auto &event : events) {
        auto cluster_it = engine_to_cluster_.find(event.instance_id);
        if (cluster_it == engine_to_cluster_.end()) {
            continue;
        }
        const std::string &cluster_id = cluster_it->second;
        if (event.kind == TierFlowEventKind::ENTER_TIER && !event.to_tier.empty()) {
            p2p_tracker_.Add(cluster_id, event.to_tier, event.block_key, event.instance_id);
        } else if (event.kind == TierFlowEventKind::LEAVE_TIER && !event.from_tier.empty()) {
            p2p_tracker_.Remove(cluster_id, event.from_tier, event.block_key, event.instance_id);
        } else if (event.kind == TierFlowEventKind::FINAL_EVICT) {
            p2p_tracker_.RemoveFromAllTiers(cluster_id, event.block_key, event.instance_id);
        }
    }
}

HierarchicalReplayManager::P2PReadResult
HierarchicalReplayManager::SelectP2PPeer(const std::string &engine_instance_id,
                                         const std::string &cluster_id,
                                         const P2PReadFlowConfig &flow,
                                         const std::vector<int64_t> &block_ids,
                                         const std::vector<bool> &satisfied_mask) const {
    std::vector<size_t> missing_indices;
    for (size_t idx = 0; idx < block_ids.size(); ++idx) {
        if (idx >= satisfied_mask.size() || !satisfied_mask[idx]) {
            missing_indices.push_back(idx);
        }
    }
    if (missing_indices.empty()) {
        return {};
    }

    size_t best_len = 0;
    std::string best_peer;
    for (const auto &peer_id : InferIdsForCluster(cluster_id)) {
        if (peer_id == engine_instance_id) {
            continue;
        }
        size_t match_len = 0;
        while (match_len < missing_indices.size()) {
            const size_t block_idx = missing_indices[match_len];
            if (!p2p_tracker_.Contains(cluster_id, flow.tier(), block_ids[block_idx], peer_id)) {
                break;
            }
            ++match_len;
        }
        if (match_len > best_len) {
            best_len = match_len;
            best_peer = peer_id;
        }
    }
    if (best_len == 0) {
        return {};
    }

    P2PReadResult result;
    result.peer_infer_id = std::move(best_peer);
    result.hit_indices.assign(missing_indices.begin(), missing_indices.begin() + best_len);
    return result;
}

void HierarchicalReplayManager::FillEngineFromHitIndices(const std::string &engine_instance_id,
                                                         const std::string &storage_pool_id,
                                                         const std::string &trace_id,
                                                         int64_t timestamp,
                                                         const std::vector<int64_t> &block_ids,
                                                         const std::vector<size_t> &hit_indices,
                                                         const StoragePoolFlowConfig &flow) {
    if (hit_indices.empty()) {
        return;
    }
    const size_t promote_path_len = *std::max_element(hit_indices.begin(), hit_indices.end()) + 1;
    std::vector<int64_t> promote_path(block_ids.begin(), block_ids.begin() + promote_path_len);
    const auto promote_res =
        engine_manager_->FillCachePathWithTtlUs(engine_instance_id, trace_id, timestamp, promote_path, hit_indices, 0);
    ApplyEngineTierEvents(promote_res.tier_flow_events);
    ApplyStoragePoolCascadingEvictions(storage_pool_id, trace_id, timestamp, 0, flow, promote_res.evicted_keys);
}

HierarchicalReplayManager::P2PReadResult
HierarchicalReplayManager::ApplyP2PReadFlow(const std::string &engine_instance_id,
                                            const std::string &storage_pool_id,
                                            const std::string &trace_id,
                                            int64_t timestamp,
                                            const std::vector<int64_t> &block_ids,
                                            const P2PReadFlowConfig &flow,
                                            const StoragePoolFlowConfig &storage_pool_flow,
                                            std::vector<bool> *satisfied_mask) {
    if (satisfied_mask == nullptr || block_ids.empty()) {
        return {};
    }
    const std::string &cluster_id = ClusterForEngine(engine_instance_id);
    P2PReadResult result = SelectP2PPeer(engine_instance_id, cluster_id, flow, block_ids, *satisfied_mask);
    if (result.hit_indices.empty()) {
        return result;
    }

    if (flow.peer_read_touch_enabled()) {
        std::vector<int64_t> peer_touch_keys;
        peer_touch_keys.reserve(result.hit_indices.size());
        for (const size_t idx : result.hit_indices) {
            if (idx < block_ids.size()) {
                peer_touch_keys.push_back(block_ids[idx]);
            }
        }
        engine_manager_->TouchCacheKeysAtTier(result.peer_infer_id, peer_touch_keys, flow.tier(), timestamp);
    }

    FillEngineFromHitIndices(
        engine_instance_id, storage_pool_id, trace_id, timestamp, block_ids, result.hit_indices, storage_pool_flow);
    MarkIndices(result.hit_indices, satisfied_mask);
    return result;
}

void HierarchicalReplayManager::DirectRun() {
    auto traces = StandardTraceLoader::LoadFromFile(config_.trace_file_path(), config_.trace_replay_config().mode());
    TraceTimeSorter::SortTracesByTimestamp(traces);
    if (config_.infer_scheduling_strategy() == "prefix_hit") {
        RunTracesWithPrefixHitScheduling(traces);
        return;
    }
    ScheduleTraces(traces);
    RunTraces(traces);
}

void HierarchicalReplayManager::ScheduleTraces(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) const {
    if (config_.infer_scheduling_strategy() == "preserve_trace" ||
        config_.infer_scheduling_strategy() == "prefix_hit") {
        return;
    }
    if (config_.infer_scheduling_strategy() != "round_robin") {
        throw std::runtime_error("Unknown infer_scheduling_strategy: " + config_.infer_scheduling_strategy());
    }
    if (sorted_engine_instance_ids_.empty()) {
        throw std::runtime_error("round_robin scheduling requires at least one engine instance");
    }

    size_t request_idx = 0;
    std::string current_engine_instance_id = sorted_engine_instance_ids_.front();
    for (auto &trace : traces) {
        if (!trace) {
            continue;
        }
        if (std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
            current_engine_instance_id = sorted_engine_instance_ids_[request_idx % sorted_engine_instance_ids_.size()];
            request_idx++;
        } else if (request_idx == 0) {
            current_engine_instance_id = sorted_engine_instance_ids_.front();
        }
        trace->set_instance_id(current_engine_instance_id);
    }
}

void HierarchicalReplayManager::RunTracesWithPrefixHitScheduling(
    const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
    if (sorted_engine_instance_ids_.empty()) {
        throw std::runtime_error("prefix_hit scheduling requires at least one engine instance");
    }

    pending_writes_ = {};
    next_pending_write_sequence_ = 0;
    size_t request_idx = 0;
    std::string current_engine_instance_id = sorted_engine_instance_ids_.front();
    for (const auto &trace : traces) {
        if (!trace) {
            continue;
        }
        FlushPendingWritesThrough(trace->timestamp_ns());
        if (auto request_trace = std::dynamic_pointer_cast<RequestSchemaTrace>(trace)) {
            current_engine_instance_id =
                ChoosePrefixHitEngineInstance(request_trace->keys(), request_trace->timestamp_ns(), request_idx);
            request_trace->set_instance_id(current_engine_instance_id);
            request_idx++;
        } else if (auto get_trace = std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
            current_engine_instance_id =
                ChoosePrefixHitEngineInstance(get_trace->keys(), get_trace->timestamp_ns(), request_idx);
            get_trace->set_instance_id(current_engine_instance_id);
            request_idx++;
        } else if (auto write_trace = std::dynamic_pointer_cast<WriteCacheSchemaTrace>(trace)) {
            write_trace->set_instance_id(current_engine_instance_id);
        }
        RunTrace(trace);
    }
    FlushAllPendingWrites();
}

std::string HierarchicalReplayManager::ChoosePrefixHitEngineInstance(const std::vector<int64_t> &block_ids,
                                                                     int64_t timestamp,
                                                                     size_t request_idx) const {
    size_t best_match = 0;
    std::vector<std::string> candidates;
    for (const auto &instance_id : sorted_engine_instance_ids_) {
        const size_t match = engine_manager_->PrefixMatchCount(instance_id, block_ids, timestamp);
        if (match > best_match) {
            best_match = match;
            candidates.clear();
            candidates.push_back(instance_id);
        } else if (match == best_match) {
            candidates.push_back(instance_id);
        }
    }
    if (candidates.empty()) {
        throw std::runtime_error("prefix_hit scheduling has no engine candidates");
    }
    if (candidates.size() == 1) {
        return candidates.front();
    }
    return candidates[request_idx % candidates.size()];
}

void HierarchicalReplayManager::RunTraces(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
    pending_writes_ = {};
    next_pending_write_sequence_ = 0;
    for (const auto &trace : traces) {
        if (trace) {
            FlushPendingWritesThrough(trace->timestamp_ns());
        }
        RunTrace(trace);
    }
    FlushAllPendingWrites();
}

void HierarchicalReplayManager::RunTrace(const std::shared_ptr<OptimizerSchemaTrace> &trace) {
    if (!trace) {
        return;
    }
    if (auto request_trace = std::dynamic_pointer_cast<RequestSchemaTrace>(trace)) {
        HandleRequest(*request_trace);
    } else if (auto get_trace = std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
        GetCacheLocation(get_trace->instance_id(),
                         get_trace->trace_id(),
                         get_trace->timestamp_ns(),
                         get_trace->keys(),
                         get_trace->input_len(),
                         get_trace->query_type());
    } else if (auto write_trace = std::dynamic_pointer_cast<WriteCacheSchemaTrace>(trace)) {
        WriteCacheWithTtlUs(write_trace->instance_id(),
                            write_trace->trace_id(),
                            write_trace->timestamp_ns(),
                            write_trace->keys(),
                            write_trace->ttl_us());
    } else {
        KVCM_LOG_WARN("Hierarchical replay skips unknown trace type.");
    }
}

void HierarchicalReplayManager::HandleRequest(const RequestSchemaTrace &trace) {
    if (!IsSupportedQueryType(trace.query_type())) {
        throw std::runtime_error("Unsupported hierarchical query_type: " + trace.query_type());
    }
    GetCacheLocation(trace.instance_id(),
                     trace.trace_id(),
                     trace.timestamp_ns(),
                     trace.keys(),
                     trace.input_len(),
                     trace.query_type());
    ScheduleRequestWrite(trace);
}

void HierarchicalReplayManager::ScheduleRequestWrite(const RequestSchemaTrace &trace) {
    if (trace.timestamp_ns() > std::numeric_limits<int64_t>::max() - write_delay_ns_) {
        throw std::runtime_error("request write timestamp overflows int64: instance_id=" + trace.instance_id() +
                                 ", trace_id=" + trace.trace_id());
    }

    WriteCacheSchemaTrace write_trace;
    write_trace.set_instance_id(trace.instance_id());
    write_trace.set_trace_id(trace.trace_id() + ":write");
    write_trace.set_timestamp_ns(trace.timestamp_ns() + write_delay_ns_);
    write_trace.set_keys(trace.keys());
    write_trace.set_ttl_us(trace.ttl_us());
    pending_writes_.push(
        PendingWrite{write_trace.timestamp_ns(), next_pending_write_sequence_++, std::move(write_trace)});
}

void HierarchicalReplayManager::FlushPendingWritesThrough(int64_t timestamp_ns) {
    while (!pending_writes_.empty() && pending_writes_.top().timestamp_ns <= timestamp_ns) {
        auto pending = pending_writes_.top();
        pending_writes_.pop();
        RunPendingWrite(pending.trace);
    }
}

void HierarchicalReplayManager::FlushAllPendingWrites() {
    while (!pending_writes_.empty()) {
        auto pending = pending_writes_.top();
        pending_writes_.pop();
        RunPendingWrite(pending.trace);
    }
}

void HierarchicalReplayManager::RunPendingWrite(const WriteCacheSchemaTrace &trace) {
    WriteCacheWithTtlUs(trace.instance_id(), trace.trace_id(), trace.timestamp_ns(), trace.keys(), trace.ttl_us());
}

HierarchicalGetCacheLocationRes HierarchicalReplayManager::GetCacheLocation(const std::string &engine_instance_id,
                                                                            const std::string &trace_id,
                                                                            int64_t timestamp,
                                                                            const std::vector<int64_t> &block_ids,
                                                                            int64_t input_len,
                                                                            const std::string &query_type) {
    if (!engine_manager_ || !storage_pool_manager_) {
        throw std::runtime_error("HierarchicalReplayManager is not initialized");
    }
    auto block_size_it = engine_block_size_.find(engine_instance_id);
    if (block_size_it == engine_block_size_.end()) {
        throw std::runtime_error("Unknown engine instance: " + engine_instance_id);
    }

    const std::string &storage_pool_id = StoragePoolForEngine(engine_instance_id);
    const std::string &engine_read_query_type = EngineReadQueryTypeForEngine(engine_instance_id);
    const BlockMask empty_mask = BlockMaskVector{};
    const auto engine_res = engine_manager_->GetCacheLocation(
        engine_instance_id, trace_id, timestamp, block_ids, empty_mask, input_len, true, true, engine_read_query_type);
    ApplyEngineTierEvents(engine_res.tier_flow_events);
    std::vector<size_t> engine_hit_indices;
    for (const size_t idx : engine_res.hit_indices) {
        if (idx < block_ids.size()) {
            engine_hit_indices.push_back(idx);
        }
    }
    const size_t engine_hit_blocks = engine_hit_indices.size();

    const auto &storage_pool_flow = StoragePoolFlowForEngine(engine_instance_id);
    ApplyStoragePoolCascadingEvictions(
        storage_pool_id, trace_id, timestamp, 0, storage_pool_flow, engine_res.evicted_keys);
    std::vector<bool> satisfied_mask(block_ids.size(), false);
    MarkIndices(engine_hit_indices, &satisfied_mask);
    std::vector<size_t> peer_hit_indices;
    for (const auto &flow : P2PReadFlowsForCluster(ClusterForEngine(engine_instance_id))) {
        const auto p2p_read = ApplyP2PReadFlow(engine_instance_id,
                                               storage_pool_id,
                                               trace_id,
                                               timestamp,
                                               block_ids,
                                               flow,
                                               storage_pool_flow,
                                               &satisfied_mask);
        peer_hit_indices.insert(peer_hit_indices.end(), p2p_read.hit_indices.begin(), p2p_read.hit_indices.end());
    }
    const std::vector<size_t> non_storage_hit_indices = IndicesFromMask(satisfied_mask);
    const auto storage_pool_read = ReadStoragePool(engine_instance_id,
                                                   storage_pool_id,
                                                   trace_id,
                                                   timestamp,
                                                   block_ids,
                                                   non_storage_hit_indices,
                                                   input_len,
                                                   query_type,
                                                   storage_pool_flow);
    const size_t storage_pool_hit_blocks = storage_pool_read.hit_blocks;
    MarkIndices(storage_pool_read.hit_indices, &satisfied_mask);

    CombinedReadRecord record;
    record.trace_id = trace_id;
    record.engine_instance_id = engine_instance_id;
    record.storage_pool_id = storage_pool_id;
    record.timestamp_ns = timestamp;
    record.read_blocks = block_ids.size();
    record.engine_hit_blocks = engine_hit_blocks;
    record.peer_hit_blocks = peer_hit_indices.size();
    record.storage_pool_hit_blocks = storage_pool_hit_blocks;
    record.input_tokens = static_cast<size_t>(input_len);
    record.block_size_tokens = block_size_it->second;
    combined_read_records_.push_back(record);

    HierarchicalGetCacheLocationRes res;
    res.trace_id = trace_id;
    res.engine_hit_length = static_cast<int64_t>(engine_hit_blocks);
    res.peer_hit_length = static_cast<int64_t>(peer_hit_indices.size());
    res.storage_pool_hit_length = static_cast<int64_t>(storage_pool_hit_blocks);
    res.total_hit_length = static_cast<int64_t>(engine_hit_blocks + peer_hit_indices.size() + storage_pool_hit_blocks);
    return res;
}

WriteCacheRes HierarchicalReplayManager::WriteCache(const std::string &engine_instance_id,
                                                    const std::string &trace_id,
                                                    int64_t timestamp,
                                                    const std::vector<int64_t> &block_ids,
                                                    int64_t ttl_seconds) {
    const int64_t ttl_us = ttl_seconds > 0 ? ttl_seconds * 1000000 : ttl_seconds;
    return WriteCacheWithTtlUs(engine_instance_id, trace_id, timestamp, block_ids, ttl_us);
}

WriteCacheRes HierarchicalReplayManager::WriteCacheWithTtlUs(const std::string &engine_instance_id,
                                                             const std::string &trace_id,
                                                             int64_t timestamp,
                                                             const std::vector<int64_t> &block_ids,
                                                             int64_t ttl_us) {
    if (!engine_manager_ || !storage_pool_manager_) {
        throw std::runtime_error("HierarchicalReplayManager is not initialized");
    }
    const std::string &storage_pool_id = StoragePoolForEngine(engine_instance_id);

    auto engine_res = engine_manager_->WriteCacheWithTtlUs(engine_instance_id, trace_id, timestamp, block_ids, ttl_us);
    ApplyEngineTierEvents(engine_res.tier_flow_events);
    const auto &storage_pool_flow = StoragePoolFlowForEngine(engine_instance_id);
    ApplyStoragePoolWriteFlow(
        engine_instance_id, storage_pool_id, trace_id, timestamp, ttl_us, storage_pool_flow, engine_res);

    CombinedWriteRecord record;
    record.timestamp_ns = timestamp;
    record.write_blocks = block_ids.size();
    combined_write_records_.push_back(record);
    return engine_res;
}

HashStoragePoolReadResult HierarchicalReplayManager::ReadStoragePool(const std::string &engine_instance_id,
                                                                     const std::string &storage_pool_id,
                                                                     const std::string &trace_id,
                                                                     int64_t timestamp,
                                                                     const std::vector<int64_t> &block_ids,
                                                                     const std::vector<size_t> &engine_hit_indices,
                                                                     int64_t input_len,
                                                                     const std::string &query_type,
                                                                     const StoragePoolFlowConfig &flow) {
    HashStoragePoolReadResult result;
    if (block_ids.empty()) {
        return result;
    }

    result = storage_pool_manager_->Read(HashStoragePoolReadRequest(storage_pool_id,
                                                                    trace_id,
                                                                    timestamp,
                                                                    block_ids,
                                                                    engine_hit_indices,
                                                                    input_len,
                                                                    query_type,
                                                                    flow.local_read_touch_enabled()));

    if (result.hit_blocks == 0) {
        return result;
    }

    FillEngineFromHitIndices(
        engine_instance_id, storage_pool_id, trace_id, timestamp, block_ids, result.hit_indices, flow);
    return result;
}

void HierarchicalReplayManager::WriteStoragePoolKeys(const std::string &storage_pool_id,
                                                     const std::string &trace_id,
                                                     int64_t timestamp,
                                                     const std::vector<int64_t> &keys,
                                                     int64_t ttl_us,
                                                     bool touch_existing) {
    if (!keys.empty()) {
        storage_pool_manager_->WriteKeys(storage_pool_id, trace_id, timestamp, keys, ttl_us, touch_existing);
    }
}

void HierarchicalReplayManager::ApplyStoragePoolWriteFlow(const std::string &engine_instance_id,
                                                          const std::string &storage_pool_id,
                                                          const std::string &trace_id,
                                                          int64_t timestamp,
                                                          int64_t ttl_us,
                                                          const StoragePoolFlowConfig &flow,
                                                          const WriteCacheRes &engine_write_res) {
    if (flow.write_mode() == TierWriteMode::WRITE_THROUGH) {
        WriteStoragePoolKeys(storage_pool_id,
                             trace_id,
                             timestamp,
                             engine_write_res.pool_source_write_keys,
                             ttl_us,
                             flow.shadow_write_touch_enabled());
    } else if (flow.write_mode() == TierWriteMode::CASCADING) {
        ApplyStoragePoolCascadingEvictions(
            storage_pool_id, trace_id, timestamp, ttl_us, flow, engine_write_res.evicted_keys);
    } else if (flow.write_mode() == TierWriteMode::WRITE_THROUGH_SELECTIVE) {
        const auto selected_keys = engine_manager_->PoolSourceWriteTouchKeysAtLeast(
            engine_instance_id, engine_write_res.pool_source_write_keys, flow.selective_write_threshold(), timestamp);
        WriteStoragePoolKeys(
            storage_pool_id, trace_id, timestamp, selected_keys, ttl_us, flow.shadow_write_touch_enabled());
    }
}

void HierarchicalReplayManager::ApplyStoragePoolCascadingEvictions(const std::string &storage_pool_id,
                                                                   const std::string &trace_id,
                                                                   int64_t timestamp,
                                                                   int64_t ttl_us,
                                                                   const StoragePoolFlowConfig &flow,
                                                                   const std::vector<int64_t> &evicted_keys) {
    if (flow.write_mode() != TierWriteMode::CASCADING) {
        return;
    }
    WriteStoragePoolKeys(storage_pool_id, trace_id, timestamp, evicted_keys, ttl_us, flow.shadow_write_touch_enabled());
}

void HierarchicalReplayManager::AnalyzeResults() {
    ExportCombinedHitRates();
    if (engine_manager_) {
        engine_manager_->AnalyzeResults();
    }
    if (storage_pool_manager_) {
        storage_pool_manager_->AnalyzeResults();
    }
}

void HierarchicalReplayManager::ExportCombinedHitRates() const {
    std::filesystem::create_directories(config_.output_result_path());
    const std::string filename = config_.output_result_path() + "/hierarchical_hit_rates.csv";
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open hierarchical hit-rate CSV: " + filename);
    }

    file << "TimestampNs,TraceId,EngineInstanceId,StoragePoolId,ReadBlocks,LocalHitBlocks,PeerHitBlocks,"
            "RemoteHitBlocks,"
            "HitBlocks,"
            "InputTokens,LocalHitTokens,PeerHitTokens,RemoteHitTokens,HitTokens,LocalHitRate,PeerHitRate,"
            "RemoteHitRate,HitRate,"
            "AccReadBlocks,AccHitBlocks,AccInputTokens,AccLocalHitTokens,AccPeerHitTokens,AccRemoteHitTokens,"
            "AccHitTokens,"
            "AccLocalHitRate,AccPeerHitRate,AccRemoteHitRate,AccHitRate,AccWriteBlocks\n";

    size_t acc_read_blocks = 0;
    size_t acc_hit_blocks = 0;
    size_t acc_input_tokens = 0;
    size_t acc_local_hit_tokens = 0;
    size_t acc_peer_hit_tokens = 0;
    size_t acc_remote_hit_tokens = 0;
    size_t acc_hit_tokens = 0;
    size_t acc_write_blocks = 0;
    size_t write_index = 0;

    for (const auto &record : combined_read_records_) {
        while (write_index < combined_write_records_.size() &&
               combined_write_records_[write_index].timestamp_ns <= record.timestamp_ns) {
            acc_write_blocks += combined_write_records_[write_index].write_blocks;
            write_index++;
        }

        const size_t hit_blocks = record.engine_hit_blocks + record.peer_hit_blocks + record.storage_pool_hit_blocks;
        const size_t local_hit_tokens = record.engine_hit_blocks * record.block_size_tokens;
        const size_t peer_hit_tokens = record.peer_hit_blocks * record.block_size_tokens;
        const size_t remote_hit_tokens = record.storage_pool_hit_blocks * record.block_size_tokens;
        const size_t hit_tokens = hit_blocks * record.block_size_tokens;

        acc_read_blocks += record.read_blocks;
        acc_hit_blocks += hit_blocks;
        acc_input_tokens += record.input_tokens;
        acc_local_hit_tokens += local_hit_tokens;
        acc_peer_hit_tokens += peer_hit_tokens;
        acc_remote_hit_tokens += remote_hit_tokens;
        acc_hit_tokens += hit_tokens;

        file << record.timestamp_ns << "," << record.trace_id << "," << record.engine_instance_id << ","
             << record.storage_pool_id << "," << record.read_blocks << "," << record.engine_hit_blocks << ","
             << record.peer_hit_blocks << "," << record.storage_pool_hit_blocks << "," << hit_blocks << ","
             << record.input_tokens << "," << local_hit_tokens << "," << peer_hit_tokens << "," << remote_hit_tokens
             << "," << hit_tokens << ","
             << (record.input_tokens > 0 ? static_cast<double>(local_hit_tokens) / record.input_tokens : 0.0) << ","
             << (record.input_tokens > 0 ? static_cast<double>(peer_hit_tokens) / record.input_tokens : 0.0) << ","
             << (record.input_tokens > 0 ? static_cast<double>(remote_hit_tokens) / record.input_tokens : 0.0) << ","
             << (record.input_tokens > 0 ? static_cast<double>(hit_tokens) / record.input_tokens : 0.0) << ","
             << acc_read_blocks << "," << acc_hit_blocks << "," << acc_input_tokens << "," << acc_local_hit_tokens
             << "," << acc_peer_hit_tokens << "," << acc_remote_hit_tokens << "," << acc_hit_tokens << ","
             << (acc_input_tokens > 0 ? static_cast<double>(acc_local_hit_tokens) / acc_input_tokens : 0.0) << ","
             << (acc_input_tokens > 0 ? static_cast<double>(acc_peer_hit_tokens) / acc_input_tokens : 0.0) << ","
             << (acc_input_tokens > 0 ? static_cast<double>(acc_remote_hit_tokens) / acc_input_tokens : 0.0) << ","
             << (acc_input_tokens > 0 ? static_cast<double>(acc_hit_tokens) / acc_input_tokens : 0.0) << ","
             << acc_write_blocks << "\n";
    }

    KVCM_LOG_INFO("Hierarchical hit rates exported to: %s", filename.c_str());
}

} // namespace kv_cache_manager
