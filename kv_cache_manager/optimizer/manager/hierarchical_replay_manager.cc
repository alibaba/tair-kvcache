#include "kv_cache_manager/optimizer/manager/hierarchical_replay_manager.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
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

    const auto engine_instances = CollectInstances(config_.engine_config());
    const auto storage_pools = CollectStoragePools(config_.storage_pool());
    std::vector<InferEngineActiveWindow> active_windows;
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
            for (const auto &window : cluster.active_windows()) {
                active_windows.push_back(
                    InferEngineActiveWindow{window.infer_id(), window.start_ns(), window.end_ns()});
            }
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

    std::vector<std::string> engine_instance_ids;
    engine_instance_ids.reserve(config_.engine_to_storage_pool().size());
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
        engine_instance_ids.push_back(engine_instance_id);
    }

    if (engine_to_storage_pool_.size() != engine_instances.size()) {
        KVCM_LOG_ERROR("Every engine instance must have exactly one engine_to_storage_pool mapping.");
        return false;
    }
    infer_engine_scheduler_.SetEngineInstanceIds(std::move(engine_instance_ids));
    infer_engine_scheduler_.SetActiveWindows(active_windows);
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
        p2p_tracker_.ApplyEvent(cluster_id, event);
    }
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
    ApplyStoragePoolCascadingEvictions(
        storage_pool_id, engine_instance_id, "fill_eviction", trace_id, timestamp, 0, flow, promote_res.evicted_keys);
}

TierGlobalPeerSelection HierarchicalReplayManager::ApplyP2PReadFlow(const std::string &engine_instance_id,
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
    const std::vector<std::string> active_infer_ids =
        infer_engine_scheduler_.ActiveInferIds(InferIdsForCluster(cluster_id), timestamp);
    TierGlobalPeerSelection result = p2p_tracker_.SelectPeer(
        engine_instance_id, cluster_id, flow.tier(), active_infer_ids, block_ids, *satisfied_mask);
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
    if (config_.infer_active_windows_from_trace() ||
        (config_.infer_scheduling_strategy() == "preserve_trace" && !infer_engine_scheduler_.has_active_windows())) {
        infer_engine_scheduler_.BuildTraceActiveWindows(traces, write_delay_ns_, true);
    }
    if (config_.infer_scheduling_strategy() == "prefix_hit") {
        RunTracesWithPrefixHitScheduling(traces);
        return;
    }
    infer_engine_scheduler_.ScheduleTraces(config_.infer_scheduling_strategy(), traces);
    RunTraces(traces);
}

void HierarchicalReplayManager::RunTracesWithPrefixHitScheduling(
    const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
    const auto &engine_instance_ids = infer_engine_scheduler_.engine_instance_ids();
    if (engine_instance_ids.empty()) {
        throw std::runtime_error("prefix_hit scheduling requires at least one engine instance");
    }

    pending_writes_ = {};
    next_pending_write_sequence_ = 0;
    size_t request_idx = 0;
    std::string current_engine_instance_id = engine_instance_ids.front();
    const auto prefix_match_count =
        [this](const std::string &instance_id, const std::vector<int64_t> &block_ids, int64_t timestamp) {
            return engine_manager_->PrefixMatchCount(instance_id, block_ids, timestamp);
        };
    for (const auto &trace : traces) {
        if (!trace) {
            continue;
        }
        FlushPendingWritesThrough(trace->timestamp_ns());
        if (auto request_trace = std::dynamic_pointer_cast<RequestSchemaTrace>(trace)) {
            current_engine_instance_id = infer_engine_scheduler_.ChoosePrefixHitEngineInstance(
                request_trace->keys(), request_trace->timestamp_ns(), request_idx, prefix_match_count);
            request_trace->set_instance_id(current_engine_instance_id);
            request_idx++;
        } else if (auto get_trace = std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
            current_engine_instance_id = infer_engine_scheduler_.ChoosePrefixHitEngineInstance(
                get_trace->keys(), get_trace->timestamp_ns(), request_idx, prefix_match_count);
            get_trace->set_instance_id(current_engine_instance_id);
            request_idx++;
        } else if (auto write_trace = std::dynamic_pointer_cast<WriteCacheSchemaTrace>(trace)) {
            write_trace->set_instance_id(current_engine_instance_id);
        }
        RunTrace(trace);
    }
    FlushAllPendingWrites();
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
    const int64_t write_timestamp_ns = trace.timestamp_ns() + write_delay_ns_;
    if (!infer_engine_scheduler_.IsInferActiveAt(trace.instance_id(), write_timestamp_ns)) {
        throw std::runtime_error("request write targets inactive engine instance: " + trace.instance_id());
    }

    WriteCacheSchemaTrace write_trace;
    write_trace.set_instance_id(trace.instance_id());
    write_trace.set_trace_id(trace.trace_id() + ":write");
    write_trace.set_timestamp_ns(write_timestamp_ns);
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
    ApplyStoragePoolCascadingEvictions(storage_pool_id,
                                       engine_instance_id,
                                       "read_eviction",
                                       trace_id,
                                       timestamp,
                                       0,
                                       storage_pool_flow,
                                       engine_res.evicted_keys);
    std::vector<bool> satisfied_mask(block_ids.size(), false);
    MarkIndices(engine_hit_indices, &satisfied_mask);
    std::vector<size_t> peer_hit_indices;
    std::string peer_source_infer_id;
    for (const auto &flow : P2PReadFlowsForCluster(ClusterForEngine(engine_instance_id))) {
        const auto p2p_read = ApplyP2PReadFlow(engine_instance_id,
                                               storage_pool_id,
                                               trace_id,
                                               timestamp,
                                               block_ids,
                                               flow,
                                               storage_pool_flow,
                                               &satisfied_mask);
        if (!p2p_read.hit_indices.empty()) {
            if (peer_source_infer_id.empty()) {
                peer_source_infer_id = p2p_read.peer_infer_id;
            } else if (peer_source_infer_id != p2p_read.peer_infer_id) {
                throw std::runtime_error("P2P read selected multiple peer infer sources for trace: " + trace_id);
            }
        }
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
    record.peer_source_infer_id = std::move(peer_source_infer_id);
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

WriteCacheRes HierarchicalReplayManager::WriteStoragePoolKeys(const std::string &engine_instance_id,
                                                              const std::string &storage_pool_id,
                                                              const std::string &reason,
                                                              const std::string &trace_id,
                                                              int64_t timestamp,
                                                              const std::vector<int64_t> &keys,
                                                              int64_t ttl_us,
                                                              bool touch_existing) {
    WriteCacheRes res;
    res.trace_id = trace_id;
    if (keys.empty()) {
        return res;
    }

    res = storage_pool_manager_->WriteKeys(storage_pool_id, trace_id, timestamp, keys, ttl_us, touch_existing);
    auto block_size_it = engine_block_size_.find(engine_instance_id);
    if (block_size_it == engine_block_size_.end()) {
        throw std::runtime_error("Unknown engine instance: " + engine_instance_id);
    }

    PoolWriteIoRecord record;
    record.trace_id = trace_id;
    record.engine_instance_id = engine_instance_id;
    record.storage_pool_id = storage_pool_id;
    record.reason = reason;
    record.timestamp_ns = timestamp;
    record.inserted_blocks = static_cast<size_t>(res.kvcm_write_length);
    record.existing_blocks = static_cast<size_t>(res.kvcm_write_hit_length);
    record.block_size_tokens = block_size_it->second;
    pool_write_io_records_.push_back(record);
    return res;
}

void HierarchicalReplayManager::ApplyStoragePoolWriteFlow(const std::string &engine_instance_id,
                                                          const std::string &storage_pool_id,
                                                          const std::string &trace_id,
                                                          int64_t timestamp,
                                                          int64_t ttl_us,
                                                          const StoragePoolFlowConfig &flow,
                                                          const WriteCacheRes &engine_write_res) {
    if (flow.write_mode() == TierWriteMode::WRITE_THROUGH) {
        WriteStoragePoolKeys(engine_instance_id,
                             storage_pool_id,
                             "write_through",
                             trace_id,
                             timestamp,
                             engine_write_res.pool_source_write_keys,
                             ttl_us,
                             flow.shadow_write_touch_enabled());
    } else if (flow.write_mode() == TierWriteMode::CASCADING) {
        ApplyStoragePoolCascadingEvictions(storage_pool_id,
                                           engine_instance_id,
                                           "write_eviction",
                                           trace_id,
                                           timestamp,
                                           ttl_us,
                                           flow,
                                           engine_write_res.evicted_keys);
    } else if (flow.write_mode() == TierWriteMode::WRITE_THROUGH_SELECTIVE) {
        const auto selected_keys = engine_manager_->PoolSourceWriteTouchKeysAtLeast(
            engine_instance_id, engine_write_res.pool_source_write_keys, flow.selective_write_threshold(), timestamp);
        WriteStoragePoolKeys(engine_instance_id,
                             storage_pool_id,
                             "write_through_selective",
                             trace_id,
                             timestamp,
                             selected_keys,
                             ttl_us,
                             flow.shadow_write_touch_enabled());
    }
}

void HierarchicalReplayManager::ApplyStoragePoolCascadingEvictions(const std::string &storage_pool_id,
                                                                   const std::string &engine_instance_id,
                                                                   const std::string &reason,
                                                                   const std::string &trace_id,
                                                                   int64_t timestamp,
                                                                   int64_t ttl_us,
                                                                   const StoragePoolFlowConfig &flow,
                                                                   const std::vector<int64_t> &evicted_keys) {
    if (flow.write_mode() != TierWriteMode::CASCADING) {
        return;
    }
    WriteStoragePoolKeys(engine_instance_id,
                         storage_pool_id,
                         reason,
                         trace_id,
                         timestamp,
                         evicted_keys,
                         ttl_us,
                         flow.shadow_write_touch_enabled());
}

void HierarchicalReplayManager::AnalyzeResults() {
    ExportCombinedHitRates();
    ExportReadIo();
    ExportPoolWriteIo();
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

void HierarchicalReplayManager::ExportReadIo() const {
    std::filesystem::create_directories(config_.output_result_path());
    const std::string filename = config_.output_result_path() + "/hierarchical_read_io.csv";
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open hierarchical read IO CSV: " + filename);
    }

    file << "TimestampNs,TraceId,EngineInstanceId,StoragePoolId,InputTokens,"
            "LocalReadTokens,PeerTransferTokens,PoolTransferTokens,"
            "PeerSourceInferId\n";

    for (const auto &record : combined_read_records_) {
        const size_t local_read_tokens = record.engine_hit_blocks * record.block_size_tokens;
        const size_t peer_transfer_tokens = record.peer_hit_blocks * record.block_size_tokens;
        const size_t pool_transfer_tokens = record.storage_pool_hit_blocks * record.block_size_tokens;

        file << record.timestamp_ns << "," << record.trace_id << "," << record.engine_instance_id << ","
             << record.storage_pool_id << "," << record.input_tokens << "," << local_read_tokens << ","
             << peer_transfer_tokens << "," << pool_transfer_tokens << "," << record.peer_source_infer_id << "\n";
    }

    KVCM_LOG_INFO("Hierarchical read IO exported to: %s", filename.c_str());
}

void HierarchicalReplayManager::ExportPoolWriteIo() const {
    std::filesystem::create_directories(config_.output_result_path());
    const std::string filename = config_.output_result_path() + "/hierarchical_pool_write_io.csv";
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open hierarchical pool write IO CSV: " + filename);
    }

    file << "TimestampNs,TraceId,EngineInstanceId,StoragePoolId,Reason,PoolWriteTokens,PoolExistingTokens\n";

    for (const auto &record : pool_write_io_records_) {
        const size_t inserted_tokens = record.inserted_blocks * record.block_size_tokens;
        const size_t existing_tokens = record.existing_blocks * record.block_size_tokens;
        file << record.timestamp_ns << "," << record.trace_id << "," << record.engine_instance_id << ","
             << record.storage_pool_id << "," << record.reason << "," << inserted_tokens << "," << existing_tokens
             << "\n";
    }

    KVCM_LOG_INFO("Hierarchical pool write IO exported to: %s", filename.c_str());
}

} // namespace kv_cache_manager
