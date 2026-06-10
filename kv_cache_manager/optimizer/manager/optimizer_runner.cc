#include "kv_cache_manager/optimizer/manager/optimizer_runner.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <variant>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/manager/optimizer_loader.h"

namespace kv_cache_manager {
namespace {
int64_t TtlUsToNs(int64_t ttl_us) { return ttl_us > 0 ? ttl_us * 1000 : ttl_us; }

void ValidateSupportedQueryTypeOrThrow(const std::string &query_type) {
    if (!IsSupportedQueryType(query_type)) {
        throw std::runtime_error("Unsupported optimizer query_type: " + query_type);
    }
}

void MergeEvictedBlocks(OptIndexerManager::EvictedBlocks *dst, const OptIndexerManager::EvictedBlocks &src) {
    if (dst == nullptr) {
        return;
    }
    for (const auto &[instance_id, blocks] : src) {
        auto &merged = (*dst)[instance_id];
        merged.insert(merged.end(), blocks.begin(), blocks.end());
    }
}

uint64_t MixUint64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

size_t ContiguousHitPrefixLength(const QueryHit &query_hit, size_t key_count) {
    if (key_count == 0) {
        return 0;
    }
    std::vector<bool> hit_mask(key_count, false);
    for (const size_t idx : query_hit.local_hit_indices) {
        if (idx < hit_mask.size()) {
            hit_mask[idx] = true;
        }
    }
    for (const size_t idx : query_hit.remote_hit_indices) {
        if (idx < hit_mask.size()) {
            hit_mask[idx] = true;
        }
    }

    size_t prefix = 0;
    while (prefix < hit_mask.size() && hit_mask[prefix]) {
        ++prefix;
    }
    return prefix;
}

void KeepHitIndicesBefore(std::vector<size_t> *indices, size_t prefix_len) {
    if (indices == nullptr) {
        return;
    }
    indices->erase(
        std::remove_if(indices->begin(), indices->end(), [prefix_len](size_t idx) { return idx >= prefix_len; }),
        indices->end());
}

std::vector<int64_t> KeysForTierEvents(const std::shared_ptr<RadixTreeIndex> &indexer,
                                       const std::string &instance_id,
                                       const TierFlowRecorder &tier_flow,
                                       const std::vector<TierFlowEventKind> &kinds,
                                       const std::vector<TierFlowEventReason> &excluded_reasons = {}) {
    if (!indexer) {
        return {};
    }
    std::vector<int64_t> keys;
    for (const auto *block :
         tier_flow.BlocksForTier(instance_id, indexer->PoolSourceTierName(), kinds, excluded_reasons)) {
        if (block != nullptr) {
            keys.push_back(block->key);
        }
    }
    return keys;
}

std::vector<int64_t> KeysForSourceTierWrites(const std::shared_ptr<RadixTreeIndex> &indexer,
                                             const std::string &instance_id,
                                             const TierFlowRecorder &tier_flow) {
    const std::vector<TierFlowEventKind> source_write_kinds = {
        TierFlowEventKind::ENTER_TIER,
        TierFlowEventKind::WRITE_TOUCH,
    };
    const std::vector<TierFlowEventReason> excluded_source_write_reasons = {
        TierFlowEventReason::WRITE_PROPAGATION,
        TierFlowEventReason::PROMOTE,
    };
    return KeysForTierEvents(indexer, instance_id, tier_flow, source_write_kinds, excluded_source_write_reasons);
}

std::vector<int64_t> KeysForSourceTierEvictions(const std::shared_ptr<RadixTreeIndex> &indexer,
                                                const std::string &instance_id,
                                                const TierFlowRecorder &tier_flow) {
    const std::vector<TierFlowEventKind> source_eviction_kinds = {
        TierFlowEventKind::LEAVE_TIER,
    };
    return KeysForTierEvents(indexer, instance_id, tier_flow, source_eviction_kinds);
}

size_t ValidateFullBlockTrace(const GetLocationSchemaTrace &trace, size_t block_size) {
    if (block_size == 0) {
        throw std::runtime_error("GetCacheLocation requires positive instance block_size");
    }

    const size_t input_tokens = trace.input_token_count();
    const size_t max_full_blocks = input_tokens / block_size;
    if (trace.keys().size() <= max_full_blocks) {
        return input_tokens;
    }

    throw std::runtime_error(
        "GetCacheLocation trace contains partial tail block keys: instance_id=" + trace.instance_id() +
        ", trace_id=" + trace.trace_id() + ", keys=" + std::to_string(trace.keys().size()) +
        ", input_len=" + std::to_string(input_tokens) + ", block_size=" + std::to_string(block_size) +
        ", max_full_blocks=" + std::to_string(max_full_blocks) +
        ". Standard optimizer traces must drop incomplete tail blocks before replay.");
}
} // namespace

void OptimizerRunner::Run(OptimizerConfig &config) {
    write_delay_ns_ = config.trace_replay_config().write_delay_ns();
    if (write_delay_ns_ <= 0) {
        throw std::runtime_error("trace_replay.write_delay_ns must be positive");
    }
    pending_writes_ = {};
    next_pending_write_sequence_ = 0;
    mamba_state_checkpoints_.clear();

    auto starting_time = std::chrono::high_resolution_clock::now();
    auto traces = OptimizerLoader::LoadTrace(config);
    auto ending_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(ending_time - starting_time).count();
    KVCM_LOG_INFO(
        "Loaded %zu traces from file: %s in %ld ms", traces.size(), config.trace_file_path().c_str(), duration);

    starting_time = std::chrono::high_resolution_clock::now();
    RunTraces(traces);
    ending_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(ending_time - starting_time).count();
    KVCM_LOG_INFO("Playback traces in %ld ms", duration);
}

void OptimizerRunner::RunTraces(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
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

void OptimizerRunner::RunTrace(std::shared_ptr<OptimizerSchemaTrace> trace) {
    if (!trace) {
        return;
    }

    if (auto request_trace = std::dynamic_pointer_cast<RequestSchemaTrace>(trace)) {
        HandleRequest(*request_trace);
    } else if (auto get_trace = std::dynamic_pointer_cast<GetLocationSchemaTrace>(trace)) {
        ValidateSupportedQueryTypeOrThrow(get_trace->query_type());
        HandleGetLocation(*get_trace);
        stats_collector_->UpdateTimestamp(get_trace->instance_id(), get_trace->timestamp_ns());
    } else if (auto write_trace = std::dynamic_pointer_cast<WriteCacheSchemaTrace>(trace)) {
        HandleWriteCache(*write_trace);
        stats_collector_->UpdateTimestamp(write_trace->instance_id(), write_trace->timestamp_ns());
    } else {
        KVCM_LOG_WARN("Unknown trace type, skipping");
    }
}

std::shared_ptr<RadixTreeIndex> OptimizerRunner::GetIndexer(const std::string &instance_id) {
    auto indexer = indexer_manager_->GetOptIndexer(instance_id);
    if (!indexer) {
        KVCM_LOG_ERROR("Optimizer indexer not found for instance_id: %s", instance_id.c_str());
    }
    return indexer;
}

void OptimizerRunner::HandleRequest(const RequestSchemaTrace &trace) {
    ValidateSupportedQueryTypeOrThrow(trace.query_type());
    HandleGetLocation(trace);
    stats_collector_->UpdateTimestamp(trace.instance_id(), trace.timestamp_ns());
    ScheduleRequestWrite(trace);
}

void OptimizerRunner::ScheduleRequestWrite(const RequestSchemaTrace &trace) {
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

void OptimizerRunner::FlushPendingWritesThrough(int64_t timestamp_ns) {
    while (!pending_writes_.empty() && pending_writes_.top().timestamp_ns <= timestamp_ns) {
        auto pending = pending_writes_.top();
        pending_writes_.pop();
        RunPendingWrite(pending.trace);
    }
}

void OptimizerRunner::FlushAllPendingWrites() {
    while (!pending_writes_.empty()) {
        auto pending = pending_writes_.top();
        pending_writes_.pop();
        RunPendingWrite(pending.trace);
    }
}

void OptimizerRunner::RunPendingWrite(const WriteCacheSchemaTrace &trace) {
    HandleWriteCache(trace);
    stats_collector_->UpdateTimestamp(trace.instance_id(), trace.timestamp_ns());
}

ReadRecord OptimizerRunner::SubmitReadRecord(const std::string &instance_id,
                                             const std::string &trace_id,
                                             const std::vector<int64_t> &keys,
                                             int64_t timestamp_ns,
                                             const QueryHit &query_hit,
                                             const std::shared_ptr<RadixTreeIndex> &indexer,
                                             size_t local_read_block_num,
                                             size_t remote_read_block_num,
                                             size_t input_tokens,
                                             size_t block_size_tokens,
                                             const MambaStateReadStats &mamba_state_stats) {
    ReadRecord record{};
    record.timestamp_ns = timestamp_ns;
    record.trace_id = trace_id;
    record.keys_ptr = &keys;
    record.current_cache_blocks = eviction_manager_->GetCurrentInstanceUsage(instance_id);

    auto indexer_map = indexer_manager_->GetAllOptIndexers();
    record.blocks_per_instance.resize(indexer_map.size(), 0);
    size_t idx = 0;
    for (const auto &pair : indexer_map) {
        record.blocks_per_instance[idx] = eviction_manager_->GetCurrentInstanceUsage(pair.first);
        idx++;
    }

    record.remote_hit_blocks = query_hit.remote_hit_block_num;
    record.local_hit_blocks = query_hit.local_hit_block_num;
    record.remote_hit_indices = query_hit.remote_hit_indices;
    record.local_hit_indices = query_hit.local_hit_indices;
    record.per_tier_hit_blocks = query_hit.per_tier_hit_block_num;
    record.input_tokens = input_tokens;
    record.block_size_tokens = block_size_tokens;
    record.tier_names = indexer->GetTierNames();
    record.per_tier_blocks = eviction_manager_->GetCurrentInstanceUsagePerTier(instance_id);
    record.local_read_blocks = local_read_block_num;
    record.remote_read_blocks = remote_read_block_num;
    record.mamba_state_enabled = mamba_state_stats.enabled;
    record.raw_kv_remote_hit_blocks = mamba_state_stats.raw_remote_hit_blocks;
    record.raw_kv_local_hit_blocks = mamba_state_stats.raw_local_hit_blocks;
    record.mamba_state_candidate_blocks = mamba_state_stats.candidate_blocks;
    record.mamba_state_hit_blocks = mamba_state_stats.hit_blocks;
    record.mamba_state_stored_checkpoints = mamba_state_stats.stored_checkpoints;
    record.mamba_state_bytes_per_state = mamba_state_stats.bytes_per_state;

    stats_collector_->OnReadComplete(instance_id, record);
    return record;
}

ReadRecord OptimizerRunner::HandleGetLocation(const GetLocationSchemaTrace &trace,
                                              bool touch_local_hits,
                                              bool local_hits_are_reads) {
    ReadRecord record{};
    std::string instance_id = trace.instance_id();
    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return record;
    }

    const size_t block_size = indexer_manager_->GetInstanceBlockSize(instance_id);
    const size_t input_tokens = ValidateFullBlockTrace(trace, block_size);

    auto pending_evicted_blocks = indexer_manager_->EvictExpiredBeforeAccess(instance_id, trace.timestamp_ns());

    bool refresh_ttl_on_read = true;
    auto it = instance_ttl_refresh_on_read_.find(instance_id);
    if (it != instance_ttl_refresh_on_read_.end()) {
        refresh_ttl_on_read = it->second;
    }

    QueryHit query_hit;
    if (IsPrefixMatchQueryType(trace.query_type())) {
        indexer->PrefixQuery(trace.keys(),
                             trace.block_mask(),
                             trace.timestamp_ns(),
                             &query_hit,
                             refresh_ttl_on_read,
                             touch_local_hits,
                             local_hits_are_reads);
    } else if (IsBatchGetQueryType(trace.query_type())) {
        indexer->BatchQuery(trace.keys(),
                            trace.block_mask(),
                            trace.timestamp_ns(),
                            &query_hit,
                            refresh_ttl_on_read,
                            touch_local_hits,
                            local_hits_are_reads);
    } else {
        ValidateSupportedQueryTypeOrThrow(trace.query_type());
    }
    TierFlowRecorder request_tier_flow = indexer->ConsumeTierFlow();
    if (indexer->ConsumeReadTriggeredTierWrite()) {
        auto capacity_eviction = indexer_manager_->CheckAndEvict(instance_id, trace.timestamp_ns());
        MergeEvictedBlocks(&pending_evicted_blocks, capacity_eviction.evicted_blocks);
        request_tier_flow.MergeFrom(capacity_eviction.tier_flow);
    }

    size_t local_read_block_num = 0;
    size_t remote_read_block_num = trace.keys().size();
    size_t local_mask_block_num = 0;
    if (std::holds_alternative<BlockMaskVector>(trace.block_mask())) {
        const auto &mask_vector = std::get<BlockMaskVector>(trace.block_mask());
        const size_t n = std::min(mask_vector.size(), trace.keys().size());
        local_mask_block_num = std::count(mask_vector.begin(), mask_vector.begin() + n, true);
    } else if (std::holds_alternative<BlockMaskOffset>(trace.block_mask())) {
        local_mask_block_num = std::min(std::get<BlockMaskOffset>(trace.block_mask()), trace.keys().size());
    }
    local_read_block_num = local_hits_are_reads ? local_mask_block_num : 0;
    remote_read_block_num = trace.keys().size() - local_mask_block_num;
    const MambaStateReadStats mamba_state_stats = ApplyMambaStateRead(instance_id, trace.keys(), &query_hit);

    record = SubmitReadRecord(instance_id,
                              trace.trace_id(),
                              trace.keys(),
                              trace.timestamp_ns(),
                              query_hit,
                              indexer,
                              local_read_block_num,
                              remote_read_block_num,
                              input_tokens,
                              block_size,
                              mamba_state_stats);
    record.evicted_keys = KeysForSourceTierEvictions(indexer, instance_id, request_tier_flow);
    record.tier_flow_events = request_tier_flow.KeyEvents();
    indexer_manager_->CleanEvictedBlocks(pending_evicted_blocks, trace.timestamp_ns(), true);
    return record;
}

WriteRecord OptimizerRunner::HandleWriteCache(const WriteCacheSchemaTrace &trace) {
    return HandleCacheInsert(trace, true, nullptr);
}

WriteRecord OptimizerRunner::HandleFillCachePath(const WriteCacheSchemaTrace &trace,
                                                 const std::vector<size_t> &materialized_indices) {
    return HandleCacheInsert(trace, false, &materialized_indices);
}

void OptimizerRunner::ClearMambaState(const std::string &instance_id) { mamba_state_checkpoints_.erase(instance_id); }

void OptimizerRunner::ClearAllMambaStates() { mamba_state_checkpoints_.clear(); }

std::vector<OptimizerRunner::PrefixSignature>
OptimizerRunner::BuildPrefixSignatures(const std::vector<int64_t> &keys) const {
    std::vector<PrefixSignature> signatures(keys.size() + 1);
    uint64_t hash1 = 1469598103934665603ULL;
    uint64_t hash2 = 1099511628211ULL;
    for (size_t idx = 0; idx < keys.size(); ++idx) {
        const uint64_t mixed_key = MixUint64(static_cast<uint64_t>(keys[idx]));
        hash1 ^= mixed_key;
        hash1 *= 1099511628211ULL;
        hash2 ^= mixed_key + 0x9e3779b97f4a7c15ULL + (hash2 << 6) + (hash2 >> 2);
        hash2 = MixUint64(hash2);
        signatures[idx + 1] = PrefixSignature{idx + 1, hash1, hash2};
    }
    return signatures;
}

std::vector<size_t> OptimizerRunner::MambaCheckpointIndices(size_t key_count) const {
    std::vector<size_t> indices;
    if (!mamba_state_config_.enabled() || key_count == 0 || mamba_state_config_.chunk_size_blocks() == 0) {
        return indices;
    }

    for (size_t next = mamba_state_config_.chunk_size_blocks(); next <= key_count;
         next += mamba_state_config_.chunk_size_blocks()) {
        indices.push_back(next - 1);
    }
    const size_t request_end = key_count - 1;
    if (indices.empty() || indices.back() != request_end) {
        indices.push_back(request_end);
    }
    return indices;
}

OptimizerRunner::MambaStateReadStats OptimizerRunner::ApplyMambaStateRead(const std::string &instance_id,
                                                                          const std::vector<int64_t> &keys,
                                                                          QueryHit *query_hit) {
    MambaStateReadStats stats;
    if (!mamba_state_config_.enabled() || query_hit == nullptr) {
        return stats;
    }

    stats.enabled = true;
    stats.raw_remote_hit_blocks = query_hit->remote_hit_block_num;
    stats.raw_local_hit_blocks = query_hit->local_hit_block_num;
    stats.bytes_per_state = mamba_state_config_.bytes_per_state();
    auto &checkpoints = mamba_state_checkpoints_[instance_id];
    stats.stored_checkpoints = checkpoints.size();

    const size_t candidate_prefix = ContiguousHitPrefixLength(*query_hit, keys.size());
    stats.candidate_blocks = candidate_prefix;
    if (candidate_prefix == 0 || checkpoints.empty()) {
        query_hit->remote_hit_block_num = 0;
        query_hit->local_hit_block_num = 0;
        query_hit->remote_hit_indices.clear();
        query_hit->local_hit_indices.clear();
        std::fill(query_hit->per_tier_hit_block_num.begin(), query_hit->per_tier_hit_block_num.end(), 0);
        return stats;
    }

    const auto signatures = BuildPrefixSignatures(keys);
    for (size_t prefix_len = candidate_prefix; prefix_len > 0; --prefix_len) {
        if (checkpoints.find(signatures[prefix_len]) != checkpoints.end()) {
            stats.hit_blocks = prefix_len;
            break;
        }
    }

    KeepHitIndicesBefore(&query_hit->local_hit_indices, stats.hit_blocks);
    KeepHitIndicesBefore(&query_hit->remote_hit_indices, stats.hit_blocks);
    query_hit->local_hit_block_num = query_hit->local_hit_indices.size();
    query_hit->remote_hit_block_num = query_hit->remote_hit_indices.size();

    size_t remaining_tier_hits = query_hit->local_hit_block_num + query_hit->remote_hit_block_num;
    for (auto &tier_hits : query_hit->per_tier_hit_block_num) {
        const size_t kept = std::min(tier_hits, remaining_tier_hits);
        tier_hits = kept;
        remaining_tier_hits -= kept;
    }
    return stats;
}

OptimizerRunner::MambaStateWriteStats OptimizerRunner::ApplyMambaStateWrite(
    const std::string &instance_id, const std::vector<int64_t> &keys, const std::vector<size_t> *materialized_indices) {
    MambaStateWriteStats stats;
    if (!mamba_state_config_.enabled() || keys.empty()) {
        return stats;
    }

    stats.enabled = true;
    stats.bytes_per_state = mamba_state_config_.bytes_per_state();
    const auto checkpoint_indices = MambaCheckpointIndices(keys.size());
    if (checkpoint_indices.empty()) {
        return stats;
    }

    std::vector<bool> allowed(keys.size(), true);
    if (materialized_indices != nullptr) {
        std::fill(allowed.begin(), allowed.end(), false);
        for (const size_t idx : *materialized_indices) {
            if (idx < allowed.size()) {
                allowed[idx] = true;
            }
        }
    }

    const auto signatures = BuildPrefixSignatures(keys);
    auto &checkpoints = mamba_state_checkpoints_[instance_id];
    for (const size_t idx : checkpoint_indices) {
        if (idx >= allowed.size() || !allowed[idx]) {
            continue;
        }
        ++stats.write_checkpoints;
        if (checkpoints.insert(signatures[idx + 1]).second) {
            ++stats.new_checkpoints;
        }
    }
    return stats;
}

WriteRecord OptimizerRunner::HandleCacheInsert(const WriteCacheSchemaTrace &trace,
                                               bool count_new_tier_write_touch,
                                               const std::vector<size_t> *materialized_indices) {
    WriteRecord record;
    record.timestamp_ns = trace.timestamp_ns();
    record.trace_id = trace.trace_id();

    std::string instance_id = trace.instance_id();
    auto indexer = GetIndexer(instance_id);
    if (!indexer) {
        return record;
    }

    auto pending_evicted_blocks = indexer_manager_->EvictExpiredBeforeAccess(instance_id, trace.timestamp_ns());

    int64_t effective_ttl_ns = TtlUsToNs(trace.ttl_us());
    auto ttl_disabled_it = instance_group_ttl_disabled_.find(instance_id);
    if (ttl_disabled_it != instance_group_ttl_disabled_.end() && ttl_disabled_it->second) {
        effective_ttl_ns = -1;
    }

    RadixTreeIndex::InsertResult result;
    if (count_new_tier_write_touch) {
        result = indexer->InsertOnly(trace.keys(), trace.timestamp_ns(), effective_ttl_ns);
    } else if (materialized_indices != nullptr) {
        result = indexer->FillPathOnly(trace.keys(), *materialized_indices, trace.timestamp_ns(), effective_ttl_ns);
    } else {
        throw std::runtime_error("HandleCacheInsert fill requires materialized indices");
    }
    auto capacity_eviction = indexer_manager_->CheckAndEvict(instance_id, trace.timestamp_ns());
    const auto &capacity_evicted_blocks = capacity_eviction.evicted_blocks;
    MergeEvictedBlocks(&pending_evicted_blocks, capacity_evicted_blocks);
    TierFlowRecorder request_tier_flow = std::move(result.tier_flow);
    request_tier_flow.MergeFrom(capacity_eviction.tier_flow);

    auto pool_source_write_keys = KeysForSourceTierWrites(indexer, instance_id, request_tier_flow);
    std::vector<int64_t> evicted_keys = KeysForSourceTierEvictions(indexer, instance_id, request_tier_flow);
    bool evicted = !capacity_evicted_blocks.empty();
    if (evicted) {
        KVCM_LOG_DEBUG("Eviction at ts=%lld for instance_id: %s",
                       static_cast<long long>(trace.timestamp_ns()),
                       instance_id.c_str());
    }

    size_t write_blocks = trace.keys().size();
    if (materialized_indices != nullptr) {
        std::vector<bool> selected(trace.keys().size(), false);
        for (const size_t idx : *materialized_indices) {
            if (idx < selected.size()) {
                selected[idx] = true;
            }
        }
        write_blocks = std::count(selected.begin(), selected.end(), true);
    }
    record.write_blocks = write_blocks;
    record.newly_inserted_blocks = result.inserted_keys.size();
    const MambaStateWriteStats mamba_state_stats =
        ApplyMambaStateWrite(instance_id, trace.keys(), materialized_indices);
    record.mamba_state_enabled = mamba_state_stats.enabled;
    record.mamba_state_write_checkpoints = mamba_state_stats.write_checkpoints;
    record.mamba_state_new_checkpoints = mamba_state_stats.new_checkpoints;
    record.mamba_state_bytes_per_state = mamba_state_stats.bytes_per_state;
    record.pool_source_write_keys = std::move(pool_source_write_keys);
    record.evicted_keys = std::move(evicted_keys);
    record.tier_flow_events = request_tier_flow.KeyEvents();
    if (count_new_tier_write_touch) {
        stats_collector_->OnWriteComplete(instance_id, record);
    }
    indexer_manager_->CleanEvictedBlocks(pending_evicted_blocks, trace.timestamp_ns(), true);
    return record;
}
} // namespace kv_cache_manager
