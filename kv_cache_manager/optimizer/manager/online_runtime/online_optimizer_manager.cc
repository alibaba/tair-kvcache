#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/index/online/cache_indexer_factory.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"
#include "kv_cache_manager/optimizer/liteHit/request_preprocess.h"

namespace kv_cache_manager {

OnlineOptimizerManager::OnlineOptimizerManager(std::shared_ptr<OptimizerRegistryManager> registry_manager)
    : registry_manager_(std::move(registry_manager)) {}

namespace {

const LocationSpecGroup *FindLocationSpecGroup(const std::vector<LocationSpecGroup> &groups, const std::string &name) {
    for (const auto &group : groups) {
        if (group.name() == name) {
            return &group;
        }
    }
    return nullptr;
}

constexpr long double kBytesPerGb = 1024.0L * 1024.0L * 1024.0L;

bool ConvertCapacitiesToBlocks(const std::vector<double> &capacity_gb,
                               int64_t block_charge_bytes,
                               std::vector<int64_t> &capacity_blocks) {
    capacity_blocks.clear();
    capacity_blocks.reserve(capacity_gb.size());
    for (double capacity : capacity_gb) {
        if (!std::isfinite(capacity) || capacity < 0.0) {
            return false;
        }
        const long double blocks = static_cast<long double>(capacity) * kBytesPerGb / block_charge_bytes;
        if (blocks >= static_cast<long double>(std::numeric_limits<int64_t>::max())) {
            capacity_blocks.push_back(std::numeric_limits<int64_t>::max());
        } else {
            capacity_blocks.push_back(static_cast<int64_t>(blocks));
        }
    }
    return true;
}

int64_t ClampToInt64(uint64_t value) {
    return value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ? std::numeric_limits<int64_t>::max()
                                                                              : static_cast<int64_t>(value);
}

int64_t SaturatingMultiplyToInt64(uint64_t lhs, uint64_t rhs) {
    if (lhs != 0 && rhs > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / lhs) {
        return std::numeric_limits<int64_t>::max();
    }
    return static_cast<int64_t>(lhs * rhs);
}

uint64_t SaturatingAddUint64(uint64_t lhs, uint64_t rhs) {
    return rhs > std::numeric_limits<uint64_t>::max() - lhs ? std::numeric_limits<uint64_t>::max() : lhs + rhs;
}

uint64_t SaturatingMultiplyUint64(uint64_t lhs, uint64_t rhs) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return std::numeric_limits<uint64_t>::max();
    }
    return lhs * rhs;
}

} // namespace

int64_t OnlineOptimizerManager::ComputeSizeForGroup(const std::vector<LocationSpecInfo> &specs,
                                                    const LocationSpecGroup &group) {
    int64_t total = 0;
    for (const auto &spec_name : group.spec_names()) {
        const LocationSpecInfo *matched_spec = nullptr;
        for (const auto &spec : specs) {
            if (spec.name() == spec_name) {
                matched_spec = &spec;
                break;
            }
        }
        if (!matched_spec) {
            return -1;
        }
        total += matched_spec->size();
    }
    return total;
}

bool OnlineOptimizerManager::HasActiveInstanceInGroup(const std::string &instance_group_name) const {
    std::shared_lock lock(instances_mutex_);
    for (const auto &[_, state] : instances_) {
        if (state && state->instance_info && state->instance_info->instance_group_name() == instance_group_name) {
            return true;
        }
    }
    return false;
}

bool OnlineOptimizerManager::HasPersistedInstanceInGroup(const std::string &instance_group_name) const {
    return registry_manager_ && !registry_manager_->ListInstanceInfos(instance_group_name).empty();
}

ErrorCode OnlineOptimizerManager::CreateInstanceGroup(const OptimizerInstanceGroup &instance_group) {
    if (!registry_manager_) {
        return EC_ERROR;
    }
    std::lock_guard admin_guard(admin_ops_mutex_);
    return registry_manager_->CreateInstanceGroup(instance_group);
}

ErrorCode OnlineOptimizerManager::UpdateInstanceGroup(const OptimizerInstanceGroup &instance_group) {
    if (!registry_manager_) {
        return EC_ERROR;
    }

    std::lock_guard admin_guard(admin_ops_mutex_);
    if (HasActiveInstanceInGroup(instance_group.name()) || HasPersistedInstanceInGroup(instance_group.name())) {
        KVCM_LOG_ERROR("UpdateInstanceGroup failed: instance group[%s] still has registered instances",
                       instance_group.name().c_str());
        return EC_BADARGS;
    }
    return registry_manager_->UpdateInstanceGroup(instance_group);
}

ErrorCode OnlineOptimizerManager::RemoveInstanceGroup(const std::string &instance_group_name) {
    if (!registry_manager_) {
        return EC_ERROR;
    }

    std::lock_guard admin_guard(admin_ops_mutex_);
    if (HasActiveInstanceInGroup(instance_group_name) || HasPersistedInstanceInGroup(instance_group_name)) {
        KVCM_LOG_ERROR("RemoveInstanceGroup failed: instance group[%s] still has registered instances",
                       instance_group_name.c_str());
        return EC_BADARGS;
    }
    return registry_manager_->RemoveInstanceGroup(instance_group_name);
}

ErrorCode OnlineOptimizerManager::RegisterInstance(const OptimizerInstanceInfo &instance_info,
                                                   RegisterInstanceResult &result) {
    const auto &instance_id = instance_info.instance_id();
    if (instance_id.empty()) {
        KVCM_LOG_ERROR("RegisterInstance failed: empty instance_id");
        return EC_BADARGS;
    }

    std::lock_guard admin_guard(admin_ops_mutex_);

    OptimizerInstanceInfo resolved_instance_info = instance_info;
    if (resolved_instance_info.optimizer_state_info().full_location_spec_group_name().empty()) {
        if (resolved_instance_info.linear_step() != 0) {
            KVCM_LOG_ERROR("RegisterInstance failed: instance[%s] with linear_step=%d requires explicit optimizer "
                           "state",
                           instance_id.c_str(),
                           resolved_instance_info.linear_step());
            return EC_BADARGS;
        }
        if (resolved_instance_info.location_spec_groups().size() != 1) {
            KVCM_LOG_ERROR("RegisterInstance failed: full-only instance[%s] without an explicit full group must have "
                           "exactly one location spec group, actual=%zu",
                           instance_id.c_str(),
                           resolved_instance_info.location_spec_groups().size());
            return EC_BADARGS;
        }
        OptimizerStateInfo state_info;
        state_info.set_full_location_spec_group_name(resolved_instance_info.location_spec_groups().front().name());
        resolved_instance_info.set_optimizer_state_info(state_info);
    }

    auto instance_group = registry_manager_->GetInstanceGroup(resolved_instance_info.instance_group_name());
    if (!instance_group) {
        KVCM_LOG_ERROR("RegisterInstance failed: instance group[%s] not found for instance[%s]",
                       resolved_instance_info.instance_group_name().c_str(),
                       instance_id.c_str());
        return EC_NOENT;
    }

    // Save old persisted info before overwriting, so we can restore on rollback.
    auto old_instance_info = registry_manager_->GetInstanceInfo(instance_id);

    auto ec = registry_manager_->SaveInstanceInfo(resolved_instance_info);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("RegisterInstance failed: persist instance_info[%s] failed", instance_id.c_str());
        return ec;
    }

    ec = RegisterInstanceInternal(resolved_instance_info, *instance_group, result);
    if (ec != EC_OK) {
        // Rollback persistence: restore old record if it existed, else delete
        if (old_instance_info) {
            registry_manager_->SaveInstanceInfo(*old_instance_info);
        } else {
            registry_manager_->DeleteInstanceInfo(instance_id);
        }
        return ec;
    }

    return EC_OK;
}

ErrorCode OnlineOptimizerManager::RegisterInstanceInternal(const OptimizerInstanceInfo &instance_info,
                                                           const OptimizerInstanceGroup &instance_group,
                                                           RegisterInstanceResult &result) {
    const auto &instance_id = instance_info.instance_id();
    const auto &specs = instance_info.location_spec_infos();
    if (specs.empty()) {
        KVCM_LOG_ERROR("RegisterInstance failed: empty location_spec_infos for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }
    for (const auto &spec : specs) {
        if (spec.name().empty()) {
            KVCM_LOG_ERROR("RegisterInstance failed: empty spec name for instance[%s]", instance_id.c_str());
            return EC_BADARGS;
        }
        if (spec.size() <= 0) {
            KVCM_LOG_ERROR("RegisterInstance failed: non-positive spec size for spec[%s] instance[%s]",
                           spec.name().c_str(),
                           instance_id.c_str());
            return EC_BADARGS;
        }
    }
    for (size_t i = 0; i < specs.size(); ++i) {
        for (size_t j = i + 1; j < specs.size(); ++j) {
            if (specs[i].name() == specs[j].name()) {
                KVCM_LOG_ERROR("RegisterInstance failed: duplicate spec name[%s] for instance[%s]",
                               specs[i].name().c_str(),
                               instance_id.c_str());
                return EC_BADARGS;
            }
        }
    }
    const auto &groups = instance_info.location_spec_groups();
    for (const auto &group : groups) {
        if (group.name().empty()) {
            KVCM_LOG_ERROR("RegisterInstance failed: empty location_spec_group name for instance[%s]",
                           instance_id.c_str());
            return EC_BADARGS;
        }
        const auto &spec_names = group.spec_names();
        for (const auto &spec_name : spec_names) {
            if (spec_name.empty()) {
                KVCM_LOG_ERROR("RegisterInstance failed: empty spec name in location_spec_group[%s] instance[%s]",
                               group.name().c_str(),
                               instance_id.c_str());
                return EC_BADARGS;
            }
        }
        for (size_t i = 0; i < spec_names.size(); ++i) {
            for (size_t j = i + 1; j < spec_names.size(); ++j) {
                if (spec_names[i] == spec_names[j]) {
                    KVCM_LOG_ERROR("RegisterInstance failed: duplicate spec name[%s] in location_spec_group[%s] "
                                   "instance[%s]",
                                   spec_names[i].c_str(),
                                   group.name().c_str(),
                                   instance_id.c_str());
                    return EC_BADARGS;
                }
            }
        }
    }
    for (size_t i = 0; i < groups.size(); ++i) {
        for (size_t j = i + 1; j < groups.size(); ++j) {
            if (groups[i].name() == groups[j].name()) {
                KVCM_LOG_ERROR("RegisterInstance failed: duplicate location_spec_group name[%s] for instance[%s]",
                               groups[i].name().c_str(),
                               instance_id.c_str());
                return EC_BADARGS;
            }
        }
    }
    if (instance_group.shared_group_quota()) {
        KVCM_LOG_ERROR(
            "RegisterInstance failed: shared_group_quota is not supported by online indexer for instance[%s]",
            instance_id.c_str());
        return EC_BADARGS;
    }

    if (instance_info.linear_step() < 0) {
        KVCM_LOG_ERROR("RegisterInstance failed: negative linear_step for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }
    int32_t linear_step = instance_info.linear_step();
    if (instance_info.block_size() <= 0) {
        KVCM_LOG_ERROR("RegisterInstance failed: non-positive token block_size for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }
    if (instance_group.eviction_policy() != "lru") {
        KVCM_LOG_ERROR("RegisterInstance failed: unsupported eviction_policy[%s] for instance[%s]",
                       instance_group.eviction_policy().c_str(),
                       instance_id.c_str());
        return EC_BADARGS;
    }
    if (instance_group.ttl_seconds() < 0) {
        KVCM_LOG_ERROR("RegisterInstance failed: negative TTL for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }

    const auto &optimizer_state_info = instance_info.optimizer_state_info();
    if (optimizer_state_info.full_location_spec_group_name().empty()) {
        KVCM_LOG_ERROR("RegisterInstance failed: empty full_location_spec_group_name for instance[%s]",
                       instance_id.c_str());
        return EC_BADARGS;
    }
    if (!optimizer_state_info.linear_location_spec_group_name().empty() &&
        optimizer_state_info.full_location_spec_group_name() ==
            optimizer_state_info.linear_location_spec_group_name()) {
        KVCM_LOG_ERROR("RegisterInstance failed: full and linear groups are the same[%s] for instance[%s]",
                       optimizer_state_info.full_location_spec_group_name().c_str(),
                       instance_id.c_str());
        return EC_BADARGS;
    }

    const auto *full_group = FindLocationSpecGroup(groups, optimizer_state_info.full_location_spec_group_name());
    if (!full_group) {
        KVCM_LOG_ERROR("RegisterInstance failed: full group[%s] not found for instance[%s]",
                       optimizer_state_info.full_location_spec_group_name().c_str(),
                       instance_id.c_str());
        return EC_BADARGS;
    }
    int64_t size_full = ComputeSizeForGroup(specs, *full_group);
    if (size_full <= 0) {
        KVCM_LOG_ERROR("RegisterInstance failed: invalid full group[%s] size[%ld] for instance[%s]",
                       full_group->name().c_str(),
                       size_full,
                       instance_id.c_str());
        return EC_BADARGS;
    }

    int64_t size_full_linear = size_full;
    if (!optimizer_state_info.linear_location_spec_group_name().empty()) {
        const auto *linear_group =
            FindLocationSpecGroup(groups, optimizer_state_info.linear_location_spec_group_name());
        if (!linear_group) {
            KVCM_LOG_ERROR("RegisterInstance failed: linear group[%s] not found for instance[%s]",
                           optimizer_state_info.linear_location_spec_group_name().c_str(),
                           instance_id.c_str());
            return EC_BADARGS;
        }
        const int64_t size_linear = ComputeSizeForGroup(specs, *linear_group);
        if (size_linear <= 0) {
            KVCM_LOG_ERROR("RegisterInstance failed: invalid linear group[%s] size[%ld] for instance[%s]",
                           linear_group->name().c_str(),
                           size_linear,
                           instance_id.c_str());
            return EC_BADARGS;
        }
        size_full_linear += size_linear;
    }

    int64_t estimated_bytes_per_block;
    if (linear_step == 0) {
        estimated_bytes_per_block = size_full;
    } else if (linear_step == 1) {
        estimated_bytes_per_block = size_full_linear;
    } else {
        estimated_bytes_per_block = ((linear_step - 1) * size_full + size_full_linear) / linear_step;
    }

    if (estimated_bytes_per_block <= 0) {
        KVCM_LOG_ERROR("RegisterInstance failed: estimated_bytes_per_block <= 0 for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }

    const auto &capacity_gb = instance_group.capacity_gb();
    std::vector<int64_t> estimated_capacity_blocks;
    if (!ConvertCapacitiesToBlocks(capacity_gb, estimated_bytes_per_block, estimated_capacity_blocks)) {
        KVCM_LOG_ERROR("RegisterInstance failed: invalid capacity for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }

    auto state = std::make_shared<InstanceState>();
    state->instance_info = std::make_shared<OptimizerInstanceInfo>(instance_info);
    state->instance_group = std::make_shared<OptimizerInstanceGroup>(instance_group);

    state->size_full = size_full;
    state->size_full_linear = size_full_linear;
    state->linear_step = linear_step;
    state->total_hits_per_capacity.resize(capacity_gb.size(), 0);
    state->interval_hits_per_capacity.resize(capacity_gb.size(), 0);

    if (linear_step == 0) {
        state->lite_hit_capacity_blocks = estimated_capacity_blocks;
        // A group TTL is layered onto the LiteHit core; online time is the
        // wall clock, mirroring the linear-path TtlCacheIndexerWrapper.
        state->lite_hit =
            std::make_unique<LiteHit>(static_cast<uint64_t>(instance_group.ttl_seconds()) * 1000000000ULL);
    } else {
        auto indexer = CacheIndexerFactory::CreateCacheIndexer(instance_group.eviction_policy(),
                                                               instance_group.enable_theoretical_max_cache(),
                                                               capacity_gb,
                                                               size_full,
                                                               size_full_linear,
                                                               linear_step,
                                                               instance_group.ttl_seconds());
        if (!indexer) {
            KVCM_LOG_ERROR("RegisterInstance failed: initialize linear indexer for instance[%s]", instance_id.c_str());
            return EC_BADARGS;
        }
        state->indexer = std::move(indexer);
    }

    {
        std::unique_lock lock(instances_mutex_);
        instances_[instance_id] = std::move(state);
    }

    result.estimated_capacity_blocks = estimated_capacity_blocks;
    result.size_full = size_full;
    result.size_full_linear = size_full_linear;

    KVCM_LOG_INFO("RegisterInstance OK: instance[%s] group[%s] linear_step=%d estimated_bytes_per_block=%ld caps=%zu",
                  instance_id.c_str(),
                  instance_info.instance_group_name().c_str(),
                  linear_step,
                  estimated_bytes_per_block,
                  estimated_capacity_blocks.size());
    return EC_OK;
}

ErrorCode OnlineOptimizerManager::RemoveInstance(const std::string &instance_id) {
    std::lock_guard admin_guard(admin_ops_mutex_);

    {
        std::shared_lock lock(instances_mutex_);
        if (instances_.find(instance_id) == instances_.end()) {
            return EC_INSTANCE_NOT_EXIST;
        }
    }

    auto ec = registry_manager_->DeleteInstanceInfo(instance_id);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("RemoveInstance failed: delete persistent instance_info[%s] failed", instance_id.c_str());
        return ec;
    }

    {
        std::unique_lock lock(instances_mutex_);
        instances_.erase(instance_id);
    }

    KVCM_LOG_INFO("RemoveInstance OK: instance[%s]", instance_id.c_str());
    return EC_OK;
}

ErrorCode OnlineOptimizerManager::TraceQuery(const std::string &instance_id,
                                             const std::vector<int64_t> &block_keys,
                                             int64_t input_token_len,
                                             int64_t timestamp_ns,
                                             TraceQueryResult &result) {
    if (input_token_len < 0 || timestamp_ns < 0) {
        return EC_BADARGS;
    }
    const int64_t replay_timestamp_ns =
        timestamp_ns == 0 ? static_cast<int64_t>(TimestampUtil::GetCurrentTimeUs()) * 1000 : timestamp_ns;

    std::shared_ptr<InstanceState> state;
    {
        std::shared_lock lock(instances_mutex_);
        auto it = instances_.find(instance_id);
        if (it == instances_.end()) {
            return EC_INSTANCE_NOT_EXIST;
        }
        state = it->second;
    }

    if (input_token_len == 0) {
        const uint64_t block_size = static_cast<uint64_t>(state->instance_info->block_size());
        if (!block_keys.empty() &&
            block_size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / block_keys.size()) {
            return EC_BADARGS;
        }
        input_token_len = static_cast<int64_t>(block_keys.size() * block_size);
    }

    std::lock_guard<std::mutex> guard(state->mutex);
    result = TraceQueryResult{};

    const int64_t total_blocks = static_cast<int64_t>(block_keys.size());
    const size_t num_caps = state->total_hits_per_capacity.size();
    result.total_blocks = total_blocks;
    result.input_token_len = input_token_len;
    result.capacity_gb = state->instance_group->capacity_gb();

    if (state->linear_step == 0) {
        if (!state->lite_hit) {
            return EC_ERROR;
        }
        NormalizedRequest normalized;
        try {
            normalized = NormalizeRequest(block_keys,
                                          input_token_len,
                                          static_cast<uint64_t>(state->instance_info->block_size()),
                                          state->instance_group->enable_prefix_hash());
        } catch (const std::invalid_argument &e) {
            KVCM_LOG_ERROR(
                "TraceQuery failed: invalid LiteHit request for instance[%s]: %s", instance_id.c_str(), e.what());
            return EC_BADARGS;
        }

        const RequestFact fact = state->lite_hit->ProcessRequest(normalized.block_keys, replay_timestamp_ns);
        result.input_token_len = ClampToInt64(normalized.input_token_len);

        const uint64_t block_size = static_cast<uint64_t>(state->instance_info->block_size());
        const double token_denominator = static_cast<double>(normalized.input_token_len);
        result.hit_count_per_capacity.reserve(num_caps);
        result.hit_rate_per_capacity.reserve(num_caps);
        for (std::size_t i = 0; i < num_caps; ++i) {
            const uint64_t hits =
                HitCurveProjector::ProjectBlocks(fact, static_cast<uint64_t>(state->lite_hit_capacity_blocks[i]));
            result.hit_count_per_capacity.push_back(ClampToInt64(hits));
            result.hit_rate_per_capacity.push_back(
                normalized.input_token_len == 0 ? 0.0 : static_cast<double>(hits * block_size) / token_denominator);
            state->total_hits_per_capacity[i] += static_cast<int64_t>(hits);
            state->interval_hits_per_capacity[i] += static_cast<int64_t>(hits);
        }

        const uint64_t unique_blocks = state->lite_hit->current_unique_blocks();
        result.unique_keys_per_capacity.reserve(state->lite_hit_capacity_blocks.size());
        for (int64_t capacity : state->lite_hit_capacity_blocks) {
            result.unique_keys_per_capacity.push_back(
                ClampToInt64(std::min(unique_blocks, static_cast<uint64_t>(capacity))));
        }

        if (state->instance_group->enable_theoretical_max_cache()) {
            const uint64_t max_hits = HitCurveProjector::ProjectInfinite(fact);
            state->mrc_window.Record(fact);
            state->quota_mrc_window.Record(fact);
            state->quota_input_tokens = SaturatingAddUint64(state->quota_input_tokens, normalized.input_token_len);
            ++state->quota_accepted_facts;
            state->quota_newest_event_ns = std::max(state->quota_newest_event_ns, replay_timestamp_ns);
            result.max_hit_count = ClampToInt64(max_hits);
            result.max_hit_rate =
                normalized.input_token_len == 0 ? 0.0 : static_cast<double>(max_hits * block_size) / token_denominator;
            result.theoretical_unique_keys = ClampToInt64(unique_blocks);
            state->total_max_hits += static_cast<int64_t>(max_hits);
            state->interval_max_hits += static_cast<int64_t>(max_hits);
        } else {
            // Match the -1 sentinel of max_hit_count/theoretical_unique_keys:
            // "not computed" must stay distinguishable from "computed as 0".
            result.max_hit_rate = -1.0;
            result.theoretical_unique_keys = -1;
        }

        state->total_queries++;
        state->total_blocks_queried += total_blocks;
        state->total_input_tokens += ClampToInt64(normalized.input_token_len);
        state->interval_input_tokens += ClampToInt64(normalized.input_token_len);
        return EC_OK;
    }

    std::vector<int64_t> hit_count;
    int64_t max_hit_count;
    if (!state->indexer) {
        return EC_ERROR;
    }
    // Legacy analyzers keep their algorithm but share the same prefix-hash
    // preprocessing switch.
    if (state->instance_group->enable_prefix_hash()) {
        state->indexer->ProcessKeysAtTimestamp(
            ApplyPrefixHash(block_keys), replay_timestamp_ns, hit_count, max_hit_count);
    } else {
        state->indexer->ProcessKeysAtTimestamp(block_keys, replay_timestamp_ns, hit_count, max_hit_count);
    }

    state->indexer->PostQueryMaintenance();

    state->total_queries++;
    state->total_blocks_queried += total_blocks;
    for (size_t j = 0; j < num_caps; j++) {
        state->total_hits_per_capacity[j] += hit_count[j];
    }
    if (max_hit_count >= 0) {
        state->total_max_hits += max_hit_count;
    }

    hit_count.resize(num_caps);
    result.hit_count_per_capacity = std::move(hit_count);
    result.hit_rate_per_capacity.reserve(num_caps);
    for (int64_t hits : result.hit_count_per_capacity) {
        result.hit_rate_per_capacity.push_back(total_blocks > 0 ? static_cast<double>(hits) / total_blocks : 0.0);
    }
    result.unique_keys_per_capacity = state->indexer->capacity_unique_counts();
    result.theoretical_unique_keys = max_hit_count >= 0 ? state->indexer->unique_count() : -1;
    result.max_hit_count = max_hit_count;
    result.max_hit_rate = total_blocks > 0 && max_hit_count >= 0
                              ? static_cast<double>(max_hit_count) / static_cast<double>(total_blocks)
                              : 0.0;

    return EC_OK;
}

ErrorCode OnlineOptimizerManager::ListInstances(const std::string &instance_group_filter,
                                                std::vector<InstanceSummary> &summaries) const {
    std::shared_lock lock(instances_mutex_);
    summaries.clear();
    summaries.reserve(instances_.size());

    for (const auto &[id, state] : instances_) {
        if (!instance_group_filter.empty() && state->instance_info->instance_group_name() != instance_group_filter) {
            continue;
        }

        std::lock_guard<std::mutex> guard(state->mutex);
        InstanceSummary s;
        s.instance_id = id;
        s.instance_group = state->instance_info->instance_group_name();
        s.block_size = state->instance_info->block_size();
        s.total_blocks_queried = state->total_blocks_queried;
        s.bytes_per_block =
            (state->linear_step == 0)
                ? state->size_full
                : ((state->linear_step - 1) * state->size_full + state->size_full_linear) / state->linear_step;
        s.linear_step = state->linear_step;

        const auto &caps = state->instance_group->capacity_gb();
        if (state->linear_step == 0) {
            if (!state->lite_hit) {
                continue;
            }
            s.total_queries = state->total_queries;
            s.total_input_tokens = state->total_input_tokens;
            // Summaries arrive without traffic; advance the TTL watermark so
            // idle instances report the alive set as of now, not as of the
            // last request.
            state->lite_hit->AdvanceTime(static_cast<int64_t>(TimestampUtil::GetCurrentTimeUs()) * 1000);
            s.unique_keys = ClampToInt64(state->lite_hit->current_unique_blocks());
            s.ttl_eviction_count = ClampToInt64(state->lite_hit->ttl_expired_blocks());
            // Total-eviction contract: LiteHit has no capacity evictions, so
            // the total equals the TTL expirations (the linear wrapper also
            // counts harvested entries in both).
            s.eviction_count = s.ttl_eviction_count;
            s.memory_usage_bytes = ClampToInt64(state->lite_hit->memory_usage_bytes());

            // Capacity-unbounded residency: without a TTL every distinct
            // block ever seen counts; with a group TTL only the alive working
            // set does. Finite tiers are min(U, C) of this same U and need no
            // separate report.
            s.kv_cache_usage_bytes = SaturatingMultiplyToInt64(state->lite_hit->current_unique_blocks(),
                                                               static_cast<uint64_t>(state->size_full));
            // Full-attention rates are token based: cumulative hit blocks are
            // converted to tokens with the fixed block size and divided by the
            // cumulative input tokens.
            const double token_denominator = static_cast<double>(state->total_input_tokens);
            const int64_t block_size_tokens = state->instance_info->block_size();
            for (size_t i = 0; i < caps.size() && i < state->total_hits_per_capacity.size(); ++i) {
                PerCapacityHitRateInfo info;
                info.capacity_gb = caps[i];
                info.total_hits = state->total_hits_per_capacity[i];
                info.hit_rate = state->total_input_tokens > 0
                                    ? static_cast<double>(info.total_hits * block_size_tokens) / token_denominator
                                    : 0.0;
                s.per_capacity_hit_rates.push_back(info);
            }

            if (state->instance_group->enable_theoretical_max_cache()) {
                s.total_max_hits = state->total_max_hits;
                s.max_hit_rate = state->total_input_tokens > 0
                                     ? static_cast<double>(s.total_max_hits * block_size_tokens) / token_denominator
                                     : 0.0;
            } else {
                // Same -1 sentinel as TraceQuery: "not computed" stays
                // distinguishable from "computed as 0".
                s.max_hit_rate = -1.0;
            }
        } else {
            if (!state->indexer) {
                continue;
            }
            s.total_queries = state->total_queries;
            s.total_max_hits = state->total_max_hits;
            s.max_hit_rate = state->total_blocks_queried > 0 ? static_cast<double>(s.total_max_hits) /
                                                                   static_cast<double>(state->total_blocks_queried)
                                                             : 0.0;
            s.unique_keys = state->indexer->unique_count();
            s.eviction_count = state->indexer->eviction_count();
            s.memory_usage_bytes = state->indexer->memory_usage_bytes();
            s.kv_cache_usage_bytes = state->indexer->kv_cache_usage_bytes();
            s.ttl_eviction_count = state->indexer->ttl_eviction_count();

            for (size_t i = 0; i < caps.size() && i < state->total_hits_per_capacity.size(); i++) {
                PerCapacityHitRateInfo info;
                info.capacity_gb = caps[i];
                info.total_hits = state->total_hits_per_capacity[i];
                info.hit_rate = state->total_blocks_queried > 0 ? static_cast<double>(info.total_hits) /
                                                                      static_cast<double>(state->total_blocks_queried)
                                                                : 0.0;
                s.per_capacity_hit_rates.push_back(info);
            }

            // Hit-age is a legacy indexer statistic and is intentionally not
            // part of LiteHit's full-attention state.
            auto age_buckets = state->indexer->GetHitAgeBuckets();
            int64_t bucket_total = 0;
            for (const auto &bucket : age_buckets) {
                bucket_total += bucket.hit_count;
            }
            int64_t age_denom = s.total_max_hits > 0 ? s.total_max_hits : bucket_total;
            for (const auto &bucket : age_buckets) {
                HitAgeBucketRatio ratio_info;
                ratio_info.threshold_seconds = bucket.threshold_seconds;
                ratio_info.hit_count = bucket.hit_count;
                ratio_info.ratio =
                    age_denom > 0 ? static_cast<double>(bucket.hit_count) / static_cast<double>(age_denom) : 0.0;
                s.hit_age_bucket_ratios.push_back(ratio_info);
            }
        }

        summaries.push_back(std::move(s));
    }
    return EC_OK;
}

ErrorCode OnlineOptimizerManager::TakeMrcMetrics(std::vector<MrcMetricInfo> &metrics) {
    std::shared_lock lock(instances_mutex_);
    metrics.clear();

    for (const auto &[id, state] : instances_) {
        std::lock_guard<std::mutex> guard(state->mutex);
        if (state->linear_step != 0 || !state->lite_hit || !state->instance_group->enable_theoretical_max_cache()) {
            continue;
        }

        for (const auto &point : state->mrc_window.Take()) {
            MrcMetricInfo metric;
            metric.instance_id = id;
            metric.instance_group = state->instance_info->instance_group_name();
            metric.target_basis_points = point.target_basis_points;
            metric.capacity_bytes =
                SaturatingMultiplyToInt64(point.required_blocks, static_cast<uint64_t>(state->size_full));
            metrics.push_back(std::move(metric));
        }
    }
    return EC_OK;
}

OnlineMrcDecisionSnapshot OnlineOptimizerManager::TakeQuotaDecisionSnapshot(
    const std::map<std::string, std::vector<uint64_t>> &capacity_bytes_by_source, int64_t now_ns) {
    if (now_ns == 0) {
        now_ns = static_cast<int64_t>(TimestampUtil::GetCurrentTimeUs()) * 1000;
    }
    OnlineMrcDecisionSnapshot snapshot;
    snapshot.snapshot_id = ++quota_snapshot_generation_;
    snapshot.created_at_ns = now_ns;

    std::shared_lock lock(instances_mutex_);
    snapshot.sources.reserve(capacity_bytes_by_source.size());
    for (const auto &[source_id, capacity_bytes] : capacity_bytes_by_source) {
        OnlineMrcSourceSnapshot source;
        source.source_id = source_id;
        const auto state_it = instances_.find(source_id);
        if (state_it == instances_.end() || !state_it->second) {
            snapshot.sources.push_back(std::move(source));
            continue;
        }
        const auto &state = state_it->second;
        std::lock_guard<std::mutex> guard(state->mutex);
        std::vector<uint64_t> capacity_blocks;
        capacity_blocks.reserve(capacity_bytes.size());
        const uint64_t bytes_per_block = state->size_full > 0 ? static_cast<uint64_t>(state->size_full) : 0;
        for (uint64_t bytes : capacity_bytes) {
            capacity_blocks.push_back(bytes_per_block == 0 ? 0 : bytes / bytes_per_block);
        }
        const auto hit_blocks = state->quota_mrc_window.TakeHitCounts(capacity_blocks);
        source.newest_event_time_ns = state->quota_newest_event_ns;
        source.accepted_facts = state->quota_accepted_facts;
        source.curve.reserve(capacity_bytes.size());
        const uint64_t block_size = static_cast<uint64_t>(state->instance_info->block_size());
        for (size_t i = 0; i < capacity_bytes.size(); ++i) {
            source.curve.push_back({capacity_bytes[i],
                                    state->quota_input_tokens,
                                    i < hit_blocks.size() ? SaturatingMultiplyUint64(hit_blocks[i], block_size) : 0});
        }
        state->quota_input_tokens = 0;
        state->quota_accepted_facts = 0;
        state->quota_newest_event_ns = 0;
        snapshot.sources.push_back(std::move(source));
    }
    return snapshot;
}

ErrorCode OnlineOptimizerManager::TakeIntervalMetrics(std::vector<IntervalMetricInfo> &metrics) {
    std::shared_lock lock(instances_mutex_);
    metrics.clear();
    metrics.reserve(instances_.size());

    for (const auto &[id, state] : instances_) {
        std::lock_guard<std::mutex> guard(state->mutex);
        if (state->linear_step != 0 || !state->lite_hit) {
            continue;
        }

        IntervalMetricInfo metric;
        metric.instance_id = id;
        metric.instance_group = state->instance_info->instance_group_name();
        metric.has_theoretical_max_hit_rate = state->instance_group->enable_theoretical_max_cache();

        const auto &capacities = state->instance_group->capacity_gb();
        const double input_tokens = static_cast<double>(state->interval_input_tokens);
        const double block_size = static_cast<double>(state->instance_info->block_size());
        metric.per_capacity_hit_rates.reserve(capacities.size());
        for (size_t i = 0; i < capacities.size() && i < state->interval_hits_per_capacity.size(); ++i) {
            PerCapacityHitRateInfo capacity_metric;
            capacity_metric.capacity_gb = capacities[i];
            capacity_metric.total_hits = state->interval_hits_per_capacity[i];
            capacity_metric.hit_rate = state->interval_input_tokens > 0
                                           ? static_cast<double>(capacity_metric.total_hits) * block_size / input_tokens
                                           : std::numeric_limits<double>::quiet_NaN();
            metric.per_capacity_hit_rates.push_back(capacity_metric);
        }
        if (metric.has_theoretical_max_hit_rate) {
            metric.max_hit_rate = state->interval_input_tokens > 0
                                      ? static_cast<double>(state->interval_max_hits) * block_size / input_tokens
                                      : std::numeric_limits<double>::quiet_NaN();
        }

        state->interval_input_tokens = 0;
        std::fill(state->interval_hits_per_capacity.begin(), state->interval_hits_per_capacity.end(), 0);
        state->interval_max_hits = 0;
        metrics.push_back(std::move(metric));
    }
    return EC_OK;
}

ErrorCode OnlineOptimizerManager::ResetStats(const std::string &instance_id) {
    std::shared_ptr<InstanceState> state;
    {
        std::shared_lock lock(instances_mutex_);
        auto it = instances_.find(instance_id);
        if (it == instances_.end()) {
            return EC_INSTANCE_NOT_EXIST;
        }
        state = it->second;
    }

    std::lock_guard<std::mutex> guard(state->mutex);
    if (state->linear_step == 0) {
        if (!state->lite_hit) {
            return EC_ERROR;
        }
        state->lite_hit->Reset();
    } else {
        auto new_indexer =
            CacheIndexerFactory::CreateCacheIndexer(state->instance_group->eviction_policy(),
                                                    state->instance_group->enable_theoretical_max_cache(),
                                                    state->instance_group->capacity_gb(),
                                                    state->size_full,
                                                    state->size_full_linear,
                                                    state->linear_step,
                                                    state->instance_group->ttl_seconds());
        if (!new_indexer) {
            KVCM_LOG_ERROR("ResetStats failed: unsupported eviction_policy[%s] for instance[%s]",
                           state->instance_group->eviction_policy().c_str(),
                           instance_id.c_str());
            return EC_ERROR;
        }
        state->indexer = std::move(new_indexer);
    }
    state->total_queries = 0;
    state->total_blocks_queried = 0;
    state->total_input_tokens = 0;
    std::fill(state->total_hits_per_capacity.begin(), state->total_hits_per_capacity.end(), 0);
    state->total_max_hits = 0;
    state->interval_input_tokens = 0;
    std::fill(state->interval_hits_per_capacity.begin(), state->interval_hits_per_capacity.end(), 0);
    state->interval_max_hits = 0;
    state->mrc_window.Reset();
    state->quota_mrc_window.Reset();
    state->quota_input_tokens = 0;
    state->quota_accepted_facts = 0;
    state->quota_newest_event_ns = 0;
    KVCM_LOG_INFO("ResetStats OK: instance[%s]", instance_id.c_str());
    return EC_OK;
}

ErrorCode OnlineOptimizerManager::GetInstanceState(const std::string &instance_id,
                                                   std::function<void(const InstanceState &)> visitor) const {
    std::shared_ptr<InstanceState> state;
    {
        std::shared_lock lock(instances_mutex_);
        auto it = instances_.find(instance_id);
        if (it == instances_.end()) {
            return EC_INSTANCE_NOT_EXIST;
        }
        state = it->second;
    }
    std::lock_guard<std::mutex> guard(state->mutex);
    visitor(*state);
    return EC_OK;
}

ErrorCode OnlineOptimizerManager::Recover() {
    if (!registry_manager_) {
        KVCM_LOG_ERROR("OnlineOptimizerManager: Recover failed, registry_manager is null");
        return EC_ERROR;
    }

    OptimizerRegistryManager::RecoveryData data;
    auto ec = registry_manager_->LoadRecoveryData(data);
    if (ec != EC_OK) {
        KVCM_LOG_WARN("OnlineOptimizerManager: LoadRecoveryData failed, ec[%d]", static_cast<int>(ec));
        return ec;
    }

    // Hold admin_ops_mutex_ to serialize with RegisterInstance/RemoveInstance,
    // preventing stale recovery data from overwriting admin-applied state.
    std::lock_guard admin_guard(admin_ops_mutex_);

    size_t error_count = 0;
    for (const auto &info : data.instance_infos) {
        const auto &instance_id = info->instance_id();

        // Skip instances already recovered from a previous attempt to avoid
        // resetting their indexer and accumulated stats.
        {
            std::shared_lock lock(instances_mutex_);
            if (instances_.count(instance_id)) {
                continue;
            }
        }

        // Re-validate against persistence: if an admin Remove deleted this
        // record after LoadRecoveryData, do not reinsert it.
        if (!registry_manager_->GetInstanceInfo(instance_id)) {
            KVCM_LOG_INFO("OnlineOptimizerManager: recover instance[%s] skipped, no longer in registry",
                          instance_id.c_str());
            continue;
        }

        auto group = registry_manager_->GetInstanceGroup(info->instance_group_name());
        if (!group) {
            KVCM_LOG_WARN("OnlineOptimizerManager: recover instance[%s] skipped, group[%s] not found",
                          instance_id.c_str(),
                          info->instance_group_name().c_str());
            ++error_count;
            continue;
        }

        RegisterInstanceResult result;
        ec = RegisterInstanceInternal(*info, *group, result);
        if (ec != EC_OK) {
            KVCM_LOG_WARN("OnlineOptimizerManager: recover instance[%s] failed, ec=%d",
                          instance_id.c_str(),
                          static_cast<int>(ec));
            ++error_count;
            continue;
        }
    }

    KVCM_LOG_INFO("OnlineOptimizerManager: recover done, error_count[%zu], instance_num[%zu]",
                  error_count,
                  data.instance_infos.size());
    return error_count > 0 ? EC_ERROR : EC_OK;
}

} // namespace kv_cache_manager
