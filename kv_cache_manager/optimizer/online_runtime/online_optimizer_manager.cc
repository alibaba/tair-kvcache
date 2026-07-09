#include "kv_cache_manager/optimizer/online_runtime/online_optimizer_manager.h"

#include <algorithm>
#include <climits>
#include <cmath>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/index/online/cache_indexer_factory.h"

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

    auto instance_group = registry_manager_->GetInstanceGroup(instance_info.instance_group_name());
    if (!instance_group) {
        KVCM_LOG_ERROR("RegisterInstance failed: instance group[%s] not found for instance[%s]",
                       instance_info.instance_group_name().c_str(),
                       instance_id.c_str());
        return EC_NOENT;
    }

    // Save old persisted info before overwriting, so we can restore on rollback.
    auto old_instance_info = registry_manager_->GetInstanceInfo(instance_id);

    auto ec = registry_manager_->SaveInstanceInfo(instance_info);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("RegisterInstance failed: persist instance_info[%s] failed", instance_id.c_str());
        return ec;
    }

    ec = RegisterInstanceInternal(instance_info, *instance_group, result);
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
    int64_t size_full_only = ComputeSizeForGroup(specs, *full_group);
    if (size_full_only <= 0) {
        KVCM_LOG_ERROR("RegisterInstance failed: invalid full group[%s] size[%ld] for instance[%s]",
                       full_group->name().c_str(),
                       size_full_only,
                       instance_id.c_str());
        return EC_BADARGS;
    }

    int64_t size_full_linear = size_full_only;
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
        estimated_bytes_per_block = size_full_only;
    } else if (linear_step == 1) {
        estimated_bytes_per_block = size_full_linear;
    } else {
        estimated_bytes_per_block = ((linear_step - 1) * size_full_only + size_full_linear) / linear_step;
    }

    if (estimated_bytes_per_block <= 0) {
        KVCM_LOG_ERROR("RegisterInstance failed: estimated_bytes_per_block <= 0 for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }

    const auto &capacity_gb = instance_group.capacity_gb();
    std::vector<int64_t> estimated_capacity_blocks(capacity_gb.size());
    for (size_t i = 0; i < capacity_gb.size(); i++) {
        int64_t bytes = static_cast<int64_t>(capacity_gb[i] * 1024.0 * 1024.0 * 1024.0);
        estimated_capacity_blocks[i] = bytes / estimated_bytes_per_block;
    }

    auto state = std::make_shared<InstanceState>();
    state->instance_info = std::make_shared<OptimizerInstanceInfo>(instance_info);
    state->instance_group = std::make_shared<OptimizerInstanceGroup>(instance_group);

    state->size_full_only = size_full_only;
    state->size_full_linear = size_full_linear;
    state->linear_step = linear_step;
    state->total_hits_per_capacity.resize(capacity_gb.size(), 0);

    auto indexer = CacheIndexerFactory::CreateCacheIndexer(instance_group.eviction_policy(),
                                                           instance_group.enable_theoretical_max_cache(),
                                                           capacity_gb,
                                                           size_full_only,
                                                           size_full_linear,
                                                           linear_step,
                                                           instance_group.ttl_seconds());
    if (!indexer) {
        KVCM_LOG_ERROR("RegisterInstance failed: unsupported eviction_policy[%s] for instance[%s]",
                       instance_group.eviction_policy().c_str(),
                       instance_id.c_str());
        return EC_BADARGS;
    }
    state->indexer = std::move(indexer);

    {
        std::unique_lock lock(instances_mutex_);
        instances_[instance_id] = std::move(state);
    }

    result.estimated_capacity_blocks = estimated_capacity_blocks;
    result.size_full_only = size_full_only;
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
                                             TraceQueryResult &result) {
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

    const int64_t total_blocks = static_cast<int64_t>(block_keys.size());
    const size_t num_caps = state->total_hits_per_capacity.size();

    std::vector<int64_t> hit_count;
    int64_t max_hit_count;
    state->indexer->ProcessKeys(block_keys, hit_count, max_hit_count);

    state->indexer->PostQueryMaintenance();

    state->total_queries++;
    state->total_blocks_queried += total_blocks;
    for (size_t j = 0; j < num_caps; j++) {
        state->total_hits_per_capacity[j] += hit_count[j];
    }
    if (max_hit_count >= 0) {
        state->total_max_hits += max_hit_count;
    }

    result.cache_hit_count = hit_count.empty() ? 0 : hit_count[0];
    result.total_blocks = total_blocks;
    hit_count.resize(num_caps);
    result.hit_count_per_capacity = std::move(hit_count);
    result.capacity_gb = state->instance_group->capacity_gb();
    result.unique_keys_per_capacity = state->indexer->capacity_unique_counts();
    result.current_unique_keys = state->indexer->unique_count();
    result.theoretical_unique_keys = max_hit_count >= 0 ? state->indexer->unique_count() : -1;
    result.max_hit_count = max_hit_count;

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
        s.total_queries = state->total_queries;
        s.total_blocks_queried = state->total_blocks_queried;

        s.total_max_hits = state->total_max_hits;
        s.max_hit_rate = state->total_blocks_queried > 0
                             ? static_cast<double>(s.total_max_hits) / static_cast<double>(state->total_blocks_queried)
                             : 0.0;
        s.unique_keys = state->indexer->unique_count();
        s.eviction_count = state->indexer->eviction_count();
        s.memory_usage_bytes = state->indexer->memory_usage_bytes();
        s.kv_cache_usage_bytes = state->indexer->kv_cache_usage_bytes();
        s.ttl_eviction_count = state->indexer->ttl_eviction_count();
        s.avg_bytes_per_block =
            (state->linear_step == 0)
                ? state->size_full_only
                : ((state->linear_step - 1) * state->size_full_only + state->size_full_linear) / state->linear_step;
        s.linear_step = state->linear_step;

        const auto &caps = state->instance_group->capacity_gb();
        for (size_t i = 0; i < caps.size() && i < state->total_hits_per_capacity.size(); i++) {
            PerCapacityHitRateInfo info;
            info.capacity_gb = caps[i];
            info.total_hits = state->total_hits_per_capacity[i];
            info.hit_rate = state->total_blocks_queried > 0 ? static_cast<double>(info.total_hits) /
                                                                  static_cast<double>(state->total_blocks_queried)
                                                            : 0.0;
            s.per_capacity_hit_rates.push_back(info);
        }

        // Collect hit age bucket distribution
        auto age_buckets = state->indexer->GetHitAgeBuckets();
        int64_t bucket_total = 0;
        for (const auto &bucket : age_buckets) {
            bucket_total += bucket.hit_count;
        }
        // Use bucket_total as denominator when total_max_hits is unavailable (bounded indexer).
        int64_t age_denom = s.total_max_hits > 0 ? s.total_max_hits : bucket_total;
        for (const auto &bucket : age_buckets) {
            HitAgeBucketRatio ratio_info;
            ratio_info.threshold_seconds = bucket.threshold_seconds;
            ratio_info.hit_count = bucket.hit_count;
            ratio_info.ratio =
                age_denom > 0 ? static_cast<double>(bucket.hit_count) / static_cast<double>(age_denom) : 0.0;
            s.hit_age_bucket_ratios.push_back(ratio_info);
        }

        summaries.push_back(std::move(s));
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
    auto new_indexer = CacheIndexerFactory::CreateCacheIndexer(state->instance_group->eviction_policy(),
                                                               state->instance_group->enable_theoretical_max_cache(),
                                                               state->instance_group->capacity_gb(),
                                                               state->size_full_only,
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
    state->total_queries = 0;
    state->total_blocks_queried = 0;
    std::fill(state->total_hits_per_capacity.begin(), state->total_hits_per_capacity.end(), 0);
    state->total_max_hits = 0;
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
