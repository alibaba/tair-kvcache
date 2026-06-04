#include "kv_cache_manager/online_optimizer/manager/online_optimizer_manager.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/online_optimizer/config/optimizer_registry_manager.h"

namespace kv_cache_manager {

OnlineOptimizerManager::OnlineOptimizerManager(std::shared_ptr<OptimizerRegistryManager> registry_manager)
    : registry_manager_(std::move(registry_manager)) {}

int64_t OnlineOptimizerManager::ComputeSizeForGroup(const std::vector<LocationSpecInfo> &specs,
                                                     const std::vector<LocationSpecGroup> &groups,
                                                     const std::string &group_name) {
    for (const auto &g : groups) {
        if (g.name() == group_name) {
            int64_t total = 0;
            for (const auto &spec_name : g.spec_names()) {
                for (const auto &spec : specs) {
                    if (spec.name() == spec_name) {
                        total += spec.size();
                        break;
                    }
                }
            }
            return total;
        }
    }
    return 0;
}

int64_t OnlineOptimizerManager::ComputeTotalSize(const std::vector<LocationSpecInfo> &specs) {
    int64_t total = 0;
    for (const auto &spec : specs) {
        total += spec.size();
    }
    return total;
}

ErrorCode OnlineOptimizerManager::RegisterInstance(const OptimizerInstanceInfo &instance_info,
                                                    const OptimizerInstanceGroup &instance_group,
                                                    RegisterInstanceResult &result) {
    const auto &instance_id = instance_info.instance_id();
    if (instance_id.empty()) {
        KVCM_LOG_ERROR("RegisterInstance failed: empty instance_id");
        return EC_BADARGS;
    }

    // Persist first, then build internal state
    if (registry_manager_) {
        auto ec = registry_manager_->SaveInstanceInfo(instance_info);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("RegisterInstance failed: persist instance_info[%s] failed", instance_id.c_str());
            return ec;
        }
    }

    auto ec = RegisterInstanceInternal(instance_info, instance_group, result);
    if (ec != EC_OK) {
        // Rollback persistence on failure
        if (registry_manager_) {
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
    if (!instance_group.enabled()) {
        KVCM_LOG_ERROR("RegisterInstance failed: optimizer not enabled for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }

    const auto &groups = instance_info.location_spec_groups();
    int32_t linear_step = std::max(instance_info.linear_step(), int32_t(1));
    const auto &full_group_name = instance_info.full_group_name();

    int64_t size_full_only;
    // 默认linear group包含所有spec，即spec total size
    int64_t size_full_linear = ComputeTotalSize(specs);

    if (linear_step <= 1 || full_group_name.empty()) {
        size_full_only = size_full_linear;
    } else {
        // Compute size for full_group_name
        size_full_only = ComputeSizeForGroup(specs, groups, full_group_name);
        if (size_full_only <= 0) {
            KVCM_LOG_WARN("full_group_name[%s] not found or zero size, falling back to total size",
                          full_group_name.c_str());
            size_full_only = size_full_linear;
        }
    }

    int64_t avg_bytes_per_block;
    if (linear_step <= 1) {
        avg_bytes_per_block = size_full_linear;
    } else {
        avg_bytes_per_block =
            ((linear_step - 1) * size_full_only + size_full_linear) / linear_step;
    }

    if (avg_bytes_per_block <= 0) {
        KVCM_LOG_ERROR("RegisterInstance failed: avg_bytes_per_block <= 0 for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }

    const auto &capacity_gb = instance_group.capacity_gb();
    std::vector<int64_t> capacity_blocks(capacity_gb.size());
    int64_t max_cap = 0;
    for (size_t i = 0; i < capacity_gb.size(); i++) {
        int64_t bytes = static_cast<int64_t>(capacity_gb[i] * 1024.0 * 1024.0 * 1024.0);
        capacity_blocks[i] = bytes / avg_bytes_per_block;
        max_cap = std::max(max_cap, capacity_blocks[i]);
    }

    auto state = std::make_shared<InstanceState>();
    state->instance_info = std::make_shared<OptimizerInstanceInfo>(instance_info);
    state->instance_group = std::make_shared<OptimizerInstanceGroup>(instance_group);

    state->size_full_only = size_full_only;
    state->size_full_linear = size_full_linear;
    state->linear_step = linear_step;
    state->avg_bytes_per_block = avg_bytes_per_block;
    state->capacity_blocks = capacity_blocks;
    state->max_capacity_blocks = max_cap;
    state->primary_capacity_index = instance_group.primary_capacity_index();
    state->total_hits_per_capacity.resize(capacity_gb.size(), 0);

    state->indexer = CreateCacheIndexer(instance_group.indexer_type(), instance_group.max_key_count());

    {
        std::unique_lock lock(instances_mutex_);
        instances_[instance_id] = std::move(state);
    }

    result.capacity_blocks = capacity_blocks;
    result.avg_bytes_per_block = avg_bytes_per_block;
    result.size_full_only = size_full_only;
    result.size_full_linear = size_full_linear;

    KVCM_LOG_INFO("RegisterInstance OK: instance[%s] group[%s] linear_step=%d avg_bytes=%ld caps=%zu",
                  instance_id.c_str(),
                  instance_info.instance_group_name().c_str(),
                  linear_step,
                  avg_bytes_per_block,
                  capacity_blocks.size());
    return EC_OK;
}

ErrorCode OnlineOptimizerManager::RemoveInstance(const std::string &instance_id) {
    {
        std::unique_lock lock(instances_mutex_);
        auto it = instances_.find(instance_id);
        if (it == instances_.end()) {
            return EC_INSTANCE_NOT_EXIST;
        }
        instances_.erase(it);
    }

    if (registry_manager_) {
        registry_manager_->DeleteInstanceInfo(instance_id);
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
    const size_t num_caps = state->capacity_blocks.size();

    std::vector<int64_t> hit_count(num_caps, 0);
    std::vector<int64_t> first_miss(num_caps, total_blocks);

    for (int64_t i = 0; i < total_blocks; i++) {
        int64_t sd = state->indexer->ProcessKey(block_keys[i]);

        for (size_t j = 0; j < num_caps; j++) {
            if (i < first_miss[j]) {
                if (sd >= state->capacity_blocks[j] || sd == INT64_MAX) {
                    first_miss[j] = i;
                }
            }
        }
    }

    for (size_t j = 0; j < num_caps; j++) {
        hit_count[j] = first_miss[j];
    }

    state->indexer->PostQueryMaintenance();

    state->total_queries++;
    state->total_blocks_queried += total_blocks;
    for (size_t j = 0; j < num_caps; j++) {
        state->total_hits_per_capacity[j] += hit_count[j];
    }

    int32_t primary_idx = state->primary_capacity_index;
    if (primary_idx >= 0 && primary_idx < static_cast<int32_t>(num_caps)) {
        result.cache_hit_count = hit_count[primary_idx];
    } else if (!hit_count.empty()) {
        result.cache_hit_count = hit_count[0];
    }
    result.total_blocks = total_blocks;
    result.hit_count_per_capacity = hit_count;
    result.capacity_gb = state->instance_group->capacity_gb();
    result.current_unique_keys = state->indexer->unique_count();

    return EC_OK;
}

ErrorCode OnlineOptimizerManager::ListInstances(const std::string &instance_group_filter,
                                                 std::vector<InstanceSummary> &summaries) const {
    std::shared_lock lock(instances_mutex_);
    summaries.clear();
    summaries.reserve(instances_.size());

    for (const auto &[id, state] : instances_) {
        if (!instance_group_filter.empty() &&
            state->instance_info->instance_group_name() != instance_group_filter) {
            continue;
        }

        std::lock_guard<std::mutex> guard(state->mutex);
        InstanceSummary s;
        s.instance_id = id;
        s.instance_group = state->instance_info->instance_group_name();
        s.block_size = state->instance_info->block_size();
        s.total_queries = state->total_queries;
        s.total_blocks_queried = state->total_blocks_queried;

        int32_t primary_idx = state->primary_capacity_index;
        if (primary_idx >= 0 && primary_idx < static_cast<int32_t>(state->total_hits_per_capacity.size())) {
            s.total_hits = state->total_hits_per_capacity[primary_idx];
        }
        s.hit_rate = state->total_blocks_queried > 0
                         ? static_cast<double>(s.total_hits) / static_cast<double>(state->total_blocks_queried)
                         : 0.0;
        s.unique_keys = state->indexer->unique_count();
        s.peak_unique_keys = state->indexer->peak_unique_count();
        s.avg_bytes_per_block = state->avg_bytes_per_block;
        s.linear_step = state->linear_step;
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
    state->indexer = CreateCacheIndexer(state->instance_group->indexer_type(), state->instance_group->max_key_count());
    state->total_queries = 0;
    state->total_blocks_queried = 0;
    std::fill(state->total_hits_per_capacity.begin(), state->total_hits_per_capacity.end(), 0);
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
        return EC_OK;
    }

    OptimizerRegistryManager::RecoveryData data;
    auto ec = registry_manager_->LoadRecoveryData(data);
    if (ec != EC_OK) {
        KVCM_LOG_WARN("OnlineOptimizerManager: LoadRecoveryData failed, ec[%d]", static_cast<int>(ec));
        return ec;
    }

    size_t error_count = 0;
    for (const auto &info : data.instance_infos) {
        auto group = registry_manager_->GetInstanceGroup(info->instance_group_name());
        if (!group) {
            KVCM_LOG_WARN("OnlineOptimizerManager: recover instance[%s] skipped, group[%s] not found",
                          info->instance_id().c_str(), info->instance_group_name().c_str());
            ++error_count;
            continue;
        }

        RegisterInstanceResult result;
        ec = RegisterInstanceInternal(*info, *group, result);
        if (ec != EC_OK) {
            KVCM_LOG_WARN("OnlineOptimizerManager: recover instance[%s] failed, ec=%d",
                          info->instance_id().c_str(), static_cast<int>(ec));
            ++error_count;
            continue;
        }
    }

    KVCM_LOG_INFO("OnlineOptimizerManager: recover done, error_count[%zu], instance_num[%zu]",
                  error_count, data.instance_infos.size());
    return error_count > 0 ? EC_ERROR : EC_OK;
}

} // namespace kv_cache_manager
