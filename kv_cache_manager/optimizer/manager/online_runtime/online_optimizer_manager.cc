#include "kv_cache_manager/optimizer/manager/online_runtime/online_optimizer_manager.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
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

bool ConvertCapacitiesToBytes(const std::vector<double> &capacity_gb, std::vector<uint64_t> &capacity_bytes) {
    capacity_bytes.clear();
    capacity_bytes.reserve(capacity_gb.size());
    for (double capacity : capacity_gb) {
        if (!std::isfinite(capacity) || capacity < 0.0) {
            return false;
        }
        const long double bytes = static_cast<long double>(capacity) * kBytesPerGb;
        if (bytes >= static_cast<long double>(std::numeric_limits<int64_t>::max())) {
            capacity_bytes.push_back(static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
        } else {
            capacity_bytes.push_back(static_cast<uint64_t>(bytes));
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

int64_t ClampToInt64(long double value) {
    if (value >= static_cast<long double>(std::numeric_limits<int64_t>::max())) {
        return std::numeric_limits<int64_t>::max();
    }
    return value <= 0.0L ? 0 : static_cast<int64_t>(value);
}

// Block count of one capacity tier for the register response.
//
// full-attention (step_blocks == 0): exact, and the same value doubles as the
// LiteHit projection slot because every block costs exactly full_charge_bytes.
//
// linear: an ESTIMATE only. Full blocks and Linear states have different
// charges, so no single block count describes the byte-axis cache; hits are
// always decided on the byte axis. The estimate amortizes one Linear state over
// step_blocks blocks.
int64_t CapacityBlocksForResponse(uint64_t capacity_bytes,
                                  int64_t full_charge_bytes,
                                  int64_t linear_charge_bytes,
                                  int32_t step_blocks) {
    if (step_blocks <= 0) {
        return ClampToInt64(capacity_bytes / static_cast<uint64_t>(full_charge_bytes));
    }
    const long double bytes_per_step =
        static_cast<long double>(step_blocks) * full_charge_bytes + static_cast<long double>(linear_charge_bytes);
    return ClampToInt64(static_cast<long double>(capacity_bytes) * step_blocks / bytes_per_step);
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
    std::string invalid_fields;
    if (!instance_group.ValidateRequiredFields(invalid_fields)) {
        KVCM_LOG_ERROR("CreateInstanceGroup failed: invalid group config %s", invalid_fields.c_str());
        return EC_BADARGS;
    }
    std::lock_guard admin_guard(admin_ops_mutex_);
    return registry_manager_->CreateInstanceGroup(instance_group);
}

ErrorCode OnlineOptimizerManager::UpdateInstanceGroup(const OptimizerInstanceGroup &instance_group) {
    if (!registry_manager_) {
        return EC_ERROR;
    }
    std::string invalid_fields;
    if (!instance_group.ValidateRequiredFields(invalid_fields)) {
        KVCM_LOG_ERROR("UpdateInstanceGroup failed: invalid group config %s", invalid_fields.c_str());
        return EC_BADARGS;
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
    const int32_t linear_step = instance_info.linear_step();
    if (instance_info.block_size() <= 0) {
        KVCM_LOG_ERROR("RegisterInstance failed: non-positive token block_size for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }
    // linear_step counts tokens; Linear states can only land on complete block
    // boundaries, so it must divide evenly into whole blocks.
    if (linear_step > 0 && linear_step % instance_info.block_size() != 0) {
        KVCM_LOG_ERROR(
            "RegisterInstance failed: linear_step=%d tokens is not a multiple of block_size=%d for instance[%s]",
            linear_step,
            instance_info.block_size(),
            instance_id.c_str());
        return EC_BADARGS;
    }
    const int32_t linear_step_blocks = linear_step / instance_info.block_size();
    std::string invalid_group_fields;
    if (!instance_group.ValidateRequiredFields(invalid_group_fields)) {
        KVCM_LOG_ERROR("RegisterInstance failed: invalid group config %s for instance[%s]",
                       invalid_group_fields.c_str(),
                       instance_id.c_str());
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
    if (linear_step == 0 && !optimizer_state_info.linear_location_spec_group_name().empty()) {
        KVCM_LOG_ERROR(
            "RegisterInstance failed: full-attention instance[%s] must not configure a linear/Mamba spec group",
            instance_id.c_str());
        return EC_BADARGS;
    }
    if (linear_step > 0 && optimizer_state_info.linear_location_spec_group_name().empty()) {
        KVCM_LOG_ERROR("RegisterInstance failed: linear instance[%s] requires a non-empty Mamba spec group",
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
    int64_t full_charge_bytes = ComputeSizeForGroup(specs, *full_group);
    if (full_charge_bytes <= 0) {
        KVCM_LOG_ERROR("RegisterInstance failed: invalid full group[%s] size[%ld] for instance[%s]",
                       full_group->name().c_str(),
                       full_charge_bytes,
                       instance_id.c_str());
        return EC_BADARGS;
    }

    int64_t linear_charge_bytes = 0;
    if (!optimizer_state_info.linear_location_spec_group_name().empty()) {
        const auto *linear_group =
            FindLocationSpecGroup(groups, optimizer_state_info.linear_location_spec_group_name());
        if (!linear_group) {
            KVCM_LOG_ERROR("RegisterInstance failed: linear group[%s] not found for instance[%s]",
                           optimizer_state_info.linear_location_spec_group_name().c_str(),
                           instance_id.c_str());
            return EC_BADARGS;
        }
        linear_charge_bytes = ComputeSizeForGroup(specs, *linear_group);
        if (linear_charge_bytes <= 0) {
            KVCM_LOG_ERROR("RegisterInstance failed: invalid linear group[%s] size[%ld] for instance[%s]",
                           linear_group->name().c_str(),
                           linear_charge_bytes,
                           instance_id.c_str());
            return EC_BADARGS;
        }
    }

    const auto &capacity_gb = instance_group.capacity_gb();
    std::vector<uint64_t> capacity_bytes;
    if (!ConvertCapacitiesToBytes(capacity_gb, capacity_bytes)) {
        KVCM_LOG_ERROR("RegisterInstance failed: invalid capacity for instance[%s]", instance_id.c_str());
        return EC_BADARGS;
    }
    std::vector<int64_t> estimated_capacity_blocks;
    estimated_capacity_blocks.reserve(capacity_bytes.size());
    for (uint64_t bytes : capacity_bytes) {
        estimated_capacity_blocks.push_back(
            CapacityBlocksForResponse(bytes, full_charge_bytes, linear_charge_bytes, linear_step_blocks));
    }

    auto state = std::make_shared<InstanceState>();
    state->instance_info = std::make_shared<OptimizerInstanceInfo>(instance_info);
    state->instance_group = std::make_shared<OptimizerInstanceGroup>(instance_group);

    state->full_charge_bytes = full_charge_bytes;
    state->linear_charge_bytes = linear_charge_bytes;
    state->linear_step = linear_step;
    state->linear_step_blocks = linear_step_blocks;
    state->total_hits_per_capacity.resize(capacity_gb.size(), 0);
    state->capacity_bytes = capacity_bytes;

    LiteHit::CacheObjectConfig object_config;
    object_config.full_charge_bytes = static_cast<uint64_t>(full_charge_bytes);
    if (linear_step != 0) {
        object_config.linear_charge_bytes = static_cast<uint64_t>(linear_charge_bytes);
        object_config.linear_step_blocks = static_cast<uint64_t>(linear_step_blocks);
    }
    const uint64_t ttl_ns = static_cast<uint64_t>(instance_group.ttl_seconds()) * 1000000000ULL;
    state->lite_hit = std::make_unique<TtlLiteHit>(object_config, ttl_ns);

    {
        std::unique_lock lock(instances_mutex_);
        instances_[instance_id] = std::move(state);
    }

    result.estimated_capacity_blocks = estimated_capacity_blocks;
    result.full_charge_bytes = full_charge_bytes;
    result.linear_charge_bytes = linear_charge_bytes;

    KVCM_LOG_INFO(
        "RegisterInstance OK: instance[%s] group[%s] linear_step=%d full_charge=%ld linear_charge=%ld caps=%zu",
        instance_id.c_str(),
        instance_info.instance_group_name().c_str(),
        linear_step,
        full_charge_bytes,
        linear_charge_bytes,
        capacity_bytes.size());
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

    const uint64_t block_size = static_cast<uint64_t>(state->instance_info->block_size());
    if (!block_keys.empty() &&
        block_size > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / block_keys.size()) {
        return EC_BADARGS;
    }
    const int64_t input_token_len = static_cast<int64_t>(block_keys.size() * block_size);
    return TraceQuery(instance_id, block_keys, input_token_len, result);
}

ErrorCode OnlineOptimizerManager::TraceQuery(const std::string &instance_id,
                                             const std::vector<int64_t> &block_keys,
                                             int64_t input_token_len,
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
        if (input_token_len < 0) {
            return EC_BADARGS;
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

        const FullRequestFact fact =
            state->lite_hit->ProcessFullRequest(normalized.block_keys, TimestampUtil::GetCurrentTimeUs() * 1000);
        result.input_token_len = ClampToInt64(normalized.input_token_len);

        const uint64_t block_size = static_cast<uint64_t>(state->instance_info->block_size());
        const double token_denominator = static_cast<double>(normalized.input_token_len);
        result.hit_count_per_capacity.reserve(num_caps);
        result.hit_rate_per_capacity.reserve(num_caps);
        for (std::size_t i = 0; i < num_caps; ++i) {
            const uint64_t hits = HitCurveProjector::ProjectFullBytes(
                fact, state->capacity_bytes[i], static_cast<uint64_t>(state->full_charge_bytes));
            result.hit_count_per_capacity.push_back(ClampToInt64(hits));
            result.hit_rate_per_capacity.push_back(
                normalized.input_token_len == 0 ? 0.0 : static_cast<double>(hits * block_size) / token_denominator);
            state->total_hits_per_capacity[i] += static_cast<int64_t>(hits);
        }

        const uint64_t unique_blocks = state->lite_hit->current_unique_blocks();
        result.unique_keys_per_capacity.reserve(state->capacity_bytes.size());
        for (uint64_t capacity_bytes : state->capacity_bytes) {
            result.unique_keys_per_capacity.push_back(
                ClampToInt64(state->lite_hit->FullObjectsWithinTotalBytes(capacity_bytes)));
        }

        if (state->instance_group->enable_theoretical_max_cache()) {
            const uint64_t max_hits = HitCurveProjector::ProjectFullInfinite(fact);
            result.max_hit_count = ClampToInt64(max_hits);
            result.max_hit_rate =
                normalized.input_token_len == 0 ? 0.0 : static_cast<double>(max_hits * block_size) / token_denominator;
            result.theoretical_unique_keys = ClampToInt64(unique_blocks);
            state->total_max_hits += static_cast<int64_t>(max_hits);
        } else {
            // Match the -1 sentinel of max_hit_count/theoretical_unique_keys:
            // "not computed" must stay distinguishable from "computed as 0".
            result.max_hit_rate = -1.0;
            result.theoretical_unique_keys = -1;
        }

        state->total_queries++;
        state->total_blocks_queried += total_blocks;
        state->total_input_tokens += ClampToInt64(normalized.input_token_len);
        return EC_OK;
    }

    if (state->lite_hit) {
        if (input_token_len < 0) {
            return EC_BADARGS;
        }

        NormalizedRequest normalized;
        try {
            normalized = NormalizeRequest(block_keys,
                                          input_token_len,
                                          static_cast<uint64_t>(state->instance_info->block_size()),
                                          state->instance_group->enable_prefix_hash());
        } catch (const std::invalid_argument &e) {
            KVCM_LOG_ERROR(
                "TraceQuery failed: invalid Mamba request for instance[%s]: %s", instance_id.c_str(), e.what());
            return EC_BADARGS;
        }

        const RequestFact fact =
            state->lite_hit->ProcessRequest(normalized.block_keys, TimestampUtil::GetCurrentTimeUs() * 1000);
        result.input_token_len = ClampToInt64(normalized.input_token_len);

        const uint64_t block_size = static_cast<uint64_t>(state->instance_info->block_size());
        const double token_denominator = static_cast<double>(normalized.input_token_len);
        result.hit_count_per_capacity.reserve(num_caps);
        result.hit_rate_per_capacity.reserve(num_caps);
        for (std::size_t i = 0; i < num_caps; ++i) {
            const uint64_t hits = HitCurveProjector::ProjectBytes(fact, state->capacity_bytes[i]);
            result.hit_count_per_capacity.push_back(ClampToInt64(hits));
            result.hit_rate_per_capacity.push_back(
                normalized.input_token_len == 0 ? 0.0 : static_cast<double>(hits * block_size) / token_denominator);
            state->total_hits_per_capacity[i] += static_cast<int64_t>(hits);
        }

        result.unique_keys_per_capacity.reserve(state->capacity_bytes.size());
        for (uint64_t capacity_bytes : state->capacity_bytes) {
            result.unique_keys_per_capacity.push_back(
                ClampToInt64(state->lite_hit->FullObjectsWithinTotalBytes(capacity_bytes)));
        }

        if (state->instance_group->enable_theoretical_max_cache()) {
            const uint64_t max_hits = HitCurveProjector::ProjectInfinite(fact);
            result.max_hit_count = ClampToInt64(max_hits);
            result.max_hit_rate =
                normalized.input_token_len == 0 ? 0.0 : static_cast<double>(max_hits * block_size) / token_denominator;
            result.theoretical_unique_keys = ClampToInt64(state->lite_hit->current_unique_blocks());
            state->total_max_hits += static_cast<int64_t>(max_hits);
        } else {
            result.max_hit_rate = -1.0;
            result.theoretical_unique_keys = -1;
        }

        state->total_queries++;
        state->total_blocks_queried += total_blocks;
        state->total_input_tokens += ClampToInt64(normalized.input_token_len);
        return EC_OK;
    }

    return EC_ERROR;
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
        // Exact configured charge for full-attention. The Mamba branch below
        // replaces it with the current resident working-set average.
        s.bytes_per_block = static_cast<double>(state->full_charge_bytes);
        s.linear_step = state->linear_step;

        if (!state->lite_hit) {
            continue;
        }
        // Summaries may arrive without traffic. Advance the shared TTL
        // watermark for both Full-only and Linear instances so observability
        // reflects the alive working set as of now.
        state->lite_hit->AdvanceTime(static_cast<int64_t>(TimestampUtil::GetCurrentTimeUs()) * 1000);
        s.total_queries = state->total_queries;
        s.total_input_tokens = state->total_input_tokens;
        s.ttl_eviction_count = ClampToInt64(state->lite_hit->ttl_expired_blocks());
        // LiteHit models an unbounded recency stack, so it has no capacity
        // evictions; the total equals expired Full blocks in both modes.
        s.eviction_count = s.ttl_eviction_count;
        s.memory_usage_bytes = ClampToInt64(state->lite_hit->memory_usage_bytes());

        const auto &caps = state->instance_group->capacity_gb();
        if (state->linear_step == 0) {
            s.unique_keys = ClampToInt64(state->lite_hit->current_unique_blocks());

            // Capacity-unbounded residency: without a TTL every distinct
            // block ever seen counts; with a group TTL only the alive working
            // set does. Finite tiers are min(U, C) of this same U and need no
            // separate report.
            s.kv_cache_usage_bytes = SaturatingMultiplyToInt64(state->lite_hit->current_unique_blocks(),
                                                               static_cast<uint64_t>(state->full_charge_bytes));

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
            // unique_keys intentionally counts only Full objects.
            const uint64_t resident_full_blocks = state->lite_hit->current_unique_blocks();
            const uint64_t resident_bytes = state->lite_hit->resident_bytes();
            s.unique_keys = ClampToInt64(resident_full_blocks);
            // Infinite-capacity working set, Full and Mamba bytes included.
            s.kv_cache_usage_bytes = ClampToInt64(resident_bytes);
            s.bytes_per_block =
                resident_full_blocks == 0
                    ? 0.0
                    : static_cast<double>(static_cast<long double>(resident_bytes) / resident_full_blocks);

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
                s.max_hit_rate = -1.0;
            }
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
    if (!state->lite_hit) {
        return EC_ERROR;
    }
    state->lite_hit->Reset();
    state->total_queries = 0;
    state->total_blocks_queried = 0;
    state->total_input_tokens = 0;
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
