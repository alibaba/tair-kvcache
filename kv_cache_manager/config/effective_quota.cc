#include "kv_cache_manager/config/effective_quota.h"

#include <algorithm>
#include <limits>
#include <set>

#include "kv_cache_manager/data_storage/data_storage_manager.h"

namespace kv_cache_manager {
namespace {

int64_t ScaleCapacity(int64_t capacity, double ratio) {
    if (ratio == 1.0) {
        return capacity;
    }
    // Use the rounded double product for decimal ratios (0.3 * 100 = 30).
    // Guard the rounded INT64 boundary before conversion and never exceed
    // the physical capacity, including when its double representation rounds up.
    const double scaled = static_cast<double>(capacity) * ratio;
    return scaled >= static_cast<double>(INT64_MAX) ? capacity : std::min(capacity, static_cast<int64_t>(scaled));
}

InstanceGroupQuota EnforcementQuota(const EffectiveGroupQuota &effective, bool reclaim) {
    std::vector<QuotaConfig> configs;
    for (const auto &[type, value] : effective.types) {
        if (reclaim && value.config.automatic() && value.capacity < 0) {
            continue;
        }
        configs.emplace_back(std::max<int64_t>(0, value.capacity), type);
    }
    return InstanceGroupQuota(effective.configured_capacity, configs);
}
} // namespace

InstanceGroupQuota EffectiveGroupQuota::ForAdmission() const { return EnforcementQuota(*this, false); }
InstanceGroupQuota EffectiveGroupQuota::ForReclaim() const { return EnforcementQuota(*this, true); }

double QuotaUsageRatio(uint64_t usage, int64_t capacity) {
    return capacity > 0 ? static_cast<double>(usage) / capacity : 0.0;
}

std::set<std::string> GetQuotaStorageNames(const InstanceGroup &group) {
    std::set<std::string> associated(group.storage_candidates().begin(), group.storage_candidates().end());
    if (group.cache_config()) {
        for (const auto &strategy : group.cache_config()->migration_strategies()) {
            if (strategy) {
                associated.insert(strategy->source_storage_name());
                associated.insert(strategy->target_storage_name());
            }
        }
    }
    associated.erase("");
    return associated;
}

EffectiveGroupQuota ResolveEffectiveQuota(const InstanceGroup &group, const DataStorageManager *storage_manager) {
    const auto &configs = group.quota().quota_config();
    const bool needs_capacity = std::any_of(configs.begin(), configs.end(), [](const auto &q) { return q.automatic(); });
    if (!needs_capacity || !storage_manager) {
        return ResolveEffectiveQuota(group, std::map<std::string, StorageCapacitySnapshot>{});
    }
    return ResolveEffectiveQuota(group, storage_manager->GetCapacitySnapshots(GetQuotaStorageNames(group)));
}

namespace {
EffectiveTypeQuota ResolveTypeQuota(const QuotaConfig &config, const std::set<std::string> &names,
                                    const std::map<std::string, StorageCapacitySnapshot> &snapshots) {
    EffectiveTypeQuota value;
    value.config = config;
    if (!config.automatic()) {
        value.capacity = config.capacity();
        return value;
    }
    int64_t total = 0;
    int64_t oldest = std::numeric_limits<int64_t>::max();
    bool matched = false;
    bool refreshed = true;
    for (const auto &name : names) {
        const auto it = snapshots.find(name);
        if (it == snapshots.end()) {
            continue;
        }
        const auto &sample = it->second;
        if (ToBaseType(sample.type) != ToBaseType(config.storage_spec())) {
            continue;
        }
        if (!sample.HasKnownCapacity() || sample.capacity > INT64_MAX - total) {
            return value;
        }
        matched = true;
        total += sample.capacity;
        oldest = std::min(oldest, sample.last_success_time_s);
        refreshed = refreshed && sample.refresh_code == EC_OK;
    }
    if (matched) {
        value.backend_capacity = total;
        value.capacity = ScaleCapacity(total, config.ratio());
        value.refresh_success = refreshed;
        value.last_success_time_s = oldest;
    }
    return value;
}
} // namespace

EffectiveGroupQuota ResolveEffectiveQuota(const InstanceGroup &group,
                                          const std::map<std::string, StorageCapacitySnapshot> &snapshots) {
    EffectiveGroupQuota result;
    result.configured_capacity = group.quota().capacity();
    const auto &configs = group.quota().quota_config();
    const auto names = snapshots.empty() ? std::set<std::string>{} : GetQuotaStorageNames(group);
    int64_t reference = configs.empty() ? -1 : 0;
    for (const auto &config : configs) {
        auto value = ResolveTypeQuota(config, names, snapshots);
        if (reference >= 0 && value.capacity >= 0 && value.capacity <= INT64_MAX - reference) {
            reference += value.capacity;
        } else {
            reference = -1;
        }
        result.types.emplace(ToBaseType(config.storage_spec()), std::move(value));
    }
    result.reference_capacity = reference;
    return result;
}

} // namespace kv_cache_manager
