#include "kv_cache_manager/config/effective_quota.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace kv_cache_manager {
namespace {
constexpr int64_t kCapacityMaxAgeUs = 30000000;

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
        if (reclaim && !value.usable) {
            continue;
        }
        configs.emplace_back(value.usable ? value.capacity : 0, type);
    }
    return InstanceGroupQuota(effective.configured_capacity, configs);
}
} // namespace

InstanceGroupQuota EffectiveGroupQuota::ForAdmission() const { return EnforcementQuota(*this, false); }
InstanceGroupQuota EffectiveGroupQuota::ForReclaim() const { return EnforcementQuota(*this, true); }

double QuotaUsageRatio(uint64_t usage, int64_t capacity) {
    if (capacity < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (capacity == 0) {
        return usage == 0 ? 0.0 : std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(usage) / capacity;
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

EffectiveGroupQuota ResolveEffectiveQuota(const InstanceGroup &group,
                                          const std::map<std::string, StorageCapacitySnapshot> &snapshots,
                                          int64_t now_steady_us) {
    EffectiveGroupQuota result;
    result.configured_capacity = group.quota().capacity();
    const auto associated = GetQuotaStorageNames(group);
    auto configs = group.quota().quota_config();
    // Unlimited mode defaults every associated type to auto 100%, including
    // types without an override when another type has a fixed/custom quota.
    // Existing positive fixed totals retain their legacy missing-type semantics.
    bool associations_complete = true;
    if (result.configured_capacity == -1) {
        result.automatic_defaults = true;
        std::set<DataStorageType> types;
        std::set<DataStorageType> configured_types;
        for (const auto &config : configs) {
            configured_types.insert(ToBaseType(config.storage_spec()));
        }
        for (const auto &name : associated) {
            const auto it = snapshots.find(name);
            if (it != snapshots.end()) {
                types.insert(ToBaseType(it->second.type));
            } else {
                associations_complete = false;
            }
        }
        for (const auto type : types) {
            if (configured_types.count(type) != 0) {
                continue;
            }
            QuotaConfig config(0, type);
            config.set_mode("auto");
            configs.push_back(config);
        }
    }
    bool complete = !configs.empty() && associations_complete;
    int64_t reference = 0;
    for (const auto &config : configs) {
        const auto type = ToBaseType(config.storage_spec());
        auto &value = result.types[type];
        value.config = config;
        std::string invalid;
        if (!config.ValidateRequiredFields(invalid)) {
            value.status = "invalid_config";
        } else {
            std::set<std::string> names = associated;
            const bool explicit_scope = !config.storage_names().empty();
            if (explicit_scope) {
                names = std::set<std::string>(config.storage_names().begin(), config.storage_names().end());
            }
            std::map<std::string, StorageCapacitySnapshot> pools;
            bool missing = false;
            bool unsupported = false;
            for (const auto &name : names) {
                const auto it = snapshots.find(name);
                if (it == snapshots.end()) {
                    // A missing association has unknown type. Do not publish a
                    // smaller partial aggregate while registration recovers.
                    missing = true;
                    continue;
                }
                if (ToBaseType(it->second.type) != type) {
                    missing = missing || explicit_scope;
                    continue;
                }
                const auto &snapshot = it->second;
                unsupported = unsupported || snapshot.refresh_code == EC_UNIMPLEMENTED;
                const auto identity = snapshot.identity.empty() ? name : snapshot.identity;
                const auto pool_it = pools.find(identity);
                if (pool_it == pools.end() ||
                    snapshot.last_success_steady_us > pool_it->second.last_success_steady_us) {
                    pools[identity] = snapshot;
                }
            }
            bool known = !missing && !pools.empty();
            bool fresh = known;
            bool failed = false;
            int64_t backend_capacity = 0;
            value.last_success_time_s = std::numeric_limits<int64_t>::max();
            value.status = missing ? "missing_backend" : "unknown";
            for (const auto &[identity, snapshot] : pools) {
                failed = failed || snapshot.refresh_code != EC_OK;
                if (snapshot.capacity < 0) {
                    known = fresh = false;
                    value.status = snapshot.refresh_code == EC_UNIMPLEMENTED ? "unsupported" : "unknown";
                    continue;
                }
                value.last_success_time_s = std::min(value.last_success_time_s, snapshot.last_success_time_s);
                fresh = fresh && now_steady_us >= snapshot.last_success_steady_us &&
                        now_steady_us - snapshot.last_success_steady_us <= kCapacityMaxAgeUs;
                if (snapshot.capacity > std::numeric_limits<int64_t>::max() - backend_capacity) {
                    known = fresh = false;
                    value.status = "overflow";
                } else {
                    backend_capacity += snapshot.capacity;
                }
            }
            if (known) {
                value.backend_capacity = backend_capacity;
                value.capacity = ScaleCapacity(backend_capacity, config.ratio());
                value.known = true;
                value.usable = fresh && !unsupported;
                value.status = unsupported ? "unsupported" : (!fresh ? "stale" : (failed ? "refresh_error" : "fresh"));
            } else {
                value.last_success_time_s = 0;
                if (unsupported) {
                    value.status = "unsupported";
                }
            }
            if (!config.automatic()) {
                value.capacity = config.capacity();
                value.known = value.usable = true;
                value.status = "fixed";
            }
        }
        if (!value.known || value.capacity < 0 || value.capacity > std::numeric_limits<int64_t>::max() - reference) {
            complete = false;
        } else {
            reference += value.capacity;
        }
    }
    result.reference_capacity = complete ? reference : -1;
    return result;
}
} // namespace kv_cache_manager
