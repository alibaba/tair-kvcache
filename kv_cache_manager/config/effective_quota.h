#pragma once

#include <map>
#include <set>
#include <string>

#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"

namespace kv_cache_manager {

class DataStorageManager;

struct EffectiveTypeQuota {
    QuotaConfig config;
    int64_t capacity = -1;
    int64_t backend_capacity = -1;
    int64_t last_success_time_s = 0;
    bool refresh_success = false;
};

struct EffectiveGroupQuota {
    int64_t configured_capacity = -1;
    int64_t reference_capacity = -1;
    std::map<DataStorageType, EffectiveTypeQuota> types;

    InstanceGroupQuota ForAdmission() const;
    InstanceGroupQuota ForReclaim() const;
};

std::set<std::string> GetQuotaStorageNames(const InstanceGroup &group);
EffectiveGroupQuota ResolveEffectiveQuota(const InstanceGroup &group,
                                          const std::map<std::string, StorageCapacitySnapshot> &snapshots);
EffectiveGroupQuota ResolveEffectiveQuota(const InstanceGroup &group, const DataStorageManager *storage_manager);
double QuotaUsageRatio(uint64_t usage, int64_t capacity);

} // namespace kv_cache_manager
