#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/config/registry_storage_backend.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_group.h"
#include "kv_cache_manager/optimizer/config/optimizer_instance_info.h"

namespace kv_cache_manager {

class OptimizerRegistryManager {
public:
    explicit OptimizerRegistryManager(const std::string &registry_uri);
    ~OptimizerRegistryManager() = default;

    OptimizerRegistryManager(const OptimizerRegistryManager &) = delete;
    OptimizerRegistryManager &operator=(const OptimizerRegistryManager &) = delete;

    bool Init();

    // InstanceGroup CRUD
    ErrorCode CreateInstanceGroup(const OptimizerInstanceGroup &group);
    ErrorCode UpdateInstanceGroup(const OptimizerInstanceGroup &group);
    ErrorCode RemoveInstanceGroup(const std::string &group_name);
    std::shared_ptr<const OptimizerInstanceGroup> GetInstanceGroup(const std::string &group_name) const;
    std::vector<std::shared_ptr<const OptimizerInstanceGroup>> ListInstanceGroups() const;

    // InstanceInfo persistence (called by OnlineOptimizerManager)
    ErrorCode SaveInstanceInfo(const OptimizerInstanceInfo &instance_info);
    ErrorCode DeleteInstanceInfo(const std::string &instance_id);
    std::shared_ptr<const OptimizerInstanceInfo> GetInstanceInfo(const std::string &instance_id) const;
    std::vector<std::shared_ptr<const OptimizerInstanceInfo>>
    ListInstanceInfos(const std::string &group_filter = "") const;

    // Recovery: returns all persisted instance_infos and groups for re-registration
    struct RecoveryData {
        std::vector<std::shared_ptr<OptimizerInstanceInfo>> instance_infos;
        std::vector<std::shared_ptr<OptimizerInstanceGroup>> instance_groups;
    };
    ErrorCode LoadRecoveryData(RecoveryData &data);

private:
    ErrorCode LoadAndSave(const std::string &key, const std::string &id, const Jsonizable *jsonizable);
    ErrorCode LoadAndDelete(const std::string &key, const std::string &id);

    std::string registry_uri_;
    std::unique_ptr<RegistryStorageBackend> storage_;

    mutable std::shared_mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<OptimizerInstanceGroup>> instance_groups_;
    std::unordered_map<std::string, std::shared_ptr<OptimizerInstanceInfo>> instance_infos_;
};

} // namespace kv_cache_manager
