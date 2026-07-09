#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/registry_storage_backend.h"
#include "kv_cache_manager/config/registry_storage_backend_factory.h"

namespace kv_cache_manager {

static constexpr const char *kOptInstanceInfoKey = "optimizer_instance_info";
static constexpr const char *kOptInstanceGroupKey = "optimizer_instance_group";

OptimizerRegistryManager::OptimizerRegistryManager(const std::string &registry_uri) : registry_uri_(registry_uri) {}

bool OptimizerRegistryManager::Init() {
    if (registry_uri_.empty()) {
        KVCM_LOG_WARN("OptimizerRegistryManager: empty registry_uri, persistence disabled");
        return true;
    }
    storage_ = RegistryStorageBackendFactory::CreateAndInitStorageBackend(registry_uri_);
    if (!storage_) {
        KVCM_LOG_ERROR("OptimizerRegistryManager: failed to create storage backend for uri[%s]", registry_uri_.c_str());
        return false;
    }
    KVCM_LOG_INFO("OptimizerRegistryManager: initialized with uri[%s]", registry_uri_.c_str());
    return true;
}

ErrorCode
OptimizerRegistryManager::LoadAndSave(const std::string &key, const std::string &id, const Jsonizable *jsonizable) {
    std::map<std::string, std::string> value_map;
    auto ec = storage_->Load(key, value_map);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_ERROR(
            "OptimizerRegistryManager: Load %s failed, id[%s], ec[%d]", key.c_str(), id.c_str(), static_cast<int>(ec));
        return ec;
    }
    if (jsonizable) {
        value_map[id] = jsonizable->ToJsonString();
    } else {
        value_map[id] = id;
    }
    ec = storage_->Save(key, value_map);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR(
            "OptimizerRegistryManager: Save %s failed, id[%s], ec[%d]", key.c_str(), id.c_str(), static_cast<int>(ec));
        return ec;
    }
    return EC_OK;
}

ErrorCode OptimizerRegistryManager::LoadAndDelete(const std::string &key, const std::string &id) {
    std::map<std::string, std::string> value_map;
    auto ec = storage_->Load(key, value_map);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_ERROR(
            "OptimizerRegistryManager: Load %s failed, id[%s], ec[%d]", key.c_str(), id.c_str(), static_cast<int>(ec));
        return ec;
    }
    value_map.erase(id);
    if (value_map.empty()) {
        ec = storage_->Delete(key);
    } else {
        ec = storage_->Save(key, value_map);
    }
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("OptimizerRegistryManager: Delete %s failed, id[%s], ec[%d]",
                       key.c_str(),
                       id.c_str(),
                       static_cast<int>(ec));
        return ec;
    }
    return EC_OK;
}

// ============== InstanceGroup CRUD ==============

ErrorCode OptimizerRegistryManager::CreateInstanceGroup(const OptimizerInstanceGroup &group) {
    std::unique_lock lock(mutex_);
    const auto &name = group.name();

    if (instance_groups_.count(name)) {
        KVCM_LOG_WARN("OptimizerRegistryManager: instance group[%s] already exists", name.c_str());
        return EC_DUPLICATE_ENTITY;
    }

    if (storage_) {
        auto ec = LoadAndSave(kOptInstanceGroupKey, name, &group);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("OptimizerRegistryManager: failed to persist instance group[%s]", name.c_str());
            return ec;
        }
    }

    instance_groups_[name] = std::make_shared<OptimizerInstanceGroup>(group);
    KVCM_LOG_INFO("OptimizerRegistryManager: created instance group[%s]", name.c_str());
    return EC_OK;
}

ErrorCode OptimizerRegistryManager::UpdateInstanceGroup(const OptimizerInstanceGroup &group) {
    std::unique_lock lock(mutex_);
    const auto &name = group.name();

    if (!instance_groups_.count(name)) {
        KVCM_LOG_WARN("OptimizerRegistryManager: instance group[%s] not found for update", name.c_str());
        return EC_NOENT;
    }

    if (storage_) {
        auto ec = LoadAndSave(kOptInstanceGroupKey, name, &group);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("OptimizerRegistryManager: failed to persist updated instance group[%s]", name.c_str());
            return ec;
        }
    }

    instance_groups_[name] = std::make_shared<OptimizerInstanceGroup>(group);
    KVCM_LOG_INFO("OptimizerRegistryManager: updated instance group[%s]", name.c_str());
    return EC_OK;
}

ErrorCode OptimizerRegistryManager::RemoveInstanceGroup(const std::string &group_name) {
    std::unique_lock lock(mutex_);

    if (!instance_groups_.count(group_name)) {
        KVCM_LOG_WARN("OptimizerRegistryManager: instance group[%s] not found for removal", group_name.c_str());
        return EC_NOENT;
    }

    for (const auto &[instance_id, info] : instance_infos_) {
        if (info && info->instance_group_name() == group_name) {
            KVCM_LOG_ERROR("OptimizerRegistryManager: cannot remove instance group[%s], instance[%s] still exists",
                           group_name.c_str(),
                           instance_id.c_str());
            return EC_BADARGS;
        }
    }

    if (storage_) {
        auto ec = LoadAndDelete(kOptInstanceGroupKey, group_name);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("OptimizerRegistryManager: failed to delete instance group[%s]", group_name.c_str());
            return ec;
        }
    }

    instance_groups_.erase(group_name);
    KVCM_LOG_INFO("OptimizerRegistryManager: removed instance group[%s]", group_name.c_str());
    return EC_OK;
}

std::shared_ptr<const OptimizerInstanceGroup>
OptimizerRegistryManager::GetInstanceGroup(const std::string &group_name) const {
    std::shared_lock lock(mutex_);
    auto it = instance_groups_.find(group_name);
    if (it == instance_groups_.end()) {
        return nullptr;
    }
    return it->second;
}

std::vector<std::shared_ptr<const OptimizerInstanceGroup>> OptimizerRegistryManager::ListInstanceGroups() const {
    std::shared_lock lock(mutex_);
    std::vector<std::shared_ptr<const OptimizerInstanceGroup>> result;
    result.reserve(instance_groups_.size());
    for (const auto &[_, group] : instance_groups_) {
        result.push_back(group);
    }
    return result;
}

// ============== InstanceInfo persistence ==============

ErrorCode OptimizerRegistryManager::SaveInstanceInfo(const OptimizerInstanceInfo &instance_info) {
    std::unique_lock lock(mutex_);
    const auto &instance_id = instance_info.instance_id();

    if (storage_) {
        auto ec = LoadAndSave(kOptInstanceInfoKey, instance_id, &instance_info);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("OptimizerRegistryManager: failed to save instance info[%s]", instance_id.c_str());
            return ec;
        }
    }

    instance_infos_[instance_id] = std::make_shared<OptimizerInstanceInfo>(instance_info);
    KVCM_LOG_INFO("OptimizerRegistryManager: saved instance info[%s] group[%s]",
                  instance_id.c_str(),
                  instance_info.instance_group_name().c_str());
    return EC_OK;
}

ErrorCode OptimizerRegistryManager::DeleteInstanceInfo(const std::string &instance_id) {
    std::unique_lock lock(mutex_);

    if (!instance_infos_.count(instance_id)) {
        KVCM_LOG_WARN("OptimizerRegistryManager: instance info[%s] not found for deletion", instance_id.c_str());
        return EC_NOENT;
    }

    if (storage_) {
        auto ec = LoadAndDelete(kOptInstanceInfoKey, instance_id);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("OptimizerRegistryManager: failed to delete instance info[%s]", instance_id.c_str());
            return ec;
        }
    }

    instance_infos_.erase(instance_id);
    KVCM_LOG_INFO("OptimizerRegistryManager: deleted instance info[%s]", instance_id.c_str());
    return EC_OK;
}

std::shared_ptr<const OptimizerInstanceInfo>
OptimizerRegistryManager::GetInstanceInfo(const std::string &instance_id) const {
    std::shared_lock lock(mutex_);
    auto it = instance_infos_.find(instance_id);
    if (it == instance_infos_.end()) {
        return nullptr;
    }
    return it->second;
}

std::vector<std::shared_ptr<const OptimizerInstanceInfo>>
OptimizerRegistryManager::ListInstanceInfos(const std::string &group_filter) const {
    std::shared_lock lock(mutex_);
    std::vector<std::shared_ptr<const OptimizerInstanceInfo>> result;
    result.reserve(instance_infos_.size());
    for (const auto &[_, info] : instance_infos_) {
        if (group_filter.empty() || info->instance_group_name() == group_filter) {
            result.push_back(info);
        }
    }
    return result;
}

// ============== Recovery ==============

ErrorCode OptimizerRegistryManager::LoadRecoveryData(RecoveryData &data) {
    if (!storage_) {
        return EC_OK;
    }

    std::unique_lock lock(mutex_);

    data.instance_groups.clear();
    data.instance_infos.clear();

    // Load into temporary maps first; only swap into in-memory state after
    // both reads succeed so a transient error does not wipe the last good view.
    std::unordered_map<std::string, std::shared_ptr<OptimizerInstanceGroup>> tmp_groups;
    std::unordered_map<std::string, std::shared_ptr<OptimizerInstanceInfo>> tmp_infos;

    // Load instance groups
    std::map<std::string, std::string> group_map;
    auto ec = storage_->Load(kOptInstanceGroupKey, group_map);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_WARN("OptimizerRegistryManager: load instance_groups failed, ec[%d]", static_cast<int>(ec));
        return EC_ERROR;
    }
    for (const auto &[name, json] : group_map) {
        auto group = std::make_shared<OptimizerInstanceGroup>();
        if (!group->FromJsonString(json)) {
            KVCM_LOG_WARN("OptimizerRegistryManager: OptimizerInstanceGroup from json failed, skip group[%s]",
                          name.c_str());
            continue;
        }
        data.instance_groups.push_back(group);
        tmp_groups[name] = group;
    }

    // Load instance infos
    std::map<std::string, std::string> info_map;
    ec = storage_->Load(kOptInstanceInfoKey, info_map);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_WARN("OptimizerRegistryManager: load instance_infos failed, ec[%d]", static_cast<int>(ec));
        return EC_ERROR;
    }
    for (const auto &[instance_id, json] : info_map) {
        auto info = std::make_shared<OptimizerInstanceInfo>();
        if (!info->FromJsonString(json)) {
            KVCM_LOG_WARN("OptimizerRegistryManager: OptimizerInstanceInfo from json failed, skip instance[%s]",
                          instance_id.c_str());
            continue;
        }
        data.instance_infos.push_back(info);
        tmp_infos[instance_id] = info;
    }

    // Both loads succeeded — commit to in-memory state.
    instance_groups_ = std::move(tmp_groups);
    instance_infos_ = std::move(tmp_infos);

    KVCM_LOG_INFO("OptimizerRegistryManager: loaded %zu groups, %zu instances for recovery",
                  data.instance_groups.size(),
                  data.instance_infos.size());
    return EC_OK;
}

} // namespace kv_cache_manager
