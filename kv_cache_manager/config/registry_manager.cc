#include "kv_cache_manager/config/registry_manager.h"

#include <memory>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/config/account.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/registry_storage_backend_factory.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

static constexpr const char *kRegistryStorageKey = "storage";
static constexpr const char *kRegistryGroupKey = "instance_group";
static constexpr const char *kRegistryInstanceKey = "instance";
static constexpr const char *kRegistryAccountKey = "account";

#define PREFIX_LOG_I(LEVEL, format, args...)                                                                           \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] instance [%s] | " format, trace_id.c_str(), instance_id.c_str(), ##args);      \
    } while (0)

#define PREFIX_LOG_G(LEVEL, format, args...)                                                                           \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL(                                                                                              \
            "trace_id [%s] instance_group [%s] | " format, trace_id.c_str(), instance_group_name.c_str(), ##args);     \
    } while (0)

#define PREFIX_LOG_S(LEVEL, format, args...)                                                                           \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL(                                                                                              \
            "trace_id [%s] storage [%s] | " format, trace_id.c_str(), global_unique_name.c_str(), ##args);             \
    } while (0)

#define PREFIX_LOG_A(LEVEL, format, args...)                                                                           \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] account_name [%s] | " format, trace_id.c_str(), user_name.c_str(), ##args);    \
    } while (0)

#define RETURN_IF_EC_NOT_OK(ec)                                                                                        \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            return ec;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_TYPE(ec, Type)                                                                        \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            return {ec, Type()};                                                                                       \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_LOG_I(LEVEL, ec, format, args...)                                                     \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG_I(LEVEL, format, ##args);                                                                       \
            return ec;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_LOG_G(LEVEL, ec, format, args...)                                                     \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG_G(LEVEL, format, ##args);                                                                       \
            return ec;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_LOG_S(LEVEL, ec, format, args...)                                                     \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG_S(LEVEL, format, ##args);                                                                       \
            return ec;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_LOG_A(LEVEL, ec, format, args...)                                                     \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG_A(LEVEL, format, ##args);                                                                       \
            return ec;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG_I(LEVEL, ec, Type, format, args...)                                          \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG_I(LEVEL, format, ##args);                                                                       \
            return {ec, Type()};                                                                                       \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG_G(LEVEL, ec, Type, format, args...)                                          \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG_G(LEVEL, format, ##args);                                                                       \
            return {ec, Type()};                                                                                       \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG_S(LEVEL, ec, Type, format, args...)                                          \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG_S(LEVEL, format, ##args);                                                                       \
            return {ec, Type()};                                                                                       \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG_A(LEVEL, ec, Type, format, args...)                                          \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG_A(LEVEL, format, ##args);                                                                       \
            return {ec, Type()};                                                                                       \
        }                                                                                                              \
    } while (0)

RegistryManager::RegistryManager(const std::string &registry_storage_uri,
                                 std::shared_ptr<MetricsRegistry> metrics_registry)
    : registry_storage_uri_(registry_storage_uri), metrics_registry_(std::move(metrics_registry)) {}

bool RegistryManager::Init() {
    data_storage_manager_.reset(new DataStorageManager(metrics_registry_));
    storage_ = RegistryStorageBackendFactory::CreateAndInitStorageBackend(registry_storage_uri_);
    if (!storage_) {
        KVCM_LOG_ERROR("registry storage backend create and init failed, storage uri[%s]",
                       registry_storage_uri_.c_str());
        return false;
    }
    KVCM_LOG_INFO("registry manager init and recover success, storage uri[%s]", registry_storage_uri_.c_str());
    return true;
}

ErrorCode RegistryManager::AddStorage(RequestContext *request_context, const StorageConfig &storage_config) {
    const auto &trace_id = request_context->request_id();
    const auto &global_unique_name = storage_config.global_unique_name();
    auto ec = LoadAndSave(kRegistryStorageKey, global_unique_name, &storage_config);
    RETURN_IF_EC_NOT_OK_WITH_LOG_S(WARN, ec, "load and save storage failed");
    ec = data_storage_manager_->RegisterStorage(request_context, global_unique_name, storage_config);
    RETURN_IF_EC_NOT_OK_WITH_LOG_S(WARN, ec, "add storage failed");
    PREFIX_LOG_S(INFO, "add storage OK");
    return ec;
}

ErrorCode RegistryManager::EnableStorage(RequestContext *request_context, const std::string &global_unique_name) {
    const auto &trace_id = request_context->request_id();
    auto ec = UpdateStorageAvailableStatus(global_unique_name, /*is_available*/ true);
    RETURN_IF_EC_NOT_OK_WITH_LOG_S(WARN, ec, "update storage available status failed");
    ec = data_storage_manager_->EnableStorage(global_unique_name);
    RETURN_IF_EC_NOT_OK_WITH_LOG_S(WARN, ec, "enable storage failed");
    PREFIX_LOG_S(INFO, "enable storage OK");
    return ec;
}

ErrorCode RegistryManager::DisableStorage(RequestContext *request_context, const std::string &global_unique_name) {
    const auto &trace_id = request_context->request_id();
    auto ec = UpdateStorageAvailableStatus(global_unique_name, /*is_available*/ false);
    RETURN_IF_EC_NOT_OK_WITH_LOG_S(WARN, ec, "update storage available status failed");
    ec = data_storage_manager_->DisableStorage(global_unique_name);
    RETURN_IF_EC_NOT_OK_WITH_LOG_S(WARN, ec, "disable storage failed");
    PREFIX_LOG_S(INFO, "disable storage OK");
    return ec;
}

ErrorCode RegistryManager::RemoveStorage(RequestContext *request_context, const std::string &global_unique_name) {
    const auto &trace_id = request_context->request_id();
    auto ec = LoadAndDelete(kRegistryStorageKey, global_unique_name);
    RETURN_IF_EC_NOT_OK_WITH_LOG_S(WARN, ec, "load and delete storage failed");
    ec = data_storage_manager_->UnRegisterStorage(global_unique_name);
    RETURN_IF_EC_NOT_OK_WITH_LOG_S(WARN, ec, "remove storage failed");
    PREFIX_LOG_S(INFO, "remove storage OK");
    return ec;
}

ErrorCode RegistryManager::UpdateStorage(RequestContext *request_context,
                                         const StorageConfig &storage_config,
                                         bool force_update) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    const auto &trace_id = request_context->request_id();
    const auto &global_unique_name = storage_config.global_unique_name();
    // 重建期间短暂不可用
    auto ec = RemoveStorage(request_context, global_unique_name);
    RETURN_IF_EC_NOT_OK_WITH_LOG_S(WARN, ec, "update storage failed: remove storage failed");
    ec = AddStorage(request_context, storage_config);
    RETURN_IF_EC_NOT_OK_WITH_LOG_S(WARN, ec, "update storage failed: add storage failed");
    PREFIX_LOG_S(INFO, "update storage OK");
    return EC_OK;
}

std::pair<ErrorCode, std::vector<StorageConfig>> RegistryManager::ListStorage(RequestContext *request_context) {
    std::vector<StorageConfig> storages = data_storage_manager_->ListStorageConfig();
    return {EC_OK, storages};
}

ErrorCode RegistryManager::CreateInstanceGroup(RequestContext *request_context, const InstanceGroup &instance_group) {
    const auto &trace_id = request_context->request_id();
    const auto &instance_group_name = instance_group.name();
    if (instance_group_configs_.find(instance_group_name) != instance_group_configs_.end()) {
        RETURN_IF_EC_NOT_OK_WITH_LOG_G(WARN, EC_EXIST, "create instance group failed: instance group already existed");
    }
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (instance_group_configs_.find(instance_group_name) != instance_group_configs_.end()) {
            RETURN_IF_EC_NOT_OK_WITH_LOG_G(
                WARN, EC_EXIST, "create instance group failed: instance group already existed");
        }
        // save to storage backend
        auto ec = LoadAndSave(kRegistryGroupKey, instance_group_name, &instance_group);
        RETURN_IF_EC_NOT_OK_WITH_LOG_G(WARN, ec, "load and save instance group failed");
        instance_group_configs_[instance_group_name] = std::make_shared<InstanceGroup>(instance_group);
    }
    PREFIX_LOG_G(INFO, "create instance group OK");
    return EC_OK;
}

ErrorCode RegistryManager::UpdateInstanceGroup(RequestContext *request_context,
                                               const InstanceGroup &instance_group,
                                               int64_t current_version) {
    const auto &trace_id = request_context->request_id();
    const auto &instance_group_name = instance_group.name();
    auto iter = instance_group_configs_.find(instance_group_name);
    if (iter == instance_group_configs_.end()) {
        RETURN_IF_EC_NOT_OK_WITH_LOG_G(WARN, EC_NOENT, "update instance group failed: instance group not found");
    }
    if (current_version != iter->second->version() || instance_group.version() <= iter->second->version()) {
        // TODO: 定义错误码
        RETURN_IF_EC_NOT_OK_WITH_LOG_G(WARN,
                                       EC_ERROR,
                                       "update instance group failed: instance group version not match, current "
                                       "version in request: %ld, request version: %ld, current version: %ld",
                                       current_version,
                                       instance_group.version(),
                                       iter->second->version());
    }
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        iter = instance_group_configs_.find(instance_group_name);
        if (iter == instance_group_configs_.end()) {
            RETURN_IF_EC_NOT_OK_WITH_LOG_G(WARN, EC_NOENT, "update instance group failed: instance group not found");
        }
        if (current_version != iter->second->version() || instance_group.version() <= iter->second->version()) {
            // TODO: 定义错误码
            RETURN_IF_EC_NOT_OK_WITH_LOG_G(WARN,
                                           EC_ERROR,
                                           "update instance group failed: instance group version not match, current "
                                           "version in request: %ld, request version: %ld, current version: %ld",
                                           current_version,
                                           instance_group.version(),
                                           iter->second->version());
        }
        // save to storage backend
        auto ec = LoadAndSave(kRegistryGroupKey, instance_group_name, &instance_group);
        RETURN_IF_EC_NOT_OK_WITH_LOG_G(WARN, ec, "load and save instance group failed");
        instance_group_configs_[instance_group_name] = std::make_shared<InstanceGroup>(instance_group);
    }
    PREFIX_LOG_G(INFO, "update instance group OK");
    return EC_OK;
}

ErrorCode RegistryManager::RemoveInstanceGroup(RequestContext *request_context,
                                               const std::string &instance_group_name) {
    const auto &trace_id = request_context->request_id();
    const auto iter = instance_group_configs_.find(instance_group_name);
    if (iter == instance_group_configs_.end()) {
        RETURN_IF_EC_NOT_OK_WITH_LOG_G(WARN, EC_NOENT, "remove instance group failed: instance group not found");
    }
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        // delete from storage backend
        auto ec = LoadAndDelete(kRegistryGroupKey, instance_group_name);
        RETURN_IF_EC_NOT_OK_WITH_LOG_G(WARN, ec, "load and delete instance group failed");
        instance_group_configs_.erase(iter);
    }
    PREFIX_LOG_G(INFO, "remove instance group OK");
    return EC_OK;
}

std::pair<ErrorCode, std::vector<std::shared_ptr<const InstanceGroup>>>
RegistryManager::ListInstanceGroup(RequestContext *request_context) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<std::shared_ptr<const InstanceGroup>> instance_groups;
    for (const auto &pair : instance_group_configs_) {
        instance_groups.emplace_back(pair.second);
    }
    return std::make_pair(ErrorCode::EC_OK, instance_groups);
}

ErrorCode RegistryManager::RegisterInstance(RequestContext *request_context,
                                            const std::string &instance_group,
                                            const std::string &instance_id,
                                            int32_t block_size,
                                            const std::vector<LocationSpecInfo> &location_spec_infos,
                                            const ModelDeployment &model_deployment,
                                            const std::vector<LocationSpecGroup> &location_spec_groups) {
    const auto &trace_id = request_context->trace_id();
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto instance_group_iter = instance_group_configs_.find(instance_group);
    if (instance_group_iter == instance_group_configs_.end()) {
        RETURN_IF_EC_NOT_OK_WITH_LOG_I(
            WARN, EC_NOENT, "register instance failed: instance group not found: %s", instance_group.c_str());
    }
    auto it = instance_infos_.find(instance_id);
    if (it != instance_infos_.end()) {
        auto exist_instance_info = it->second;
        if (exist_instance_info->block_size() != block_size ||
            exist_instance_info->location_spec_infos() != location_spec_infos ||
            exist_instance_info->model_deployment() != model_deployment ||
            exist_instance_info->location_spec_groups() != location_spec_groups) {
            RETURN_IF_EC_NOT_OK_WITH_LOG_I(WARN, EC_DUPLICATE_ENTITY, "register instance failed: duplicate instance");
        }
        return EC_OK;
    }
    auto instance_info = std::make_shared<InstanceInfo>(instance_group_iter->second->global_quota_group_name(),
                                                        instance_group,
                                                        instance_id,
                                                        block_size,
                                                        location_spec_infos,
                                                        model_deployment,
                                                        location_spec_groups);
    // save the instance info to storage backend in such a way that one key corresponds to one instance
    auto ec = LoadAndSave(instance_id, instance_id, instance_info.get());
    RETURN_IF_EC_NOT_OK_WITH_LOG_I(WARN, ec, "load and save instance info failed");
    // save the instance_id as value to instance key for recover
    ec = LoadAndSave(kRegistryInstanceKey, instance_id, nullptr);
    RETURN_IF_EC_NOT_OK_WITH_LOG_I(WARN, ec, "load and save instance id failed");
    // TODO: add quota_group_name
    instance_infos_[instance_id] = instance_info;
    PREFIX_LOG_I(INFO, "register instance OK");
    return EC_OK;
}

ErrorCode RegistryManager::RemoveInstance(RequestContext *request_context,
                                          const std::string &instance_group,
                                          const std::string &instance_id) {
    const auto &trace_id = request_context->trace_id();
    if (instance_infos_.find(instance_id) == instance_infos_.end()) {
        // TODO: 添加错误码
        RETURN_IF_EC_NOT_OK_WITH_LOG_I(
            WARN, EC_NOENT, "remove instance failed: instance not found, group: %s", instance_group.c_str());
    }
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        // first, delete instance id from instance key
        auto ec = LoadAndDelete(kRegistryInstanceKey, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_LOG_I(WARN, ec, "load and delete instance info failed");
        // second, delete instance info
        ec = LoadAndDelete(instance_id, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_LOG_I(WARN, ec, "load and delete instance id failed");
        instance_infos_.erase(instance_id);
    }
    PREFIX_LOG_I(INFO, "remove instance OK");
    return EC_OK;
}

std::pair<ErrorCode, std::shared_ptr<const InstanceGroup>>
RegistryManager::GetInstanceGroup(RequestContext *request_context, const std::string &instance_group_name) {
    const auto &trace_id = request_context->request_id();
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto iter = instance_group_configs_.find(instance_group_name);
    if (iter == instance_group_configs_.end()) {
        PREFIX_LOG_G(WARN, "get instance group failed: instance group not found");
        return {EC_NOENT, nullptr};
    }
    std::shared_ptr<const InstanceGroup> instance_group = iter->second;
    return {EC_OK, instance_group};
}

InstanceInfoConstPtr RegistryManager::GetInstanceInfo(RequestContext *request_context, const std::string &instance_id) {
    const auto &trace_id = request_context->trace_id();
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = instance_infos_.find(instance_id);
    if (it != instance_infos_.end()) {
        return it->second;
    }
    PREFIX_LOG_I(WARN, "instance not found");
    return nullptr;
}

std::pair<ErrorCode, std::vector<InstanceInfoConstPtr>>
RegistryManager::ListInstanceInfo(RequestContext *request_context, const std::string &instance_group) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<InstanceInfoConstPtr> instances;
    for (auto &instance : instance_infos_) {
        if (instance.second->instance_group_name() == instance_group) {
            instances.push_back(instance.second);
        }
    }
    return {EC_OK, instances};
}

ErrorCode RegistryManager::AddAccount(RequestContext *request_context,
                                      const std::string &user_name,
                                      const std::string &password,
                                      const AccountRole &role) {
    const auto &trace_id = request_context->request_id();
    if (accounts_.find(user_name) != accounts_.end()) {
        RETURN_IF_EC_NOT_OK_WITH_LOG_A(WARN, EC_EXIST, "add account failed: account already existed");
    }
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (accounts_.find(user_name) != accounts_.end()) {
            RETURN_IF_EC_NOT_OK_WITH_LOG_A(WARN, EC_EXIST, "add account failed: account already existed");
        }
        std::shared_ptr<Account> account = std::make_shared<Account>(user_name, password, role);
        auto ec = LoadAndSave(kRegistryAccountKey, user_name, account.get());
        RETURN_IF_EC_NOT_OK_WITH_LOG_A(WARN, ec, "load and save account failed");
        accounts_[user_name] = account;
    }
    PREFIX_LOG_A(INFO, "add account OK");
    return EC_OK;
}

ErrorCode RegistryManager::DeleteAccount(RequestContext *request_context, const std::string &user_name) {
    const auto &trace_id = request_context->request_id();
    if (accounts_.find(user_name) == accounts_.end()) {
        RETURN_IF_EC_NOT_OK_WITH_LOG_A(WARN, EC_NOENT, "delete account failed: account not found");
    }
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        if (accounts_.find(user_name) == accounts_.end()) {
            RETURN_IF_EC_NOT_OK_WITH_LOG_A(WARN, EC_NOENT, "delete account failed: account not found");
        }
        auto ec = LoadAndDelete(kRegistryAccountKey, user_name);
        RETURN_IF_EC_NOT_OK_WITH_LOG_A(WARN, ec, "load and delete account failed");
        accounts_.erase(user_name);
    }
    PREFIX_LOG_A(INFO, "delete account OK");
    return EC_OK;
}

std::pair<ErrorCode, std::vector<std::shared_ptr<const Account>>>
RegistryManager::ListAccount(RequestContext *request_context) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<std::shared_ptr<const Account>> accounts;
    for (auto &account : accounts_) {
        accounts.push_back(account.second);
    }
    return {EC_OK, accounts};
}

std::pair<ErrorCode, std::string> RegistryManager::GenConfigSnapshot(RequestContext *request_context) {
    return {EC_UNIMPLEMENTED, ""};
}

ErrorCode RegistryManager::LoadConfigSnapshot(RequestContext *request_context, const std::string &config_snapshot) {
    return EC_UNIMPLEMENTED;
}

CacheConfigConstPtr RegistryManager::GetCacheConfig(const std::string &instance_group) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = instance_group_configs_.find(instance_group);
    if (it != instance_group_configs_.end()) {
        return it->second->cache_config();
    }
    return nullptr;
}

std::shared_ptr<DataStorageManager> RegistryManager::data_storage_manager() const { return data_storage_manager_; }

ErrorCode RegistryManager::LoadAndSave(const std::string &key, const std::string &id, const Jsonizable *jsonizable) {
    std::map<std::string, std::string> value_map;
    auto ec = storage_->Load(key, value_map);
    // maybe value_map is empty
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_ERROR("Load %s from storage backend failed, id[%s], ec[%d]", key.c_str(), id.c_str(), ec);
        return ec;
    }
    if (jsonizable) {
        value_map[id] = jsonizable->ToJsonString();
    } else {
        value_map[id] = id;
    }
    ec = storage_->Save(key, value_map);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Save %s to storage backend failed, id[%s], ec[%d]", key.c_str(), id.c_str(), ec);
        return ec;
    }
    return EC_OK;
}

ErrorCode RegistryManager::LoadAndDelete(const std::string &key, const std::string &id) {
    std::map<std::string, std::string> value_map;
    auto ec = storage_->Load(key, value_map);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_ERROR("Load %s from storage backend failed, id[%s], ec[%d]", key.c_str(), id.c_str(), ec);
        return ec;
    }
    value_map.erase(id);
    if (value_map.empty()) {
        ec = storage_->Delete(key);
    } else {
        ec = storage_->Save(key, value_map);
    }
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Save %s to storage backend failed, id[%s], ec[%d]", key.c_str(), id.c_str(), ec);
        return ec;
    }
    return EC_OK;
}

ErrorCode RegistryManager::UpdateStorageAvailableStatus(const std::string &global_unique_name, bool is_available) {
    auto storage_backend = data_storage_manager_->GetDataStorageBackend(global_unique_name);
    if (!storage_backend) {
        KVCM_LOG_ERROR("get storage backend failed, name: %s not exist", global_unique_name.c_str());
        return EC_NOENT;
    }
    auto storage_config = storage_backend->GetStorageConfig();
    storage_config.set_is_available(is_available); // for recover
    return LoadAndSave(kRegistryStorageKey, global_unique_name, &storage_config);
}

ErrorCode RegistryManager::DoRecover() {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    size_t storage_num = 0;
    size_t not_available_storage_num = 0;
    auto request_context = std::make_shared<RequestContext>("registry_manager_recover_trace");
    // recover storage
    std::map<std::string, std::string> value_map;
    auto ec = storage_->Load(kRegistryStorageKey, value_map);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_ERROR("load registry storage backend failed");
        return ec;
    }
    for (const auto &[id, val] : value_map) {
        StorageConfig storage_config;
        if (!storage_config.FromJsonString(val)) {
            KVCM_LOG_ERROR("storage config from json string failed, json string[%s]", val.c_str());
            return EC_CONFIG_ERROR;
        }
        auto name = storage_config.global_unique_name();
        ec = data_storage_manager_->RegisterStorage(request_context.get(), name, storage_config);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR(
                "register storage failed when recover, name[%s], json string[%s]", name.c_str(), val.c_str());
            return ec;
        }
        if (!storage_config.is_available()) {
            ec = data_storage_manager_->DisableStorage(name);
            if (ec != EC_OK) {
                KVCM_LOG_ERROR(
                    "disable storage failed when recover, name[%s], json string[%s]", name.c_str(), val.c_str());
                return ec;
            }
            ++not_available_storage_num;
        }
        ++storage_num;
    }

    // recover account
    value_map.clear();
    ec = storage_->Load(kRegistryAccountKey, value_map);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_ERROR("load registry account failed");
        return ec;
    }
    for (const auto &[id, val] : value_map) {
        auto account = std::make_shared<Account>();
        if (!account->FromJsonString(val)) {
            KVCM_LOG_ERROR("account from json string failed, json string[%s]", val.c_str());
            return EC_CONFIG_ERROR;
        }
        accounts_[account->user_name()] = account;
    }
    // recover instance group
    value_map.clear();
    ec = storage_->Load(kRegistryGroupKey, value_map);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_ERROR("load registry instance group failed");
        return ec;
    }
    for (const auto &[id, val] : value_map) {
        auto instance_group = std::make_shared<InstanceGroup>();
        if (!instance_group->FromJsonString(val)) {
            KVCM_LOG_ERROR("instance group from json string failed, json string[%s]", val.c_str());
            return EC_CONFIG_ERROR;
        }
        instance_group_configs_[instance_group->name()] = instance_group;
    }

    // recover instance info
    value_map.clear();
    ec = storage_->Load(kRegistryInstanceKey, value_map);
    if (ec != EC_OK && ec != EC_NOENT) {
        KVCM_LOG_ERROR("load registry instance group failed");
        return ec;
    }
    std::vector<std::string> instance_ids;
    for (const auto &[id, val] : value_map) {
        std::map<std::string, std::string> instance_map;
        ec = storage_->Load(id, instance_map);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("load registry instance info failed, instance id[%s]", id.c_str());
            return ec;
        }
        auto instance_info = std::make_shared<InstanceInfo>();
        if (!instance_info->FromJsonString(instance_map[id])) {
            KVCM_LOG_ERROR("instance info from json string failed, json string[%s]", instance_map[id].c_str());
            return EC_CONFIG_ERROR;
        }
        instance_infos_[id] = instance_info;
    }
    KVCM_LOG_INFO(
        "registry manager do recover success, recover storage num[%lu], available storage num[%lu], account num[%lu], "
        "instance group num[%lu], instance info num[%lu]",
        storage_num,
        storage_num - not_available_storage_num,
        accounts_.size(),
        instance_group_configs_.size(),
        instance_infos_.size());
    return EC_OK;
}
ErrorCode RegistryManager::DoCleanup() {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    KVCM_LOG_INFO("registry manager start cleanup");

    auto ec = data_storage_manager_->DoCleanup();
    if (ec != EC_OK) {
        KVCM_LOG_WARN("failed during cleanup data_storage_manager, ec[%d]", ec);
    }
    // clear config and infos
    instance_group_configs_.clear();
    instance_infos_.clear();
    accounts_.clear();

    KVCM_LOG_INFO("registry manager cleanup completed");
    return EC_OK;
}

std::string RegistryManager::GetInstanceGroupName(const std::string &instance_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = instance_infos_.find(instance_id);
    if (it != instance_infos_.end()) {
        return it->second->instance_group_name();
    }
    return "";
}

} // namespace kv_cache_manager
