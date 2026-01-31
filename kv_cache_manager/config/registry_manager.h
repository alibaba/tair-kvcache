#pragma once

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/config/registry_storage_backend.h"

namespace kv_cache_manager {

class LocationSpecInfo;
class LocationSpecGroup;
class Account;
enum class AccountRole : uint8_t;
class InstanceGroup;
class CacheConfig;
class MetricsRegistry;
class DataStorageManager;
class InstanceInfo;
class ModelDeployment;
class StorageConfig;
class RequestContext;
class Jsonizable;

class RegistryManager {
public:
    RegistryManager(const std::string &registry_uri, std::shared_ptr<MetricsRegistry> metrics_registry);
    bool Init();
    ErrorCode DoRecover();
    ErrorCode DoCleanup();

public:
    ErrorCode AddStorage(RequestContext *request_context, const StorageConfig &storage_config);
    ErrorCode EnableStorage(RequestContext *request_context, const std::string &global_unique_name);
    ErrorCode DisableStorage(RequestContext *request_context, const std::string &global_unique_name);
    ErrorCode RemoveStorage(RequestContext *request_context, const std::string &global_unique_name);
    ErrorCode UpdateStorage(RequestContext *request_context, const StorageConfig &storage_config, bool force_update);
    std::pair<ErrorCode, std::vector<StorageConfig>> ListStorage(RequestContext *request_context);

    ErrorCode CreateInstanceGroup(RequestContext *request_context, const InstanceGroup &instance_group);
    ErrorCode
    UpdateInstanceGroup(RequestContext *request_context, const InstanceGroup &instance_group, int64_t current_version);
    ErrorCode RemoveInstanceGroup(RequestContext *request_context, const std::string &name);
    std::pair<ErrorCode, std::shared_ptr<const InstanceGroup>> GetInstanceGroup(RequestContext *request_context,
                                                                                const std::string &name);
    std::pair<ErrorCode, std::vector<std::shared_ptr<const InstanceGroup>>>
    ListInstanceGroup(RequestContext *request_context) const; // list all the instance_groups
    ErrorCode RegisterInstance(RequestContext *request_context,
                               const std::string &instance_group,
                               const std::string &instance_id,
                               int32_t block_size,
                               const std::vector<LocationSpecInfo> &location_spec_infos,
                               const ModelDeployment &model_deployment,
                               const std::vector<LocationSpecGroup> &location_spec_groups = {});

    ErrorCode
    RemoveInstance(RequestContext *request_context, const std::string &instance_group, const std::string &instance_id);
    std::shared_ptr<const InstanceInfo> GetInstanceInfo(RequestContext *request_context,
                                                        const std::string &instance_id);
    std::pair<ErrorCode, std::vector<std::shared_ptr<const InstanceInfo>>>
    ListInstanceInfo(RequestContext *request_context, const std::string &instance_group);

    ErrorCode AddAccount(RequestContext *request_context,
                         const std::string &user_name,
                         const std::string &password,
                         const AccountRole &role);
    ErrorCode DeleteAccount(RequestContext *request_context, const std::string &user_name);
    std::pair<ErrorCode, std::vector<std::shared_ptr<const Account>>> ListAccount(RequestContext *request_context);
    std::pair<ErrorCode, std::string> GenConfigSnapshot(RequestContext *request_context);
    ErrorCode LoadConfigSnapshot(RequestContext *request_context, const std::string &config_snapshot);

    std::shared_ptr<const CacheConfig> GetCacheConfig(const std::string &instance_group);

    std::shared_ptr<DataStorageManager> data_storage_manager() const;
    std::string GetInstanceGroupName(const std::string &instance_id) const;

private:
    ErrorCode LoadAndSave(const std::string &key, const std::string &id, const Jsonizable *jsonizable);
    ErrorCode LoadAndDelete(const std::string &key, const std::string &id);
    ErrorCode UpdateStorageAvailableStatus(const std::string &global_unique_name, bool is_available);

private:
    /***
     * === 成员变量清理说明 ===
     * 所有成员变量必须添加注释说明在主备切换时是否需要清理，并按需要在DoCleanup中添加清理实现。
     * 1. 需要清理的成员：包含DoRecover加载的信息、运行时状态等，必须在 DoCleanup() 中正确处理。
     * 2. 无需清理的成员：只读配置、共享引用、主备切换时无需释放的长期对象（如RegistryStorageBackend）等。
     * ============================================
     */

    // 无需清理 - 只读配置，无运行时状态
    std::string registry_storage_uri_;
    // 无需清理 - 共享引用，且RegistryManager本身没有根据动态信息添加Metrics
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    // 需要清理 - 在DoCleanup()中调用UnRegisterStorage所有存储
    std::shared_ptr<DataStorageManager> data_storage_manager_;
    // 需要清理 - 在DoCleanup()中clear()
    std::unordered_map<std::string, std::shared_ptr<InstanceGroup>> instance_group_configs_;
    // 需要清理 - 在DoCleanup()中clear()
    std::unordered_map<std::string, std::shared_ptr<InstanceInfo>> instance_infos_;
    // 需要清理 - 在DoCleanup()中clear()
    std::unordered_map<std::string, std::shared_ptr<Account>> accounts_;
    // 无需清理 - 主备切换不改变到registry后端存储的配置
    std::unique_ptr<RegistryStorageBackend> storage_;
    // 无需清理 - 不包含数据
    mutable std::shared_mutex mutex_;
};

} // namespace kv_cache_manager
