#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/manager/cache_location_view.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/manager/write_location_manager.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"

namespace kv_cache_manager {

class RegistryManager;
class MetaSearcherManager;
class MetaSearcher;
class MetaIndexerManager;
class MetricsRegistry;
class CacheReclaimer;
class SchedulePlanExecutor;
class ReclaimerTaskSupervisor;
class StartupConfigLoader;
class RequestContext;
class SelectLocationPolicy;
class EventManager;
class CacheManagerMetricsRecorder;
constexpr unsigned int DEFAULT_SCHEDULE_PLAN_EXECUTOR_THREAD_COUNT = 2;

class CacheManager {
    // TODO should not public
public:
    enum class QueryType {
        QT_UNSPECIFIED = 0,
        QT_BATCH_GET = 1,
        QT_PREFIX_MATCH = 2,
        QT_REVERSE_ROLL_SW_MATCH = 3,
    };
    std::string QueryTypeToString(QueryType query_type) const {
        switch (query_type) {
        case QueryType::QT_BATCH_GET:
            return "batch_get";
        case QueryType::QT_PREFIX_MATCH:
            return "prefix_match";
        case QueryType::QT_REVERSE_ROLL_SW_MATCH:
            return "reverse_roll_sw_match";
        default:
            return "unspecified";
        }
    }
    using KeyType = int64_t;
    using KeyVector = std::vector<KeyType>;
    using TokenIds = int64_t;
    using TokenIdsVector = std::vector<KeyType>;
    using UriType = std::string;
    using UriVector = std::vector<UriType>;

    CacheManager(std::shared_ptr<MetricsRegistry> metrics_registry, std::shared_ptr<RegistryManager> registry_manager);
    ~CacheManager();

    bool Init(int32_t schedule_plan_executor_thread_count = DEFAULT_SCHEDULE_PLAN_EXECUTOR_THREAD_COUNT);
    ErrorCode DoRecover();
    ErrorCode DoCleanup();
    std::shared_ptr<RegistryManager> GetRegistryManager() { return registry_manager_; }

    std::pair<ErrorCode, std::string> RegisterInstance(RequestContext *request_context,
                                                       const std::string &instance_group,
                                                       const std::string &instance_id,
                                                       int32_t block_size,
                                                       const std::vector<LocationSpecInfo> &location_spec_infos,
                                                       const ModelDeployment &model_deployment,
                                                       const std::vector<LocationSpecGroup> &location_spec_groups);

    ErrorCode
    RemoveInstance(RequestContext *request_context, const std::string &instance_group, const std::string &instance_id);

    std::pair<ErrorCode, std::shared_ptr<const InstanceInfo>> GetInstanceInfo(RequestContext *request_context,
                                                                              const std::string &instance_id);

    std::pair<ErrorCode, CacheMetaVecWrapper> GetCacheMeta(RequestContext *request_context,
                                                           const std::string &instance_id,
                                                           const KeyVector &keys,
                                                           const TokenIdsVector &tokens,
                                                           const BlockMask &block_mask,
                                                           int32_t detail_level /*TODO*/);

    std::pair<ErrorCode, CacheLocationViewVecWrapper>
    GetCacheLocation(RequestContext *request_context,
                     const std::string &instance_id,
                     QueryType query_type,
                     const KeyVector &keys,
                     const TokenIdsVector &tokens,
                     const BlockMask &block_mask,
                     int32_t sw_size,
                     const std::vector<std::string> &location_spec_names);

    std::pair<ErrorCode, StartWriteCacheInfo> StartWriteCache(RequestContext *request_context,
                                                              const std::string &instance_id,
                                                              const KeyVector &keys,
                                                              const TokenIdsVector &tokens,
                                                              const std::vector<std::string> &location_spec_group_names,
                                                              int64_t write_timeout_seconds);
    ErrorCode FinishWriteCache(RequestContext *request_context,
                               const std::string &instance_id,
                               const std::string &write_session_id,
                               const BlockMask &success_block_mask);

    ErrorCode RemoveCache(RequestContext *request_context,
                          const std::string &instance_id,
                          const KeyVector &keys,
                          const TokenIdsVector &tokens,
                          const BlockMask &block_mask /*TODO*/);
    ErrorCode TrimCache(RequestContext *request_context,
                        const std::string &instance_id,
                        const proto::meta::TrimStrategy &trim_strategy,
                        std::int32_t begin_ts = -1,
                        std::int32_t end_ts = -1) const noexcept;

    void PauseReclaimer();
    void ResumeReclaimer();

    std::shared_ptr<MetaIndexerManager> meta_indexer_manager() { return meta_indexer_manager_; }
    std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor() { return schedule_plan_executor_; }
    std::shared_ptr<CacheReclaimer> cache_reclaimer() { return cache_reclaimer_; }
    std::shared_ptr<EventManager> event_manager() { return event_manager_; }
    std::shared_ptr<CacheManagerMetricsRecorder> metrics_recorder() { return metrics_recorder_; }

private:
    ErrorCode FilterWriteCache(RequestContext *request_context,
                               const std::string &instance_id,
                               MetaSearcher *meta_searcher,
                               const KeyVector &keys,
                               KeyVector &new_keys,
                               const std::vector<std::string> &location_spec_group_names,
                               std::vector<std::string_view> &new_location_spec_group_names,
                               BlockMask &block_mask);
    ErrorCode GenWriteLocation(RequestContext *request_context,
                               const std::string &instance_id,
                               const CacheManager::KeyVector &keys,
                               const std::vector<std::string_view> &location_spec_group_names,
                               CacheLocationVector &new_locations);
    ErrorCode CreateInSingleBatch(RequestContext *request_context,
                                  const std::string &instance_id,
                                  const CacheManager::KeyVector &keys,
                                  const std::vector<std::string_view> &location_spec_group_names,
                                  const std::shared_ptr<const InstanceInfo> &instance_info,
                                  const std::shared_ptr<DataStorageManager> &data_storage_manager,
                                  const std::string &unique_name,
                                  std::vector<DataStorageUri> &allocated_uris,
                                  std::vector<std::vector<std::pair<size_t, const LocationSpecInfo *>>> &key_to_uris,
                                  bool &is_create_success,
                                  int64_t common_size);
    ErrorCode CreateBySpec(RequestContext *request_context,
                           const std::string &instance_id,
                           const CacheManager::KeyVector &keys,
                           const std::vector<std::string_view> &location_spec_group_names,
                           const std::shared_ptr<const InstanceInfo> &instance_info,
                           const std::shared_ptr<DataStorageManager> &data_storage_manager,
                           const std::string &unique_name,
                           std::vector<DataStorageUri> &allocated_uris,
                           std::vector<std::vector<std::pair<size_t, const LocationSpecInfo *>>> &key_to_uris,
                           bool &is_create_success);

    ErrorCode TryCreateMetaSearcher(RequestContext *request_context, const std::string &instance_id);
    std::pair<ErrorCode, MetaSearcher *> CheckInputAndGetMetaSearcher(RequestContext *request_context,
                                                                      const std::string &instance_id,
                                                                      const KeyVector &keys,
                                                                      const TokenIdsVector &tokens) const;
    std::pair<ErrorCode, int64_t> GetBlockSize(RequestContext *request_context, const std::string &instance_id) const;
    void FilterLocationSpecByName(CacheLocationVector &locations, const std::vector<std::string> &location_spec_names);
    std::string GetStorageConfigStr(RequestContext *request_context, const std::string &instance_id) const;
    ErrorCode GetCacheLocationByQueryType(MetaSearcher *meta_searcher,
                                          RequestContext *request_context,
                                          const std::string &instance_id,
                                          QueryType query_type,
                                          const KeyVector &keys,
                                          const BlockMask &block_mask,
                                          int32_t sw_size,
                                          CacheLocationVector &cache_locations) const;
    std::unique_ptr<SelectLocationPolicy> genSelectLocationPolicy(RequestContext *request_context,
                                                                  const std::string &instance_id) const;

private:
    /***
     * === 成员变量清理说明 ===
     * 所有成员变量必须添加注释说明在主备切换时是否需要清理，并按需要在DoCleanup中添加清理实现。
     * 1. 需要清理的成员：包含DoRecover加载的信息、运行时状态等，必须在 DoCleanup() 中正确处理。
     * 2. 无需清理的成员：只读配置、共享引用、主备切换时无需释放的长期对象（StartupConfigLoader）等。
     * ============================================
     */

    // 需要清理
    std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    // 需要清理 - 所有正在执行的写入均按失败处理
    std::shared_ptr<WriteLocationManager> write_location_manager_;
    // 需要清理
    std::shared_ptr<MetaSearcherManager> meta_searcher_manager_;
    // 需要清理
    std::unique_ptr<DataStorageSelector> data_storage_selector_;
    // 无需清理 - CacheManager当前没有给MetricsRegistry动态添加新的监控指标
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    // 无需清理 - RegistryManager单独进行了清理，不由CacheManager负责
    std::shared_ptr<RegistryManager> registry_manager_;
    // 无需清理 - 让遗留的Plan自行跑完
    std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor_;
    // 无需清理 - 仅需要暂停
    std::shared_ptr<CacheReclaimer> cache_reclaimer_;
    // 无需清理 - SchedulePlanExecutor遗留的Plan会继续跑完
    std::unique_ptr<ReclaimerTaskSupervisor> reclaimer_task_supervisor_;
    // 无需清理 - 不包含运行时状态
    std::shared_ptr<EventManager> event_manager_;
    // 需要清理 - 避免有metrics遗留
    std::shared_ptr<CacheManagerMetricsRecorder> metrics_recorder_;
};

} // namespace kv_cache_manager
