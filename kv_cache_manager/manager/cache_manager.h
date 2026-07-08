#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"
#include "kv_cache_manager/manager/cache_garbage_collector.h"
#include "kv_cache_manager/manager/cache_location_view.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/select_location_policy.h"
#include "kv_cache_manager/manager/write_location_manager.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"

namespace kv_cache_manager {

class RegistryManager;
class MetaSearcherManager;
class MetaIndexerManager;
class MetricsRegistry;
class CacheReclaimer;
class SchedulePlanExecutor;
class ReclaimerTaskSupervisor;
class StartupConfigLoader;
class EventManager;
class CacheManagerMetricsRecorder;
class EventReportBackend;
struct MetricsLifecycle;
class MigrationManager;
constexpr unsigned int DEFAULT_SCHEDULE_PLAN_EXECUTOR_THREAD_COUNT = 2;
constexpr unsigned int DEFAULT_SCHEDULE_PLAN_MIGRATION_WORKER_BUDGET = 1;
constexpr unsigned int DEFAULT_META_QUERY_WORKER_COUNT = 4;
constexpr std::size_t DEFAULT_META_QUERY_PARALLEL_THRESHOLD = 256;
constexpr std::size_t DEFAULT_META_QUERY_CHUNK_SIZE = 128;

class CacheManager {
    // TODO should not public
public:
    enum class QueryType {
        QT_UNSPECIFIED = 0,
        QT_BATCH_GET = 1,
        QT_PREFIX_MATCH = 2,
        QT_REVERSE_ROLL_SW_MATCH = 3,
        QT_PREFIX_MATCH_WITH_MAMBA = 4,
    };
    std::string QueryTypeToString(QueryType query_type) const {
        switch (query_type) {
        case QueryType::QT_BATCH_GET:
            return "batch_get";
        case QueryType::QT_PREFIX_MATCH:
            return "prefix_match";
        case QueryType::QT_REVERSE_ROLL_SW_MATCH:
            return "reverse_roll_sw_match";
        case QueryType::QT_PREFIX_MATCH_WITH_MAMBA:
            return "prefix_match_with_mamba";
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

    struct HostCacheMatch {
        std::string host_ip_port;
        int64_t local;
        int64_t p2p_1_fetch;
        int64_t p2p_1_total_match;
    };

    CacheManager(std::shared_ptr<MetricsRegistry> metrics_registry,
                 std::shared_ptr<RegistryManager> registry_manager,
                 std::shared_ptr<MetricsLifecycle> metrics_lifecycle = nullptr);
    ~CacheManager();

    bool Init(int32_t schedule_plan_executor_thread_count = DEFAULT_SCHEDULE_PLAN_EXECUTOR_THREAD_COUNT,
              uint64_t cache_reclaimer_key_sampling_size_total = 1000,
              uint64_t cache_reclaimer_key_sampling_size_per_task = 100,
              uint64_t cache_reclaimer_del_batch_size = 100,
              uint32_t cache_reclaimer_idle_interval_ms = 100,
              uint32_t cache_reclaimer_worker_size = 16,
              CacheReclaimerAsyncDeleteConfig cache_reclaimer_async_delete_config = {},
              uint32_t schedule_plan_migration_worker_budget = DEFAULT_SCHEDULE_PLAN_MIGRATION_WORKER_BUDGET,
              uint32_t meta_query_worker_count = DEFAULT_META_QUERY_WORKER_COUNT,
              std::size_t meta_query_parallel_threshold = DEFAULT_META_QUERY_PARALLEL_THRESHOLD,
              std::size_t meta_query_chunk_size = DEFAULT_META_QUERY_CHUNK_SIZE,
              CacheGarbageCollector::Config cache_gc_config = {});
    ErrorCode DoRecover();
    ErrorCode DoRecoverOnce();
    void StartRecoverRetryLoop();
    void StopRecoverRetryLoop();
    ErrorCode DoCleanup();
    std::shared_ptr<RegistryManager> GetRegistryManager() { return registry_manager_; }
    [[nodiscard]] std::shared_ptr<MetricsLifecycle> metrics_lifecycle() const { return metrics_lifecycle_; }

    // register a callback invoked after an instance is fully removed;
    // must be set before serving traffic (not thread-safe for writes)
    // the callback runs on the RemoveInstance call stack — callees
    // must not re-enter CacheManager methods
    using OnInstanceRemovedFn = std::function<void(const std::string &instance_id)>;
    void SetOnInstanceRemoved(OnInstanceRemovedFn fn) { on_instance_removed_ = std::move(fn); }

    std::string GetExtraInfo(RequestContext *request_context, const std::string &instance_id);

    std::pair<ErrorCode, std::string> RegisterInstance(RequestContext *request_context,
                                                       const std::string &instance_group,
                                                       const std::string &instance_id,
                                                       int32_t block_size,
                                                       const std::vector<LocationSpecInfo> &location_spec_infos,
                                                       const ModelDeployment &model_deployment,
                                                       const std::vector<LocationSpecGroup> &location_spec_groups,
                                                       QueryType default_query_type = QueryType::QT_UNSPECIFIED);

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

    std::pair<ErrorCode, CacheMetaDetailVec> GetCacheMetaDetail(RequestContext *request_context,
                                                                const std::string &instance_id,
                                                                const KeyVector &keys,
                                                                const TokenIdsVector &tokens,
                                                                const BlockMask &block_mask,
                                                                int32_t detail_level /*reserved*/);

    std::pair<ErrorCode, CacheLocationViewVecWrapper>
    GetCacheLocation(RequestContext *request_context,
                     const std::string &instance_id,
                     QueryType query_type,
                     const KeyVector &keys,
                     const TokenIdsVector &tokens,
                     const BlockMask &block_mask,
                     int32_t sw_size,
                     const std::vector<std::string> &location_spec_names);

    std::pair<ErrorCode, BatchLocationsView>
    GetCacheLocationsByBackend(RequestContext *request_context,
                               const std::string &instance_id,
                               QueryType query_type,
                               const KeyVector &keys,
                               const TokenIdsVector &tokens,
                               const BlockMask &block_mask,
                               int32_t sw_size,
                               const std::vector<std::string> &location_spec_names,
                               const std::vector<BackendSelector> &backend_selectors = {});

    std::pair<ErrorCode, int64_t> GetCacheLocationLen(RequestContext *request_context,
                                                      const std::string &instance_id,
                                                      QueryType query_type,
                                                      const KeyVector &keys,
                                                      const TokenIdsVector &tokens,
                                                      int32_t sw_size);

    std::pair<ErrorCode, StartWriteCacheInfo> StartWriteCache(RequestContext *request_context,
                                                              const std::string &instance_id,
                                                              const KeyVector &keys,
                                                              const TokenIdsVector &tokens,
                                                              const std::vector<std::string> &location_spec_group_names,
                                                              int64_t write_timeout_seconds,
                                                              int32_t min_replica_count = 1);
    ErrorCode
    FinishWriteCache(RequestContext *request_context,
                     const std::string &instance_id,
                     const std::string &write_session_id,
                     const BlockMask &success_block_mask,
                     std::unique_ptr<WriteLocationManager::WriteLocationInfo> write_location_info_internal = nullptr);

    ErrorCode RemoveCache(RequestContext *request_context,
                          const std::string &instance_id,
                          const KeyVector &keys,
                          const TokenIdsVector &tokens,
                          const BlockMask &block_mask /*TODO*/);

    // 分层迁移编排 facade：把原本散在 AdminServiceImpl::MigrateCache 的业务编排
    // （候选采样 / location 批查 / 逐 block 准入 / Copy+Mark 分发与 fallback / 计数）收到 manager 层，
    // service 层只做 proto glue。返回内部 ErrorCode + accepted/rejected + message，由调用方映射 proto。
    struct MigrateCacheResult {
        ErrorCode ec = EC_OK;
        int64_t accepted = 0;
        int64_t rejected = 0;
        std::string message;
    };
    // do_copy/do_mark 由 method 翻译而来；explicit_block_keys 非空则优先，否则按 sample_count 采样。
    MigrateCacheResult MigrateCache(RequestContext *request_context,
                                    const std::string &trace_id,
                                    const std::string &instance_id,
                                    const std::string &src_name,
                                    const std::string &dst_name,
                                    bool do_copy,
                                    bool do_mark,
                                    const std::vector<int64_t> &explicit_block_keys,
                                    int64_t sample_count);

    ErrorCode ReportEvent(RequestContext *request_context,
                          const proto::meta::ReportEventRequest *request,
                          proto::meta::ReportEventResponse *response);
    std::pair<ErrorCode, std::vector<HostCacheMatch>>
    GetHostCacheState(RequestContext *request_context,
                      const std::string &instance_id,
                      QueryType query_type,
                      const KeyVector &block_cache_keys,
                      const std::vector<std::string> &medium_filter = {},
                      size_t p2p_host_count = 0);
    ErrorCode TrimCache(RequestContext *request_context,
                        const std::string &instance_id,
                        const proto::meta::TrimStrategy &trim_strategy,
                        std::int32_t begin_ts = -1,
                        std::int32_t end_ts = -1) const noexcept;

    void PauseReclaimer();
    void ResumeReclaimer();
    ErrorCode StartCacheGarbageCollector();
    void RequestStopCacheGarbageCollector();
    void JoinCacheGarbageCollector();

    // 多层存储迁移控制面随 leader 生命周期启停（OnBecomeLeader/OnNoLongerLeader）。
    void StartMigrationManager();
    void StopMigrationManager();

    std::shared_ptr<MetaIndexerManager> meta_indexer_manager() { return meta_indexer_manager_; }
    std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor() { return schedule_plan_executor_; }
    std::shared_ptr<CacheReclaimer> cache_reclaimer() { return cache_reclaimer_; }
    std::shared_ptr<CacheGarbageCollector> cache_garbage_collector() { return cache_garbage_collector_; }
    std::shared_ptr<EventManager> event_manager() { return event_manager_; }
    std::shared_ptr<CacheManagerMetricsRecorder> metrics_recorder() { return metrics_recorder_; }
    std::shared_ptr<MigrationManager> migration_manager() { return migration_manager_; }

    // Set revisit interval histogram configuration for per-instance tracking.
    // Must be called before any MetaIndexer is created.
    void SetRevisitHistogramConfig(const std::vector<double> &boundaries);

private:
    struct EventCleanupCallbackState {
        std::shared_mutex mutex;
        bool accepting = true;
        // Advances whenever cleanup is deactivated. A task admitted before a
        // leader cleanup must stay stale even if the same CacheManager is later
        // activated again during recovery.
        uint64_t epoch = 1;
    };

    ErrorCode FilterWriteCache(RequestContext *request_context,
                               const std::string &instance_id,
                               MetaSearcher *meta_searcher,
                               const KeyVector &keys,
                               KeyVector &new_keys,
                               const std::vector<std::string> &location_spec_group_names,
                               std::vector<std::string_view> &new_location_spec_group_names,
                               BlockMask &block_mask,
                               int32_t min_replica_count,
                               std::vector<std::string> &new_keys_tiered_targets);
    ErrorCode FilterWriteCacheWithMinReplica(RequestContext *request_context,
                                             const std::string &instance_id,
                                             MetaSearcher *meta_searcher,
                                             const KeyVector &keys,
                                             KeyVector &new_keys,
                                             const std::vector<std::string> &location_spec_group_names,
                                             std::vector<std::string_view> &new_location_spec_group_names,
                                             BlockMask &block_mask,
                                             int32_t min_replica_count,
                                             std::vector<std::string> &new_keys_tiered_targets);
    ErrorCode GenWriteLocation(RequestContext *request_context,
                               const std::string &instance_id,
                               const CacheManager::KeyVector &keys,
                               const std::vector<std::string_view> &location_spec_group_names,
                               const std::vector<std::string> &tiered_targets,
                               CacheLocationVector &new_locations);
    // 按 LocationSpec URI 汇总各 storage 的预估写入字节并记录到 per-storage 指标。
    // URI 约定：hostname = storage unique_name（DataStorageManager::Create 设置），
    // size 参数 = 该 spec 的字节数（各 backend Create 时写入）。解析失败的 spec 跳过。
    void RecordWriteBytesForLocations(const CacheLocationVector &locations);
    // 在指定 storage 上为 keys 分配写 location（GenWriteLocation 的单 storage 分配单元，
    // 供多层存储 Mark 路径按 block 路由到不同 storage 复用）。out_locations 与 keys 同序追加。
    ErrorCode GenWriteLocationOnStorage(RequestContext *request_context,
                                        const std::string &instance_id,
                                        const CacheManager::KeyVector &keys,
                                        const std::vector<std::string_view> &location_spec_group_names,
                                        const std::shared_ptr<const InstanceInfo> &instance_info,
                                        const std::shared_ptr<DataStorageManager> &data_storage_manager,
                                        const std::string &storage_name,
                                        DataStorageType storage_type,
                                        CacheLocationVector &out_locations);
    void RollbackAddLocations(RequestContext *request_context,
                              const std::string &instance_id,
                              const KeyVector &keys,
                              const CacheLocationVector &locations,
                              const std::vector<MetaSearcher::AddLocationResult> &add_results);
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
    ErrorCode CheckLocationSpecGroupNames(RequestContext *request_context,
                                          const std::string &instance_id,
                                          size_t key_count,
                                          const std::vector<std::string> &location_spec_group_names);
    static void FillEmptyLocationSpecs(const std::vector<LocationSpecInfo> &location_spec_infos,
                                       CacheLocationVector &locations);
    std::string GetStorageConfigStr(RequestContext *request_context, const std::string &instance_id) const;

    void CleanupHostLocations(const std::string &instance_id,
                              const std::string &host_ip_port,
                              uint64_t cleanup_generation,
                              DataStorageType storage_type,
                              const std::shared_ptr<EventReportBackend> &expected_backend);
    ErrorCode CleanupStaleSnapshotLocations(const ReporterSnapshotKey &reporter_key,
                                            const std::string &snapshot_version,
                                            DataStorageType storage_type,
                                            const std::shared_ptr<EventReportBackend> &event_backend,
                                            uint64_t snapshot_attempt_epoch = 0,
                                            uint64_t lifecycle_generation = 0);
    ErrorCode GetCacheLocationByQueryType(MetaSearcher *meta_searcher,
                                          RequestContext *request_context,
                                          const std::string &instance_id,
                                          QueryType query_type,
                                          const KeyVector &keys,
                                          const BlockMask &block_mask,
                                          int32_t sw_size,
                                          CacheLocationVector &cache_locations) const;
    ErrorCode PerformCacheLocationQuery(RequestContext *request_context,
                                        ServiceMetricsCollector *service_metrics_collector,
                                        MetaSearcher *meta_searcher,
                                        const std::string &instance_id,
                                        QueryType query_type,
                                        const KeyVector &keys,
                                        const TokenIdsVector &tokens,
                                        const BlockMask &block_mask,
                                        int32_t sw_size,
                                        KeyVector &query_keys,
                                        CacheLocationVector &cache_locations) const;
    std::unique_ptr<SelectLocationPolicy> genSelectLocationPolicy(RequestContext *request_context,
                                                                  const std::string &instance_id) const;
    CheckLocDataExistFunc GetCheckLocDataExistFunc(const std::string &instance_id) const;
    MetaSearcher::CheckHostCacheLocationFunc
    GetHostCacheStateCheckLocDataExistFunc(const std::string &instance_id) const;
    SubmitDelReqFunc GetSubmitDelReqFunc(const std::string &instance_id) const;
    void ClearEventCleanupCallbacks();
    void DeactivateEventCleanupCallbacks();
    void ActivateEventCleanupCallbacks();

    // purge metrics registry entries and invoke the removal callback
    // for a given instance_id
    void InvalidateInstanceMetrics(const std::string &instance_id) const;

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
    std::shared_ptr<DataStorageSelector> data_storage_selector_;
    // 无需清理 - CacheManager当前没有给MetricsRegistry动态添加新的监控指标
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    // 无需清理 - RegistryManager单独进行了清理，不由CacheManager负责
    std::shared_ptr<RegistryManager> registry_manager_;
    // 无需清理 - 让遗留的Plan自行跑完
    std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor_;
    // leader demotion 和析构时先停止 GC 线程；已接受的删除任务由 Executor best effort 继续执行
    std::shared_ptr<CacheGarbageCollector> cache_garbage_collector_;
    // 无需清理 - 仅需要暂停
    std::shared_ptr<CacheReclaimer> cache_reclaimer_;
    // 无需清理 - 随 leader 生命周期 Start/Stop；活跃任务在切主时由 Reclaimer 孤儿检测兜底
    std::shared_ptr<MigrationManager> migration_manager_;
    // 无需清理 - SchedulePlanExecutor遗留的Plan会继续跑完
    std::unique_ptr<ReclaimerTaskSupervisor> reclaimer_task_supervisor_;
    // 无需清理 - 不包含运行时状态
    std::shared_ptr<EventManager> event_manager_;
    // 无需清理
    std::shared_ptr<MetricsLifecycle> metrics_lifecycle_;
    // EventReportBackend owns a callback that cannot retain CacheManager.
    // This separate gate lets destruction drain a callback copy that was
    // already taken by the liveness thread and reject copies invoked later.
    std::shared_ptr<EventCleanupCallbackState> event_cleanup_callback_state_ =
        std::make_shared<EventCleanupCallbackState>();
    // 需要清理 - 避免有metrics遗留
    std::shared_ptr<CacheManagerMetricsRecorder> metrics_recorder_;
    // 无需清理
    OnInstanceRemovedFn on_instance_removed_;
    // 需要清理 - recover 重试线程相关，在DoCleanup()中StopRecoverRetryLoop()
    std::thread recover_retry_thread_;
    std::atomic<bool> recover_retry_stop_{false};
};

} // namespace kv_cache_manager
