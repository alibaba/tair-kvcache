#include "kv_cache_manager/manager/cache_manager.h"

#include <algorithm>
#include <array>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/env_util.h"
#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/spec_events/optimizer_event.h"
#include "kv_cache_manager/manager/cache_manager_metrics_recorder.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/manager/hash_util.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/manager/reclaimer_task_supervisor.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/manager/select_location_policy.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_collector.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"

namespace kv_cache_manager {

#define PREFIX_LOG(LEVEL, format, args...)                                                                             \
    do {                                                                                                               \
        KVCM_LOG_##LEVEL("trace_id [%s] instance [%s] | " format, trace_id.c_str(), instance_id.c_str(), ##args);      \
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

#define RETURN_IF_EC_NOT_OK_WITH_LOG(LEVEL, ec, format, args...)                                                       \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG(LEVEL, format, ##args);                                                                         \
            return ec;                                                                                                 \
        }                                                                                                              \
    } while (0)

#define RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(LEVEL, ec, Type, format, args...)                                            \
    do {                                                                                                               \
        if ((ec) != EC_OK) {                                                                                           \
            PREFIX_LOG(LEVEL, format, ##args);                                                                         \
            return {ec, Type()};                                                                                       \
        }                                                                                                              \
    } while (0)

namespace {
CacheManager::KeyVector GenKeyVector(const CacheManager::TokenIdsVector &tokens, int64_t block_size) {
    std::vector<int64_t> block_keys;
    size_t total_blocks = tokens.size() / block_size;

    int64_t hash = 0;
    for (int index = 0; index < total_blocks; index++) {
        auto pos = index * block_size;
        hash = hashInt64Array(hash, &tokens[pos], &tokens[pos + block_size]);
        block_keys.push_back(hash);
    }
    return block_keys;
}

inline std::pair<kv_cache_manager::ErrorCode, bool>
IsSpecNameInSpecGroup(const std::string &trace_id,
                      const std::string &instance_id,
                      std::string_view spec_name,
                      std::string_view group_name,
                      const std::vector<kv_cache_manager::LocationSpecGroup> &location_spec_groups) {
    // we have sorted location_spec_groups before
    auto it_group = std::lower_bound(location_spec_groups.begin(),
                                     location_spec_groups.end(),
                                     group_name,
                                     [](const auto &location_spec_group, std::string_view group_name) {
                                         return location_spec_group.name() < group_name;
                                     });
    if (it_group == location_spec_groups.end() || it_group->name() != group_name) {
        PREFIX_LOG(WARN, "not find group [%s]", group_name.data());
        return {EC_ERROR, false};
    }
    const auto &group = *it_group;
    auto it_spec_name = std::lower_bound(
        group.spec_names().begin(),
        group.spec_names().end(),
        spec_name,
        [](const std::string &src_spec_name, std::string_view dst_spec_name) { return src_spec_name < dst_spec_name; });
    if (it_spec_name == group.spec_names().end() || *it_spec_name != spec_name) {
        PREFIX_LOG(DEBUG, "not find spec_name [%s] in group [%s]", spec_name.data(), group_name.data());
        return {EC_OK, false};
    }
    return {EC_OK, true};
}

} // namespace

CacheManager::CacheManager(std::shared_ptr<MetricsRegistry> metrics_registry,
                           std::shared_ptr<RegistryManager> registry_manager)
    : meta_indexer_manager_(std::make_shared<MetaIndexerManager>())
    , write_location_manager_(std::make_shared<WriteLocationManager>())
    , meta_searcher_manager_(std::make_shared<MetaSearcherManager>(registry_manager, meta_indexer_manager_))
    , data_storage_selector_(std::make_unique<DataStorageSelector>(meta_indexer_manager_, registry_manager))
    , metrics_registry_(std::move(metrics_registry))
    , registry_manager_(std::move(registry_manager))
    , metrics_recorder_(std::make_shared<CacheManagerMetricsRecorder>(
          meta_indexer_manager_, write_location_manager_, registry_manager_)) {}

CacheManager::~CacheManager() {}

bool CacheManager::Init(int32_t schedule_plan_executor_thread_count) {
    schedule_plan_executor_ = std::make_shared<SchedulePlanExecutor>(schedule_plan_executor_thread_count,
                                                                     meta_indexer_manager_,
                                                                     registry_manager_->data_storage_manager(),
                                                                     metrics_registry_);
    event_manager_ = std::make_shared<EventManager>();
    if (!event_manager_) {
        KVCM_LOG_WARN("create EventManager failed");
    }
    if (!event_manager_->Init()) {
        KVCM_LOG_ERROR("event_manager init failed");
    }

    cache_reclaimer_ = std::make_shared<CacheReclaimer>(registry_manager_,
                                                        meta_indexer_manager_,
                                                        meta_searcher_manager_,
                                                        schedule_plan_executor_,
                                                        metrics_registry_,
                                                        event_manager_);
    if (cache_reclaimer_->Start() != EC_OK) {
        KVCM_LOG_ERROR("CacheManager init failed");
        return false;
    }
    reclaimer_task_supervisor_ = std::make_unique<ReclaimerTaskSupervisor>(schedule_plan_executor_);
    reclaimer_task_supervisor_->Start();
    write_location_manager_->Start();
    metrics_recorder_->Start();
    KVCM_LOG_INFO("CacheManager init OK");
    return true;
}

std::pair<ErrorCode, std::string>
CacheManager::RegisterInstance(RequestContext *request_context,
                               const std::string &instance_group,
                               const std::string &instance_id,
                               int32_t block_size,
                               const std::vector<LocationSpecInfo> &location_spec_infos,
                               const ModelDeployment &model_deployment,
                               const std::vector<LocationSpecGroup> &location_spec_groups) {
    SPAN_TRACER(request_context);
    // TODO : not thread safe now
    const auto &trace_id = request_context->trace_id();
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (instance_info) {
        if (instance_info->block_size() != block_size || instance_info->location_spec_infos() != location_spec_infos ||
            instance_info->model_deployment() != model_deployment ||
            instance_info->location_spec_groups() != location_spec_groups) {
            PREFIX_LOG(WARN, "register instance failed: duplicate instance");
            request_context->error_tracer()->AddErrorMsg("register instance failed: duplicate instance");
            return {EC_DUPLICATE_ENTITY, {}};
        }
        auto ec = TryCreateMetaSearcher(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, std::string, "register instance failed with errorcode: %d", ec);
        PREFIX_LOG(INFO, "register instance OK");
        return {ec, GetStorageConfigStr(request_context, instance_id)};
    }
    auto ec = registry_manager_->RegisterInstance(request_context,
                                                  instance_group,
                                                  instance_id,
                                                  block_size,
                                                  location_spec_infos,
                                                  model_deployment,
                                                  location_spec_groups);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, std::string, "register instance failed with errorcode: %d", ec);
    ec = TryCreateMetaSearcher(request_context, instance_id);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, std::string, "register instance failed with errorcode: %d", ec);
    PREFIX_LOG(INFO, "register instance OK");
    return {ec, GetStorageConfigStr(request_context, instance_id)};
}

ErrorCode CacheManager::RemoveInstance(RequestContext *request_context,
                                       const std::string &instance_group,
                                       const std::string &instance_id) {
    SPAN_TRACER(request_context);
    const auto &trace_id = request_context->trace_id();

    auto ec = registry_manager_->RemoveInstance(request_context, instance_group, instance_id);
    RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "remove instance failed");

    ec = TrimCache(request_context, instance_id, proto::meta::TrimStrategy::TS_REMOVE_ALL_CACHE);
    RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "remove instance failed");
    PREFIX_LOG(INFO, "remove instance OK");
    return ec;
}

std::pair<ErrorCode, InstanceInfoConstPtr> CacheManager::GetInstanceInfo(RequestContext *request_context,
                                                                         const std::string &instance_id) {
    SPAN_TRACER(request_context);
    const auto &trace_id = request_context->trace_id();
    InstanceInfoConstPtr info_ptr = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (info_ptr == nullptr) {
        PREFIX_LOG(DEBUG, "get instance info failed");
        return {EC_INSTANCE_NOT_EXIST, nullptr};
    }
    return {EC_OK, info_ptr};
}

std::pair<ErrorCode, std::vector<InstanceInfoConstPtr>> ListInstanceInfo(RequestContext *request_context,
                                                                         const std::string &instance_group) {
    SPAN_TRACER(request_context);
    return {EC_OK, std::vector<InstanceInfoConstPtr>()};
}

std::pair<ErrorCode, CacheMetaVecWrapper> CacheManager::GetCacheMeta(RequestContext *request_context,
                                                                     const std::string &instance_id,
                                                                     const KeyVector &keys,
                                                                     const TokenIdsVector &tokens,
                                                                     const BlockMask &block_mask,
                                                                     int32_t detail_level /*TODO*/) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    auto [ec, meta_searcher] = CheckInputAndGetMetaSearcher(request_context, instance_id, keys, tokens);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(DEBUG, ec, CacheMetaVecWrapper, "get cache meta failed");
    std::vector<CacheLocationMap> location_maps;
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, ManagerBatchGetLocation);
    if (!keys.empty()) {
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, keys.size());
        ec = meta_searcher->BatchGetLocation(request_context, keys, block_mask, location_maps);
    } else {
        auto [ec_temp, block_size] = GetBlockSize(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(DEBUG, ec_temp, CacheMetaVecWrapper, "get cache meta failed");
        auto gen_keys = GenKeyVector(tokens, block_size);
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, gen_keys.size());
        ec = meta_searcher->BatchGetLocation(request_context, gen_keys, block_mask, location_maps);
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, ManagerBatchGetLocation);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(DEBUG, ec, CacheMetaVecWrapper, "get cache meta failed: BatchGetLocation fail");
    // TODO, 现在BatchGetLocation接口还未返回 location properties 信息, 先置空
    // 另外现在BatchGetLocation接口返回的是一个block key对应的location map, 和proto定义不同,
    // 先临时只返回map里的第一个 location(不管是不是在serving状态), 将serving状态保存在meta里 这里现在非常 ugly
    CacheLocationVector cache_locations;
    std::vector<std::string> metas;
    std::map<std::string, std::string> meta;
    for (CacheLocationMap &location_map : location_maps) {
        auto iter = location_map.begin();
        if (iter != location_map.end()) {
            auto nh = location_map.extract(iter);
            cache_locations.push_back(std::move(nh.mapped()));
            meta["id"] = cache_locations.back().id();
        } else {
            CacheLocation cache_location;
            cache_location.set_status(CacheLocationStatus::CLS_NOT_FOUND);
            cache_locations.push_back(std::move(cache_location));
        }
        meta["status"] = CacheLocation::CacheLocationStatusToString(cache_locations.back().status());
        metas.push_back(Jsonizable::ToJsonString(meta));
    }

    return {ec, CacheMetaVecWrapper(std::move(metas), std::move(cache_locations))};
}

std::pair<ErrorCode, CacheLocationViewVecWrapper>
CacheManager::GetCacheLocation(RequestContext *request_context,
                               const std::string &instance_id,
                               QueryType query_type,
                               const KeyVector &keys,
                               const TokenIdsVector &tokens,
                               const BlockMask &block_mask,
                               int32_t sw_size,
                               const std::vector<std::string> &location_spec_names) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    auto [ec, meta_searcher] = CheckInputAndGetMetaSearcher(request_context, instance_id, keys, tokens);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, CacheLocationViewVecWrapper, "check input or get meta searcher failed");
    if (query_type == QueryType::QT_UNSPECIFIED) {
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_ERROR, CacheLocationViewVecWrapper, "unknown query type");
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, ManagerPrefixMatch);
    CacheLocationVector cache_locations;
    KeyVector query_keys = keys;
    if (!keys.empty()) {
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, keys.size());
        ec = GetCacheLocationByQueryType(
            meta_searcher, request_context, instance_id, query_type, keys, block_mask, sw_size, cache_locations);
    } else {
        auto [ec_temp, block_size] = GetBlockSize(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec_temp, CacheLocationViewVecWrapper, "get block_size failed");
        auto gen_keys = GenKeyVector(tokens, block_size);
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, gen_keys.size());
        query_keys = gen_keys;
        ec = GetCacheLocationByQueryType(
            meta_searcher, request_context, instance_id, query_type, gen_keys, block_mask, sw_size, cache_locations);
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, ManagerPrefixMatch);
    KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, prefix_match_len, cache_locations.size());
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, CacheLocationViewVecWrapper, "get cache location failed");
    FilterLocationSpecByName(cache_locations, location_spec_names);

    auto cache_get_event = std::make_shared<CacheGetEvent>(instance_id);
    cache_get_event->SetEventTriggerTime();
    cache_get_event->SetAddtionalArgs(
        QueryTypeToString(query_type), query_keys, tokens, block_mask, sw_size, location_spec_names);
    if (event_manager_)
        event_manager_->Publish(cache_get_event);
    return {ec, CacheLocationViewVecWrapper(std::move(cache_locations))};
}

std::pair<ErrorCode, StartWriteCacheInfo>
CacheManager::StartWriteCache(RequestContext *request_context,
                              const std::string &instance_id,
                              const KeyVector &keys,
                              const TokenIdsVector &tokens,
                              const std::vector<std::string> &location_spec_group_names,
                              int64_t write_timeout_seconds) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    if (!location_spec_group_names.empty() && keys.size() != location_spec_group_names.size()) {
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN,
                                          EC_ERROR,
                                          StartWriteCacheInfo,
                                          "location_spec_group_names size not match , expect[%zu], real[%zu]",
                                          keys.size(),
                                          location_spec_group_names.size());
    }
    auto [ec, meta_searcher] = CheckInputAndGetMetaSearcher(request_context, instance_id, keys, tokens);
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, StartWriteCacheInfo, "start write cache failed");
    CacheLocationVector new_locations;
    BlockMask block_mask;
    KeyVector new_keys;
    std::vector<std::string_view> new_location_spec_group_names;
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, ManagerFilterWriteCache);
    KeyVector query_keys = keys;

    ErrorCode filter_ec;
    if (!keys.empty()) {
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, keys.size());
        filter_ec = FilterWriteCache(request_context,
                                     instance_id,
                                     meta_searcher,
                                     keys,
                                     new_keys,
                                     location_spec_group_names,
                                     new_location_spec_group_names,
                                     block_mask);
    } else {
        auto [ec_temp, block_size] = GetBlockSize(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec_temp, StartWriteCacheInfo, "start write cache failed");
        auto gen_keys = GenKeyVector(tokens, block_size);
        query_keys = gen_keys;
        KVCM_METRICS_COLLECTOR_SET_METRICS(service_metrics_collector, manager, request_key_count, gen_keys.size());
        filter_ec = FilterWriteCache(request_context,
                                     instance_id,
                                     meta_searcher,
                                     gen_keys,
                                     new_keys,
                                     location_spec_group_names,
                                     new_location_spec_group_names,
                                     block_mask);
    }
    RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, filter_ec, StartWriteCacheInfo, "filter write cache failed");

    std::vector<std::string> location_ids;
    std::string write_session_id = StringUtil::GenerateRandomString(32);
    if (new_keys.empty()) {
        // if no new keys, delete this write_session_id as soon as possible
        write_timeout_seconds = 10; // seconds
    } else {
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, ManagerFilterWriteCache);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, StartWriteCacheInfo, "start write cache failed");
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, GenWriteLocation);
        ec = GenWriteLocation(request_context, instance_id, new_keys, new_location_spec_group_names, new_locations);
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, GenWriteLocation);
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, StartWriteCacheInfo, "start write cache failed");
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, ManagerBatchAddLocation);
        ec = meta_searcher->BatchAddLocation(request_context, new_keys, new_locations, location_ids);
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, ManagerBatchAddLocation);
        // FIXME(rui): PartialOK and return will cause storage leak (not possible to reclaim)
        //             example case -- indexer reach max key count after partial write
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, ec, StartWriteCacheInfo, "start write cache failed");
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, PutWriteLocationManager);
    write_location_manager_->Put(write_session_id,
                                 std::move(new_keys),
                                 std::move(location_ids),
                                 write_timeout_seconds,
                                 [this, trace_id, instance_id, write_session_id] {
                                     RequestContext temp_request_context(trace_id);
                                     BlockMaskOffset failed_mask = 0;
                                     auto ec = this->FinishWriteCache(
                                         &temp_request_context, instance_id, write_session_id, failed_mask);
                                     static_cast<void>(ec);
                                 });
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, PutWriteLocationManager);
    auto start_write_event = std::make_shared<StartWriteCacheEvent>(instance_id);
    start_write_event->SetEventTriggerTime();
    start_write_event->SetAddtionalArgs(
        write_session_id, query_keys, tokens, block_mask, location_spec_group_names, write_timeout_seconds);
    if (event_manager_)
        event_manager_->Publish(start_write_event);
    return {EC_OK,
            StartWriteCacheInfo(std::move(write_session_id),
                                std::move(block_mask),
                                CacheLocationViewVecWrapper(std::move(new_locations)))};
}

ErrorCode CacheManager::FinishWriteCache(RequestContext *request_context,
                                         const std::string &instance_id,
                                         const std::string &write_session_id,
                                         const BlockMask &success_block_mask) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    WriteLocationManager::WriteLocationInfo location_info;
    if (!write_location_manager_->GetAndDelete(write_session_id, location_info)) {
        request_context->error_tracer()->AddErrorMsg("write_session_id has been deleted");
        RETURN_IF_EC_NOT_OK_WITH_LOG(
            WARN, EC_ERROR, "finish write cache failed: write_session_id not found: %s", write_session_id.c_str());
    }

    if (!IsBlockMaskValid(success_block_mask, location_info.keys.size())) {
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN,
                                     EC_BADARGS,
                                     "invalid block mask, mask type: %zu, size: %zu",
                                     success_block_mask.index(),
                                     location_info.keys.size());
    }

    MetaSearcher *meta_searcher = meta_searcher_manager_->GetMetaSearcher(instance_id);
    if (!meta_searcher) {
        request_context->error_tracer()->AddErrorMsg("instance not exist");
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_INSTANCE_NOT_EXIST, "finish write cache failed: meta searcher not found");
    }
    std::vector<KeyType> success_batch_keys;
    std::vector<std::vector<MetaSearcher::LocationUpdateTask>> success_batch_update_tasks;
    CacheLocationDelRequest failed_del_request{.instance_id = instance_id, .delay = std::chrono::seconds(0)};

    for (size_t block_key_idx = 0; block_key_idx < location_info.keys.size(); block_key_idx++) {
        if (IsIndexInMaskRange(success_block_mask, block_key_idx)) {
            // success
            success_batch_keys.push_back(location_info.keys[block_key_idx]);
            success_batch_update_tasks.push_back(
                {{location_info.location_ids[block_key_idx], CacheLocationStatus::CLS_SERVING}});
        } else {
            // failed
            failed_del_request.block_keys.push_back(location_info.keys[block_key_idx]);
            failed_del_request.location_ids.push_back({location_info.location_ids[block_key_idx]});
        }
    }

    ErrorCode ec = ErrorCode::EC_OK;
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, manager, request_key_count, success_batch_keys.size());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, ManagerBatchUpdateLocation);
    if (!success_batch_keys.empty()) {
        std::vector<std::vector<ErrorCode>> out_batch_results;
        ec = meta_searcher->BatchUpdateLocationStatus(
            request_context, success_batch_keys, success_batch_update_tasks, out_batch_results);
        if (ec != EC_OK) {
            std::string detail_ec_str = MetaSearcher::BatchErrorCodeToStr(out_batch_results);
            PREFIX_LOG(WARN, "update location status failed, ec: %d, ec_batches: %s", ec, detail_ec_str.c_str());
        }
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, ManagerBatchUpdateLocation);

    if (!failed_del_request.block_keys.empty()) {
        reclaimer_task_supervisor_->Submit(request_context->trace_id(), std::move(failed_del_request));
        // no need to wait delete finish here
    }
    auto finish_write_event = std::make_shared<FinishWriteCacheEvent>(instance_id);
    finish_write_event->SetEventTriggerTime();
    finish_write_event->SetAddtionalArgs(write_session_id, success_block_mask);
    if (event_manager_)
        event_manager_->Publish(finish_write_event);
    return ec;
}

ErrorCode CacheManager::RemoveCache(RequestContext *request_context,
                                    const std::string &instance_id,
                                    const KeyVector &keys,
                                    const TokenIdsVector &tokens,
                                    const BlockMask &block_mask /*TODO*/) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    assert(schedule_plan_executor_);
    if (keys.empty() && tokens.empty()) {
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_BADARGS, "remove cache failed: empty input");
    }
    CacheMetaDelRequest request;
    request.instance_id = instance_id;
    if (!keys.empty()) {
        request.block_keys = keys;
    } else {
        auto [ec, block_size] = GetBlockSize(request_context, instance_id);
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "remove cache failed");
        auto gen_keys = GenKeyVector(tokens, block_size);
        request.block_keys = std::move(gen_keys);
    }
    reclaimer_task_supervisor_->Submit(trace_id, std::move(request));
    return EC_OK;
}

ErrorCode CacheManager::TrimCache(RequestContext *request_context,
                                  const std::string &instance_id,
                                  const proto::meta::TrimStrategy &trim_strategy,
                                  std::int32_t begin_ts,
                                  std::int32_t end_ts) const noexcept {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();

    if (trim_strategy != proto::meta::TS_REMOVE_ALL_CACHE) {
        PREFIX_LOG(WARN, "trim strategy not implemented");
        return ErrorCode::EC_UNIMPLEMENTED;
    }

    const auto meta_indexer = meta_indexer_manager_->GetMetaIndexer(instance_id);
    if (meta_indexer == nullptr) {
        PREFIX_LOG(WARN, "meta indexer is nullptr");
        return ErrorCode::EC_INSTANCE_NOT_EXIST;
    }

    std::string cursor = SCAN_BASE_CURSOR;
    do {
        constexpr std::size_t limit = 64;
        std::string next_cursor;

        CacheMetaDelRequest request;
        request.instance_id = instance_id;

        if (const auto ec = meta_indexer->Scan(cursor, limit, next_cursor, request.block_keys);
            ec != ErrorCode::EC_OK) {
            // TODO (rui): cache reclaimer should reclaim the dangling blocks
            RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "trim cache failed");
        }

        reclaimer_task_supervisor_->Submit(trace_id, std::move(request));
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);

    return ErrorCode::EC_OK;
}
void CacheManager::PauseReclaimer() { cache_reclaimer_->Pause(); }
void CacheManager::ResumeReclaimer() { cache_reclaimer_->Resume(); }

void CacheManager::FilterLocationSpecByName(CacheLocationVector &locations,
                                            const std::vector<std::string> &location_spec_names) {
    if (location_spec_names.empty()) {
        return;
    }

    const std::unordered_set<std::string> names_set(location_spec_names.begin(), location_spec_names.end());
    for (auto &location : locations) {
        std::vector<LocationSpec> new_specs;
        for (auto &spec : location.location_specs()) {
            if (names_set.count(spec.name()) == 0) {
                continue;
            }
            new_specs.push_back(spec);
        }
        location.set_location_specs(std::move(new_specs));
    }
}

ErrorCode CacheManager::FilterWriteCache(RequestContext *request_context,
                                         const std::string &instance_id,
                                         MetaSearcher *meta_searcher,
                                         const KeyVector &keys,
                                         KeyVector &new_keys,
                                         const std::vector<std::string> &location_spec_group_names,
                                         std::vector<std::string_view> &new_location_spec_group_names,
                                         BlockMask &block_mask) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    static BlockMask empty_block_mask = static_cast<size_t>(0);
    std::vector<CacheLocationMap> location_maps;
    auto ec = meta_searcher->BatchGetLocation(request_context, keys, empty_block_mask, location_maps);
    RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "BatchGetLocation failed");
    assert(keys.size() == location_maps.size());
    auto policy = genSelectLocationPolicy(request_context, instance_id);
    if (!policy) {
        return EC_ERROR;
    }
    auto first_empty = std::find_if(
        location_maps.begin(), location_maps.end(), [&policy](const auto &m) { return !policy->ExistsForWrite(m); });
    bool only_prefix_not_empty =
        std::all_of(first_empty, location_maps.end(), [&policy](const auto &m) { return !policy->ExistsForWrite(m); });
    if (only_prefix_not_empty) {
        size_t offset = first_empty - location_maps.begin();
        block_mask = static_cast<BlockMaskOffset>(offset);
        new_keys.insert(new_keys.end(), keys.begin() + offset, keys.end());
        if (!location_spec_group_names.empty()) {
            new_location_spec_group_names.insert(new_location_spec_group_names.end(),
                                                 location_spec_group_names.begin() + offset,
                                                 location_spec_group_names.end());
        }
        return EC_OK;
    }
    block_mask = BlockMaskVector(location_maps.size(), false);
    for (size_t i = 0; i < location_maps.size(); ++i) {
        if (!location_maps[i].empty()) {
            std::get<BlockMaskVector>(block_mask)[i] = true;
        } else {
            new_keys.push_back(keys[i]);
            if (!location_spec_group_names.empty()) {
                new_location_spec_group_names.push_back(location_spec_group_names[i]);
            }
        }
    }
    return EC_OK;
}

ErrorCode
CacheManager::CreateInSingleBatch(RequestContext *request_context,
                                  const std::string &instance_id,
                                  const CacheManager::KeyVector &keys,
                                  const std::vector<std::string_view> &location_spec_group_names,
                                  const std::shared_ptr<const InstanceInfo> &instance_info,
                                  const std::shared_ptr<DataStorageManager> &data_storage_manager,
                                  const std::string &unique_name,
                                  std::vector<DataStorageUri> &allocated_uris,
                                  std::vector<std::vector<std::pair<size_t, const LocationSpecInfo *>>> &key_to_uris,
                                  bool &is_create_success,
                                  int64_t common_size) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    std::vector<std::string> merged_block_keys;
    std::vector<size_t> merged_keys_idx;
    std::vector<const LocationSpecInfo *> spec_info_mapping;
    merged_block_keys.reserve(instance_info->location_spec_infos().size() * keys.size());
    merged_keys_idx.reserve(instance_info->location_spec_infos().size() * keys.size());
    spec_info_mapping.reserve(instance_info->location_spec_infos().size() * keys.size());

    for (const auto &spec_info : instance_info->location_spec_infos()) {
        if (location_spec_group_names.empty()) {
            for (size_t i = 0; i < keys.size(); i++) {
                std::string block_key = instance_id + "/" + spec_info.name() + "/" + StringUtil::Uint64ToHex(keys[i]);
                merged_block_keys.push_back(block_key);
                merged_keys_idx.push_back(i);
                spec_info_mapping.push_back(&spec_info);
            }
        } else {
            for (size_t i = 0; i < keys.size(); i++) {
                auto [ec, found] = IsSpecNameInSpecGroup(trace_id,
                                                         instance_id,
                                                         spec_info.name(),
                                                         location_spec_group_names[i],
                                                         instance_info->location_spec_groups());
                RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "IsSpecNameInSpecGroup failed");
                if (found) {
                    std::string block_key =
                        instance_id + "/" + spec_info.name() + "/" + StringUtil::Uint64ToHex(keys[i]);
                    merged_block_keys.push_back(block_key);
                    merged_keys_idx.push_back(i);
                    spec_info_mapping.push_back(&spec_info);
                }
            }
        }
    }

    std::vector<std::pair<ErrorCode, DataStorageUri>> results = data_storage_manager->Create(
        request_context, unique_name, merged_block_keys, common_size, []() { /* do nothing */ });

    for (size_t i = 0; i < results.size(); i++) {
        if (results[i].first == ErrorCode::EC_OK) {
            allocated_uris.push_back(results[i].second);
            key_to_uris[merged_keys_idx[i]].push_back({allocated_uris.size() - 1, spec_info_mapping[i]});
        }
    }
    // TODO: move check to another function
    if (results.size() != merged_block_keys.size()) {
        is_create_success = false;
        PREFIX_LOG(WARN,
                   "create data storage fail, results size:%ld, request size: %ld",
                   results.size(),
                   merged_block_keys.size());
    }
    for (auto &result : results) {
        if (result.first != ErrorCode::EC_OK) {
            is_create_success = false;
            PREFIX_LOG(WARN, "create data storage fail, ec_code: %d", result.first);
            break;
        }
    }
    return EC_OK;
}

ErrorCode CacheManager::CreateBySpec(RequestContext *request_context,
                                     const std::string &instance_id,
                                     const CacheManager::KeyVector &keys,
                                     const std::vector<std::string_view> &location_spec_group_names,
                                     const std::shared_ptr<const InstanceInfo> &instance_info,
                                     const std::shared_ptr<DataStorageManager> &data_storage_manager,
                                     const std::string &unique_name,
                                     std::vector<DataStorageUri> &allocated_uris,
                                     std::vector<std::vector<std::pair<size_t, const LocationSpecInfo *>>> &key_to_uris,
                                     bool &is_create_success) {
    // avoid use file across tp ranks
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    for (const auto &spec_info : instance_info->location_spec_infos()) {
        std::vector<std::string> block_keys;
        std::vector<size_t> keys_idx;
        block_keys.reserve(keys.size());
        keys_idx.reserve(keys.size());
        if (location_spec_group_names.empty()) {
            for (size_t i = 0; i < keys.size(); i++) {
                std::string block_key = instance_id + "/" + spec_info.name() + "/" + StringUtil::Uint64ToHex(keys[i]);
                block_keys.push_back(block_key);
                keys_idx.push_back(i);
            }
        } else {
            for (size_t i = 0; i < keys.size(); i++) {
                auto [ec, found] = IsSpecNameInSpecGroup(trace_id,
                                                         instance_id,
                                                         spec_info.name(),
                                                         location_spec_group_names[i],
                                                         instance_info->location_spec_groups());
                RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "IsSpecNameInSpecGroup failed");
                if (found) {
                    std::string block_key =
                        instance_id + "/" + spec_info.name() + "/" + StringUtil::Uint64ToHex(keys[i]);
                    block_keys.push_back(block_key);
                    keys_idx.push_back(i);
                }
            }
        }

        std::vector<std::pair<ErrorCode, DataStorageUri>> results = data_storage_manager->Create(
            request_context, unique_name, block_keys, spec_info.size(), []() { /* do nothing */ });

        for (size_t i = 0; i < results.size(); i++) {
            if (results[i].first == ErrorCode::EC_OK) {
                allocated_uris.push_back(results[i].second);
                key_to_uris[keys_idx[i]].push_back({allocated_uris.size() - 1, &spec_info});
            }
        }

        // TODO: move check to another function
        if (results.size() != block_keys.size()) {
            is_create_success = false;
            PREFIX_LOG(WARN,
                       "create data storage fail, results size:%ld, request size: %ld",
                       results.size(),
                       block_keys.size());
        }
        for (auto &result : results) {
            if (result.first != ErrorCode::EC_OK) {
                is_create_success = false;
                PREFIX_LOG(WARN, "create data storage fail, ec_code: %d", result.first);
                break;
            }
        }

        if (!is_create_success) {
            break;
        }
    }
    return EC_OK;
}

ErrorCode CacheManager::GenWriteLocation(RequestContext *request_context,
                                         const std::string &instance_id,
                                         const CacheManager::KeyVector &keys,
                                         const std::vector<std::string_view> &location_spec_group_names,
                                         CacheLocationVector &new_locations) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();

    if (keys.empty()) {
        PREFIX_LOG(INFO, "new keys empty, no need to generate write location");
        return EC_OK;
    }

    auto data_storage_manager = registry_manager_->data_storage_manager();
    if (data_storage_manager == nullptr) {
        request_context->error_tracer()->AddErrorMsg("data storage manager not found");
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_ERROR, "data storage manager not found");
    }

    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (instance_info == nullptr) {
        request_context->error_tracer()->AddErrorMsg("instance not found");
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_INSTANCE_NOT_EXIST, "instance not found");
    }

    // select storage type and unique name
    const auto select_result = data_storage_selector_->SelectCacheWriteDataStorageBackend(
        request_context, instance_info->instance_group_name());
    RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, select_result.ec, "select storage backend failed");

    std::vector<DataStorageUri> allocated_uris;
    allocated_uris.reserve(instance_info->location_spec_infos().size() * keys.size());
    std::vector<std::vector<std::pair<size_t, const LocationSpecInfo *>>> key_to_uris(keys.size());
    bool is_create_success = true;

    bool merge = instance_info->location_spec_infos().empty() ||
                 std::all_of(instance_info->location_spec_infos().begin() + 1,
                             instance_info->location_spec_infos().end(),
                             [&instance_info](const auto &spec) {
                                 return spec.size() == instance_info->location_spec_infos().front().size();
                             });
    int64_t common_size = merge && !instance_info->location_spec_infos().empty()
                              ? instance_info->location_spec_infos().front().size()
                              : 0;

    if (merge) {
        auto ec = CreateInSingleBatch(request_context,
                                      instance_id,
                                      keys,
                                      location_spec_group_names,
                                      instance_info,
                                      data_storage_manager,
                                      select_result.name,
                                      allocated_uris,
                                      key_to_uris,
                                      is_create_success,
                                      common_size);
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "CreateInSingleBatch failed");
    } else {
        auto ec = CreateBySpec(request_context,
                               instance_id,
                               keys,
                               location_spec_group_names,
                               instance_info,
                               data_storage_manager,
                               select_result.name,
                               allocated_uris,
                               key_to_uris,
                               is_create_success);
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, ec, "CreateBySpec failed");
    }

    if (!is_create_success) {
        request_context->error_tracer()->AddErrorMsg("some internal error when GenWriteLocation");
        auto error_codes = data_storage_manager->Delete(
            request_context, select_result.name, allocated_uris, []() { /* do nothing */ });
        for (size_t i = 0; i < error_codes.size(); i++) {
            if (i >= allocated_uris.size()) {
                PREFIX_LOG(WARN,
                           "wrong error code num from Delete, results size:%ld, request size: %ld",
                           error_codes.size(),
                           allocated_uris.size());
                break;
            }
            if (error_codes[i] != ErrorCode::EC_OK) {
                PREFIX_LOG(WARN,
                           "delete data uri failed, storage unique name: %s, uri: %s",
                           select_result.name.c_str(),
                           allocated_uris[i].ToUriString().c_str());
            }
        }
        return EC_ERROR;
    }

    for (const auto &uris : key_to_uris) {
        CacheLocation cache_location;
        cache_location.set_type(select_result.type);
        for (const auto &[data_storage_uri_idx, location_spec_info] : uris) {
            LocationSpec location_spec;
            location_spec.set_name(location_spec_info->name());
            location_spec.set_uri(allocated_uris[data_storage_uri_idx].ToUriString());
            cache_location.push_location_spec(std::move(location_spec));
        }
        cache_location.set_spec_size(uris.size());
        new_locations.push_back(std::move(cache_location));
    }
    return EC_OK;
}

ErrorCode CacheManager::TryCreateMetaSearcher(RequestContext *request_context, const std::string &instance_id) {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    MetaSearcher *meta_searcher = meta_searcher_manager_->TryCreateMetaSearcher(request_context, instance_id);
    if (!meta_searcher) {
        RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_ERROR, "create meta searcher failed");
    }
    PREFIX_LOG(INFO, "create meta searcher success");
    return EC_OK;
}

std::pair<ErrorCode, MetaSearcher *> CacheManager::CheckInputAndGetMetaSearcher(RequestContext *request_context,
                                                                                const std::string &instance_id,
                                                                                const KeyVector &keys,
                                                                                const TokenIdsVector &tokens) const {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    MetaSearcher *meta_searcher = meta_searcher_manager_->GetMetaSearcher(instance_id);
    if (!meta_searcher) {
        PREFIX_LOG(WARN, "meta searcher not found");
        request_context->error_tracer()->AddErrorMsg("instance not exist");
        return {EC_INSTANCE_NOT_EXIST, nullptr};
    }
    if (keys.empty() && tokens.empty()) {
        PREFIX_LOG(WARN, "empty input");
        request_context->error_tracer()->AddErrorMsg("empty input");
        return {EC_BADARGS, nullptr};
    }
    return {EC_OK, meta_searcher};
}

std::pair<ErrorCode, int64_t> CacheManager::GetBlockSize(RequestContext *request_context,
                                                         const std::string &instance_id) const {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (!instance_info) {
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_INSTANCE_NOT_EXIST, int64_t, "instance not found");
    };
    int64_t block_size = instance_info->block_size();
    if (block_size <= 0) {
        request_context->error_tracer()->AddErrorMsg("tokens size per block error");
        RETURN_IF_EC_NOT_OK_WITH_TYPE_LOG(WARN, EC_BADARGS, int64_t, "tokens size per block error [%ld]", block_size);
    };
    return {EC_OK, block_size};
}

std::string CacheManager::GetStorageConfigStr(RequestContext *request_context, const std::string &instance_id) const {
    SPAN_TRACER(request_context);
    const auto &trace_id = request_context->trace_id();
    auto all_configs = registry_manager_->data_storage_manager()->ListStorageConfig();
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (instance_info == nullptr) {
        PREFIX_LOG(WARN, "instance not found");
        return {};
    }
    const auto &instance_group_name = instance_info->instance_group_name();
    auto [ec, instance_group] = registry_manager_->GetInstanceGroup(request_context, instance_group_name);
    if (instance_group == nullptr) {
        PREFIX_LOG(WARN, "instance group not found: %s", instance_group_name.c_str());
        return {};
    }
    const auto &storage_candadates = instance_group->storage_candidates();
    std::set<std::string_view> storage_candadate_set(storage_candadates.begin(), storage_candadates.end());
    // TODO : try optimize these copy operation
    std::vector<const StorageConfig *> result;
    for (const auto &config : all_configs) {
        if (storage_candadate_set.find(config.global_unique_name()) != storage_candadate_set.end()) {
            result.push_back(&config);
        }
    }
    return Jsonizable::ToJsonString(result);
}

ErrorCode CacheManager::GetCacheLocationByQueryType(MetaSearcher *meta_searcher,
                                                    RequestContext *request_context,
                                                    const std::string &instance_id,
                                                    QueryType query_type,
                                                    const KeyVector &keys,
                                                    const BlockMask &block_mask,
                                                    int32_t sw_size,
                                                    CacheLocationVector &cache_locations) const {
    SPAN_TRACER(request_context);
    const std::string &trace_id = request_context->trace_id();
    auto policy = genSelectLocationPolicy(request_context, instance_id);
    if (policy == nullptr) {
        return EC_ERROR;
    }
    ErrorCode ec = EC_ERROR;
    switch (query_type) {
    case QueryType::QT_BATCH_GET: {
        ec = meta_searcher->BatchGetBestLocation(request_context, keys, cache_locations, policy.get());
        break;
    }
    case QueryType::QT_PREFIX_MATCH: {
        ec = meta_searcher->PrefixMatch(request_context, keys, block_mask, cache_locations, policy.get());
        break;
    }
    case QueryType::QT_REVERSE_ROLL_SW_MATCH: {
        if (keys.size() < sw_size || sw_size < 0) {
            request_context->error_tracer()->AddErrorMsg("QT_REVERSE_ROLL_SW_MATCH bad args");
            RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_BADARGS, "bad keys size: %zu, %d", keys.size(), sw_size);
        }
        ec = meta_searcher->ReverseRollSlideWindowMatch(request_context, keys, sw_size, cache_locations, policy.get());
        break;
    }
    default:
        assert(false);
    }
    if (ec == EC_OK && query_type != QueryType::QT_PREFIX_MATCH) {
        auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
        if (instance_info == nullptr) {
            request_context->error_tracer()->AddErrorMsg("instance not found");
            RETURN_IF_EC_NOT_OK_WITH_LOG(WARN, EC_INSTANCE_NOT_EXIST, "instance not found");
        }
        for (auto &location : cache_locations) {
            if (location.spec_size() == 0) {
                location.set_spec_size(instance_info->location_spec_infos().size());
                for (auto &spec_info : instance_info->location_spec_infos()) {
                    location.push_location_spec(LocationSpec(spec_info.name(), ""));
                }
            }
        }
    }
    return ec;
}

ErrorCode CacheManager::DoRecover() {
    if (!registry_manager_) {
        KVCM_LOG_ERROR("CacheManager do recover failed, registry_manager is nullptr");
        return EC_ERROR;
    }
    auto request_context = std::make_shared<RequestContext>("cache_manager_recover_trace");
    auto [ec1, instance_groups] = registry_manager_->ListInstanceGroup(request_context.get());
    if (ec1 != EC_OK) {
        KVCM_LOG_ERROR("CacheManager ListInstanceGroup failed when recover, ec[%d]", ec1);
        return ec1;
    }
    for (const auto &instance_group : instance_groups) {
        std::string group_name = instance_group->name();
        auto [ec2, instance_infos] = registry_manager_->ListInstanceInfo(request_context.get(), group_name);
        if (ec2 != EC_OK) {
            KVCM_LOG_ERROR("CacheManager ListInstanceInfo failed when recover, ec[%d] instance_group name[%s]",
                           ec2,
                           group_name.c_str());
            return ec2;
        }
        for (const auto &instance_info : instance_infos) {
            auto [ec3, config_str] = RegisterInstance(request_context.get(),
                                                      group_name,
                                                      instance_info->instance_id(),
                                                      instance_info->block_size(),
                                                      instance_info->location_spec_infos(),
                                                      instance_info->model_deployment(),
                                                      instance_info->location_spec_groups());
            if (ec3 != EC_OK) {
                KVCM_LOG_ERROR("CacheManager RegisterInstance failed when recover, ec[%d] instance_group "
                               "name[%s] instance_id[%s]",
                               ec3,
                               group_name.c_str(),
                               instance_info->instance_id().c_str());
                return ec3;
            }
            KVCM_LOG_INFO("CacheManager RegisterInstance success when recover, instance_id[%s], storage_config[%s]",
                          instance_info->instance_id().c_str(),
                          config_str.c_str());
        }
    }
    return EC_OK;
}
ErrorCode CacheManager::DoCleanup() {
    // aborting write session need meta indexer
    write_location_manager_->DoCleanup();
    meta_searcher_manager_->DoCleanup();
    meta_indexer_manager_->DoCleanup();
    metrics_recorder_->DoCleanup();
    data_storage_selector_->DoCleanup();

    return EC_OK;
}

std::unique_ptr<SelectLocationPolicy> CacheManager::genSelectLocationPolicy(RequestContext *request_context,
                                                                            const std::string &instance_id) const {
    const auto &trace_id = request_context->trace_id();
    auto all_storages = registry_manager_->data_storage_manager()->GetAllStorageNames();
    auto all_available_storages = registry_manager_->data_storage_manager()->GetAvailableStorages();
    if (all_available_storages.size() >= all_storages.size()) {
        return std::make_unique<StaticWeightSLPolicy>();
    }
    auto instance_info = registry_manager_->GetInstanceInfo(request_context, instance_id);
    if (instance_info == nullptr) {
        request_context->error_tracer()->AddErrorMsg("instance not found");
        PREFIX_LOG(WARN, "instance not found");
        return nullptr;
    }
    const auto &instance_group_name = instance_info->instance_group_name();
    auto [ec, instance_group] = registry_manager_->GetInstanceGroup(request_context, instance_group_name);
    if (instance_group == nullptr) {
        request_context->error_tracer()->AddErrorMsg("instance group not found");
        PREFIX_LOG(WARN, "instance group not found: %s", instance_group_name.c_str());
        return nullptr;
    }
    const auto &storage_candadates = instance_group->storage_candidates();
    std::vector<std::shared_ptr<DataStorageBackend>> group_storages;
    std::vector<std::shared_ptr<DataStorageBackend>> group_available_storages;
    group_storages.reserve(all_available_storages.size());
    group_available_storages.reserve(group_storages.size());
    for (const auto &candadate : storage_candadates) {
        if (auto iter = std::find_if(
                all_storages.begin(), all_storages.end(), [&candadate](const auto &name) { return candadate == name; });
            iter != all_storages.end()) {
            if (auto storage_backend = registry_manager_->data_storage_manager()->GetDataStorageBackend(*iter);
                storage_backend != nullptr) {
                group_storages.push_back(storage_backend);
            }
        }
        if (auto iter = std::find_if(all_available_storages.begin(),
                                     all_available_storages.end(),
                                     [&candadate](const auto &backend) {
                                         return candadate == backend->GetStorageConfig().global_unique_name();
                                     });
            iter != all_available_storages.end()) {
            group_available_storages.push_back(*iter);
        }
    }
    if (group_available_storages.size() >= group_storages.size()) {
        return std::make_unique<StaticWeightSLPolicy>();
    }
    if (group_available_storages.empty()) {
        request_context->error_tracer()->AddErrorMsg("all storages are unavailable");
        KVCM_INTERVAL_LOG_WARN(10, "all storages are unavailable!");
        return nullptr;
    }
    std::array<uint32_t, 5> data_storage_counts{0};
    bool is_all_type_only_one = true;
    for (const auto &storage : group_storages) {
        size_t idx = static_cast<size_t>(storage->GetType());
        if (++data_storage_counts[idx] > 1) {
            is_all_type_only_one = false;
            break;
        }
    }
    if (is_all_type_only_one) {
        StaticWeightSLPolicy::WeightArray weight_array{0};
        for (const auto &storage : group_available_storages) {
            size_t idx = static_cast<size_t>(storage->GetType());
            weight_array[idx] = 1;
        }
        return std::make_unique<DynamicWeightSLPoliy>(weight_array);
    }

    NamedStorageWeightedSLPolicy::WeightMap weight_map;
    for (const auto &storage : group_available_storages) {
        weight_map[storage->GetStorageConfig().global_unique_name()] = 1;
    }
    return std::make_unique<NamedStorageWeightedSLPolicy>(std::move(weight_map));
}

} // namespace kv_cache_manager
