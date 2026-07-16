#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <exception>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_group_quota.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/meta_cache_policy_config.h"
#include "kv_cache_manager/config/meta_indexer_config.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/config/migration_strategy.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/quota_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/config/trigger_strategy.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/manager/migration_manager.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/manager/write_location_manager.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"

using namespace kv_cache_manager;

bool VecContains(const std::vector<std::int64_t> &vec, const std::int64_t v) {
    return std::any_of(vec.cbegin(), vec.cend(), [v](const std::int64_t &e) { return e == v; });
}

std::vector<CacheLocationMap>
MakeServingLocationMaps(const std::size_t count, const DataStorageType type = DataStorageType::DATA_STORAGE_TYPE_NFS) {
    std::vector<CacheLocationMap> maps;
    maps.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const std::string location_id = "serving_location_" + std::to_string(i);
        maps.push_back(CacheLocationMap{
            {location_id,
             std::make_shared<CacheLocation>(
                 location_id, CacheLocationStatus::CLS_SERVING, type, 0, std::vector<LocationSpec>{})},
        });
    }
    return maps;
}

CacheLocationConstPtr MakeCacheLocation(const std::string &location_id,
                                        const CacheLocationStatus status,
                                        const DataStorageType type,
                                        const std::string &uri) {
    std::vector<LocationSpec> specs;
    if (!uri.empty()) {
        specs.emplace_back("test_spec", uri);
    }
    return std::make_shared<CacheLocation>(location_id, status, type, specs.size(), std::move(specs));
}

class TypedTestBackend : public DataStorageBackend {
public:
    TypedTestBackend(std::shared_ptr<MetricsRegistry> metrics_registry, DataStorageType type)
        : DataStorageBackend(std::move(metrics_registry)), type_(type) {}

    DataStorageType GetType() override { return type_; }
    bool Available() override { return true; }
    double GetStorageUsageRatio(const std::string &trace_id) const override {
        static_cast<void>(trace_id);
        return 0.0;
    }
    ErrorCode DoOpen(const StorageConfig &config, const std::string &trace_id) override {
        static_cast<void>(config);
        static_cast<void>(trace_id);
        return EC_OK;
    }
    ErrorCode Close() override { return EC_OK; }
    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t size_per_key,
                                                             const std::string &trace_id,
                                                             std::function<void()> cb) override {
        static_cast<void>(size_per_key);
        static_cast<void>(trace_id);
        static_cast<void>(cb);
        return std::vector<std::pair<ErrorCode, DataStorageUri>>(keys.size(), {EC_OK, DataStorageUri{}});
    }
    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string &trace_id,
                                  std::function<void()> cb) override {
        static_cast<void>(trace_id);
        static_cast<void>(cb);
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }
    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<bool>(storage_uris.size(), true);
    }
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

private:
    DataStorageType type_;
};

/* ---------------- RegistryManager_ListInstanceGroup_stub ---------------- */

using ins_group_ptr_vec = std::vector<std::shared_ptr<const InstanceGroup>>;
ErrorCode list_ins_group_result;
ins_group_ptr_vec instance_groups;
int list_ins_group_call_counter;
std::mutex list_ins_group_mut;

std::shared_ptr<InstanceGroup> InstanceGroupFactory() {
    const auto instance_group = std::make_shared<InstanceGroup>();

    // set basic instance group properties
    instance_group->set_name("default_test_group");
    instance_group->set_storage_candidates({"3fs_storage_01"});
    instance_group->set_global_quota_group_name("default_quota_group");
    instance_group->set_max_instance_count(100);
    instance_group->set_user_data(R"({"description": "Default instance group for KV Cache Manager"})");
    instance_group->set_version(1);

    // set quota configuration
    QuotaConfig quota_config;
    quota_config.set_capacity(10737418240LL); // 10GB
    quota_config.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);

    InstanceGroupQuota quota;
    quota.set_capacity(10737418240LL); // 10GB
    quota.set_quota_config({quota_config});
    instance_group->set_quota(quota);

    // set cache configuration
    // create trigger strategy
    TriggerStrategy trigger_strategy;
    trigger_strategy.set_used_size(1073741824); // 1GB
    trigger_strategy.set_used_percentage(0.8);

    // create reclaim strategy
    const auto reclaim_strategy = std::make_shared<CacheReclaimStrategy>();
    reclaim_strategy->set_storage_unique_name("3fs_storage_01");
    reclaim_strategy->set_reclaim_policy(ReclaimPolicy::POLICY_LRU);
    reclaim_strategy->set_trigger_strategy(trigger_strategy);
    reclaim_strategy->set_trigger_period_seconds(60);
    reclaim_strategy->set_reclaim_step_size(1073741824); // 1GB
    reclaim_strategy->set_reclaim_step_percentage(10);
    reclaim_strategy->set_delay_before_delete_ms(1000);

    // create meta storage backend config
    const auto meta_storage_backend_config = std::make_shared<MetaStorageBackendConfig>();
    meta_storage_backend_config->SetStorageType("local");
    meta_storage_backend_config->SetStorageUri("file:///tmp/meta_storage");

    // create meta cache policy config
    const auto meta_cache_policy_config = std::make_shared<MetaCachePolicyConfig>();
    meta_cache_policy_config->SetCapacity(10000);
    meta_cache_policy_config->SetType("LRU");

    // create meta indexer config
    const auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
    meta_indexer_config->SetMaxKeyCount(1000000);
    meta_indexer_config->SetMutexShardNum(16);
    meta_indexer_config->SetBatchKeySize(16);
    meta_indexer_config->SetMetaStorageBackendConfig(meta_storage_backend_config);
    meta_indexer_config->SetMetaCachePolicyConfig(meta_cache_policy_config);

    // create cache config
    const auto cache_config = std::make_shared<CacheConfig>();
    cache_config->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS);
    cache_config->set_reclaim_strategy(reclaim_strategy);
    cache_config->set_meta_indexer_config(meta_indexer_config);

    instance_group->set_cache_config(cache_config);
    return instance_group;
}

std::pair<ErrorCode, ins_group_ptr_vec> RegistryManager_ListInstanceGroup_stub(void *obj, RequestContext *rc) {
    std::lock_guard<std::mutex> lock(list_ins_group_mut);
    ++list_ins_group_call_counter;
    return std::make_pair(list_ins_group_result, instance_groups);
}

/* ---------------- RegistryManager_ListInstanceInfo_stub ---------------- */

using ins_info_ptr_vec = std::vector<std::shared_ptr<const InstanceInfo>>;
ErrorCode list_ins_info_result;
ins_info_ptr_vec instance_infos;

std::shared_ptr<InstanceInfo> InstanceInfoFactory() {
    ModelDeployment model_deployment;
    model_deployment.set_model_name("test_model");
    model_deployment.set_dtype("test_dtype");
    model_deployment.set_use_mla(false);
    model_deployment.set_tp_size(2);
    model_deployment.set_dp_size(4);
    model_deployment.set_pp_size(2);
    model_deployment.set_lora_name("test_lora_name");
    model_deployment.set_extra("test_extra");
    model_deployment.set_user_data("test_user_data");

    const auto instance_info = std::make_shared<InstanceInfo>();
    instance_info->set_instance_id("test_instance_id");
    instance_info->set_instance_group_name("default_test_group");
    instance_info->set_quota_group_name("default_quota_group");
    instance_info->set_block_size(8);
    LocationSpecInfo spec_info{"test", 1024};
    instance_info->set_location_spec_infos({spec_info});
    instance_info->set_model_deployment(model_deployment);
    return instance_info;
}

std::pair<ErrorCode, ins_info_ptr_vec>
RegistryManager_ListInstanceInfo_stub(void *obj, RequestContext *rc, const std::string &ig) {
    ins_info_ptr_vec iv;
    for (const auto &i : instance_infos) {
        if (!i // nullptr is reserved for testing purpose
            || i->instance_group_name() == ig) {
            iv.emplace_back(i);
        }
    }
    return std::make_pair(list_ins_info_result, iv);
}

/* ---------------- SchedulePlanExecutor_SubmitAsync_stub ---------------- */

std::chrono::milliseconds spe_submit_delay{0};
PlanExecuteResult del_result;
bool spe_submit_accepted;
bool spe_submit_auto_complete;
bool spe_submit_invalid_future;
std::vector<CacheLocationDelRequest> submitted_del_requests;
std::vector<std::shared_ptr<std::promise<PlanExecuteResult>>> submitted_del_promises;
std::mutex submitted_del_requests_mutex;
using spe_submit_async_loc = AsyncDeleteSubmitResult (SchedulePlanExecutor::*)(const CacheLocationDelRequest &);

AsyncDeleteSubmitResult SchedulePlanExecutor_SubmitAsync_stub(void *obj, const CacheLocationDelRequest &request) {
    const auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    {
        std::lock_guard<std::mutex> lock(submitted_del_requests_mutex);
        submitted_del_requests.emplace_back(request);
        if (!spe_submit_auto_complete) {
            submitted_del_promises.emplace_back(promise);
        }
    }
    std::this_thread::sleep_for(spe_submit_delay);
    if (!spe_submit_accepted) {
        return {};
    }
    if (spe_submit_invalid_future) {
        return AsyncDeleteSubmitResult{true, {}};
    }
    auto future = promise->get_future();
    if (spe_submit_auto_complete) {
        promise->set_value(del_result);
    }
    return AsyncDeleteSubmitResult{true, std::move(future)};
}

/* ---------------- MetaIndexerManager_GetMetaIndexer_stub ---------------- */

// the dummy meta_indexer that the meta_indexer_manager would return
std::shared_ptr<MetaIndexer> dummy_meta_indexer;

std::shared_ptr<MetaIndexer> MetaIndexerManager_GetMetaIndexer_stub(void *obj, const std::string &i) {
    return dummy_meta_indexer;
}

/* ---------------- MetaIndexer_GetProperties_stub ---------------- */

std::chrono::milliseconds mi_getprop_delay{0};
ErrorCode get_result;
PropertyMapVector get_out_properties;

MetaIndexer::Result MetaIndexer_GetProperties_stub(void *obj,
                                                   RequestContext *rc,
                                                   const KeyVector &k,
                                                   const std::vector<std::string> &p,
                                                   PropertyMapVector &out_properties) noexcept {
    if (get_result == ErrorCode::EC_OK) {
        if (k.size() == get_out_properties.size()) {
            out_properties = get_out_properties;
        } else {
            out_properties = PropertyMapVector(k.size());
        }
    } else {
        out_properties = PropertyMapVector(k.size());
    }
    std::this_thread::sleep_for(mi_getprop_delay);
    MetaIndexer::Result result(k.size());
    result.ec = get_result;
    if (get_result != ErrorCode::EC_OK) {
        std::fill(result.error_codes.begin(), result.error_codes.end(), get_result);
    }
    return result;
}

/* ---------------- MetaIndexer_RandomSample_stub ---------------- */

std::chrono::milliseconds mi_randsample_delay{0};
ErrorCode random_sample_result;
KeyVector random_sample_keys;

ErrorCode
MetaIndexer_RandomSample_stub(void *obj, RequestContext *rc, const std::size_t c, KeyVector &out_keys) noexcept {
    if (random_sample_result == ErrorCode::EC_OK) {
        if (c == random_sample_keys.size()) {
            out_keys = random_sample_keys;
        } else if (c == 11) {
            // special case
            out_keys = random_sample_keys;
        } else {
            out_keys = KeyVector(c);
        }
    }
    std::this_thread::sleep_for(mi_randsample_delay);
    return random_sample_result;
}

/* ---------------- MetaIndexer_SampleReclaimKeys_stub ---------------- */

std::chrono::milliseconds mi_sample_reclaim_delay{0};
ErrorCode sample_reclaim_result;
KeyVector sample_reclaim_keys;
int sample_reclaim_call_counter;

ErrorCode
MetaIndexer_SampleReclaimKeys_stub(void *obj, RequestContext *rc, const std::int64_t c, KeyVector &out_keys) noexcept {
    ++sample_reclaim_call_counter;
    if (sample_reclaim_result == ErrorCode::EC_OK) {
        if (c == static_cast<std::int64_t>(sample_reclaim_keys.size())) {
            out_keys = sample_reclaim_keys;
        } else if (c == 11) {
            // special case
            out_keys = sample_reclaim_keys;
        } else {
            out_keys = KeyVector(c);
        }
    }
    std::this_thread::sleep_for(mi_sample_reclaim_delay);
    return sample_reclaim_result;
}

/* ---------------- MetaIndexer KeyCount stubs ---------------- */

std::size_t key_count;
std::size_t max_key_count;

size_t MetaIndexer_GetKeyCount_stub(void *obj) noexcept { return key_count; }

size_t MetaIndexer_GetMaxKeyCount_stub(void *obj) noexcept { return max_key_count; }

void MetaIndexer_PersistMetaData_stub(void *obj) noexcept {}

/* ---------------- MetaSearcherManager_GetMetaSearcher_stub ---------------- */

// the dummy meta_searcher that the meta_searcher_manager would return
std::shared_ptr<MetaSearcher> dummy_meta_searcher;

MetaSearcher *MetaSearcherManager_GetMetaSearcher_stub(void *obj, const std::string &i) {
    return dummy_meta_searcher.get();
}

/* ---------------- MetaSearcher_BatchGetLocation_stub ---------------- */

std::chrono::milliseconds ms_batchgetloc_delay{0};
ErrorCode batch_get_loc_result;
std::vector<CacheLocationMap> batch_get_loc_out_maps;
int batch_get_loc_call_counter;

ErrorCode MetaSearcher_BatchGetLocation_stub(void *obj,
                                             RequestContext *rc,
                                             const std::vector<std::int64_t> &kv,
                                             const BlockMask &bm,
                                             std::vector<CacheLocationMap> &out_loc_maps) {
    ++batch_get_loc_call_counter;
    if (batch_get_loc_result == ErrorCode::EC_OK) {
        out_loc_maps = batch_get_loc_out_maps;
    }
    std::this_thread::sleep_for(ms_batchgetloc_delay);
    return batch_get_loc_result;
}

std::vector<MigrationManager::MigrationRequest> captured_copy_reqs;
std::vector<std::vector<MigrationManager::MigrationRequest>> captured_copy_req_batches;
std::vector<MigrationManager::CopyConcurrencyLimit> captured_copy_limits;
std::string captured_copy_trace;
std::vector<std::int64_t> captured_mark_keys;
std::string captured_mark_target;

class CacheReclaimerTest : public TESTBASE {
public:
    void SetUp() override {
        // set up stubs
        stub_.set(ADDR(RegistryManager, ListInstanceGroup), RegistryManager_ListInstanceGroup_stub);
        stub_.set(ADDR(RegistryManager, ListInstanceInfo), RegistryManager_ListInstanceInfo_stub);
        stub_.set(static_cast<spe_submit_async_loc>(ADDR(SchedulePlanExecutor, SubmitAsync)),
                  SchedulePlanExecutor_SubmitAsync_stub);
        stub_.set(ADDR(MetaIndexerManager, GetMetaIndexer), MetaIndexerManager_GetMetaIndexer_stub);
        stub_.set(ADDR(MetaIndexer, GetProperties), MetaIndexer_GetProperties_stub);
        stub_.set(ADDR(MetaIndexer, RandomSample), MetaIndexer_RandomSample_stub);
        stub_.set(ADDR(MetaIndexer, SampleReclaimKeys), MetaIndexer_SampleReclaimKeys_stub);
        stub_.set(ADDR(MetaIndexer, GetKeyCount), MetaIndexer_GetKeyCount_stub);
        stub_.set(ADDR(MetaIndexer, GetMaxKeyCount), MetaIndexer_GetMaxKeyCount_stub);
        stub_.set(ADDR(MetaIndexer, PersistMetaData), MetaIndexer_PersistMetaData_stub);
        stub_.set(ADDR(MetaSearcherManager, GetMetaSearcher), MetaSearcherManager_GetMetaSearcher_stub);
        stub_.set(ADDR(MetaSearcher, BatchGetLocation), MetaSearcher_BatchGetLocation_stub);

        // set up the global testing facilities
        list_ins_group_result = ErrorCode::EC_OK;
        list_ins_group_call_counter = 0;
        instance_groups.emplace_back(InstanceGroupFactory());

        list_ins_info_result = ErrorCode::EC_OK;
        instance_infos.emplace_back(InstanceInfoFactory());

        dummy_meta_indexer = std::make_shared<MetaIndexer>();
        dummy_meta_searcher = std::make_shared<MetaSearcher>(nullptr);

        del_result = {ErrorCode::EC_OK, ""};
        spe_submit_accepted = true;
        spe_submit_auto_complete = true;
        spe_submit_invalid_future = false;
        captured_copy_reqs.clear();
        captured_copy_req_batches.clear();
        captured_copy_limits.clear();
        captured_copy_trace.clear();
        captured_mark_keys.clear();
        captured_mark_target.clear();
        get_result = ErrorCode::EC_OK;
        random_sample_result = ErrorCode::EC_OK;
        sample_reclaim_result = ErrorCode::EC_OK;
        batch_get_loc_result = ErrorCode::EC_OK;
        sample_reclaim_call_counter = 0;
        batch_get_loc_call_counter = 0;

        key_count = 1;
        max_key_count = 16;

        spe_submit_delay = std::chrono::milliseconds{0};
        mi_getprop_delay = std::chrono::milliseconds{0};
        mi_randsample_delay = std::chrono::milliseconds{0};
        mi_sample_reclaim_delay = std::chrono::milliseconds{0};
        ms_batchgetloc_delay = std::chrono::milliseconds{0};

        request_context_ = std::make_shared<RequestContext>("cache_reclaimer_test_trace");

        // set up our target being tested
        mr_ = std::make_shared<MetricsRegistry>();
        em_ = std::make_shared<EventManager>();
        rm_ = std::make_shared<RegistryManager>("", mr_);
        mim_ = std::make_shared<MetaIndexerManager>();
        msm_ = std::make_shared<MetaSearcherManager>(rm_, mim_);
        dsm_ = std::make_shared<DataStorageManager>(mr_);
        spe_ = std::make_shared<SchedulePlanExecutor>(0, mim_, dsm_, mr_);
        mm_ = std::make_shared<MigrationManager>(spe_, mim_, dsm_);

        cache_reclaimer_ = std::make_unique<CacheReclaimer>(
            10, 100, 1, 10, 16, rm_, mim_, msm_, spe_, mr_, em_, nullptr, CacheReclaimerAsyncDeleteConfig{}, mm_);

        // avoid nullptr issue when testing methods that involve metrics
        // counter but no need to start the working thread
        cache_reclaimer_->METRICS_(cache_reclaimer, reclaim_cron_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, reclaim_cron_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, reclaim_job_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, reclaim_job_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, block_submit_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, block_submit_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, location_submit_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, location_submit_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, block_del_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, block_del_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, location_del_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, location_del_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, credit_timeout_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, credit_timeout_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, pending_limit_reject_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, pending_limit_reject_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, duplicate_pending_location_filtered_count) = mr_->GetCounter(
            SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, duplicate_pending_location_filtered_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, reclaim_no_progress_backoff_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, reclaim_no_progress_backoff_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, delete_submit_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, delete_submit_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, delete_complete_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, delete_complete_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, delete_fail_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, delete_fail_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, migration_copy_submitted_total) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, migration_copy_submitted_total));
        cache_reclaimer_->METRICS_(cache_reclaimer, migration_mark_submitted_total) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, migration_mark_submitted_total));

        cache_reclaimer_->METRICS_(cache_reclaimer, reclaim_cron_duration_us) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, reclaim_cron_duration_us));
        cache_reclaimer_->METRICS_(cache_reclaimer, reclaim_job_duration_us) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, reclaim_job_duration_us));
        cache_reclaimer_->METRICS_(cache_reclaimer, reclaim_lru_sample_duration_us) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, reclaim_lru_sample_duration_us));
        cache_reclaimer_->METRICS_(cache_reclaimer, reclaim_lru_batch_duration_us) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, reclaim_lru_batch_duration_us));
        cache_reclaimer_->METRICS_(cache_reclaimer, reclaim_lru_filter_duration_us) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, reclaim_lru_filter_duration_us));
        cache_reclaimer_->METRICS_(cache_reclaimer, reclaim_lru_submit_duration_us) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, reclaim_lru_submit_duration_us));
        cache_reclaimer_->METRICS_(cache_reclaimer, pending_delete_handler_count) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, pending_delete_handler_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, pending_location_count) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, pending_location_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, pending_delete_bytes) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, pending_delete_bytes));
        cache_reclaimer_->METRICS_(cache_reclaimer, credited_delete_bytes) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, credited_delete_bytes));
        cache_reclaimer_->METRICS_(cache_reclaimer, predicted_deleted_key_count) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, predicted_deleted_key_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, oldest_pending_request_age_ms) =
            mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, oldest_pending_request_age_ms));
    }

    void TearDown() override {
        cache_reclaimer_->Stop();

        instance_groups.clear();
        instance_infos.clear();

        dummy_meta_indexer.reset();
        dummy_meta_searcher.reset();

        {
            std::lock_guard<std::mutex> lock(submitted_del_requests_mutex);
            submitted_del_requests.clear();
            submitted_del_promises.clear();
        }
        captured_copy_reqs.clear();
        captured_copy_req_batches.clear();
        captured_copy_limits.clear();
        captured_copy_trace.clear();
        captured_mark_keys.clear();
        captured_mark_target.clear();

        get_out_properties.clear();
        random_sample_keys.clear();
        sample_reclaim_keys.clear();
        batch_get_loc_out_maps.clear();
        sample_reclaim_call_counter = 0;
        batch_get_loc_call_counter = 0;

        stub_.reset(ADDR(RegistryManager, ListInstanceGroup));
        stub_.reset(ADDR(RegistryManager, ListInstanceInfo));
        stub_.reset(static_cast<spe_submit_async_loc>(ADDR(SchedulePlanExecutor, SubmitAsync)));
        stub_.reset(ADDR(MetaIndexerManager, GetMetaIndexer));
        stub_.reset(ADDR(MetaIndexer, GetProperties));
        stub_.reset(ADDR(MetaIndexer, RandomSample));
        stub_.reset(ADDR(MetaIndexer, SampleReclaimKeys));
        stub_.reset(ADDR(MetaIndexer, GetKeyCount));
        stub_.reset(ADDR(MetaIndexer, GetMaxKeyCount));
        stub_.reset(ADDR(MetaIndexer, PersistMetaData));
        stub_.reset(ADDR(MetaSearcherManager, GetMetaSearcher));
        stub_.reset(ADDR(MetaSearcher, BatchGetLocation));
    }

    template <typename Predicate>
    bool WaitUntil(Predicate predicate, std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return predicate();
    }

    std::vector<CacheLocationDelRequest> SubmittedDelRequestsSnapshot() {
        std::lock_guard<std::mutex> lock(submitted_del_requests_mutex);
        return submitted_del_requests;
    }

    void ClearSubmittedDelRequests() {
        std::lock_guard<std::mutex> lock(submitted_del_requests_mutex);
        submitted_del_requests.clear();
    }

    bool HasSubmittedDelRequests() {
        std::lock_guard<std::mutex> lock(submitted_del_requests_mutex);
        return !submitted_del_requests.empty();
    }

    bool HasNoSubmittedDelRequests() {
        std::lock_guard<std::mutex> lock(submitted_del_requests_mutex);
        return submitted_del_requests.empty();
    }

    std::size_t SubmittedDelRequestCount() {
        std::lock_guard<std::mutex> lock(submitted_del_requests_mutex);
        return submitted_del_requests.size();
    }

    bool WaitUntilSubmittedDelRequests(std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        return WaitUntil([this] { return HasSubmittedDelRequests(); }, timeout);
    }

    void CompleteSubmittedDelete(const std::size_t index, const PlanExecuteResult &result) {
        std::shared_ptr<std::promise<PlanExecuteResult>> promise;
        {
            std::lock_guard<std::mutex> lock(submitted_del_requests_mutex);
            ASSERT_LT(index, submitted_del_promises.size());
            promise = submitted_del_promises[index];
        }
        promise->set_value(result);
    }

    void ReplaceReclaimer(const CacheReclaimerAsyncDeleteConfig &config) {
        cache_reclaimer_->Stop();
        cache_reclaimer_ =
            std::make_unique<CacheReclaimer>(10, 100, 10, 10, 16, rm_, mim_, msm_, spe_, mr_, em_, nullptr, config);
    }

    int ListInstanceGroupCallCount() {
        std::lock_guard<std::mutex> lock(list_ins_group_mut);
        return list_ins_group_call_counter;
    }

    Stub stub_;
    std::unique_ptr<CacheReclaimer> cache_reclaimer_;
    std::shared_ptr<RegistryManager> rm_;
    std::shared_ptr<MetaIndexerManager> mim_;
    std::shared_ptr<MetaSearcherManager> msm_;
    std::shared_ptr<DataStorageManager> dsm_;
    std::shared_ptr<SchedulePlanExecutor> spe_;
    std::shared_ptr<MetricsRegistry> mr_;
    std::shared_ptr<EventManager> em_;
    std::shared_ptr<MigrationManager> mm_;
    std::shared_ptr<RequestContext> request_context_;
};

TEST_F(CacheReclaimerTest, TestStartStop) {
    stub_.reset(ADDR(RegistryManager, ListInstanceGroup));
    stub_.reset(ADDR(RegistryManager, ListInstanceInfo));
    stub_.reset(static_cast<spe_submit_async_loc>(ADDR(SchedulePlanExecutor, SubmitAsync)));
    stub_.reset(ADDR(MetaIndexerManager, GetMetaIndexer));
    stub_.reset(ADDR(MetaIndexer, GetProperties));
    stub_.reset(ADDR(MetaIndexer, RandomSample));
    stub_.reset(ADDR(MetaIndexer, SampleReclaimKeys));
    stub_.reset(ADDR(MetaIndexer, GetKeyCount));
    stub_.reset(ADDR(MetaIndexer, GetMaxKeyCount));
    stub_.reset(ADDR(MetaIndexer, PersistMetaData));
    stub_.reset(ADDR(MetaSearcherManager, GetMetaSearcher));
    stub_.reset(ADDR(MetaSearcher, BatchGetLocation));

    {
        // test the normal start and stop case
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->IsPaused());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());

        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
        ASSERT_TRUE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->IsPaused());
        ASSERT_TRUE(cache_reclaimer_->reclaimer_.joinable());

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        cache_reclaimer_->Stop();
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->IsPaused());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    }

    {
        // test multiple (sequential) calls on start and stop
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());

        // round 1
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
        ASSERT_TRUE(cache_reclaimer_->IsRunning());
        ASSERT_TRUE(cache_reclaimer_->reclaimer_.joinable());

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        cache_reclaimer_->Stop();
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());

        // round 2
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
        ASSERT_TRUE(cache_reclaimer_->IsRunning());
        ASSERT_TRUE(cache_reclaimer_->reclaimer_.joinable());

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        cache_reclaimer_->Stop();
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    }

    {
        // test the case that all the dependencies are given as nullptr
        auto cache_reclaimer = std::make_unique<CacheReclaimer>(
            1000, 100, 100, 100, 16, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer->Start());
        ASSERT_FALSE(cache_reclaimer->IsRunning());
        ASSERT_FALSE(cache_reclaimer->reclaimer_.joinable());
    }

    {
        // test the case that RegisterManager is nullptr
        auto cache_reclaimer =
            std::make_unique<CacheReclaimer>(1000, 100, 100, 100, 16, nullptr, mim_, msm_, spe_, mr_, em_, nullptr);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer->Start());
        ASSERT_FALSE(cache_reclaimer->IsRunning());
        ASSERT_FALSE(cache_reclaimer->reclaimer_.joinable());
    }

    {
        // test the case that MetaIndexerManager is nullptr
        auto cache_reclaimer =
            std::make_unique<CacheReclaimer>(1000, 100, 100, 100, 16, rm_, nullptr, msm_, spe_, mr_, em_, nullptr);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer->Start());
        ASSERT_FALSE(cache_reclaimer->IsRunning());
        ASSERT_FALSE(cache_reclaimer->reclaimer_.joinable());
    }

    {
        // test the case that MetaSearcherManager is nullptr
        auto cache_reclaimer =
            std::make_unique<CacheReclaimer>(1000, 100, 100, 100, 16, rm_, mim_, nullptr, spe_, mr_, em_, nullptr);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer->Start());
        ASSERT_FALSE(cache_reclaimer->IsRunning());
        ASSERT_FALSE(cache_reclaimer->reclaimer_.joinable());
    }

    {
        // test the case that SchedulePlanExecutor is nullptr
        auto cache_reclaimer =
            std::make_unique<CacheReclaimer>(1000, 100, 100, 100, 16, rm_, mim_, msm_, nullptr, mr_, em_, nullptr);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer->Start());
        ASSERT_FALSE(cache_reclaimer->IsRunning());
        ASSERT_FALSE(cache_reclaimer->reclaimer_.joinable());
    }

    {
        // test the case that MetricsRegistry is nullptr
        auto cache_reclaimer =
            std::make_unique<CacheReclaimer>(1000, 100, 100, 100, 16, rm_, mim_, msm_, spe_, nullptr, em_, nullptr);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer->Start());
        ASSERT_FALSE(cache_reclaimer->IsRunning());
        ASSERT_FALSE(cache_reclaimer->reclaimer_.joinable());
    }

    {
        // test the case that MetricsRegistry is nullptr
        auto cache_reclaimer =
            std::make_unique<CacheReclaimer>(1000, 100, 100, 100, 16, rm_, mim_, msm_, spe_, mr_, nullptr, nullptr);
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer->Start());
        ASSERT_TRUE(cache_reclaimer->IsRunning());
        ASSERT_TRUE(cache_reclaimer->reclaimer_.joinable());
    }
}

TEST_F(CacheReclaimerTest, TestFastExiting) {
    cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), 1000);

    const auto start_tp = std::chrono::steady_clock::now();

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
    ASSERT_TRUE(cache_reclaimer_->IsRunning());
    ASSERT_TRUE(cache_reclaimer_->reclaimer_.joinable());

    std::this_thread::sleep_for(std::chrono::milliseconds(8));

    cache_reclaimer_->Stop();
    const auto stop_tp = std::chrono::steady_clock::now();

    ASSERT_FALSE(cache_reclaimer_->IsRunning());
    ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    ASSERT_GT(std::chrono::milliseconds(100), stop_tp - start_tp);
}

TEST_F(CacheReclaimerTest, TestPauseResume) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    // update the trigger strategy to trigger the reclaiming
    // so that the reclaiming method shall be entered
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 2));
    batch_get_loc_out_maps = MakeServingLocationMaps(cache_reclaimer_->GetBatchingSize(request_context_.get()));

    {
        cache_reclaimer_->Pause();
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
        ASSERT_TRUE(cache_reclaimer_->IsPaused());  // the worker thread is in paused state
        ASSERT_TRUE(HasNoSubmittedDelRequests());
    }

    {
        cache_reclaimer_->Resume();
        ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
        ASSERT_FALSE(cache_reclaimer_->IsPaused()); // the worker thread is not in paused state

        ASSERT_TRUE(WaitUntilSubmittedDelRequests());
        const auto requests = SubmittedDelRequestsSnapshot();
        ASSERT_FALSE(requests.empty());
        const auto &req = requests.back();
        ASSERT_EQ(2, req.block_keys.size());
        ASSERT_TRUE(VecContains(req.block_keys, 0));
        ASSERT_TRUE(VecContains(req.block_keys, 1));
    }
}

TEST_F(CacheReclaimerTest, TestDoubleStarts) {
    // calling Start() while reclaimer job is running should be prohibited
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
    ASSERT_TRUE(cache_reclaimer_->IsRunning());
    ASSERT_TRUE(cache_reclaimer_->reclaimer_.joinable());
    const auto tid0 = cache_reclaimer_->reclaimer_.get_id();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_EQ(ErrorCode::EC_EXIST, cache_reclaimer_->Start());
    ASSERT_TRUE(cache_reclaimer_->IsRunning());
    ASSERT_TRUE(cache_reclaimer_->reclaimer_.joinable());
    const auto tid1 = cache_reclaimer_->reclaimer_.get_id();

    // thread id should not change
    ASSERT_EQ(tid0, tid1);
}

TEST_F(CacheReclaimerTest, TestDoubleStops) {
    {
        // test stop while not started
        cache_reclaimer_->Stop();
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    }

    {
        // test double stops, which is allowed and should work fine
        // the 2nd call (and the after, if any) should have no effect
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
        ASSERT_TRUE(cache_reclaimer_->IsRunning());

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 1st stop
        cache_reclaimer_->Stop();
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());

        // 2nd stop
        cache_reclaimer_->Stop();
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    }
}

TEST_F(CacheReclaimerTest, TestWorkerConfigValues) {
    cache_reclaimer_ = std::make_unique<CacheReclaimer>(
        1000, 100, 100, 100, 16, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

    {
        // default values
        ASSERT_EQ(1000, cache_reclaimer_->GetSamplingSize(request_context_.get()));
        ASSERT_EQ(100, cache_reclaimer_->GetBatchingSize(request_context_.get()));
        ASSERT_EQ(100, cache_reclaimer_->GetSleepIntervalMs(request_context_.get()));
    }

    {
        // sampling size boundary
        constexpr std::size_t limit = 1 << 16;

        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), 0));
        ASSERT_EQ(0, cache_reclaimer_->GetSamplingSize(request_context_.get()));

        ASSERT_EQ(ErrorCode::EC_OK,
                  cache_reclaimer_->SetSamplingSize(request_context_.get(), std::numeric_limits<std::size_t>::min()));
        ASSERT_EQ(std::numeric_limits<std::size_t>::min(), cache_reclaimer_->GetSamplingSize(request_context_.get()));

        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), limit - 1));
        ASSERT_EQ(limit - 1, cache_reclaimer_->GetSamplingSize(request_context_.get()));

        ASSERT_EQ(ErrorCode::EC_OUT_OF_RANGE, cache_reclaimer_->SetSamplingSize(request_context_.get(), limit));
        ASSERT_EQ(limit - 1, cache_reclaimer_->GetSamplingSize(request_context_.get()));

        ASSERT_EQ(ErrorCode::EC_OUT_OF_RANGE, cache_reclaimer_->SetSamplingSize(request_context_.get(), -1));
        ASSERT_EQ(limit - 1, cache_reclaimer_->GetSamplingSize(request_context_.get()));

        ASSERT_EQ(ErrorCode::EC_OUT_OF_RANGE,
                  cache_reclaimer_->SetSamplingSize(request_context_.get(), std::numeric_limits<std::size_t>::max()));
        ASSERT_EQ(limit - 1, cache_reclaimer_->GetSamplingSize(request_context_.get()));
    }

    {
        // batching size boundary
        constexpr std::size_t limit = 1 << 16;

        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 0));
        ASSERT_EQ(0, cache_reclaimer_->GetBatchingSize(request_context_.get()));

        ASSERT_EQ(ErrorCode::EC_OK,
                  cache_reclaimer_->SetBatchingSize(request_context_.get(), std::numeric_limits<std::size_t>::min()));
        ASSERT_EQ(std::numeric_limits<std::size_t>::min(), cache_reclaimer_->GetBatchingSize(request_context_.get()));

        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), limit - 1));
        ASSERT_EQ(limit - 1, cache_reclaimer_->GetBatchingSize(request_context_.get()));

        ASSERT_EQ(ErrorCode::EC_OUT_OF_RANGE, cache_reclaimer_->SetBatchingSize(request_context_.get(), limit));
        ASSERT_EQ(limit - 1, cache_reclaimer_->GetBatchingSize(request_context_.get()));

        ASSERT_EQ(ErrorCode::EC_OUT_OF_RANGE, cache_reclaimer_->SetBatchingSize(request_context_.get(), -1));
        ASSERT_EQ(limit - 1, cache_reclaimer_->GetBatchingSize(request_context_.get()));

        ASSERT_EQ(ErrorCode::EC_OUT_OF_RANGE,
                  cache_reclaimer_->SetBatchingSize(request_context_.get(), std::numeric_limits<std::size_t>::max()));
        ASSERT_EQ(limit - 1, cache_reclaimer_->GetBatchingSize(request_context_.get()));
    }

    {
        // sleep time boundary
        cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), 0);
        ASSERT_EQ(0, cache_reclaimer_->GetSleepIntervalMs(request_context_.get()));

        cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), std::numeric_limits<std::uint32_t>::max());
        ASSERT_EQ(std::numeric_limits<std::uint32_t>::max(),
                  cache_reclaimer_->GetSleepIntervalMs(request_context_.get()));

        cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), -1);
        ASSERT_EQ(static_cast<uint32_t>(-1), cache_reclaimer_->GetSleepIntervalMs(request_context_.get()));
    }
}

TEST_F(CacheReclaimerTest, TestWorkerConfigWhenRunning) {
    cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), 0); // no sleeping
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
    for (int i = 0; i != 32738; ++i) {
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), i));
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), i));
        cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), 0);

        ASSERT_EQ(i, cache_reclaimer_->GetSamplingSize(request_context_.get()));
        ASSERT_EQ(i, cache_reclaimer_->GetBatchingSize(request_context_.get()));
        ASSERT_EQ(0, cache_reclaimer_->GetSleepIntervalMs(request_context_.get()));
    }
}

TEST_F(CacheReclaimerTest, TestCopyControl) {
    ASSERT_FALSE(std::is_default_constructible<CacheReclaimer>::value);
    ASSERT_FALSE(std::is_copy_constructible<CacheReclaimer>::value);
    ASSERT_FALSE(std::is_copy_assignable<CacheReclaimer>::value);
    ASSERT_FALSE(std::is_move_constructible<CacheReclaimer>::value);
    ASSERT_FALSE(std::is_move_assignable<CacheReclaimer>::value);
    ASSERT_FALSE(std::is_swappable<CacheReclaimer>::value);
}

TEST_F(CacheReclaimerTest, TestRegistryManagerListInstanceGroupUnexpectedReturn) {
    list_ins_group_result = ErrorCode::EC_ERROR;

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
}

TEST_F(CacheReclaimerTest, TestNullInstanceGroup) {
    instance_groups.emplace_back(nullptr);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
}

TEST_F(CacheReclaimerTest, TestRegistryManagerListInstanceInfoUnexpectedReturn) {
    list_ins_info_result = ErrorCode::EC_ERROR;

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
}

TEST_F(CacheReclaimerTest, TestNullInstanceInfo) {
    // craft a case that can trigger the actual reclaiming
    // so that the reclaiming method shall be entered
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);
    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    instance_infos.emplace_back(nullptr);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
}

TEST_F(CacheReclaimerTest, TestNullCacheConfig) {
    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->set_cache_config(nullptr);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
}

TEST_F(CacheReclaimerTest, TestNullReclaimStrategy) {
    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->cache_config_->set_reclaim_strategy(nullptr);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
}

TEST_F(CacheReclaimerTest, TestNullMetaIndexer) {
    dummy_meta_indexer = nullptr;

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
}

TEST_F(CacheReclaimerTest, TestNullMetaSearcher) {
    dummy_meta_searcher = nullptr;

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming04) {
    // instance 0 block byte size = 1024, key count = 1
    // instance 1 block byte size = 1024, key count = 1
    // (1024 * 1 + 1024 * 1) / 2048 > 0.8
    // should trigger reclaiming by the used_percentage strategy
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024); // x2 instances

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    cache_reclaimer_->job_state_flag_ = true;
    auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                     ins_group->name(),
                                                     ins_group->quota(),
                                                     ins_group->cache_config()->reclaim_strategy(),
                                                     instance_infos);
    ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
    ASSERT_TRUE(wle->CheckGroupWaterLevelExceed());
    ASSERT_FALSE(wle->CheckStorageTypeWaterLevelExceed());
    ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming05) {
    // instance 0 block byte size = 1024, key count = 1
    // (1024 * 1) / 2048 < 0.8
    // should *not* trigger reclaiming by the used_percentage strategy
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024);

    // use instance 0 from setup()

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    cache_reclaimer_->job_state_flag_ = true;
    auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                     ins_group->name(),
                                                     ins_group->quota(),
                                                     ins_group->cache_config()->reclaim_strategy(),
                                                     instance_infos);
    ASSERT_FALSE(CacheReclaimer::IsTriggerReclaiming(wle));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming06) {
    // instance 0 block byte size = 1024, key count = 1
    // (double)(1024 * 1) / 2048.0 is very close to 0.5
    // should trigger reclaiming by the used_percentage strategy
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024);

    // use instance 0 from setup()

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.5);
    cache_reclaimer_->job_state_flag_ = true;
    auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                     ins_group->name(),
                                                     ins_group->quota(),
                                                     ins_group->cache_config()->reclaim_strategy(),
                                                     instance_infos);
    ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
    ASSERT_TRUE(wle->CheckGroupWaterLevelExceed());
    ASSERT_FALSE(wle->CheckStorageTypeWaterLevelExceed());
    ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming07) {
    // instance 0 block byte size = 1024, key count = 1
    // instance 1 block byte size = 1024, key count = 1
    // (1024 * 1 + 1024 * 1) / 2048 < 1.2
    // should *not* trigger reclaiming by the used_percentage strategy
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024); // x2 instances

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(1.2);
    cache_reclaimer_->job_state_flag_ = true;
    auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                     ins_group->name(),
                                                     ins_group->quota(),
                                                     ins_group->cache_config()->reclaim_strategy(),
                                                     instance_infos);
    ASSERT_FALSE(CacheReclaimer::IsTriggerReclaiming(wle));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming08) {
    // instance 0 block byte size = 1024, key count = 1
    // instance 1 block byte size = 1024, key count = 1
    // instance 2 block byte size = 1024, key count = 1
    // (1024 * 1 + 1024 * 1 + 1024 * 1) / 2048 > 1.2
    // should trigger reclaiming by the used_percentage strategy
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024); // x3 instances

    // use instance 0 from setup()

    // construct instance 1 and 2
    {
        const auto ins_info = InstanceInfoFactory();
        ins_info->set_instance_id("test_instance_id_2");
        instance_infos.emplace_back(ins_info);
    }

    {
        const auto ins_info = InstanceInfoFactory();
        ins_info->set_instance_id("test_instance_id3");
        instance_infos.emplace_back(ins_info);
    }

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(1.2);
    cache_reclaimer_->job_state_flag_ = true;
    auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                     ins_group->name(),
                                                     ins_group->quota(),
                                                     ins_group->cache_config()->reclaim_strategy(),
                                                     instance_infos);
    ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
    ASSERT_TRUE(wle->CheckGroupWaterLevelExceed());
    ASSERT_FALSE(wle->CheckStorageTypeWaterLevelExceed());
    ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming09) {
    // instance 0 block byte size = 1024, key count = 16, max key count = 32
    // instance 1 block byte size = 1024, key count = 16, max key count = 32
    // (16 + 16) / (32 + 32) < 0.8
    // should not trigger reclaiming by the used_percentage strategy
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024); // x2 instances
    key_count = 16;
    max_key_count = 32;

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto &ins_group = instance_groups.at(0);
    cache_reclaimer_->job_state_flag_ = true;
    auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                     ins_group->name(),
                                                     ins_group->quota(),
                                                     ins_group->cache_config()->reclaim_strategy(),
                                                     instance_infos);
    ASSERT_FALSE(CacheReclaimer::IsTriggerReclaiming(wle));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming10) {
    // instance 0 block byte size = 1024, key count = 32, max key count = 32
    // instance 1 block byte size = 1024, key count = 32, max key count = 32
    // (double)((32 + 32) / (32 + 32)) is very close to 1.0
    // should trigger reclaiming by the used_percentage strategy
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024); // x2 isntances
    key_count = 32;
    max_key_count = 32;

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto ins_group = InstanceGroupFactory();
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(1.0);
    cache_reclaimer_->job_state_flag_ = true;
    auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                     ins_group->name(),
                                                     ins_group->quota(),
                                                     ins_group->cache_config()->reclaim_strategy(),
                                                     instance_infos);
    ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
    ASSERT_TRUE(wle->CheckGroupWaterLevelExceed());
    ASSERT_FALSE(wle->CheckStorageTypeWaterLevelExceed());
    ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming11) {
    // instance 0 block byte size = 1024, key count = 32, max key count = 32
    // instance 1 block byte size = 1024, key count = 32, max key count = 32
    // (double)((32 + 32) / (32 + 32)) > 0.8
    // should trigger reclaiming by the used_percentage strategy
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024); // x2 instances
    key_count = 32;
    max_key_count = 32;

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto &ins_group = instance_groups.at(0);
    cache_reclaimer_->job_state_flag_ = true;
    auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                     ins_group->name(),
                                                     ins_group->quota(),
                                                     ins_group->cache_config()->reclaim_strategy(),
                                                     instance_infos);
    ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
    ASSERT_TRUE(wle->CheckGroupWaterLevelExceed());
    ASSERT_FALSE(wle->CheckStorageTypeWaterLevelExceed());
    ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
    ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming15) {
    // test empty instance info list
    // should *not* trigger reclaiming

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_infos.clear();
    cache_reclaimer_->job_state_flag_ = true;
    auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                     ins_group->name(),
                                                     ins_group->quota(),
                                                     ins_group->cache_config()->reclaim_strategy(),
                                                     instance_infos);
    ASSERT_FALSE(CacheReclaimer::IsTriggerReclaiming(wle));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming16) {
    // test edge cases: divide by zero, negative quota

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024); // x2 isntances

    {
        // instance 0 block byte size = 1024, key count = 32, max key count = 0
        // instance 1 block byte size = 1024, key count = 32, max key count = 0
        // (double)((32 + 32) / (0 + 0)) = inf > 0.8
        // should trigger reclaiming when group_used_key_count > 0

        key_count = 32;
        max_key_count = 0;

        const auto &ins_group = instance_groups.at(0);
        cache_reclaimer_->job_state_flag_ = true;
        auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                         ins_group->name(),
                                                         ins_group->quota(),
                                                         ins_group->cache_config()->reclaim_strategy(),
                                                         instance_infos);
        ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
        ASSERT_TRUE(wle->CheckGroupWaterLevelExceed());
        ASSERT_FALSE(wle->CheckStorageTypeWaterLevelExceed());
        ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
    }

    {
        // instance 0 block byte size = 1024, key count = 0, max key count = 0
        // instance 1 block byte size = 1024, key count = 0, max key count = 0
        // (double)((0 + 0) / (0 + 0))
        // should *not* trigger reclaiming when group_used_key_count = 0

        key_count = 0;
        max_key_count = 0;

        const auto &ins_group = instance_groups.at(0);
        cache_reclaimer_->job_state_flag_ = true;
        auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                         ins_group->name(),
                                                         ins_group->quota(),
                                                         ins_group->cache_config()->reclaim_strategy(),
                                                         instance_infos);
        ASSERT_FALSE(CacheReclaimer::IsTriggerReclaiming(wle));
    }

    {
        // instance 0 block byte size = 1024, key count = 32, max key count = 32
        // instance 1 block byte size = 1024, key count = 32, max key count = 32
        // group quota capacity set to zero
        // should trigger reclaiming when group_used_byte_size > 0

        key_count = 32;
        max_key_count = 32;

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(0);
        cache_reclaimer_->job_state_flag_ = true;
        auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                         ins_group->name(),
                                                         ins_group->quota(),
                                                         ins_group->cache_config()->reclaim_strategy(),
                                                         instance_infos);
        ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
        ASSERT_TRUE(wle->CheckGroupWaterLevelExceed());
        ASSERT_FALSE(wle->CheckStorageTypeWaterLevelExceed());
        ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
    }

    {
        // instance 0 storage usage = 1024, key count = 0, max key count = 32
        // instance 1 storage usage = 1024, key count = 0, max key count = 32
        // group quota capacity set to zero
        // Byte and key-count water levels are independent, so positive official
        // storage usage still triggers reclaiming when the key count is zero.

        key_count = 0;
        max_key_count = 32;

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(0);
        cache_reclaimer_->job_state_flag_ = true;
        auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                         ins_group->name(),
                                                         ins_group->quota(),
                                                         ins_group->cache_config()->reclaim_strategy(),
                                                         instance_infos);
        ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
        ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
    }

    {
        // instance 0 block byte size = 1024, key count = 32, max key count = 32
        // instance 1 block byte size = 1024, key count = 32, max key count = 32
        // group quota capacity set to -1
        // should trigger reclaiming when group_used_byte_size > 0

        key_count = 32;
        max_key_count = 32;

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(-1); // means no capacity, same as 0
        cache_reclaimer_->job_state_flag_ = true;
        auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                         ins_group->name(),
                                                         ins_group->quota(),
                                                         ins_group->cache_config()->reclaim_strategy(),
                                                         instance_infos);
        ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
        ASSERT_TRUE(wle->CheckGroupWaterLevelExceed());
        ASSERT_FALSE(wle->CheckStorageTypeWaterLevelExceed());
        ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
    }

    {
        // instance 0 storage usage = 1024, key count = 0, max key count = 32
        // instance 1 storage usage = 1024, key count = 0, max key count = 32
        // group quota capacity set to -1
        // A negative capacity has the same zero-capacity semantics.

        key_count = 0;
        max_key_count = 32;

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(-1); // means no capacity, same as 0
        cache_reclaimer_->job_state_flag_ = true;
        auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                         ins_group->name(),
                                                         ins_group->quota(),
                                                         ins_group->cache_config()->reclaim_strategy(),
                                                         instance_infos);
        ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
        ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
    }
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming17) {
    key_count = 2;

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 1024);      // x2 instances
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE, 1024); // x2 instances

    {
        // instance 0 block byte size = 1024, key count = 2
        // instance 1 block byte size = 1024, key count = 2
        // (1024 * 2 + 1024 * 2 + 512 * 2) / 10240 < 0.9, total waterlevel not exceed
        // (512 + 512) / 1024 > 0.9, waterlevel exceed

        dummy_meta_indexer->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 0);
        dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 512); // x2 instances

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(10240);
        QuotaConfig qc(1024, DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        ins_group->quota_.set_quota_config({qc});
        ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.9);
        cache_reclaimer_->job_state_flag_ = true;
        auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                         ins_group->name(),
                                                         ins_group->quota(),
                                                         ins_group->cache_config()->reclaim_strategy(),
                                                         instance_infos);
        ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
        ASSERT_TRUE(wle->CheckGroupWaterLevelExceed());
        ASSERT_TRUE(wle->CheckStorageTypeWaterLevelExceed());
        ASSERT_FALSE(wle->GetGeneralWaterLevelExceed());
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN));
        ASSERT_TRUE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
        ASSERT_TRUE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
    }

    {
        // instance 0 block byte size = 1024, key count = 2
        // instance 1 block byte size = 1024, key count = 2
        // (1024 * 2 + 1024 * 2 + 128 * 2) / 10240 < 0.9, total waterlevel not exceed
        // (128 + 128) / 1024 < 0.9, waterlevel not exceed

        dummy_meta_indexer->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 0);
        dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 128); // x2 instances

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(10240);
        QuotaConfig qc(1024, DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        ins_group->quota_.set_quota_config({qc});
        ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.9);
        cache_reclaimer_->job_state_flag_ = true;
        auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                         ins_group->name(),
                                                         ins_group->quota(),
                                                         ins_group->cache_config()->reclaim_strategy(),
                                                         instance_infos);
        ASSERT_FALSE(CacheReclaimer::IsTriggerReclaiming(wle));
    }

    {
        // instance 0 block byte size = 1024, key count = 2
        // instance 1 block byte size = 1024, key count = 2
        // instance 2 block byte size = 1024, key count = 2
        // (1024 * 2 + 1024 * 2 + 512 * 2) / 5120 > 0.9, total waterlevel exceed
        // (512 + 512) / 1024 > 0.9, waterlevel exceed

        // construct another instance
        const auto ins_info3 = InstanceInfoFactory();
        ins_info3->set_instance_id("test_instance_id_3");
        instance_infos.emplace_back(ins_info3);

        dummy_meta_indexer->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 0);
        dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS, 512); // x2 instances

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(5120);
        QuotaConfig qc(1024, DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        ins_group->quota_.set_quota_config({qc});
        ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.9);
        cache_reclaimer_->job_state_flag_ = true;
        auto wle = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                         ins_group->name(),
                                                         ins_group->quota(),
                                                         ins_group->cache_config()->reclaim_strategy(),
                                                         instance_infos);
        ASSERT_TRUE(CacheReclaimer::IsTriggerReclaiming(wle));
        ASSERT_TRUE(wle->CheckGroupWaterLevelExceed());
        ASSERT_TRUE(wle->CheckStorageTypeWaterLevelExceed());
        ASSERT_TRUE(wle->GetGeneralWaterLevelExceed());
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN));
        ASSERT_TRUE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_HF3FS));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_MOONCAKE));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL));
        ASSERT_FALSE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_NFS));
        ASSERT_TRUE(wle->GetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS));
    }
}

TEST_F(CacheReclaimerTest, TestInsufficientSampledKeys) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    // update the trigger strategy to trigger the reclaiming
    // so that the reclaiming method shall be entered
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    // batching_size default to 100 which is larger than the size of sampled keys (10)
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 100));
    batch_get_loc_out_maps = MakeServingLocationMaps(sample_reclaim_keys.size());
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    // main thread sleeps for 10ms to ensure the worker thread do
    // reclaiming at least once (not 100% but should have reasonable
    // high probability)
    ASSERT_TRUE(WaitUntilSubmittedDelRequests());
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();
    // all blocks should be submitted when sampled keys are insufficient
    const auto requests = SubmittedDelRequestsSnapshot();
    ASSERT_FALSE(requests.empty());
    const auto &req = requests.back();
    ASSERT_EQ(10, req.block_keys.size());
    for (std::int64_t i = 0; i != 10; ++i) {
        ASSERT_TRUE(VecContains(req.block_keys, i));
    }
    ASSERT_EQ(req.block_keys.size(), req.location_ids.size());
}

TEST_F(CacheReclaimerTest, TestReclaimByLRU00) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    // update the trigger strategy to trigger the reclaiming
    // so that the reclaiming method shall be entered
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 2));
    batch_get_loc_out_maps = MakeServingLocationMaps(cache_reclaimer_->GetBatchingSize(request_context_.get()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    ASSERT_TRUE(WaitUntilSubmittedDelRequests());
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    const auto requests = SubmittedDelRequestsSnapshot();
    ASSERT_FALSE(requests.empty());
    const auto &req = requests.back();
    ASSERT_EQ(2, req.block_keys.size());
    ASSERT_TRUE(VecContains(req.block_keys, 0));
    ASSERT_TRUE(VecContains(req.block_keys, 1));
}

TEST_F(CacheReclaimerTest, TestReclaimByLRU01) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "9"}, // block key id -> 0
        },
        {
            {PROPERTY_LRU_TIME, "2"}, // block key id -> 1
        },
        {
            {PROPERTY_LRU_TIME, "128"}, // block key id -> 2
        },
        {
            {PROPERTY_LRU_TIME, "31"}, // block key id -> 3
        },
        {
            {PROPERTY_LRU_TIME, "6"}, // block key id -> 4
        },
        {
            {PROPERTY_LRU_TIME, "4"}, // block key id -> 5
        },
        {
            {PROPERTY_LRU_TIME, "5"}, // block key id -> 6
        },
        {
            {PROPERTY_LRU_TIME, "8"}, // block key id -> 7
        },
        {
            {PROPERTY_LRU_TIME, "100"}, // block key id -> 8
        },
        {
            {PROPERTY_LRU_TIME, "8"}, // block key id -> 9
        },
    };

    // update the trigger strategy to trigger the reclaiming
    // so that the reclaiming method shall be entered
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 3));
    batch_get_loc_out_maps = MakeServingLocationMaps(cache_reclaimer_->GetBatchingSize(request_context_.get()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    ASSERT_TRUE(WaitUntilSubmittedDelRequests());
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    const auto requests = SubmittedDelRequestsSnapshot();
    ASSERT_FALSE(requests.empty());
    const auto &req = requests.back();
    ASSERT_EQ(3, req.block_keys.size());
    // the 3 keys with minimal time point should be included
    ASSERT_TRUE(VecContains(req.block_keys, 1)); // time point -> 2
    ASSERT_TRUE(VecContains(req.block_keys, 5)); // time point -> 4
    ASSERT_TRUE(VecContains(req.block_keys, 6)); // time point -> 5
}

TEST_F(CacheReclaimerTest, TestReclaimByLRU02) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "9"}, // block key id -> 0
        },
        {
            {PROPERTY_LRU_TIME, "8"}, // block key id -> 1
        },
        {
            {PROPERTY_LRU_TIME, "128"}, // block key id -> 2
        },
        {
            {PROPERTY_LRU_TIME, "31"}, // block key id -> 3
        },
        {
            {PROPERTY_LRU_TIME, "6"}, // block key id -> 4
        },
        {
            {PROPERTY_LRU_TIME, "4"}, // block key id -> 5
        },
        {
            {PROPERTY_LRU_TIME, "5"}, // block key id -> 6
        },
        {
            {PROPERTY_LRU_TIME, "8"}, // block key id -> 7
        },
        {
            {PROPERTY_LRU_TIME, "100"}, // block key id -> 8
        },
        {
            {PROPERTY_LRU_TIME, "2"}, // block key id -> 9
        },
    };

    // update the trigger strategy to trigger the reclaiming
    // so that the reclaiming method shall be entered
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 3));
    batch_get_loc_out_maps = MakeServingLocationMaps(cache_reclaimer_->GetBatchingSize(request_context_.get()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    ASSERT_TRUE(WaitUntilSubmittedDelRequests());
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    const auto requests = SubmittedDelRequestsSnapshot();
    ASSERT_FALSE(requests.empty());
    const auto &req = requests.back();
    ASSERT_EQ(3, req.block_keys.size());
    // the 3 keys with minimal time point should be included
    ASSERT_TRUE(VecContains(req.block_keys, 9)); // time point -> 2
    ASSERT_TRUE(VecContains(req.block_keys, 5)); // time point -> 4
    ASSERT_TRUE(VecContains(req.block_keys, 6)); // time point -> 5
}

TEST_F(CacheReclaimerTest, TestReclaimByLRU03) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "2"}, // block key id -> 0
        },
        {
            {PROPERTY_LRU_TIME, "8"}, // block key id -> 1
        },
        {
            {PROPERTY_LRU_TIME, "128"}, // block key id -> 2
        },
        {
            {PROPERTY_LRU_TIME, "31"}, // block key id -> 3
        },
        {
            {PROPERTY_LRU_TIME, "6"}, // block key id -> 4
        },
        {
            {PROPERTY_LRU_TIME, "4"}, // block key id -> 5
        },
        {
            {PROPERTY_LRU_TIME, "5"}, // block key id -> 6
        },
        {
            {PROPERTY_LRU_TIME, "8"}, // block key id -> 7
        },
        {
            {PROPERTY_LRU_TIME, "100"}, // block key id -> 8
        },
        {
            {PROPERTY_LRU_TIME, "9"}, // block key id -> 9
        },
    };

    // update the trigger strategy to trigger the reclaiming
    // so that the reclaiming method shall be entered
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 0));
    batch_get_loc_out_maps = std::vector<CacheLocationMap>(sample_reclaim_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    ASSERT_TRUE(HasNoSubmittedDelRequests());
}

TEST_F(CacheReclaimerTest, TestReclaimByLRU04) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "9"}, // block key id -> 0
        },
        {
            {PROPERTY_LRU_TIME, "8"}, // block key id -> 1
        },
        {
            {PROPERTY_LRU_TIME, "128"}, // block key id -> 2
        },
        {
            {PROPERTY_LRU_TIME, "31"}, // block key id -> 3
        },
        {
            {PROPERTY_LRU_TIME, "6"}, // block key id -> 4
        },
        {
            {PROPERTY_LRU_TIME, "4"}, // block key id -> 5
        },
        {
            {PROPERTY_LRU_TIME, "5"}, // block key id -> 6
        },
        {
            {PROPERTY_LRU_TIME, "8"}, // block key id -> 7
        },
        {
            {PROPERTY_LRU_TIME, "100"}, // block key id -> 8
        },
        {
            {PROPERTY_LRU_TIME, "2"}, // block key id -> 9
        },
    };

    // update the trigger strategy to trigger the reclaiming
    // so that the reclaiming method shall be entered
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));
    batch_get_loc_out_maps = MakeServingLocationMaps(cache_reclaimer_->GetBatchingSize(request_context_.get()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    ASSERT_TRUE(WaitUntilSubmittedDelRequests());
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    const auto requests = SubmittedDelRequestsSnapshot();
    ASSERT_FALSE(requests.empty());
    const auto &req = requests.back();
    ASSERT_EQ(1, req.block_keys.size());
    // the 1 keys with minimal time point should be included
    ASSERT_TRUE(VecContains(req.block_keys, 9)); // time point -> 2
}

TEST_F(CacheReclaimerTest, TestMetaIndexerGetPropertiesFailure) {
    // set up test data
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

    // configure the GetProperties stub to return an error
    get_result = ErrorCode::EC_ERROR;

    // update the trigger strategy to trigger the reclaiming

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(sample_reclaim_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted when GetProperties fails
    ASSERT_TRUE(HasNoSubmittedDelRequests());
}

TEST_F(CacheReclaimerTest, TestMetaIndexerSampleReclaimFailure) {
    // configure the SampleReclaim stub to return an error
    sample_reclaim_result = ErrorCode::EC_ERROR;

    // update the trigger strategy to trigger the reclaiming

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(sample_reclaim_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted when SampleReclaim fails
    ASSERT_TRUE(HasNoSubmittedDelRequests());
}

TEST_F(CacheReclaimerTest, TestMetaIndexerSampleKeys00) {
    // test case that sampled keys size and properties size not match
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // size is 10
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "9"}, // size is 1
        },
    };

    // update the trigger strategy to trigger the reclaiming

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(sample_reclaim_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted
    ASSERT_TRUE(HasNoSubmittedDelRequests());
}

TEST_F(CacheReclaimerTest, TestMetaIndexerSampleKeys01) {
    // test case that properties size match but has wrong field
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // size is 10
    get_out_properties = {
        {
            {PROPERTY_HIT_COUNT, "0"}, // wrong field
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    // update the trigger strategy to trigger the reclaiming

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(sample_reclaim_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted
    ASSERT_TRUE(HasNoSubmittedDelRequests());
}

TEST_F(CacheReclaimerTest, TestSchedulePlanExecutorDelFailure) {
    // set up test data
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    // configure the SchedulePlanExecutor stub to return an error
    del_result = {ErrorCode::EC_ERROR, "unknown"};

    // update the trigger strategy to trigger the reclaiming
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    batch_get_loc_out_maps = MakeServingLocationMaps(cache_reclaimer_->GetBatchingSize(request_context_.get()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    ASSERT_TRUE(WaitUntilSubmittedDelRequests());
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();
}

TEST_F(CacheReclaimerTest, TestEmptyInstanceGroups) {
    // clear all instance groups
    instance_groups.clear();

    // the mocking sample keys are set but should never be accessed
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(sample_reclaim_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted when there are no instance groups
    ASSERT_TRUE(HasNoSubmittedDelRequests());
}

TEST_F(CacheReclaimerTest, TestEmptyInstanceInfos) {
    // clear all instance infos
    instance_infos.clear();

    // the mocking sample keys are set but should never be accessed
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    // update the trigger strategy to trigger the reclaiming

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(sample_reclaim_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted when there are no instance infos
    ASSERT_TRUE(HasNoSubmittedDelRequests());
}

TEST_F(CacheReclaimerTest, TestMultipleInstanceGroups) {
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    // create multiple instance groups
    instance_groups.clear();

    // first instance group
    const auto ins_group1 = InstanceGroupFactory();
    ins_group1->set_name("test_group_1");
    ins_group1->quota_.set_capacity(512);
    instance_groups.emplace_back(ins_group1);

    // second instance group
    const auto ins_group2 = InstanceGroupFactory();
    ins_group2->set_name("test_group_2");
    ins_group2->quota_.set_capacity(512);
    instance_groups.emplace_back(ins_group2);

    // create instance infos for both groups
    instance_infos.clear();

    // instance info for first group
    const auto ins_info1 = InstanceInfoFactory();
    ins_info1->set_instance_id("test_instance_id_1");
    ins_info1->set_instance_group_name("test_group_1");
    instance_infos.emplace_back(ins_info1);

    // instance info for second group
    const auto ins_info2 = InstanceInfoFactory();
    ins_info2->set_instance_id("test_instance_id_2");
    ins_info2->set_instance_group_name("test_group_2");
    instance_infos.emplace_back(ins_info2);

    // set up test data
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 5));
    batch_get_loc_out_maps = MakeServingLocationMaps(cache_reclaimer_->GetBatchingSize(request_context_.get()));
    cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), 0);
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
    ASSERT_TRUE(WaitUntil([this] {
        bool found_instance_1 = false;
        bool found_instance_2 = false;
        for (const auto &req : SubmittedDelRequestsSnapshot()) {
            found_instance_1 = found_instance_1 || req.instance_id == "test_instance_id_1";
            found_instance_2 = found_instance_2 || req.instance_id == "test_instance_id_2";
        }
        return found_instance_1 && found_instance_2;
    }));

    cache_reclaimer_->Stop();

    // deletion requests should be submitted for both instance groups
    const auto requests = SubmittedDelRequestsSnapshot();
    ASSERT_FALSE(requests.empty());

    // check that we have requests for both instances
    bool found_instance_1 = false;
    bool found_instance_2 = false;

    for (const auto &req : requests) {
        if (req.instance_id == "test_instance_id_1") {
            found_instance_1 = true;
        } else if (req.instance_id == "test_instance_id_2") {
            found_instance_2 = true;
        }
    }

    ASSERT_TRUE(found_instance_1);
    ASSERT_TRUE(found_instance_2);
}

TEST_F(CacheReclaimerTest, TestKeyCountEdgeCases) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), sample_reclaim_keys.size()));

    // usage size not zero so key count is tested
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    {
        // test with zero key count
        key_count = 0;
        max_key_count = 100;

        // update the trigger strategy to trigger the reclaiming based on percentage
        instance_groups.clear();
        const auto &ins_group = InstanceGroupFactory();
        ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.01); // 1%
        instance_groups.emplace_back(ins_group);

        batch_get_loc_out_maps = MakeServingLocationMaps(sample_reclaim_keys.size());
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

        cache_reclaimer_->Stop();

        // with zero key count, the percentage usage would be 0%, so no reclaiming should happen
        ASSERT_TRUE(HasNoSubmittedDelRequests());
    }

    {
        // test with max key count equal to key count (100% usage)
        key_count = 100;
        max_key_count = 100;

        // update the trigger strategy to trigger at 90%
        instance_groups.clear();
        const auto &ins_group = InstanceGroupFactory();
        ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.9); // 90%
        instance_groups.emplace_back(ins_group);

        // clear requests from previous test
        ClearSubmittedDelRequests();

        batch_get_loc_out_maps = MakeServingLocationMaps(sample_reclaim_keys.size());
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

        ASSERT_TRUE(WaitUntilSubmittedDelRequests());
        ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

        cache_reclaimer_->Stop();

        // with 100% key usage and trigger at 90%, reclaiming should happen
        ASSERT_TRUE(HasSubmittedDelRequests());
    }

    {
        // test with zero max key count (divide by zero)
        key_count = 100;
        max_key_count = 0;

        // update the trigger strategy to trigger at 90%
        instance_groups.clear();
        const auto &ins_group = InstanceGroupFactory();
        ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.9); // 90%
        instance_groups.emplace_back(ins_group);

        // clear requests from previous test
        ClearSubmittedDelRequests();

        batch_get_loc_out_maps = MakeServingLocationMaps(sample_reclaim_keys.size());
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

        ASSERT_TRUE(WaitUntilSubmittedDelRequests());
        ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

        cache_reclaimer_->Stop();

        // group max key count is zero, reclaiming should happen
        ASSERT_TRUE(HasSubmittedDelRequests());
    }
}

TEST_F(CacheReclaimerTest, TestCronJobAdaptiveSleepInterval) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    // update the trigger strategy to trigger the reclaiming
    // so that the reclaiming method shall be entered
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));
    batch_get_loc_out_maps = MakeServingLocationMaps(cache_reclaimer_->GetBatchingSize(request_context_.get()));
    const auto initial_sleep_interval = std::chrono::milliseconds(100);
    cache_reclaimer_->SetSleepIntervalMs(request_context_.get(),
                                         static_cast<std::uint32_t>(initial_sleep_interval.count()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    // The worker should first sleep for initial_sleep_interval, do one
    // reclaiming round, then reduce its sleep interval to 0 and immediately
    // enter following rounds. Use a two-phase wait so the test does not depend
    // on absolute scheduler timing, while still requiring the second round to
    // arrive before the original sleep interval would have elapsed again.
    ASSERT_TRUE(WaitUntilSubmittedDelRequests(initial_sleep_interval + std::chrono::milliseconds(1000)));
    ASSERT_TRUE(WaitUntil([this] { return ListInstanceGroupCallCount() > 1 && SubmittedDelRequestCount() > 1; },
                          initial_sleep_interval / 2));
    cache_reclaimer_->Stop(); // join the worker thread

    ASSERT_LT(1, ListInstanceGroupCallCount());
    ASSERT_LT(1, SubmittedDelRequestCount());
}

TEST_F(CacheReclaimerTest, TestCronJobAdaptiveSleepIntervalRecovery) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    {
        // update the trigger strategy to trigger the reclaiming
        // so that the reclaiming method shall be entered
        dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 4096);

        // use instance 0 from setup()
        // construct instance 1
        const auto ins_info = InstanceInfoFactory();
        ins_info->set_instance_id("test_instance_id_2");
        instance_infos.emplace_back(ins_info);

        instance_groups.clear();
        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(2048);
        instance_groups.emplace_back(ins_group);
    }

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), sample_reclaim_keys.size()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));
    batch_get_loc_out_maps = MakeServingLocationMaps(cache_reclaimer_->GetBatchingSize(request_context_.get()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    // worker thread should first sleep for 10ms then do the reclaiming
    // multiple times with 0 sleep interval in between
    std::this_thread::sleep_for(std::chrono::milliseconds(16));

    {
        // worker thread still running
        std::lock_guard<std::mutex> lock(list_ins_group_mut);

        // update the trigger strategy to *not* trigger the reclaiming
        instance_groups.clear();
        const auto ins_group = InstanceGroupFactory();
        instance_groups.emplace_back(ins_group);

        // reset the stub call counter
        list_ins_group_call_counter = 0;
        KVCM_LOG_INFO("list_ins_group_call_counter reset to: %d", list_ins_group_call_counter);
    }

    // verify the sleep interval is recovered to 10ms
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    cache_reclaimer_->Stop(); // join the worker thread

    // the worker thread is synchronised by join(), so the count should
    // be (at most) 2 if sleep interval reverting is working
    //
    // [1st turn to not triggered]
    // <sleep interval recovered to 10ms>
    //       |
    //       |
    //   sleep 10ms
    //       |
    //       |
    //       V
    // [2nd not triggered]
    //       |
    //       |
    // sleep 6ms (stopping requested by main thread)
    //       |
    //       |
    //       V
    //    [finish]
    ASSERT_GE(2, list_ins_group_call_counter);
}

TEST_F(CacheReclaimerTest, TestGenTraceID) {
    int i = 32768;
    while (i-- != 0) {
        const auto trace_id = CacheReclaimer::GenTraceID();
        ASSERT_EQ(trace_id.size(), CacheReclaimer::kTraceIDPrefix.size() + 16);
    }
}

template <typename T>
std::size_t GetFwdListSize(const std::forward_list<T> &fwd_list) {
    std::size_t size = 0;
    for (auto it = fwd_list.cbegin(); it != fwd_list.cend(); ++it) {
        ++size;
    }
    return size;
}

TEST_F(CacheReclaimerTest, TestHandleDelRes00) {
    // test empty list
    cache_reclaimer_->delete_handlers_.clear();
    cache_reclaimer_->HandleDelRes();
}

TEST_F(CacheReclaimerTest, TestHandleDelRes01) {
    // test one handler only
    const auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto fut = promise->get_future();

    cache_reclaimer_->delete_handlers_.clear();
    const auto now = std::chrono::steady_clock::now();
    cache_reclaimer_->delete_handlers_.emplace_front(request_context_,
                                                     "test_instance",
                                                     "test_instance_group",
                                                     2,
                                                     3,
                                                     std::vector<CacheReclaimer::PendingLocationKey>{},
                                                     CacheReclaimer::BytesByStorageType{},
                                                     CacheReclaimer::CountsByStorageType{},
                                                     0,
                                                     now,
                                                     now + std::chrono::hours(1),
                                                     std::move(fut));

    cache_reclaimer_->HandleDelRes();
    ASSERT_FALSE(cache_reclaimer_->delete_handlers_.empty());
    ASSERT_EQ(1, GetFwdListSize(cache_reclaimer_->delete_handlers_));

    promise->set_value(PlanExecuteResult{ErrorCode::EC_OK, "ok"});

    cache_reclaimer_->HandleDelRes();
    ASSERT_TRUE(cache_reclaimer_->delete_handlers_.empty());

    ASSERT_EQ(2, mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, block_del_count)).Get());
    ASSERT_EQ(3, mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, location_del_count)).Get());
}

TEST_F(CacheReclaimerTest, TestHandleDelRes02) {
    // test multiple handlers
    cache_reclaimer_->delete_handlers_.clear();
    std::vector<std::shared_ptr<std::promise<PlanExecuteResult>>> promises;
    for (int i = 0; i != 16; ++i) {
        const auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
        promises.emplace_back(promise);

        auto fut = promise->get_future();
        const auto now = std::chrono::steady_clock::now();
        cache_reclaimer_->delete_handlers_.emplace_front(request_context_,
                                                         "test_instance" + std::to_string(i),
                                                         "test_instance_group" + std::to_string(i),
                                                         0,
                                                         0,
                                                         std::vector<CacheReclaimer::PendingLocationKey>{},
                                                         CacheReclaimer::BytesByStorageType{},
                                                         CacheReclaimer::CountsByStorageType{},
                                                         0,
                                                         now,
                                                         now + std::chrono::hours(1),
                                                         std::move(fut));
    }

    cache_reclaimer_->HandleDelRes();
    ASSERT_FALSE(cache_reclaimer_->delete_handlers_.empty());
    ASSERT_EQ(16, GetFwdListSize(cache_reclaimer_->delete_handlers_));

    for (int i = 0; i != 4; ++i) {
        promises[i]->set_value(PlanExecuteResult{ErrorCode::EC_OK, "ok"});
    }

    for (int i = 6; i != 8; ++i) {
        promises[i]->set_value(PlanExecuteResult{ErrorCode::EC_ERROR, "not ok"});
    }

    for (int i = 12; i != 16; ++i) {
        promises[i]->set_value(PlanExecuteResult{ErrorCode::EC_OK, "ok"});
    }

    cache_reclaimer_->HandleDelRes();
    ASSERT_FALSE(cache_reclaimer_->delete_handlers_.empty());
    ASSERT_EQ(16 - 4 - 2 - 4, GetFwdListSize(cache_reclaimer_->delete_handlers_));

    for (int i = 0; i != 16; ++i) {
        try {
            promises[i]->set_value(PlanExecuteResult{ErrorCode::EC_OK, "ok"});
        } catch (...) {}
    }

    cache_reclaimer_->HandleDelRes();
    ASSERT_TRUE(cache_reclaimer_->delete_handlers_.empty());
}

TEST_F(CacheReclaimerTest, TestHandleDelRes03) {
    // test promise set exception
    const auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto fut = promise->get_future();

    cache_reclaimer_->delete_handlers_.clear();
    const auto now = std::chrono::steady_clock::now();
    cache_reclaimer_->delete_handlers_.emplace_front(request_context_,
                                                     "test_instance",
                                                     "test_instance_group",
                                                     0,
                                                     0,
                                                     std::vector<CacheReclaimer::PendingLocationKey>{},
                                                     CacheReclaimer::BytesByStorageType{},
                                                     CacheReclaimer::CountsByStorageType{},
                                                     0,
                                                     now,
                                                     now + std::chrono::hours(1),
                                                     std::move(fut));

    try {
        throw std::runtime_error("test exception");
    } catch (...) {
        promise->set_exception(std::current_exception());
    }

    cache_reclaimer_->HandleDelRes();
    ASSERT_FALSE(cache_reclaimer_->delete_handlers_.empty());
    ASSERT_FALSE(cache_reclaimer_->delete_handlers_.front().credit_enabled_);
}

TEST_F(CacheReclaimerTest, TestHandleDelRes04) {
    // test invalid future
    const auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto fut = promise->get_future();

    promise->set_value(PlanExecuteResult{ErrorCode::EC_OK, "ok"});
    fut.get(); // fut is not valid anymore

    cache_reclaimer_->delete_handlers_.clear();
    const auto now = std::chrono::steady_clock::now();
    cache_reclaimer_->delete_handlers_.emplace_front(request_context_,
                                                     "test_instance",
                                                     "test_instance_group",
                                                     0,
                                                     0,
                                                     std::vector<CacheReclaimer::PendingLocationKey>{},
                                                     CacheReclaimer::BytesByStorageType{},
                                                     CacheReclaimer::CountsByStorageType{},
                                                     0,
                                                     now,
                                                     now + std::chrono::hours(1),
                                                     std::move(fut));

    cache_reclaimer_->HandleDelRes();
    ASSERT_FALSE(cache_reclaimer_->delete_handlers_.empty());
    ASSERT_FALSE(cache_reclaimer_->delete_handlers_.front().credit_enabled_);
}

TEST_F(CacheReclaimerTest, TestAsyncDeleteStateIsInstanceIsolatedAndReleasedOnTerminalFuture) {
    spe_submit_auto_complete = false;
    cache_reclaimer_->job_state_flag_ = true;

    CacheReclaimer::BytesByStorageType bytes_by_type{};
    CacheReclaimer::CountsByStorageType counts_by_type{};
    const auto nfs_idx = ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS);
    bytes_by_type[nfs_idx] = 100;
    counts_by_type[nfs_idx] = 1;
    CacheLocationDelRequest request{
        .instance_id = "ignored_by_reclaimer",
        .block_keys = {7},
        .location_ids = {{"same_location"}},
    };

    const auto instance_1 = InstanceInfoFactory();
    ASSERT_TRUE(
        cache_reclaimer_->SubmitDelReq(request_context_, instance_1, request, bytes_by_type, counts_by_type, 1));
    ASSERT_EQ(1, SubmittedDelRequestCount());
    ASSERT_EQ(1, cache_reclaimer_->pending_locations_.size());
    ASSERT_EQ(100, cache_reclaimer_->credited_delete_bytes_by_group_.at(instance_1->instance_group_name())[nfs_idx]);
    ASSERT_EQ(1, cache_reclaimer_->predicted_deleted_keys_by_group_.at(instance_1->instance_group_name()));
    ASSERT_EQ(1, cache_reclaimer_->pending_delete_handler_count_);
    ASSERT_EQ(100, cache_reclaimer_->pending_delete_bytes_);

    // The same instance/block/location is pending before Executor CAS and must not be submitted again.
    ASSERT_FALSE(
        cache_reclaimer_->SubmitDelReq(request_context_, instance_1, request, bytes_by_type, counts_by_type, 1));
    ASSERT_EQ(1, SubmittedDelRequestCount());

    // The same block/location in another instance remains independent.
    const auto instance_2 = InstanceInfoFactory();
    instance_2->set_instance_id("second_instance");
    ASSERT_TRUE(
        cache_reclaimer_->SubmitDelReq(request_context_, instance_2, request, bytes_by_type, counts_by_type, 1));
    ASSERT_EQ(2, SubmittedDelRequestCount());
    ASSERT_EQ(2, cache_reclaimer_->pending_locations_.size());
    ASSERT_EQ(200, cache_reclaimer_->credited_delete_bytes_by_group_.at(instance_1->instance_group_name())[nfs_idx]);
    ASSERT_EQ(2, cache_reclaimer_->pending_delete_handler_count_);
    // Direct submissions bypass the cron's end-of-batch metrics refresh.
    cache_reclaimer_->UpdateAsyncDeleteMetrics();
    const MetricsTags credit_tags{{"instance_group", instance_1->instance_group_name()},
                                  {"storage_type", ToString(DataStorageType::DATA_STORAGE_TYPE_NFS)}};
    EXPECT_EQ(
        200,
        mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, credited_delete_bytes), credit_tags).Get());

    CompleteSubmittedDelete(0, PlanExecuteResult{ErrorCode::EC_PARTIAL_OK, "partial"});
    CompleteSubmittedDelete(1, PlanExecuteResult{ErrorCode::EC_ERROR, "failed"});
    cache_reclaimer_->HandleDelRes();

    EXPECT_TRUE(cache_reclaimer_->delete_handlers_.empty());
    EXPECT_TRUE(cache_reclaimer_->pending_locations_.empty());
    EXPECT_TRUE(cache_reclaimer_->credited_delete_bytes_by_group_.empty());
    EXPECT_TRUE(cache_reclaimer_->predicted_deleted_keys_by_group_.empty());
    EXPECT_TRUE(cache_reclaimer_->pending_quota_by_group_type_.empty());
    EXPECT_EQ(0, cache_reclaimer_->pending_delete_handler_count_);
    EXPECT_EQ(0, cache_reclaimer_->pending_delete_bytes_);
    EXPECT_EQ(
        0,
        mr_->GetGauge(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, credited_delete_bytes), credit_tags).Get());
}

TEST_F(CacheReclaimerTest, TestFilterLocationCreditsNormalizeTypeAndPredictKeysConservatively) {
    const auto instance = InstanceInfoFactory();
    batch_get_loc_out_maps = {
        CacheLocationMap{
            {"hf3fs",
             MakeCacheLocation("hf3fs",
                               CacheLocationStatus::CLS_SERVING,
                               DataStorageType::DATA_STORAGE_TYPE_HF3FS,
                               "nfs://store/hf3fs?size=10")},
            {"vcns",
             MakeCacheLocation("vcns",
                               CacheLocationStatus::CLS_SERVING,
                               DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS,
                               "nfs://store/vcns?size=20")},
        },
        CacheLocationMap{
            {"unknown_size",
             MakeCacheLocation("unknown_size",
                               CacheLocationStatus::CLS_SERVING,
                               DataStorageType::DATA_STORAGE_TYPE_NFS,
                               "nfs://store/unknown?size=not-a-number")},
            {"active_writer",
             MakeCacheLocation("active_writer",
                               CacheLocationStatus::CLS_WRITING,
                               DataStorageType::DATA_STORAGE_TYPE_NFS,
                               "nfs://store/writing?size=40")},
        },
    };

    std::vector<std::vector<std::string>> location_ids;
    CacheReclaimer::BytesByStorageType bytes_by_type{};
    CacheReclaimer::CountsByStorageType counts_by_type{};
    std::uint64_t predicted_keys = 0;
    CacheReclaimer::AgeStats create_age_stats;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(),
                                              instance,
                                              {10, 11},
                                              CacheReclaimer::WaterLevelExceed{},
                                              location_ids,
                                              bytes_by_type,
                                              counts_by_type,
                                              predicted_keys,
                                              create_age_stats));

    const auto hf3fs_idx = ToIndex(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
    const auto vcns_idx = ToIndex(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS);
    const auto nfs_idx = ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS);
    ASSERT_EQ(2, location_ids.size());
    EXPECT_EQ(2, location_ids[0].size());
    EXPECT_EQ(1, location_ids[1].size());
    EXPECT_EQ(30, bytes_by_type[hf3fs_idx]);
    EXPECT_EQ(2, counts_by_type[hf3fs_idx]);
    EXPECT_EQ(0, bytes_by_type[vcns_idx]);
    EXPECT_EQ(0, counts_by_type[vcns_idx]);
    EXPECT_EQ(0, bytes_by_type[nfs_idx]);
    EXPECT_EQ(1, counts_by_type[nfs_idx]);
    EXPECT_EQ(1, predicted_keys);

    // Pending is scoped by instance_id: it is filtered in the same instance but not another one.
    batch_get_loc_out_maps = {batch_get_loc_out_maps.front()};
    cache_reclaimer_->pending_locations_.insert({instance->instance_id(), 10, "hf3fs"});
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(),
                                              instance,
                                              {10},
                                              CacheReclaimer::WaterLevelExceed{},
                                              location_ids,
                                              bytes_by_type,
                                              counts_by_type,
                                              predicted_keys,
                                              create_age_stats));
    EXPECT_EQ(1, location_ids.front().size());
    EXPECT_EQ("vcns", location_ids.front().front());
    EXPECT_EQ(0, predicted_keys);

    const auto other_instance = InstanceInfoFactory();
    other_instance->set_instance_id("other_instance");
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(),
                                              other_instance,
                                              {10},
                                              CacheReclaimer::WaterLevelExceed{},
                                              location_ids,
                                              bytes_by_type,
                                              counts_by_type,
                                              predicted_keys,
                                              create_age_stats));
    EXPECT_EQ(2, location_ids.front().size());
    EXPECT_EQ(1, predicted_keys);
}

TEST_F(CacheReclaimerTest, TestCreditDeadlineDisablesCreditButKeepsPendingAndHardQuota) {
    CacheReclaimerAsyncDeleteConfig config;
    config.inflight_delete_timeout_ms = 1;
    config.pending_delete_handler_limit = 1;
    ReplaceReclaimer(config);
    cache_reclaimer_->job_state_flag_ = true;
    spe_submit_auto_complete = false;

    CacheReclaimer::BytesByStorageType bytes_by_type{};
    CacheReclaimer::CountsByStorageType counts_by_type{};
    const auto type_idx = ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS);
    bytes_by_type[type_idx] = 64;
    counts_by_type[type_idx] = 1;
    const auto instance = InstanceInfoFactory();
    CacheLocationDelRequest request{
        .block_keys = {1},
        .location_ids = {{"deadline_location"}},
    };
    ASSERT_TRUE(cache_reclaimer_->SubmitDelReq(request_context_, instance, request, bytes_by_type, counts_by_type, 1));

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    cache_reclaimer_->HandleDelRes();
    ASSERT_FALSE(cache_reclaimer_->delete_handlers_.empty());
    EXPECT_FALSE(cache_reclaimer_->delete_handlers_.front().credit_enabled_);
    EXPECT_TRUE(cache_reclaimer_->credited_delete_bytes_by_group_.empty());
    EXPECT_TRUE(cache_reclaimer_->predicted_deleted_keys_by_group_.empty());
    EXPECT_EQ(1, cache_reclaimer_->pending_locations_.size());
    EXPECT_EQ(1, cache_reclaimer_->pending_delete_handler_count_);
    EXPECT_EQ(64, cache_reclaimer_->pending_delete_bytes_);
    EXPECT_EQ(1,
              cache_reclaimer_->pending_quota_by_group_type_
                  .at({instance->instance_group_name(), DataStorageType::DATA_STORAGE_TYPE_NFS})
                  .location_count);

    CacheLocationDelRequest second_request{
        .block_keys = {2},
        .location_ids = {{"second_location"}},
    };
    EXPECT_FALSE(
        cache_reclaimer_->SubmitDelReq(request_context_, instance, second_request, bytes_by_type, counts_by_type, 1));
    EXPECT_EQ(1, SubmittedDelRequestCount());

    CompleteSubmittedDelete(0, PlanExecuteResult{ErrorCode::EC_OK, ""});
    cache_reclaimer_->HandleDelRes();
    EXPECT_TRUE(cache_reclaimer_->delete_handlers_.empty());
    EXPECT_TRUE(cache_reclaimer_->pending_locations_.empty());
    EXPECT_TRUE(cache_reclaimer_->pending_quota_by_group_type_.empty());
    EXPECT_EQ(0, cache_reclaimer_->pending_delete_handler_count_);
    EXPECT_EQ(0, cache_reclaimer_->pending_delete_bytes_);
}

TEST_F(CacheReclaimerTest, TestInvalidFutureDisablesCreditButKeepsPendingAndQuota) {
    cache_reclaimer_->job_state_flag_ = true;
    spe_submit_invalid_future = true;

    CacheReclaimer::BytesByStorageType bytes_by_type{};
    CacheReclaimer::CountsByStorageType counts_by_type{};
    const auto type_idx = ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS);
    bytes_by_type[type_idx] = 32;
    counts_by_type[type_idx] = 1;
    const auto instance = InstanceInfoFactory();
    CacheLocationDelRequest request{
        .block_keys = {2},
        .location_ids = {{"invalid_future_location"}},
    };
    ASSERT_TRUE(cache_reclaimer_->SubmitDelReq(request_context_, instance, request, bytes_by_type, counts_by_type, 1));
    cache_reclaimer_->HandleDelRes();

    ASSERT_FALSE(cache_reclaimer_->delete_handlers_.empty());
    EXPECT_FALSE(cache_reclaimer_->delete_handlers_.front().credit_enabled_);
    EXPECT_TRUE(cache_reclaimer_->credited_delete_bytes_by_group_.empty());
    EXPECT_EQ(1, cache_reclaimer_->pending_locations_.size());
    EXPECT_EQ(1, cache_reclaimer_->pending_delete_handler_count_);
    EXPECT_EQ(32, cache_reclaimer_->pending_delete_bytes_);
}

TEST_F(CacheReclaimerTest, TestBackpressureCropsOnlySaturatedGroupTypeAndProcessLimitStopsAll) {
    CacheReclaimerAsyncDeleteConfig config;
    config.pending_location_limit_per_group_type = 1;
    config.pending_bytes_limit_per_group_type = 1024;
    ReplaceReclaimer(config);

    const auto instance = InstanceInfoFactory();
    cache_reclaimer_->pending_quota_by_group_type_[{instance->instance_group_name(),
                                                    DataStorageType::DATA_STORAGE_TYPE_HF3FS}] = {1, 1};
    batch_get_loc_out_maps = {CacheLocationMap{
        {"hf3fs",
         MakeCacheLocation("hf3fs",
                           CacheLocationStatus::CLS_SERVING,
                           DataStorageType::DATA_STORAGE_TYPE_HF3FS,
                           "nfs://store/hf3fs?size=10")},
        {"nfs",
         MakeCacheLocation("nfs",
                           CacheLocationStatus::CLS_SERVING,
                           DataStorageType::DATA_STORAGE_TYPE_NFS,
                           "nfs://store/nfs?size=20")},
    }};

    std::vector<std::vector<std::string>> location_ids;
    CacheReclaimer::BytesByStorageType bytes_by_type{};
    CacheReclaimer::CountsByStorageType counts_by_type{};
    std::uint64_t predicted_keys = 0;
    CacheReclaimer::AgeStats create_age_stats;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(),
                                              instance,
                                              {1},
                                              CacheReclaimer::WaterLevelExceed{},
                                              location_ids,
                                              bytes_by_type,
                                              counts_by_type,
                                              predicted_keys,
                                              create_age_stats));
    ASSERT_EQ(1, location_ids.size());
    ASSERT_EQ(1, location_ids.front().size());
    EXPECT_EQ("nfs", location_ids.front().front());
    EXPECT_EQ(0, bytes_by_type[ToIndex(DataStorageType::DATA_STORAGE_TYPE_HF3FS)]);
    EXPECT_EQ(20, bytes_by_type[ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS)]);
    EXPECT_EQ(0, predicted_keys);
    const MetricsTags hf3fs_limit_tags{{"instance_group", instance->instance_group_name()},
                                       {"storage_type", ToString(DataStorageType::DATA_STORAGE_TYPE_HF3FS)}};
    EXPECT_GT(mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, pending_limit_reject_count),
                              hf3fs_limit_tags)
                  .Get(),
              0);

    cache_reclaimer_
        ->pending_quota_by_group_type_[{instance->instance_group_name(), DataStorageType::DATA_STORAGE_TYPE_HF3FS}] = {
        0, config.pending_bytes_limit_per_group_type};
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(),
                                              instance,
                                              {1},
                                              CacheReclaimer::WaterLevelExceed{},
                                              location_ids,
                                              bytes_by_type,
                                              counts_by_type,
                                              predicted_keys,
                                              create_age_stats));
    ASSERT_EQ(1, location_ids.front().size());
    EXPECT_EQ("nfs", location_ids.front().front());

    cache_reclaimer_->pending_delete_handler_count_ = config.pending_delete_handler_limit;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(),
                                              instance,
                                              {1},
                                              CacheReclaimer::WaterLevelExceed{},
                                              location_ids,
                                              bytes_by_type,
                                              counts_by_type,
                                              predicted_keys,
                                              create_age_stats));
    ASSERT_EQ(1, location_ids.size());
    EXPECT_TRUE(location_ids.front().empty());
    EXPECT_EQ(0, create_age_stats.min_us);
    EXPECT_EQ(0, create_age_stats.max_us);
    EXPECT_EQ(0, create_age_stats.avg_us);

    cache_reclaimer_->pending_delete_handler_count_ = 0;
    cache_reclaimer_->pending_delete_bytes_ = config.pending_bytes_limit;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(),
                                              instance,
                                              {1},
                                              CacheReclaimer::WaterLevelExceed{},
                                              location_ids,
                                              bytes_by_type,
                                              counts_by_type,
                                              predicted_keys,
                                              create_age_stats));
    ASSERT_EQ(1, location_ids.size());
    EXPECT_TRUE(location_ids.front().empty());
}

TEST_F(CacheReclaimerTest, TestEmptyAndRejectedRequestsLeaveNoAsyncDeleteState) {
    cache_reclaimer_->job_state_flag_ = true;
    const auto instance = InstanceInfoFactory();
    CacheReclaimer::BytesByStorageType bytes_by_type{};
    CacheReclaimer::CountsByStorageType counts_by_type{};

    EXPECT_FALSE(cache_reclaimer_->SubmitDelReq(
        request_context_, instance, CacheLocationDelRequest{}, bytes_by_type, counts_by_type, 0));
    EXPECT_EQ(0, SubmittedDelRequestCount());
    EXPECT_TRUE(cache_reclaimer_->delete_handlers_.empty());
    EXPECT_TRUE(cache_reclaimer_->pending_locations_.empty());

    spe_submit_accepted = false;
    CacheLocationDelRequest request{
        .block_keys = {1},
        .location_ids = {{"rejected_location"}},
    };
    counts_by_type[ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS)] = 1;
    EXPECT_FALSE(cache_reclaimer_->SubmitDelReq(request_context_, instance, request, bytes_by_type, counts_by_type, 0));
    EXPECT_EQ(1, SubmittedDelRequestCount());
    EXPECT_TRUE(cache_reclaimer_->delete_handlers_.empty());
    EXPECT_TRUE(cache_reclaimer_->pending_locations_.empty());
    EXPECT_TRUE(cache_reclaimer_->pending_quota_by_group_type_.empty());
    EXPECT_EQ(0, cache_reclaimer_->pending_delete_handler_count_);
}

TEST_F(CacheReclaimerTest, TestWaterLevelCreditsUseSaturatingSubtraction) {
    EXPECT_EQ(0, CacheReclaimer::SaturatingSub(1, 2));
    EXPECT_EQ(0, CacheReclaimer::SaturatingSub(0, std::numeric_limits<std::uint64_t>::max()));
    EXPECT_EQ(3, CacheReclaimer::SaturatingSub(5, 2));

    const auto instance_group = InstanceGroupFactory();
    instance_group->quota_.set_capacity(100);
    instance_group->quota_.quota_config_.clear();
    QuotaConfig nfs_quota;
    nfs_quota.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
    nfs_quota.set_capacity(100);
    instance_group->quota_.quota_config_.push_back(nfs_quota);
    instance_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.8);
    dummy_meta_indexer->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 90);
    key_count = 79;
    max_key_count = 100;

    CacheReclaimer::BytesByStorageType credit{};
    credit[ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS)] = 20;
    cache_reclaimer_->credited_delete_bytes_by_group_[instance_group->name()] = credit;
    cache_reclaimer_->predicted_deleted_keys_by_group_[instance_group->name()] =
        std::numeric_limits<std::uint64_t>::max();
    cache_reclaimer_->job_state_flag_ = true;
    const auto water_level = cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                                                   instance_group->name(),
                                                                   instance_group->quota(),
                                                                   instance_group->cache_config()->reclaim_strategy(),
                                                                   instance_infos);
    ASSERT_NE(nullptr, water_level);
    EXPECT_FALSE(water_level->CheckGroupWaterLevelExceed());

    // A byte credit that saturates effective bytes at zero must not
    // suppress an independently exceeded key-count water level.
    credit[ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS)] = std::numeric_limits<std::uint64_t>::max();
    cache_reclaimer_->credited_delete_bytes_by_group_[instance_group->name()] = credit;
    cache_reclaimer_->predicted_deleted_keys_by_group_[instance_group->name()] = 0;
    key_count = 90;
    const auto key_water_level =
        cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                              instance_group->name(),
                                              instance_group->quota(),
                                              instance_group->cache_config()->reclaim_strategy(),
                                              instance_infos);
    ASSERT_NE(nullptr, key_water_level);
    EXPECT_TRUE(key_water_level->GetGeneralWaterLevelExceed());

    // The symmetric case must still evaluate byte usage when predicted
    // key credit saturates the effective key count at zero.
    credit.fill(0);
    cache_reclaimer_->credited_delete_bytes_by_group_[instance_group->name()] = credit;
    cache_reclaimer_->predicted_deleted_keys_by_group_[instance_group->name()] =
        std::numeric_limits<std::uint64_t>::max();
    const auto byte_water_level =
        cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                              instance_group->name(),
                                              instance_group->quota(),
                                              instance_group->cache_config()->reclaim_strategy(),
                                              instance_infos);
    ASSERT_NE(nullptr, byte_water_level);
    EXPECT_TRUE(byte_water_level->GetGeneralWaterLevelExceed());

    // A zero-capacity storage type is satisfied once its effective usage
    // has saturated to zero; otherwise credit could never stop eviction.
    instance_group->quota_.quota_config_.front().set_capacity(0);
    credit[ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS)] = std::numeric_limits<std::uint64_t>::max();
    cache_reclaimer_->credited_delete_bytes_by_group_[instance_group->name()] = credit;
    const auto zero_effective_usage =
        cache_reclaimer_->GetWaterLevelExceed(request_context_.get(),
                                              instance_group->name(),
                                              instance_group->quota(),
                                              instance_group->cache_config()->reclaim_strategy(),
                                              instance_infos);
    ASSERT_NE(nullptr, zero_effective_usage);
    EXPECT_FALSE(zero_effective_usage->CheckGroupWaterLevelExceed());
}

TEST_F(CacheReclaimerTest, TestSameGroupRechecksCreditBeforeSubmittingNextInstance) {
    cache_reclaimer_->job_state_flag_ = true;
    spe_submit_auto_complete = false;
    sample_reclaim_keys = {1};
    get_out_properties = {{{PROPERTY_LRU_TIME, "1"}}};
    batch_get_loc_out_maps = {CacheLocationMap{
        {"fifty_bytes",
         MakeCacheLocation("fifty_bytes",
                           CacheLocationStatus::CLS_SERVING,
                           DataStorageType::DATA_STORAGE_TYPE_NFS,
                           "nfs://store/key?size=50")},
    }};
    dummy_meta_indexer->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 60);

    const auto instance_1 = InstanceInfoFactory();
    const auto instance_2 = InstanceInfoFactory();
    instance_2->set_instance_id("same_group_second_instance");
    instance_infos = {instance_1, instance_2};

    const auto group = InstanceGroupFactory();
    group->quota_.set_capacity(100);
    group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.8);
    group->cache_config_->reclaim_strategy_->set_delay_before_delete_ms(1000);
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), 1));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));

    const auto result = cache_reclaimer_->TryReclaimOnGroup(request_context_, group);
    EXPECT_TRUE(result.water_level_exceeded);
    EXPECT_TRUE(result.made_progress);
    EXPECT_EQ(1, SubmittedDelRequestCount());
    EXPECT_EQ(instance_1->instance_id(), SubmittedDelRequestsSnapshot().front().instance_id);
}

TEST_F(CacheReclaimerTest, TestReclaimCronHandlesReadyFutureBeforeReadingOfficialUsage) {
    sample_reclaim_keys = {1};
    get_out_properties = {{{PROPERTY_LRU_TIME, "1"}}};
    batch_get_loc_out_maps = MakeServingLocationMaps(1);
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), 1));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));
    cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), 1);

    dummy_meta_indexer->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 100);
    const auto group = InstanceGroupFactory();
    group->quota_.set_capacity(100);
    group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.8);
    instance_groups = {group};

    CacheReclaimer::BytesByStorageType bytes_by_type{};
    CacheReclaimer::CountsByStorageType counts_by_type{};
    bytes_by_type[ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS)] = 30;
    counts_by_type[ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS)] = 1;
    std::promise<PlanExecuteResult> completed_promise;
    auto completed_future = completed_promise.get_future();
    completed_promise.set_value(PlanExecuteResult{ErrorCode::EC_OK, ""});
    const auto now = std::chrono::steady_clock::now();
    cache_reclaimer_->delete_handlers_.emplace_front(
        request_context_,
        instance_infos.front()->instance_id(),
        group->name(),
        1,
        1,
        std::vector<CacheReclaimer::PendingLocationKey>{{instance_infos.front()->instance_id(), 999, "old"}},
        bytes_by_type,
        counts_by_type,
        0,
        now,
        now + std::chrono::hours(1),
        std::move(completed_future));
    cache_reclaimer_->AddDeleteHandlerState(cache_reclaimer_->delete_handlers_.front());

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
    ASSERT_TRUE(WaitUntilSubmittedDelRequests());
    cache_reclaimer_->Stop();
    EXPECT_EQ(0, cache_reclaimer_->pending_locations_.count({instance_infos.front()->instance_id(), 999, "old"}));
    EXPECT_GT(SubmittedDelRequestCount(), 0);
}

TEST_F(CacheReclaimerTest, TestPausedCronStillReleasesTerminalFutureState) {
    CacheReclaimer::BytesByStorageType bytes_by_type{};
    CacheReclaimer::CountsByStorageType counts_by_type{};
    const auto nfs_idx = ToIndex(DataStorageType::DATA_STORAGE_TYPE_NFS);
    bytes_by_type[nfs_idx] = 16;
    counts_by_type[nfs_idx] = 1;

    std::promise<PlanExecuteResult> promise;
    auto future = promise.get_future();
    promise.set_value(PlanExecuteResult{ErrorCode::EC_OK, ""});
    const auto now = std::chrono::steady_clock::now();
    cache_reclaimer_->delete_handlers_.emplace_front(
        request_context_,
        instance_infos.front()->instance_id(),
        instance_groups.front()->name(),
        1,
        1,
        std::vector<CacheReclaimer::PendingLocationKey>{{instance_infos.front()->instance_id(), 1, "paused"}},
        bytes_by_type,
        counts_by_type,
        1,
        now,
        now + std::chrono::hours(1),
        std::move(future));
    cache_reclaimer_->AddDeleteHandlerState(cache_reclaimer_->delete_handlers_.front());
    // Direct state setup bypasses the cron's end-of-batch metrics refresh.
    cache_reclaimer_->UpdateAsyncDeleteMetrics();
    ASSERT_EQ(1, cache_reclaimer_->get_cache_reclaimer_pending_delete_handler_count_metrics());
    cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), 1);
    cache_reclaimer_->Pause();

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
    ASSERT_TRUE(WaitUntil(
        [this] { return cache_reclaimer_->get_cache_reclaimer_pending_delete_handler_count_metrics() == 0; }));
    cache_reclaimer_->Stop();

    EXPECT_TRUE(cache_reclaimer_->pending_locations_.empty());
    EXPECT_TRUE(cache_reclaimer_->credited_delete_bytes_by_group_.empty());
    EXPECT_TRUE(cache_reclaimer_->pending_quota_by_group_type_.empty());
    EXPECT_EQ(0, cache_reclaimer_->pending_delete_handler_count_);
    EXPECT_EQ(0, cache_reclaimer_->pending_delete_bytes_);
}

TEST_F(CacheReclaimerTest, TestReclaimCronBacksOffWhenExecutorRejects) {
    sample_reclaim_keys = {1};
    get_out_properties = {{{PROPERTY_LRU_TIME, "1"}}};
    batch_get_loc_out_maps = MakeServingLocationMaps(1);
    dummy_meta_indexer->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 100);
    const auto group = InstanceGroupFactory();
    group->quota_.set_capacity(10);
    instance_groups = {group};
    spe_submit_accepted = false;
    const auto polling_interval = std::chrono::milliseconds(80);
    cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), static_cast<std::uint32_t>(polling_interval.count()));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), 1));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
    ASSERT_TRUE(WaitUntilSubmittedDelRequests(std::chrono::seconds(1)));
    const auto submit_count_after_first_round = SubmittedDelRequestCount();
    const auto list_count_after_first_round = ListInstanceGroupCallCount();
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_EQ(submit_count_after_first_round, SubmittedDelRequestCount());
    EXPECT_EQ(list_count_after_first_round, ListInstanceGroupCallCount());
    cache_reclaimer_->Stop();
    EXPECT_GT(cache_reclaimer_->get_cache_reclaimer_reclaim_no_progress_backoff_count_metrics(), 0);
}

TEST_F(CacheReclaimerTest, TestReclaimCronKeepsPositiveBackoffWhenPollingIntervalIsZero) {
    sample_reclaim_keys = {1};
    get_out_properties = {{{PROPERTY_LRU_TIME, "1"}}};
    batch_get_loc_out_maps = MakeServingLocationMaps(1);
    dummy_meta_indexer->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 100);
    const auto group = InstanceGroupFactory();
    group->quota_.set_capacity(10);
    instance_groups = {group};
    spe_submit_accepted = false;
    cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), 0);
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), 1));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
    ASSERT_TRUE(WaitUntilSubmittedDelRequests(std::chrono::seconds(1)));
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    cache_reclaimer_->Stop();

    EXPECT_LT(ListInstanceGroupCallCount(), 100);
    EXPECT_GT(cache_reclaimer_->get_cache_reclaimer_reclaim_no_progress_backoff_count_metrics(), 0);
}

TEST_F(CacheReclaimerTest, TestTryReclaimReportsNoProgressForPendingOrMissingVictim) {
    cache_reclaimer_->job_state_flag_ = true;
    sample_reclaim_keys = {1};
    get_out_properties = {{{PROPERTY_LRU_TIME, "1"}}};
    dummy_meta_indexer->SetStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 100);
    const auto group = InstanceGroupFactory();
    group->quota_.set_capacity(10);
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), 1));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));

    batch_get_loc_out_maps = {CacheLocationMap{
        {"pending",
         MakeCacheLocation("pending",
                           CacheLocationStatus::CLS_SERVING,
                           DataStorageType::DATA_STORAGE_TYPE_NFS,
                           "nfs://store/pending?size=1")},
    }};
    cache_reclaimer_->pending_locations_.insert({instance_infos.front()->instance_id(), 1, "pending"});
    auto result = cache_reclaimer_->TryReclaimOnGroup(request_context_, group);
    EXPECT_TRUE(result.water_level_exceeded);
    EXPECT_FALSE(result.made_progress);
    EXPECT_EQ(0, SubmittedDelRequestCount());

    cache_reclaimer_->pending_locations_.clear();
    batch_get_loc_out_maps = {CacheLocationMap{
        {"deleting",
         MakeCacheLocation("deleting",
                           CacheLocationStatus::CLS_DELETING,
                           DataStorageType::DATA_STORAGE_TYPE_NFS,
                           "nfs://store/deleting?size=1")},
    }};
    result = cache_reclaimer_->TryReclaimOnGroup(request_context_, group);
    EXPECT_TRUE(result.water_level_exceeded);
    EXPECT_FALSE(result.made_progress);
    EXPECT_EQ(0, SubmittedDelRequestCount());
}

TEST_F(CacheReclaimerTest, TestDoKeySampling) {
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {
            {PROPERTY_LRU_TIME, "0"},
        },
        {
            {PROPERTY_LRU_TIME, "1"},
        },
        {
            {PROPERTY_LRU_TIME, "2"},
        },
        {
            {PROPERTY_LRU_TIME, "3"},
        },
        {
            {PROPERTY_LRU_TIME, "4"},
        },
        {
            {PROPERTY_LRU_TIME, "5"},
        },
        {
            {PROPERTY_LRU_TIME, "6"},
        },
        {
            {PROPERTY_LRU_TIME, "7"},
        },
        {
            {PROPERTY_LRU_TIME, "8"},
        },
        {
            {PROPERTY_LRU_TIME, "9"},
        },
    };

    {
        cache_reclaimer_->sampling_size_.store(sample_reclaim_keys.size());
        cache_reclaimer_->sampling_size_per_task_.store(100);

        std::vector<std::int64_t> keys;
        std::vector<std::map<std::string, std::string>> maps;
        ASSERT_TRUE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
        ASSERT_EQ(sample_reclaim_keys.size(), keys.size());
        ASSERT_EQ(get_out_properties.size(), maps.size());
    }

    {
        cache_reclaimer_->sampling_size_.store(0);
        cache_reclaimer_->sampling_size_per_task_.store(100);

        std::vector<std::int64_t> keys;
        std::vector<std::map<std::string, std::string>> maps;
        ASSERT_FALSE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
    }

    {
        cache_reclaimer_->sampling_size_.store(sample_reclaim_keys.size());
        cache_reclaimer_->sampling_size_per_task_.store(0); // 0 means single thread key sampling

        std::vector<std::int64_t> keys;
        std::vector<std::map<std::string, std::string>> maps;
        ASSERT_TRUE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
        ASSERT_EQ(sample_reclaim_keys.size(), keys.size());
        ASSERT_EQ(get_out_properties.size(), maps.size());
    }
    {
        // sampling_size <= sampling_size_per_task means single thread key sampling
        cache_reclaimer_->sampling_size_.store(1000);
        cache_reclaimer_->sampling_size_per_task_.store(1000);

        std::vector<std::int64_t> keys;
        std::vector<std::map<std::string, std::string>> maps;
        ASSERT_TRUE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
        ASSERT_EQ(1000, keys.size());
        ASSERT_EQ(1000, maps.size());
    }

    {
        cache_reclaimer_->sampling_size_.store(1000);
        cache_reclaimer_->sampling_size_per_task_.store(100);

        std::vector<std::int64_t> keys;
        std::vector<std::map<std::string, std::string>> maps;
        ASSERT_TRUE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
        ASSERT_EQ(1000, keys.size());
        ASSERT_EQ(1000, maps.size());
    }

    {
        cache_reclaimer_->sampling_size_.store(999);
        cache_reclaimer_->sampling_size_per_task_.store(99);

        std::vector<std::int64_t> keys;
        std::vector<std::map<std::string, std::string>> maps;
        ASSERT_TRUE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
        ASSERT_EQ(999, keys.size());
        ASSERT_EQ(999, maps.size());
    }

    {
        cache_reclaimer_->sampling_size_.store(1001);
        cache_reclaimer_->sampling_size_per_task_.store(100);

        std::vector<std::int64_t> keys;
        std::vector<std::map<std::string, std::string>> maps;
        ASSERT_TRUE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
        ASSERT_EQ(1001, keys.size());
        ASSERT_EQ(1001, maps.size());
    }

    {
        cache_reclaimer_->sampling_size_.store(1);
        cache_reclaimer_->sampling_size_per_task_.store(999);

        std::vector<std::int64_t> keys;
        std::vector<std::map<std::string, std::string>> maps;
        ASSERT_TRUE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
        ASSERT_EQ(1, keys.size());
        ASSERT_EQ(1, maps.size());
    }

    {
        // test less sampled keys returnd
        cache_reclaimer_->sampling_size_.store(100);
        cache_reclaimer_->sampling_size_per_task_.store(11); // trigger the specially crafted case
        // 100 = 11 * 9 + 1
        // 9 + 1 sampling tasks would be despatched
        // the specially crafted size 11 would cause the mock func return 10 sampled keys
        // 10 * 9 + 1 = 91

        std::vector<std::int64_t> keys;
        std::vector<std::map<std::string, std::string>> maps;
        ASSERT_TRUE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
        ASSERT_EQ(91, keys.size());
        ASSERT_EQ(91, maps.size());
    }
}

TEST_F(CacheReclaimerTest, TestDoKeySamplingFutureTimeout_SampleReclaimKeysHangs) {
    // when SampleReclaimKeys blocks longer than the future timeout,
    // DoKeySampling should return false instead of blocking forever
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {{PROPERTY_LRU_TIME, "0"}},
        {{PROPERTY_LRU_TIME, "1"}},
        {{PROPERTY_LRU_TIME, "2"}},
        {{PROPERTY_LRU_TIME, "3"}},
        {{PROPERTY_LRU_TIME, "4"}},
        {{PROPERTY_LRU_TIME, "5"}},
        {{PROPERTY_LRU_TIME, "6"}},
        {{PROPERTY_LRU_TIME, "7"}},
        {{PROPERTY_LRU_TIME, "8"}},
        {{PROPERTY_LRU_TIME, "9"}},
    };

    // set a very short future timeout (50ms) and a long sample delay (500ms)
    cache_reclaimer_->future_timeout_ms_.store(50);
    mi_sample_reclaim_delay = std::chrono::milliseconds{500};

    // use multi-thread path: sampling_size > sampling_size_per_task
    cache_reclaimer_->sampling_size_.store(10);
    cache_reclaimer_->sampling_size_per_task_.store(5);

    std::vector<std::int64_t> keys;
    std::vector<std::map<std::string, std::string>> maps;

    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_FALSE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
    const auto elapsed = std::chrono::steady_clock::now() - t0;

    // should return within the timeout budget (~50ms), not wait for the full 500ms task
    ASSERT_LT(elapsed, std::chrono::milliseconds(300));

    // in-flight counter was incremented for 2 tasks
    ASSERT_GT(cache_reclaimer_->in_flight_sampling_tasks_.load(), 0u);

    // wait for background tasks to finish naturally
    while (cache_reclaimer_->in_flight_sampling_tasks_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

TEST_F(CacheReclaimerTest, TestDoKeySamplingFutureTimeout_GetPropertiesHangs) {
    // when GetProperties blocks longer than the future timeout,
    // DoKeySampling should return false instead of blocking forever
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {{PROPERTY_LRU_TIME, "0"}},
        {{PROPERTY_LRU_TIME, "1"}},
        {{PROPERTY_LRU_TIME, "2"}},
        {{PROPERTY_LRU_TIME, "3"}},
        {{PROPERTY_LRU_TIME, "4"}},
        {{PROPERTY_LRU_TIME, "5"}},
        {{PROPERTY_LRU_TIME, "6"}},
        {{PROPERTY_LRU_TIME, "7"}},
        {{PROPERTY_LRU_TIME, "8"}},
        {{PROPERTY_LRU_TIME, "9"}},
    };

    // set a very short future timeout (50ms) and a long GetProperties delay (500ms)
    cache_reclaimer_->future_timeout_ms_.store(50);
    mi_getprop_delay = std::chrono::milliseconds{500};

    // use multi-thread path with sampling_size > batching_size to trigger GetProperties
    cache_reclaimer_->sampling_size_.store(10);
    cache_reclaimer_->sampling_size_per_task_.store(5);
    cache_reclaimer_->batching_size_.store(1);

    std::vector<std::int64_t> keys;
    std::vector<std::map<std::string, std::string>> maps;
    ASSERT_FALSE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));

    // wait for background tasks to finish naturally
    while (cache_reclaimer_->in_flight_sampling_tasks_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

TEST_F(CacheReclaimerTest, TestDoKeySamplingFutureTimeout_NoTimeoutOnFastTasks) {
    // when tasks complete quickly, DoKeySampling should succeed normally
    // even with a timeout configured
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {{PROPERTY_LRU_TIME, "0"}},
        {{PROPERTY_LRU_TIME, "1"}},
        {{PROPERTY_LRU_TIME, "2"}},
        {{PROPERTY_LRU_TIME, "3"}},
        {{PROPERTY_LRU_TIME, "4"}},
        {{PROPERTY_LRU_TIME, "5"}},
        {{PROPERTY_LRU_TIME, "6"}},
        {{PROPERTY_LRU_TIME, "7"}},
        {{PROPERTY_LRU_TIME, "8"}},
        {{PROPERTY_LRU_TIME, "9"}},
    };

    // set a reasonable future timeout (5000ms) and no delay
    cache_reclaimer_->future_timeout_ms_.store(5000);
    mi_sample_reclaim_delay = std::chrono::milliseconds{0};
    mi_getprop_delay = std::chrono::milliseconds{0};

    // use multi-thread path
    cache_reclaimer_->sampling_size_.store(10);
    cache_reclaimer_->sampling_size_per_task_.store(5);

    std::vector<std::int64_t> keys;
    std::vector<std::map<std::string, std::string>> maps;
    ASSERT_TRUE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
    ASSERT_EQ(10u, keys.size());

    // all tasks completed, in-flight should be zero
    ASSERT_EQ(0u, cache_reclaimer_->in_flight_sampling_tasks_.load());
}

TEST_F(CacheReclaimerTest, TestDoKeySamplingFutureTimeout_DeadlineBoundsAllFutures) {
    // verify that the total wait is bounded by a single deadline, not
    // N * timeout (where N is the number of worker tasks)
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {{PROPERTY_LRU_TIME, "0"}},
        {{PROPERTY_LRU_TIME, "1"}},
        {{PROPERTY_LRU_TIME, "2"}},
        {{PROPERTY_LRU_TIME, "3"}},
        {{PROPERTY_LRU_TIME, "4"}},
        {{PROPERTY_LRU_TIME, "5"}},
        {{PROPERTY_LRU_TIME, "6"}},
        {{PROPERTY_LRU_TIME, "7"}},
        {{PROPERTY_LRU_TIME, "8"}},
        {{PROPERTY_LRU_TIME, "9"}},
    };

    // 100ms timeout, 500ms delay; 5 worker tasks should all time out
    // but total wait should be ~100ms, not 5 * 100ms
    cache_reclaimer_->future_timeout_ms_.store(100);
    mi_sample_reclaim_delay = std::chrono::milliseconds{500};

    cache_reclaimer_->sampling_size_.store(10);
    cache_reclaimer_->sampling_size_per_task_.store(2); // 5 tasks

    std::vector<std::int64_t> keys;
    std::vector<std::map<std::string, std::string>> maps;

    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_FALSE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
    const auto elapsed = std::chrono::steady_clock::now() - t0;

    // should be bounded by ~100ms deadline, not 500ms (5 tasks * 100ms)
    ASSERT_LT(elapsed, std::chrono::milliseconds(300));

    // wait for background tasks to finish
    while (cache_reclaimer_->in_flight_sampling_tasks_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

TEST_F(CacheReclaimerTest, TestDoKeySamplingFutureTimeout_WorkerSaturationGuard) {
    // when all workers are occupied by timed-out tasks, subsequent
    // DoKeySampling calls should fail fast without submitting more work
    sample_reclaim_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    get_out_properties = {
        {{PROPERTY_LRU_TIME, "0"}},
        {{PROPERTY_LRU_TIME, "1"}},
        {{PROPERTY_LRU_TIME, "2"}},
        {{PROPERTY_LRU_TIME, "3"}},
        {{PROPERTY_LRU_TIME, "4"}},
        {{PROPERTY_LRU_TIME, "5"}},
        {{PROPERTY_LRU_TIME, "6"}},
        {{PROPERTY_LRU_TIME, "7"}},
        {{PROPERTY_LRU_TIME, "8"}},
        {{PROPERTY_LRU_TIME, "9"}},
    };

    // simulate saturated worker pool by setting in_flight >= workers_.size()
    cache_reclaimer_->in_flight_sampling_tasks_.store(cache_reclaimer_->workers_.size());

    cache_reclaimer_->future_timeout_ms_.store(5000);
    mi_sample_reclaim_delay = std::chrono::milliseconds{0};
    cache_reclaimer_->sampling_size_.store(10);
    cache_reclaimer_->sampling_size_per_task_.store(5);

    std::vector<std::int64_t> keys;
    std::vector<std::map<std::string, std::string>> maps;

    // should immediately return false without submitting new work
    const auto t0 = std::chrono::steady_clock::now();
    ASSERT_FALSE(cache_reclaimer_->DoKeySampling(request_context_, instance_infos.front(), keys, maps));
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    ASSERT_LT(elapsed, std::chrono::milliseconds(50));

    // restore
    cache_reclaimer_->in_flight_sampling_tasks_.store(0);
}

TEST_F(CacheReclaimerTest, TestDupKeys) {
    {
        sample_reclaim_keys = {0, 0, 2, 3, 4, 5, 6, 7, 8, 9};
        get_out_properties = {
            {
                {PROPERTY_LRU_TIME, "1"},
            },
            {
                {PROPERTY_LRU_TIME, "1"},
            },
            {
                {PROPERTY_LRU_TIME, "2"},
            },
            {
                {PROPERTY_LRU_TIME, "3"},
            },
            {
                {PROPERTY_LRU_TIME, "4"},
            },
            {
                {PROPERTY_LRU_TIME, "5"},
            },
            {
                {PROPERTY_LRU_TIME, "6"},
            },
            {
                {PROPERTY_LRU_TIME, "7"},
            },
            {
                {PROPERTY_LRU_TIME, "8"},
            },
            {
                {PROPERTY_LRU_TIME, "9"},
            },
        };

        cache_reclaimer_->sampling_size_.store(sample_reclaim_keys.size());
        cache_reclaimer_->sampling_size_per_task_.store(100);
        cache_reclaimer_->batching_size_.store(sample_reclaim_keys.size());

        std::vector<std::int64_t> keys(sample_reclaim_keys);
        std::vector<std::map<std::string, std::string>> maps(get_out_properties);
        std::vector<std::int64_t> batch;
        CacheReclaimer::AgeStats lru_age_stats;
        ASSERT_TRUE(cache_reclaimer_->MakeBatchByLRU(
            request_context_.get(), instance_infos.front(), keys, maps, batch, lru_age_stats));
        ASSERT_EQ(9, batch.size());
        // keys 1..10 unique, lru_times 0..9; tp=0 excluded from stats, tp=1..9 included (9 entries)
        // ages: now_us-1, now_us-2, ..., now_us-9 → min=now_us-9, max=now_us-1, diff=8
        EXPECT_GT(lru_age_stats.min_us, 0);
        EXPECT_GT(lru_age_stats.max_us, 0);
        EXPECT_GT(lru_age_stats.avg_us, 0);
        EXPECT_LE(lru_age_stats.min_us, lru_age_stats.avg_us);
        EXPECT_LE(lru_age_stats.avg_us, lru_age_stats.max_us);
        EXPECT_EQ(lru_age_stats.max_us - lru_age_stats.min_us, 8);
    }

    {
        sample_reclaim_keys = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        get_out_properties = {
            {
                {PROPERTY_LRU_TIME, "0"},
            },
            {
                {PROPERTY_LRU_TIME, "1"},
            },
            {
                {PROPERTY_LRU_TIME, "2"},
            },
            {
                {PROPERTY_LRU_TIME, "3"},
            },
            {
                {PROPERTY_LRU_TIME, "4"},
            },
            {
                {PROPERTY_LRU_TIME, "5"},
            },
            {
                {PROPERTY_LRU_TIME, "6"},
            },
            {
                {PROPERTY_LRU_TIME, "7"},
            },
            {
                {PROPERTY_LRU_TIME, "8"},
            },
            {
                {PROPERTY_LRU_TIME, "9"},
            },
        };

        cache_reclaimer_->sampling_size_.store(sample_reclaim_keys.size());
        cache_reclaimer_->sampling_size_per_task_.store(100);
        cache_reclaimer_->batching_size_.store(2);

        std::vector<std::int64_t> keys(sample_reclaim_keys);
        std::vector<std::map<std::string, std::string>> maps(get_out_properties);
        std::vector<std::int64_t> batch;
        CacheReclaimer::AgeStats lru_age_stats;
        ASSERT_TRUE(cache_reclaimer_->MakeBatchByLRU(
            request_context_.get(), instance_infos.front(), keys, maps, batch, lru_age_stats));
        ASSERT_EQ(1, batch.size());
        // all keys are 1 (only 1 unique key), first occurrence has tp=0 → excluded
        // age_count=0 → Clear() called → all stats zeroed
        EXPECT_EQ(lru_age_stats.min_us, 0);
        EXPECT_EQ(lru_age_stats.max_us, 0);
        EXPECT_EQ(lru_age_stats.avg_us, 0);
    }

    {
        sample_reclaim_keys = {1, 1, 1, 1, 1, 1, 1, 2, 1, 1};
        get_out_properties = {
            {
                {PROPERTY_LRU_TIME, "9"},
            },
            {
                {PROPERTY_LRU_TIME, "9"},
            },
            {
                {PROPERTY_LRU_TIME, "9"},
            },
            {
                {PROPERTY_LRU_TIME, "9"},
            },
            {
                {PROPERTY_LRU_TIME, "9"},
            },
            {
                {PROPERTY_LRU_TIME, "9"},
            },
            {
                {PROPERTY_LRU_TIME, "9"},
            },
            {
                {PROPERTY_LRU_TIME, "10"},
            },
            {
                {PROPERTY_LRU_TIME, "9"},
            },
            {
                {PROPERTY_LRU_TIME, "9"},
            },
        };

        cache_reclaimer_->sampling_size_.store(sample_reclaim_keys.size());
        cache_reclaimer_->sampling_size_per_task_.store(100);
        cache_reclaimer_->batching_size_.store(2);

        std::vector<std::int64_t> keys(sample_reclaim_keys);
        std::vector<std::map<std::string, std::string>> maps(get_out_properties);
        std::vector<std::int64_t> batch;
        CacheReclaimer::AgeStats lru_age_stats;
        ASSERT_TRUE(cache_reclaimer_->MakeBatchByLRU(
            request_context_.get(), instance_infos.front(), keys, maps, batch, lru_age_stats));
        ASSERT_EQ(2, batch.size());
        // keys={1*7,2,1,1}, tp={9*7,10,9,9}; sorted → key=1(tp=9) then key=2(tp=10)
        // ages: now_us-9 and now_us-10 → min=now_us-10, max=now_us-9, diff=1
        EXPECT_GT(lru_age_stats.min_us, 0);
        EXPECT_GT(lru_age_stats.max_us, 0);
        EXPECT_GT(lru_age_stats.avg_us, 0);
        EXPECT_LE(lru_age_stats.min_us, lru_age_stats.avg_us);
        EXPECT_LE(lru_age_stats.avg_us, lru_age_stats.max_us);
        EXPECT_EQ(lru_age_stats.max_us - lru_age_stats.min_us, 1);
    }
}

// 活跃迁移任务不影响 FilterLocID 的驱逐选择；HasMigrationTask 只用于防重复迁移提交。
TEST_F(CacheReclaimerTest, TestFilterLocIDDoesNotSkipActiveMigrationTask) {
    const auto ins_info = InstanceInfoFactory();

    // seed 一个活跃迁移任务，使 HasMigrationTask(42) 为真（43 无）。
    mm_->DebugInsertActiveCopyTask(ins_info->instance_id(), 42, "");
    ASSERT_TRUE(mm_->HasMigrationTask(ins_info->instance_id(), 42));
    ASSERT_FALSE(mm_->HasMigrationTask(ins_info->instance_id(), 43));

    auto make_serving_map = [](const std::string &loc_id) {
        CacheLocationMap m;
        auto loc =
            std::make_shared<CacheLocation>(loc_id,
                                            CacheLocationStatus::CLS_SERVING,
                                            DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                            1,
                                            std::vector<LocationSpec>{LocationSpec("TP0", "dummy://d/" + loc_id)});
        m.emplace(loc_id, loc);
        return m;
    };

    // case 1: 仅 group 级（general）水位超阈值 -> 收集所有可驱逐 location。
    {
        batch_get_loc_out_maps = {make_serving_map("loc_a"), make_serving_map("loc_b")};
        batch_get_loc_result = ErrorCode::EC_OK;

        CacheReclaimer::WaterLevelExceed wl;
        wl.SetGeneralWaterLevelExceed(true);
        ASSERT_FALSE(wl.CheckStorageTypeWaterLevelExceed());

        std::vector<std::vector<std::string>> out;
        CacheReclaimer::AgeStats age_stats;
        ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(), ins_info, {42, 43}, wl, out, age_stats));
        ASSERT_EQ(2u, out.size());
        ASSERT_EQ(1u, out[0].size()) << "active migration task should not suppress eviction";
        ASSERT_EQ("loc_a", out[0][0]);
        ASSERT_EQ(1u, out[1].size());
        ASSERT_EQ("loc_b", out[1][0]);
    }

    // case 2: 某存储类型水位超阈值 -> 同样不因活跃迁移任务跳过。
    {
        batch_get_loc_out_maps = {make_serving_map("loc_a"), make_serving_map("loc_b")};
        batch_get_loc_result = ErrorCode::EC_OK;

        CacheReclaimer::WaterLevelExceed wl;
        wl.SetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_DUMMY, true);
        ASSERT_TRUE(wl.CheckStorageTypeWaterLevelExceed());

        std::vector<std::vector<std::string>> out;
        CacheReclaimer::AgeStats age_stats;
        ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(), ins_info, {42, 43}, wl, out, age_stats));
        ASSERT_EQ(2u, out.size());
        ASSERT_EQ(1u, out[0].size()) << "migrating block not protected in storage-type eviction zone";
        ASSERT_EQ("loc_a", out[0][0]);
        ASSERT_EQ(1u, out[1].size());
        ASSERT_EQ("loc_b", out[1][0]);
    }
}

TEST_F(CacheReclaimerTest, TestFilterLocIDSkipsActiveWritingLocations) {
    auto write_location_manager = std::make_shared<WriteLocationManager>();
    cache_reclaimer_ = std::make_unique<CacheReclaimer>(
        10, 100, 1, 10, 16, rm_, mim_, msm_, spe_, mr_, em_, write_location_manager, mm_);

    const auto ins_info = InstanceInfoFactory();
    mm_->DebugInsertActiveCopyTask(ins_info->instance_id(), 42, "migration_writing");
    MigrationManager::MigrationRequest preparing_req;
    preparing_req.instance_id = ins_info->instance_id();
    preparing_req.block_key = 45;
    preparing_req.src_storage_name = "hot_01";
    preparing_req.dst_storage_name = "cold_01";
    {
        std::lock_guard<std::mutex> lock(mm_->task_mutex_);
        ASSERT_TRUE(mm_->ReservePreparingTaskLocked(preparing_req));
    }
    write_location_manager->Put(
        "write_session", {44}, {"client_writing"}, 60, [](std::unique_ptr<WriteLocationManager::WriteLocationInfo>) {});

    auto make_writing_map = [](const std::string &loc_id) {
        CacheLocationMap m;
        auto loc =
            std::make_shared<CacheLocation>(loc_id,
                                            CacheLocationStatus::CLS_WRITING,
                                            DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                            1,
                                            std::vector<LocationSpec>{LocationSpec("TP0", "dummy://d/" + loc_id)});
        m.emplace(loc_id, loc);
        return m;
    };

    batch_get_loc_out_maps = {make_writing_map("migration_writing"),
                              make_writing_map("orphan_writing"),
                              make_writing_map("client_writing"),
                              make_writing_map("preparing_writing")};
    batch_get_loc_result = ErrorCode::EC_OK;

    CacheReclaimer::WaterLevelExceed wl;
    wl.SetGeneralWaterLevelExceed(true);

    std::vector<std::vector<std::string>> out;
    CacheReclaimer::AgeStats age_stats;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(), ins_info, {42, 43, 44, 45}, wl, out, age_stats));
    ASSERT_EQ(4u, out.size());
    EXPECT_TRUE(out[0].empty()) << "active copy target should not be treated as orphan";
    ASSERT_EQ(1u, out[1].size());
    EXPECT_EQ("orphan_writing", out[1][0]);
    EXPECT_TRUE(out[2].empty()) << "active client write location should not be treated as orphan";
    EXPECT_TRUE(out[3].empty()) << "preparing copy target should be protected before location id is bound";
}

/* -------- keep_both 分层淘汰优先级（多副本保冷弃热）stub + test -------- */
namespace {
std::shared_ptr<const InstanceGroup> g_cold_ig;
std::pair<ErrorCode, std::shared_ptr<const InstanceGroup>>
RegistryManager_GetInstanceGroup_cold_stub(void * /*obj*/, RequestContext * /*rc*/, const std::string & /*ig*/) {
    return {ErrorCode::EC_OK, g_cold_ig};
}
} // namespace

// 多层存储 keep_both 配套：总水位超限时，对同时有冷/热副本的 block 只淘汰热(源)副本、保留冷副本；
// 仅热副本的 block 维持正常淘汰。
TEST_F(CacheReclaimerTest, TestFilterLocIDKeepColdEvictHot) {
    // 构建带 target_storage=cold_01 的迁移策略的 IG，并通过 stub 让 GetInstanceGroup 返回它。
    auto ig = InstanceGroupFactory();
    {
        auto cfg = std::make_shared<CacheConfig>();
        auto strategy = std::make_shared<MigrationStrategy>();
        strategy->set_source_storage_name("hot_01");
        strategy->set_target_storage_name("cold_01");
        cfg->set_migration_strategies({strategy});
        ig->set_cache_config(cfg);
    }
    g_cold_ig = ig;
    stub_.set(ADDR(RegistryManager, GetInstanceGroup), RegistryManager_GetInstanceGroup_cold_stub);

    const auto ins_info = InstanceInfoFactory();

    auto make_loc = [](const std::string &loc_id, const std::string &storage) {
        return std::make_shared<CacheLocation>(
            loc_id,
            CacheLocationStatus::CLS_SERVING,
            DataStorageType::DATA_STORAGE_TYPE_DUMMY,
            1,
            std::vector<LocationSpec>{LocationSpec("TP0", "dummy://" + storage + "/" + loc_id)});
    };
    // block 0: 冷+热 双副本; block 1: 仅热副本
    CacheLocationMap m0;
    m0.emplace("hot_a", make_loc("hot_a", "hot_01"));
    m0.emplace("cold_a", make_loc("cold_a", "cold_01"));
    CacheLocationMap m1;
    m1.emplace("hot_b", make_loc("hot_b", "hot_01"));
    batch_get_loc_out_maps = {std::move(m0), std::move(m1)};
    batch_get_loc_result = ErrorCode::EC_OK;

    CacheReclaimer::WaterLevelExceed wl;
    wl.SetGeneralWaterLevelExceed(true); // 总水位超限分支
    ASSERT_FALSE(wl.CheckStorageTypeWaterLevelExceed());

    std::vector<std::vector<std::string>> out;
    CacheReclaimer::AgeStats age_stats;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(), ins_info, {0, 1}, wl, out, age_stats));
    ASSERT_EQ(2u, out.size());
    // block 0 多副本 -> 只淘汰热副本 hot_a，保留 cold_a
    ASSERT_EQ(1u, out[0].size()) << "multi-replica block should evict only the hot copy";
    ASSERT_EQ("hot_a", out[0][0]);
    // block 1 仅热副本 -> 正常淘汰
    ASSERT_EQ(1u, out[1].size());
    ASSERT_EQ("hot_b", out[1][0]);

    stub_.reset(ADDR(RegistryManager, GetInstanceGroup));
    g_cold_ig.reset();
}

TEST_F(CacheReclaimerTest, TestFilterLocIDDoesNotEvictHotWhenColdSpecsIncomplete) {
    auto ig = InstanceGroupFactory();
    {
        auto cfg = std::make_shared<CacheConfig>();
        auto strategy = std::make_shared<MigrationStrategy>();
        strategy->set_source_storage_name("hot_01");
        strategy->set_target_storage_name("cold_01");
        cfg->set_migration_strategies({strategy});
        ig->set_cache_config(cfg);
    }
    g_cold_ig = ig;
    stub_.set(ADDR(RegistryManager, GetInstanceGroup), RegistryManager_GetInstanceGroup_cold_stub);

    const auto ins_info = InstanceInfoFactory();
    auto hot_loc = std::make_shared<CacheLocation>("hot_full",
                                                   CacheLocationStatus::CLS_SERVING,
                                                   DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                                   2,
                                                   std::vector<LocationSpec>{
                                                       LocationSpec("TP0", "dummy://hot_01/hot_full/tp0"),
                                                       LocationSpec("TP1", "dummy://hot_01/hot_full/tp1"),
                                                   });
    auto partial_cold_loc = std::make_shared<CacheLocation>(
        "cold_partial",
        CacheLocationStatus::CLS_SERVING,
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold_01/cold_partial/tp0")});

    CacheLocationMap loc_map;
    loc_map.emplace("hot_full", hot_loc);
    loc_map.emplace("cold_partial", partial_cold_loc);
    batch_get_loc_out_maps = {std::move(loc_map)};
    batch_get_loc_result = ErrorCode::EC_OK;

    CacheReclaimer::WaterLevelExceed wl;
    wl.SetGeneralWaterLevelExceed(true);

    std::vector<std::vector<std::string>> out;
    CacheReclaimer::AgeStats age_stats;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(), ins_info, {0}, wl, out, age_stats));
    ASSERT_EQ(1u, out.size());
    EXPECT_TRUE(out[0].empty()) << "hot location should stay until cold tier covers all of its specs";

    stub_.reset(ADDR(RegistryManager, GetInstanceGroup));
    g_cold_ig.reset();
}

TEST_F(CacheReclaimerTest, TestFilterLocIDDoesNotPreserveColdOrphanWriting) {
    auto write_location_manager = std::make_shared<WriteLocationManager>();
    cache_reclaimer_ = std::make_unique<CacheReclaimer>(
        10, 100, 1, 10, 16, rm_, mim_, msm_, spe_, mr_, em_, write_location_manager, mm_);

    auto ig = InstanceGroupFactory();
    {
        auto cfg = std::make_shared<CacheConfig>();
        auto strategy = std::make_shared<MigrationStrategy>();
        strategy->set_source_storage_name("hot_01");
        strategy->set_target_storage_name("cold_01");
        cfg->set_migration_strategies({strategy});
        ig->set_cache_config(cfg);
    }
    g_cold_ig = ig;
    stub_.set(ADDR(RegistryManager, GetInstanceGroup), RegistryManager_GetInstanceGroup_cold_stub);

    const auto ins_info = InstanceInfoFactory();
    auto make_loc = [](const std::string &loc_id, const std::string &storage, CacheLocationStatus status) {
        return std::make_shared<CacheLocation>(
            loc_id,
            status,
            DataStorageType::DATA_STORAGE_TYPE_DUMMY,
            1,
            std::vector<LocationSpec>{LocationSpec("TP0", "dummy://" + storage + "/" + loc_id)});
    };

    CacheLocationMap loc_map;
    loc_map.emplace("hot_serving", make_loc("hot_serving", "hot_01", CacheLocationStatus::CLS_SERVING));
    loc_map.emplace("cold_orphan", make_loc("cold_orphan", "cold_01", CacheLocationStatus::CLS_WRITING));
    batch_get_loc_out_maps = {std::move(loc_map)};
    batch_get_loc_result = ErrorCode::EC_OK;

    CacheReclaimer::WaterLevelExceed wl;
    wl.SetGeneralWaterLevelExceed(true);

    std::vector<std::vector<std::string>> out;
    CacheReclaimer::AgeStats age_stats;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(), ins_info, {0}, wl, out, age_stats));
    ASSERT_EQ(1u, out.size());
    ASSERT_EQ(2u, out[0].size());
    std::sort(out[0].begin(), out[0].end());
    EXPECT_EQ("cold_orphan", out[0][0]);
    EXPECT_EQ("hot_serving", out[0][1]);

    stub_.reset(ADDR(RegistryManager, GetInstanceGroup));
    g_cold_ig.reset();
}

/* ---------------- increment B: migration strategy stubs ---------------- */

namespace {
CacheLocationConstPtr
MakeLocWithStatus(const std::string &id, const std::string &storage, const CacheLocationStatus status) {
    return std::make_shared<CacheLocation>(
        id,
        status,
        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
        1,
        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://" + storage + "/p_" + id)});
}

CacheLocationConstPtr MakeServingLoc(const std::string &id, const std::string &storage) {
    return MakeLocWithStatus(id, storage, CacheLocationStatus::CLS_SERVING);
}
} // namespace

// 按请求 key 构造对齐的 location map（不依赖 batch 顺序）：
//   10,20,21,22,30 -> 仅在 hot_01
//   11             -> hot_01 + cold_01(目标已有副本)
//   12             -> 仅在 other_storage(非源)
//   13             -> hot_01 + cold_01(目标已有 WRITING location)
//   40             -> 仅在 hot_02
ErrorCode MigrateTest_BatchGetLocation_stub(void *obj,
                                            RequestContext *rc,
                                            const std::vector<std::int64_t> &kv,
                                            const BlockMask &bm,
                                            std::vector<CacheLocationMap> &out_loc_maps) {
    ++batch_get_loc_call_counter;
    out_loc_maps.clear();
    for (const auto k : kv) {
        CacheLocationMap m;
        switch (k) {
        case 10:
        case 20:
        case 21:
        case 22:
        case 30:
            m.emplace("src_" + std::to_string(k), MakeServingLoc("src_" + std::to_string(k), "hot_01"));
            break;
        case 40:
            m.emplace("src_" + std::to_string(k), MakeServingLoc("src_" + std::to_string(k), "hot_02"));
            break;
        case 50:
            m.emplace("src_50", MakeServingLoc("src_50", "hot_vcns"));
            break;
        case 11:
            m.emplace("src_11", MakeServingLoc("src_11", "hot_01"));
            m.emplace("dst_11", MakeServingLoc("dst_11", "cold_01"));
            break;
        case 13:
            m.emplace("src_13", MakeServingLoc("src_13", "hot_01"));
            m.emplace("dst_13", MakeLocWithStatus("dst_13", "cold_01", CacheLocationStatus::CLS_WRITING));
            break;
        case 12:
            m.emplace("oth_12", MakeServingLoc("oth_12", "other_storage"));
            break;
        default:
            break;
        }
        out_loc_maps.emplace_back(std::move(m));
    }
    return ErrorCode::EC_OK;
}

std::vector<ErrorCode> MigrationManager_BatchSubmit_stub(void *obj,
                                                         const std::string &trace_id,
                                                         std::vector<MigrationManager::MigrationRequest> requests,
                                                         MigrationManager::CopyConcurrencyLimit copy_limit) {
    captured_copy_trace = trace_id;
    captured_copy_reqs = requests;
    captured_copy_req_batches.emplace_back(requests);
    captured_copy_limits.emplace_back(std::move(copy_limit));
    return std::vector<ErrorCode>(requests.size(), ErrorCode::EC_OK);
}

ErrorCode MigrationManager_MarkForTieredWrite_stub(void *obj,
                                                   const std::string &instance_id,
                                                   const std::vector<std::int64_t> &block_keys,
                                                   const std::string &dst_storage_name) {
    static_cast<void>(obj);
    static_cast<void>(instance_id);
    captured_mark_keys = block_keys;
    captured_mark_target = dst_storage_name;
    return ErrorCode::EC_OK;
}

static MigrationStrategy
MakeStrategy(const std::string &src, const std::string &dst, bool copy_enabled, bool mark_enabled) {
    MigrationStrategy strategy;
    strategy.set_source_storage_name(src);
    strategy.set_target_storage_name(dst);
    strategy.set_trigger_threshold(0.5);
    MigrationMethods methods;
    methods.set_copy(MigrationCopyMethod(copy_enabled));
    methods.set_mark(MigrationMarkMethod(mark_enabled));
    strategy.set_methods(methods);
    strategy.set_retention(MigrationRetention::MIGRATION_RETENTION_DELETE_SOURCE);
    return strategy;
}

TEST_F(CacheReclaimerTest, TestMigrateByStrategyOnBatch) {
    stub_.set(ADDR(MigrationManager, BatchSubmit), MigrationManager_BatchSubmit_stub);
    stub_.set(ADDR(MigrationManager, MarkForTieredWrite), MigrationManager_MarkForTieredWrite_stub);

    cache_reclaimer_->job_state_flag_ = true; // 让 IsRunning() 为真，无需启动 cron 线程

    const auto ins_info = InstanceInfoFactory();
    auto make_loc_maps = [&](const std::vector<std::int64_t> &batch) {
        std::vector<CacheLocationMap> loc_maps;
        const BlockMask blk_mask(std::in_place_type<BlockMaskVector>, batch.size(), false);
        const auto ec = MigrateTest_BatchGetLocation_stub(nullptr, request_context_.get(), batch, blk_mask, loc_maps);
        EXPECT_EQ(ErrorCode::EC_OK, ec);
        EXPECT_EQ(batch.size(), loc_maps.size());
        return loc_maps;
    };

    // ---- 场景 A：准入过滤（仅 copy）----
    {
        captured_copy_reqs.clear();
        captured_copy_req_batches.clear();
        const std::vector<std::int64_t> batch = {10, 11, 12, 13};
        const auto loc_maps = make_loc_maps(batch);

        const auto strategy = MakeStrategy("hot_01", "cold_01", /*copy*/ true, /*mark*/ false);
        const std::size_t submitted = cache_reclaimer_->MigrateByStrategyOnBatch(
            request_context_, ins_info, strategy, /*available_copy_slots*/ 5, batch, loc_maps);

        // 仅 block 10 合格：11 目标已有 SERVING 副本，12 不在源 storage 上，13 目标已有 WRITING location
        ASSERT_EQ(1u, submitted);
        ASSERT_EQ(1u, captured_copy_reqs.size());
        ASSERT_EQ(10, captured_copy_reqs[0].block_key);
        ASSERT_EQ("src_10", captured_copy_reqs[0].src_location_id);
        ASSERT_EQ("hot_01", captured_copy_reqs[0].src_storage_name);
        ASSERT_EQ("cold_01", captured_copy_reqs[0].dst_storage_name);
    }

    // ---- 场景 B：并发预算上限 ----
    {
        captured_copy_reqs.clear();
        captured_copy_req_batches.clear();
        const std::vector<std::int64_t> batch = {20, 21, 22};
        const auto loc_maps = make_loc_maps(batch);

        const auto strategy = MakeStrategy("hot_01", "cold_01", /*copy*/ true, /*mark*/ false);
        const std::size_t submitted = cache_reclaimer_->MigrateByStrategyOnBatch(
            request_context_, ins_info, strategy, /*available_copy_slots*/ 2, batch, loc_maps);

        // 3 个合格候选，预算 2 -> 只提交 2 个
        ASSERT_EQ(2u, submitted);
        ASSERT_EQ(2u, captured_copy_reqs.size());
        for (const auto &req : captured_copy_reqs) {
            ASSERT_TRUE(req.block_key == 20 || req.block_key == 21 || req.block_key == 22);
            ASSERT_EQ("hot_01", req.src_storage_name);
        }
    }

    // ---- 场景 C：mark 路径 ----
    {
        captured_copy_reqs.clear();
        captured_copy_req_batches.clear();
        captured_mark_keys.clear();
        const std::vector<std::int64_t> batch = {30};
        const auto loc_maps = make_loc_maps(batch);

        const auto strategy = MakeStrategy("hot_01", "cold_01", /*copy*/ false, /*mark*/ true);
        const std::size_t submitted = cache_reclaimer_->MigrateByStrategyOnBatch(
            request_context_, ins_info, strategy, /*available_copy_slots*/ 0, batch, loc_maps);

        ASSERT_EQ(0u, submitted); // 无 copy
        ASSERT_TRUE(captured_copy_reqs.empty());
        ASSERT_EQ(1u, captured_mark_keys.size());
        ASSERT_EQ(30, captured_mark_keys[0]);
        ASSERT_EQ("cold_01", captured_mark_target);
    }

    // ---- 场景 D：copy + mark 互斥分发，copy 优先；copy 槽位外才 mark ----
    {
        captured_copy_reqs.clear();
        captured_copy_req_batches.clear();
        captured_mark_keys.clear();
        captured_mark_target.clear();
        const std::vector<std::int64_t> batch = {20, 21, 22};
        const auto loc_maps = make_loc_maps(batch);

        const auto strategy = MakeStrategy("hot_01", "cold_01", /*copy*/ true, /*mark*/ true);
        const std::size_t submitted = cache_reclaimer_->MigrateByStrategyOnBatch(
            request_context_, ins_info, strategy, /*available_copy_slots*/ 2, batch, loc_maps);

        ASSERT_EQ(2u, submitted);
        ASSERT_EQ(2u, captured_copy_reqs.size());
        ASSERT_EQ(1u, captured_mark_keys.size());
        std::vector<std::int64_t> copied_keys;
        for (const auto &req : captured_copy_reqs) {
            copied_keys.push_back(req.block_key);
            ASSERT_FALSE(VecContains(captured_mark_keys, req.block_key));
        }
        ASSERT_TRUE(VecContains(copied_keys, 20) || VecContains(captured_mark_keys, 20));
        ASSERT_TRUE(VecContains(copied_keys, 21) || VecContains(captured_mark_keys, 21));
        ASSERT_TRUE(VecContains(copied_keys, 22) || VecContains(captured_mark_keys, 22));
        ASSERT_EQ("cold_01", captured_mark_target);
    }

    stub_.reset(ADDR(MigrationManager, BatchSubmit));
    stub_.reset(ADDR(MigrationManager, MarkForTieredWrite));
}

TEST_F(CacheReclaimerTest, TestTryMigrateOnGroupReusesSampledBatchAcrossStrategies) {
    stub_.set(ADDR(MetaSearcher, BatchGetLocation), MigrateTest_BatchGetLocation_stub);
    stub_.set(ADDR(MigrationManager, BatchSubmit), MigrationManager_BatchSubmit_stub);

    cache_reclaimer_->job_state_flag_ = true;
    rm_->data_storage_manager_ = dsm_;

    auto register_nfs_storage = [this](const std::string &name) {
        auto spec = std::make_shared<NfsStorageSpec>();
        spec->set_root_path(GetPrivateTestRuntimeDataPath() + name + "/");
        spec->set_key_count_per_file(1);
        StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, name, spec);
        ASSERT_EQ(ErrorCode::EC_OK, dsm_->RegisterStorage(request_context_.get(), name, storage_config));
    };
    register_nfs_storage("hot_01");
    register_nfs_storage("hot_02");

    auto ins_group = InstanceGroupFactory();
    {
        QuotaConfig quota_config;
        quota_config.set_capacity(1000);
        quota_config.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
        InstanceGroupQuota quota;
        quota.set_capacity(1000);
        quota.set_quota_config({quota_config});
        ins_group->set_quota(quota);

        auto cfg = std::make_shared<CacheConfig>();
        cfg->set_reclaim_strategy(ins_group->cache_config()->reclaim_strategy());
        cfg->set_meta_indexer_config(ins_group->cache_config()->meta_indexer_config());
        cfg->set_migration_copy_max_concurrency(2);
        cfg->set_migration_strategies({
            std::make_shared<MigrationStrategy>(MakeStrategy("hot_01", "cold_01", /*copy*/ true, /*mark*/ false)),
            std::make_shared<MigrationStrategy>(MakeStrategy("hot_02", "cold_02", /*copy*/ true, /*mark*/ false)),
        });
        ins_group->set_cache_config(cfg);
        // MigrateByStrategyOnBatch 从 RegistryManager 读取同一份 group 并发配置。
        rm_->instance_group_configs_[ins_group->name()] = ins_group;
    }

    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 800);
    key_count = 2;
    max_key_count = 16;
    sample_reclaim_keys = {10, 40};
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), 2));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 2));

    captured_copy_reqs.clear();
    captured_copy_req_batches.clear();
    captured_copy_limits.clear();
    cache_reclaimer_->TryMigrateOnGroup(request_context_, ins_group, {InstanceInfoFactory()});

    ASSERT_EQ(1, sample_reclaim_call_counter);
    ASSERT_EQ(1, batch_get_loc_call_counter);
    ASSERT_EQ(2u, captured_copy_req_batches.size());
    ASSERT_EQ(1u, captured_copy_req_batches[0].size());
    ASSERT_EQ(10, captured_copy_req_batches[0][0].block_key);
    ASSERT_EQ("hot_01", captured_copy_req_batches[0][0].src_storage_name);
    ASSERT_EQ(1u, captured_copy_req_batches[1].size());
    ASSERT_EQ(40, captured_copy_req_batches[1][0].block_key);
    ASSERT_EQ("hot_02", captured_copy_req_batches[1][0].src_storage_name);
    ASSERT_EQ(2u, captured_copy_limits.size());
    for (const auto &limit : captured_copy_limits) {
        ASSERT_EQ(ins_group->name(), limit.instance_group_name);
        ASSERT_EQ(2u, limit.max_concurrency);
    }

    stub_.reset(ADDR(MetaSearcher, BatchGetLocation));
    stub_.reset(ADDR(MigrationManager, BatchSubmit));
}

// 水位低于 trigger_threshold 时 TryMigrateOnGroup 不下发任何 copy/mark。
TEST_F(CacheReclaimerTest, TestTryMigrateOnGroupSkipsBelowThreshold) {
    stub_.set(ADDR(MetaSearcher, BatchGetLocation), MigrateTest_BatchGetLocation_stub);
    stub_.set(ADDR(MigrationManager, BatchSubmit), MigrationManager_BatchSubmit_stub);
    stub_.set(ADDR(MigrationManager, MarkForTieredWrite), MigrationManager_MarkForTieredWrite_stub);

    cache_reclaimer_->job_state_flag_ = true;
    rm_->data_storage_manager_ = dsm_;

    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_root_path(GetPrivateTestRuntimeDataPath() + "hot_01/");
    spec->set_key_count_per_file(1);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "hot_01", spec);
    ASSERT_EQ(ErrorCode::EC_OK, dsm_->RegisterStorage(request_context_.get(), "hot_01", storage_config));

    auto ins_group = InstanceGroupFactory();
    {
        QuotaConfig quota_config;
        quota_config.set_capacity(1000);
        quota_config.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_NFS);
        InstanceGroupQuota quota;
        quota.set_capacity(1000);
        quota.set_quota_config({quota_config});
        ins_group->set_quota(quota);

        auto cfg = std::make_shared<CacheConfig>();
        cfg->set_reclaim_strategy(ins_group->cache_config()->reclaim_strategy());
        cfg->set_meta_indexer_config(ins_group->cache_config()->meta_indexer_config());
        cfg->set_migration_strategies({
            std::make_shared<MigrationStrategy>(MakeStrategy("hot_01", "cold_01", /*copy*/ true, /*mark*/ true)),
        });
        ins_group->set_cache_config(cfg);
    }

    // 用量 200 / 容量 1000 = 0.2 < threshold 0.5 -> 水位门控早返回。
    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 200);
    sample_reclaim_keys = {10};
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), 1));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));

    captured_copy_reqs.clear();
    captured_copy_req_batches.clear();
    captured_mark_keys.clear();
    cache_reclaimer_->TryMigrateOnGroup(request_context_, ins_group, {InstanceInfoFactory()});

    ASSERT_TRUE(captured_copy_reqs.empty());
    ASSERT_TRUE(captured_copy_req_batches.empty());
    ASSERT_TRUE(captured_mark_keys.empty());
    ASSERT_EQ(0, sample_reclaim_call_counter); // 未触发采样
    ASSERT_EQ(0, batch_get_loc_call_counter);

    stub_.reset(ADDR(MetaSearcher, BatchGetLocation));
    stub_.reset(ADDR(MigrationManager, BatchSubmit));
    stub_.reset(ADDR(MigrationManager, MarkForTieredWrite));
}

// quota 中不含源 storage 对应 type 时（capacity<=0），TryMigrateOnGroup 跳过该 strategy。
TEST_F(CacheReclaimerTest, TestTryMigrateOnGroupSkipsWhenNoCapacity) {
    stub_.set(ADDR(MetaSearcher, BatchGetLocation), MigrateTest_BatchGetLocation_stub);
    stub_.set(ADDR(MigrationManager, BatchSubmit), MigrationManager_BatchSubmit_stub);
    stub_.set(ADDR(MigrationManager, MarkForTieredWrite), MigrationManager_MarkForTieredWrite_stub);

    cache_reclaimer_->job_state_flag_ = true;
    rm_->data_storage_manager_ = dsm_;

    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_root_path(GetPrivateTestRuntimeDataPath() + "hot_01/");
    spec->set_key_count_per_file(1);
    StorageConfig storage_config(DataStorageType::DATA_STORAGE_TYPE_NFS, "hot_01", spec);
    ASSERT_EQ(ErrorCode::EC_OK, dsm_->RegisterStorage(request_context_.get(), "hot_01", storage_config));

    auto ins_group = InstanceGroupFactory();
    {
        // quota 不为 NFS 类型设置 capacity -> capacity=0 -> 早返回。
        InstanceGroupQuota quota;
        quota.set_capacity(0);
        quota.set_quota_config({});
        ins_group->set_quota(quota);

        auto cfg = std::make_shared<CacheConfig>();
        cfg->set_reclaim_strategy(ins_group->cache_config()->reclaim_strategy());
        cfg->set_meta_indexer_config(ins_group->cache_config()->meta_indexer_config());
        cfg->set_migration_strategies({
            std::make_shared<MigrationStrategy>(MakeStrategy("hot_01", "cold_01", /*copy*/ true, /*mark*/ true)),
        });
        ins_group->set_cache_config(cfg);
    }

    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_NFS, 800);
    sample_reclaim_keys = {10};
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), 1));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));

    captured_copy_reqs.clear();
    captured_copy_req_batches.clear();
    captured_mark_keys.clear();
    cache_reclaimer_->TryMigrateOnGroup(request_context_, ins_group, {InstanceInfoFactory()});

    ASSERT_TRUE(captured_copy_reqs.empty());
    ASSERT_TRUE(captured_copy_req_batches.empty());
    ASSERT_TRUE(captured_mark_keys.empty());
    ASSERT_EQ(0, sample_reclaim_call_counter);
    ASSERT_EQ(0, batch_get_loc_call_counter);

    stub_.reset(ADDR(MetaSearcher, BatchGetLocation));
    stub_.reset(ADDR(MigrationManager, BatchSubmit));
    stub_.reset(ADDR(MigrationManager, MarkForTieredWrite));
}

// VCNS_HF3FS 在水位/用量统计上归并到 HF3FS；迁移策略查 source type quota 时也应一致归并。
TEST_F(CacheReclaimerTest, TestTryMigrateOnGroupNormalizesVcnsSourceTypeForQuota) {
    stub_.set(ADDR(MetaSearcher, BatchGetLocation), MigrateTest_BatchGetLocation_stub);
    stub_.set(ADDR(MigrationManager, BatchSubmit), MigrationManager_BatchSubmit_stub);

    cache_reclaimer_->job_state_flag_ = true;
    rm_->data_storage_manager_ = dsm_;

    dsm_->storage_map_["hot_vcns"] =
        std::make_shared<TypedTestBackend>(mr_, DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS);

    auto ins_group = InstanceGroupFactory();
    {
        QuotaConfig quota_config;
        quota_config.set_capacity(1000);
        quota_config.set_storage_type(DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        InstanceGroupQuota quota;
        quota.set_capacity(1000);
        quota.set_quota_config({quota_config});
        ins_group->set_quota(quota);

        auto cfg = std::make_shared<CacheConfig>();
        cfg->set_reclaim_strategy(ins_group->cache_config()->reclaim_strategy());
        cfg->set_meta_indexer_config(ins_group->cache_config()->meta_indexer_config());
        cfg->set_migration_strategies({
            std::make_shared<MigrationStrategy>(MakeStrategy("hot_vcns", "cold_01", /*copy*/ true, /*mark*/ false)),
        });
        ins_group->set_cache_config(cfg);
    }

    dummy_meta_indexer->AddStorageUsageByType(DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS, 800);
    sample_reclaim_keys = {50};
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetSamplingSize(request_context_.get(), 1));
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));

    captured_copy_reqs.clear();
    captured_copy_req_batches.clear();
    cache_reclaimer_->TryMigrateOnGroup(request_context_, ins_group, {InstanceInfoFactory()});

    ASSERT_EQ(1, sample_reclaim_call_counter);
    ASSERT_EQ(1, batch_get_loc_call_counter);
    ASSERT_EQ(1u, captured_copy_reqs.size());
    ASSERT_EQ(50, captured_copy_reqs[0].block_key);
    ASSERT_EQ("hot_vcns", captured_copy_reqs[0].src_storage_name);

    stub_.reset(ADDR(MetaSearcher, BatchGetLocation));
    stub_.reset(ADDR(MigrationManager, BatchSubmit));
}

// F-23: BOTH 方法下 copy 提交部分失败时，失败的 block 应回落到 mark（与 Admin BOTH 对齐）。
// 用一个让第 2 个 copy request 失败的 stub 验证回落行为。
namespace {
std::vector<ErrorCode> BatchSubmit_PartialFail_stub(void * /*obj*/,
                                                    const std::string &trace_id,
                                                    std::vector<MigrationManager::MigrationRequest> requests,
                                                    MigrationManager::CopyConcurrencyLimit /*copy_limit*/) {
    captured_copy_trace = trace_id;
    captured_copy_reqs = requests;
    captured_copy_req_batches.emplace_back(requests);
    std::vector<ErrorCode> results(requests.size(), ErrorCode::EC_OK);
    // 让第 2 个(index 1)失败
    if (results.size() > 1) {
        results[1] = ErrorCode::EC_ERROR;
    }
    return results;
}
} // namespace

TEST_F(CacheReclaimerTest, TestMigrateByStrategyBothCopyFailFallbackMark) {
    cache_reclaimer_->job_state_flag_ = true;
    const auto ins_info = InstanceInfoFactory();
    captured_copy_reqs.clear();
    captured_copy_req_batches.clear();
    captured_mark_keys.clear();
    captured_mark_target.clear();

    stub_.set(ADDR(MetaSearcher, BatchGetLocation), MigrateTest_BatchGetLocation_stub);
    // 用部分失败的 BatchSubmit stub
    stub_.set(ADDR(MigrationManager, BatchSubmit), BatchSubmit_PartialFail_stub);
    stub_.set(ADDR(MigrationManager, MarkForTieredWrite), MigrationManager_MarkForTieredWrite_stub);

    // batch = {20, 21, 22}：全部合格(loc_maps 里有源)；slots 足够(5)
    const std::vector<std::int64_t> batch = {20, 21, 22};
    std::vector<CacheLocationMap> loc_maps;
    {
        const BlockMask blk_mask(std::in_place_type<BlockMaskVector>, batch.size(), false);
        ASSERT_EQ(ErrorCode::EC_OK,
                  MigrateTest_BatchGetLocation_stub(nullptr, request_context_.get(), batch, blk_mask, loc_maps));
        ASSERT_EQ(batch.size(), loc_maps.size());
    }

    const auto strategy = MakeStrategy("hot_01", "cold_01", /*copy*/ true, /*mark*/ true);
    const std::size_t submitted = cache_reclaimer_->MigrateByStrategyOnBatch(
        request_context_, ins_info, strategy, /*available_copy_slots*/ 5, batch, loc_maps);

    // 3 个都进 copy；BatchSubmit 返回 [OK, ERROR, OK]
    ASSERT_EQ(3u, captured_copy_reqs.size());
    // submitted_copy = 2 (index 0 和 2 成功)
    ASSERT_EQ(2u, submitted);
    // 失败的 index 1 (block_key=21) 应回落到 mark
    ASSERT_EQ(1u, captured_mark_keys.size());
    ASSERT_EQ(21, captured_mark_keys[0]);
    ASSERT_EQ("cold_01", captured_mark_target);

    stub_.reset(ADDR(MetaSearcher, BatchGetLocation));
    stub_.reset(ADDR(MigrationManager, BatchSubmit));
    stub_.reset(ADDR(MigrationManager, MarkForTieredWrite));
}

// F-26: multi-cold union — 多个 cold SERVING location 联合覆盖 hot 的多 specs → 允许删 hot。
TEST_F(CacheReclaimerTest, TestFilterLocIDMultiColdUnionCoversHot) {
    auto ig = InstanceGroupFactory();
    {
        auto cfg = std::make_shared<CacheConfig>();
        auto strategy = std::make_shared<MigrationStrategy>();
        strategy->set_source_storage_name("hot_01");
        strategy->set_target_storage_name("cold_01");
        cfg->set_migration_strategies({strategy});
        ig->set_cache_config(cfg);
    }
    g_cold_ig = ig;
    stub_.set(ADDR(RegistryManager, GetInstanceGroup), RegistryManager_GetInstanceGroup_cold_stub);
    const auto ins_info = InstanceInfoFactory();

    auto hot_loc = std::make_shared<CacheLocation>("hot_full",
                                                   CacheLocationStatus::CLS_SERVING,
                                                   DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                                   2,
                                                   std::vector<LocationSpec>{
                                                       LocationSpec("TP0", "dummy://hot_01/hot_full/tp0"),
                                                       LocationSpec("TP1", "dummy://hot_01/hot_full/tp1"),
                                                   });
    auto cold_a =
        std::make_shared<CacheLocation>("cold_a",
                                        CacheLocationStatus::CLS_SERVING,
                                        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold_01/cold_a/tp0")});
    auto cold_b =
        std::make_shared<CacheLocation>("cold_b",
                                        CacheLocationStatus::CLS_SERVING,
                                        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("TP1", "dummy://cold_01/cold_b/tp1")});

    CacheLocationMap loc_map;
    loc_map.emplace("hot_full", hot_loc);
    loc_map.emplace("cold_a", cold_a);
    loc_map.emplace("cold_b", cold_b);
    batch_get_loc_out_maps = {std::move(loc_map)};
    batch_get_loc_result = ErrorCode::EC_OK;

    CacheReclaimer::WaterLevelExceed wl;
    wl.SetGeneralWaterLevelExceed(true);

    std::vector<std::vector<std::string>> out;
    CacheReclaimer::AgeStats age_stats;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(), ins_info, {0}, wl, out, age_stats));
    ASSERT_EQ(1u, out.size());
    ASSERT_EQ(1u, out[0].size()) << "hot should be evicted when cold union covers all specs";
    ASSERT_EQ("hot_full", out[0][0]);

    stub_.reset(ADDR(RegistryManager, GetInstanceGroup));
    g_cold_ig.reset();
}

// F-26: storage-type eviction zone 下不保冷 — 硬驱逐时 keep-cold-evict-hot 不生效。
TEST_F(CacheReclaimerTest, TestFilterLocIDStorageTypeEvictionIgnoresCold) {
    auto ig = InstanceGroupFactory();
    {
        auto cfg = std::make_shared<CacheConfig>();
        auto strategy = std::make_shared<MigrationStrategy>();
        strategy->set_source_storage_name("hot_01");
        strategy->set_target_storage_name("cold_01");
        cfg->set_migration_strategies({strategy});
        ig->set_cache_config(cfg);
    }
    g_cold_ig = ig;
    stub_.set(ADDR(RegistryManager, GetInstanceGroup), RegistryManager_GetInstanceGroup_cold_stub);
    const auto ins_info = InstanceInfoFactory();

    auto make_loc = [](const std::string &loc_id, const std::string &storage) {
        return std::make_shared<CacheLocation>(
            loc_id,
            CacheLocationStatus::CLS_SERVING,
            DataStorageType::DATA_STORAGE_TYPE_DUMMY,
            1,
            std::vector<LocationSpec>{LocationSpec("TP0", "dummy://" + storage + "/" + loc_id)});
    };
    CacheLocationMap m;
    m.emplace("hot_a", make_loc("hot_a", "hot_01"));
    m.emplace("cold_a", make_loc("cold_a", "cold_01"));
    batch_get_loc_out_maps = {std::move(m)};
    batch_get_loc_result = ErrorCode::EC_OK;

    CacheReclaimer::WaterLevelExceed wl;
    wl.SetWaterLevelExceedByType(DataStorageType::DATA_STORAGE_TYPE_DUMMY, true);
    ASSERT_TRUE(wl.CheckStorageTypeWaterLevelExceed());

    std::vector<std::vector<std::string>> out;
    CacheReclaimer::AgeStats age_stats;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(), ins_info, {0}, wl, out, age_stats));
    ASSERT_EQ(1u, out.size());
    ASSERT_EQ(2u, out[0].size()) << "storage-type eviction should not preserve cold";

    stub_.reset(ADDR(RegistryManager, GetInstanceGroup));
    g_cold_ig.reset();
}

// F-26: WRITING 状态的冷副本不算覆盖——迁移中的半成品不能作为删热依据。
TEST_F(CacheReclaimerTest, TestFilterLocIDWritingColdDoesNotProtectHot) {
    auto ig = InstanceGroupFactory();
    {
        auto cfg = std::make_shared<CacheConfig>();
        auto strategy = std::make_shared<MigrationStrategy>();
        strategy->set_source_storage_name("hot_01");
        strategy->set_target_storage_name("cold_01");
        cfg->set_migration_strategies({strategy});
        ig->set_cache_config(cfg);
    }
    g_cold_ig = ig;
    stub_.set(ADDR(RegistryManager, GetInstanceGroup), RegistryManager_GetInstanceGroup_cold_stub);
    const auto ins_info = InstanceInfoFactory();

    // hot: SERVING, cold: WRITING(迁移中半成品)
    auto hot_loc =
        std::make_shared<CacheLocation>("hot_a",
                                        CacheLocationStatus::CLS_SERVING,
                                        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://hot_01/hot_a")});
    auto cold_writing =
        std::make_shared<CacheLocation>("cold_w",
                                        CacheLocationStatus::CLS_WRITING,
                                        DataStorageType::DATA_STORAGE_TYPE_DUMMY,
                                        1,
                                        std::vector<LocationSpec>{LocationSpec("TP0", "dummy://cold_01/cold_w")});

    CacheLocationMap loc_map;
    loc_map.emplace("hot_a", hot_loc);
    loc_map.emplace("cold_w", cold_writing);
    batch_get_loc_out_maps = {std::move(loc_map)};
    batch_get_loc_result = ErrorCode::EC_OK;

    CacheReclaimer::WaterLevelExceed wl;
    wl.SetGeneralWaterLevelExceed(true);

    std::vector<std::vector<std::string>> out;
    CacheReclaimer::AgeStats age_stats;
    ASSERT_TRUE(cache_reclaimer_->FilterLocID(request_context_.get(), ins_info, {0}, wl, out, age_stats));
    ASSERT_EQ(1u, out.size());
    // WRITING cold 不算覆盖 → has_cold=false → keep_cold_evict_hot=false → hot 正常被选
    ASSERT_EQ(1u, out[0].size());
    ASSERT_EQ("hot_a", out[0][0]);

    stub_.reset(ADDR(RegistryManager, GetInstanceGroup));
    g_cold_ig.reset();
}
