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
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/quota_config.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/config/trigger_strategy.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/manager/cache_location.h"
#include "kv_cache_manager/manager/cache_reclaimer.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"

using namespace kv_cache_manager;

bool VecContains(const std::vector<std::int64_t> &vec, const std::int64_t v) {
    return std::any_of(vec.cbegin(), vec.cend(), [v](const std::int64_t &e) { return e == v; });
}

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

/* ---------------- SchedulePlanExecutor_Submit_stub ---------------- */
PlanExecuteResult del_result;
std::vector<CacheLocationDelRequest> submitted_del_requests;
// spe_submit_loc is used to help casting the func addr in the stub def below
// the using declarations is the same as
// typedef std::future<PlanExecuteResult> (SchedulePlanExecutor::*spe_submit_loc)(const CacheLocationDelRequest &);
using spe_submit_loc = std::future<PlanExecuteResult> (SchedulePlanExecutor::*)(const CacheLocationDelRequest &);

std::future<PlanExecuteResult> SchedulePlanExecutor_Submit_stub(void *obj, const CacheLocationDelRequest &request) {
    const auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    submitted_del_requests.emplace_back(request);
    promise->set_value(del_result);
    return promise->get_future();
}

/* ---------------- MetaIndexerManager_GetMetaIndexer_stub ---------------- */

// the dummy meta_indexer that the meta_indexer_manager would return
std::shared_ptr<MetaIndexer> dummy_meta_indexer;

std::shared_ptr<MetaIndexer> MetaIndexerManager_GetMetaIndexer_stub(void *obj, const std::string &i) {
    return dummy_meta_indexer;
}

/* ---------------- MetaIndexer_GetProperties_stub ---------------- */

ErrorCode get_result;
MetaIndexer::PropertyMapVector get_out_properties;

MetaIndexer::Result MetaIndexer_GetProperties_stub(void *obj,
                                                   RequestContext *rc,
                                                   const MetaIndexer::KeyVector &k,
                                                   const std::vector<std::string> &p,
                                                   MetaIndexer::PropertyMapVector &out_properties) noexcept {
    if (get_result == ErrorCode::EC_OK) {
        out_properties = get_out_properties;
    }
    return {get_result};
}

/* ---------------- MetaIndexer_RandomSample_stub ---------------- */

ErrorCode random_sample_result;
MetaIndexer::KeyVector random_sample_keys;

ErrorCode MetaIndexer_RandomSample_stub(void *obj, const std::size_t c, MetaIndexer::KeyVector &out_keys) noexcept {
    if (random_sample_result == ErrorCode::EC_OK) {
        out_keys = random_sample_keys;
    }
    return random_sample_result;
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

ErrorCode batch_get_loc_result;
std::vector<CacheLocationMap> batch_get_loc_out_maps;

ErrorCode MetaSearcher_BatchGetLocation_stub(void *obj,
                                             RequestContext *rc,
                                             const std::vector<std::int64_t> &kv,
                                             const BlockMask &bm,
                                             std::vector<CacheLocationMap> &out_loc_maps) {
    if (batch_get_loc_result == ErrorCode::EC_OK) {
        out_loc_maps = batch_get_loc_out_maps;
    }
    return batch_get_loc_result;
}

class CacheReclaimerTest : public TESTBASE {
public:
    void SetUp() override {
        // set up stubs
        stub_.set(ADDR(RegistryManager, ListInstanceGroup), RegistryManager_ListInstanceGroup_stub);
        stub_.set(ADDR(RegistryManager, ListInstanceInfo), RegistryManager_ListInstanceInfo_stub);
        stub_.set(static_cast<spe_submit_loc>(ADDR(SchedulePlanExecutor, Submit)), SchedulePlanExecutor_Submit_stub);
        stub_.set(ADDR(MetaIndexerManager, GetMetaIndexer), MetaIndexerManager_GetMetaIndexer_stub);
        stub_.set(ADDR(MetaIndexer, GetProperties), MetaIndexer_GetProperties_stub);
        stub_.set(ADDR(MetaIndexer, RandomSample), MetaIndexer_RandomSample_stub);
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
        get_result = ErrorCode::EC_OK;
        random_sample_result = ErrorCode::EC_OK;
        batch_get_loc_result = ErrorCode::EC_OK;

        key_count = 1;
        max_key_count = 16;

        request_context_ = std::make_shared<RequestContext>("cache_reclaimer_test_trace");

        // set up our target being tested
        mr_ = std::make_shared<MetricsRegistry>();
        em_ = std::make_shared<EventManager>();
        rm_ = std::make_shared<RegistryManager>("", mr_);
        mim_ = std::make_shared<MetaIndexerManager>();
        msm_ = std::make_shared<MetaSearcherManager>(rm_, mim_);
        dsm_ = std::make_shared<DataStorageManager>(mr_);
        spe_ = std::make_shared<SchedulePlanExecutor>(0, mim_, dsm_, mr_);

        cache_reclaimer_ = std::make_unique<CacheReclaimer>(rm_, mim_, msm_, spe_, mr_, em_);
        cache_reclaimer_->SetSleepIntervalMs(request_context_.get(), 10);

        // avoid nullptr issue when testing methods that involve metrics
        // counter but no need to start the working thread
        cache_reclaimer_->METRICS_(cache_reclaimer, block_submit_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, block_submit_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, location_submit_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, location_submit_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, block_del_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, block_del_count));
        cache_reclaimer_->METRICS_(cache_reclaimer, location_del_count) =
            mr_->GetCounter(SCOPED_METRICS_NAME_(CacheReclaimer, cache_reclaimer, location_del_count));
    }

    void TearDown() override {
        cache_reclaimer_->Stop();

        instance_groups.clear();
        instance_infos.clear();

        dummy_meta_indexer.reset();
        dummy_meta_searcher.reset();

        submitted_del_requests.clear();

        get_out_properties.clear();
        random_sample_keys.clear();
        batch_get_loc_out_maps.clear();

        stub_.reset(ADDR(RegistryManager, ListInstanceGroup));
        stub_.reset(ADDR(RegistryManager, ListInstanceInfo));
        stub_.reset(static_cast<spe_submit_loc>(ADDR(SchedulePlanExecutor, Submit)));
        stub_.reset(ADDR(MetaIndexerManager, GetMetaIndexer));
        stub_.reset(ADDR(MetaIndexer, GetProperties));
        stub_.reset(ADDR(MetaIndexer, RandomSample));
        stub_.reset(ADDR(MetaIndexer, GetKeyCount));
        stub_.reset(ADDR(MetaIndexer, GetMaxKeyCount));
        stub_.reset(ADDR(MetaIndexer, PersistMetaData));
        stub_.reset(ADDR(MetaSearcherManager, GetMetaSearcher));
        stub_.reset(ADDR(MetaSearcher, BatchGetLocation));
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
    std::shared_ptr<RequestContext> request_context_;
};

TEST_F(CacheReclaimerTest, TestStartStop) {
    stub_.reset(ADDR(RegistryManager, ListInstanceGroup));
    stub_.reset(ADDR(RegistryManager, ListInstanceInfo));
    stub_.reset(static_cast<spe_submit_loc>(ADDR(SchedulePlanExecutor, Submit)));
    stub_.reset(ADDR(MetaIndexerManager, GetMetaIndexer));
    stub_.reset(ADDR(MetaIndexer, GetProperties));
    stub_.reset(ADDR(MetaIndexer, RandomSample));
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
        cache_reclaimer_ = std::make_unique<CacheReclaimer>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer_->Start());
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    }

    {
        // test the case that RegisterManager is nullptr
        cache_reclaimer_ = std::make_unique<CacheReclaimer>(nullptr, mim_, msm_, spe_, mr_, em_);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer_->Start());
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    }

    {
        // test the case that MetaIndexerManager is nullptr
        cache_reclaimer_ = std::make_unique<CacheReclaimer>(rm_, nullptr, msm_, spe_, mr_, em_);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer_->Start());
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    }

    {
        // test the case that MetaSearcherManager is nullptr
        cache_reclaimer_ = std::make_unique<CacheReclaimer>(rm_, mim_, nullptr, spe_, mr_, em_);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer_->Start());
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    }

    {
        // test the case that SchedulePlanExecutor is nullptr
        cache_reclaimer_ = std::make_unique<CacheReclaimer>(rm_, mim_, msm_, nullptr, mr_, em_);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer_->Start());
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    }

    {
        // test the case that MetricsRegistry is nullptr
        cache_reclaimer_ = std::make_unique<CacheReclaimer>(rm_, mim_, msm_, spe_, nullptr, em_);
        ASSERT_EQ(ErrorCode::EC_ERROR, cache_reclaimer_->Start());
        ASSERT_FALSE(cache_reclaimer_->IsRunning());
        ASSERT_FALSE(cache_reclaimer_->reclaimer_.joinable());
    }

    {
        // test the case that MetricsRegistry is nullptr
        cache_reclaimer_ = std::make_unique<CacheReclaimer>(rm_, mim_, msm_, spe_, mr_, nullptr);
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());
        ASSERT_TRUE(cache_reclaimer_->IsRunning());
        ASSERT_TRUE(cache_reclaimer_->reclaimer_.joinable());
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
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 2));
    batch_get_loc_out_maps =
        std::vector<CacheLocationMap>(cache_reclaimer_->GetBatchingSize(request_context_.get()), CacheLocationMap{});

    {
        cache_reclaimer_->Pause();
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
        ASSERT_TRUE(cache_reclaimer_->IsPaused());  // the worker thread is in paused state
        ASSERT_TRUE(submitted_del_requests.empty());
    }

    {
        cache_reclaimer_->Resume();
        ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running
        ASSERT_FALSE(cache_reclaimer_->IsPaused()); // the worker thread is not in paused state

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        ASSERT_FALSE(submitted_del_requests.empty());
        const auto &req = submitted_del_requests.back();
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
    cache_reclaimer_ = std::make_unique<CacheReclaimer>(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);

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
    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_size(16);
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

TEST_F(CacheReclaimerTest, TestTriggerReclaiming00) {
    // instance 0 block byte size = 1024, key count = 1
    // 1024 * 1 > 16
    // should *not* trigger reclaiming by the used_size strategy

    // use instance 0 from setup()

    const auto ins_group = InstanceGroupFactory();
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_size(16);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                       ins_group->name(),
                                                       ins_group->quota(),
                                                       ins_group->cache_config()->reclaim_strategy(),
                                                       instance_infos,
                                                       water_level_exceed_results));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming01) {
    // instance 0 block byte size = 1024, key count = 1
    // 1024 * 1 == 1024
    // should *not* trigger reclaiming by the used_size strategy

    // use instance 0 from setup()

    const auto ins_group = InstanceGroupFactory();
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_size(1024);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                       ins_group->name(),
                                                       ins_group->quota(),
                                                       ins_group->cache_config()->reclaim_strategy(),
                                                       instance_infos,
                                                       water_level_exceed_results));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming02) {
    // instance 0 block byte size = 1024, key count = 1
    // 1024 * 1 < 1025
    // should *not* trigger reclaiming by the used_size strategy

    // use instance 0 from setup()

    const auto ins_group = InstanceGroupFactory();
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_size(1025);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                       ins_group->name(),
                                                       ins_group->quota(),
                                                       ins_group->cache_config()->reclaim_strategy(),
                                                       instance_infos,
                                                       water_level_exceed_results));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming03) {
    // test multiple instances
    // instance 0 block byte size = 1024, key count = 1
    // instance 1 block byte size = 256, key count = 1
    // 1024 * 1 + 256 * 1 > 1025
    // should *not* trigger reclaiming by the used_size strategy

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto ins_group = InstanceGroupFactory();
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_size(1025);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                       ins_group->name(),
                                                       ins_group->quota(),
                                                       ins_group->cache_config()->reclaim_strategy(),
                                                       instance_infos,
                                                       water_level_exceed_results));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming04) {
    // instance 0 block byte size = 1024, key count = 1
    // instance 1 block byte size = 1024, key count = 1
    // (1024 * 1 + 1024 * 1) / 2048 > 0.8
    // should trigger reclaiming by the used_percentage strategy

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_TRUE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                      ins_group->name(),
                                                      ins_group->quota(),
                                                      ins_group->cache_config()->reclaim_strategy(),
                                                      instance_infos,
                                                      water_level_exceed_results));
    ASSERT_TRUE(water_level_exceed_results[0]);
    ASSERT_FALSE(water_level_exceed_results[1]);
    ASSERT_FALSE(water_level_exceed_results[2]);
    ASSERT_FALSE(water_level_exceed_results[3]);
    ASSERT_FALSE(water_level_exceed_results[4]);
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming05) {
    // instance 0 block byte size = 1024, key count = 1
    // (1024 * 1) / 2048 < 0.8
    // should *not* trigger reclaiming by the used_percentage strategy

    // use instance 0 from setup()

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                       ins_group->name(),
                                                       ins_group->quota(),
                                                       ins_group->cache_config()->reclaim_strategy(),
                                                       instance_infos,
                                                       water_level_exceed_results));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming06) {
    // instance 0 block byte size = 1024, key count = 1
    // (double)(1024 * 1) / 2048.0 is very close to 0.5
    // should trigger reclaiming by the used_percentage strategy

    // use instance 0 from setup()

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.5);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_TRUE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                      ins_group->name(),
                                                      ins_group->quota(),
                                                      ins_group->cache_config()->reclaim_strategy(),
                                                      instance_infos,
                                                      water_level_exceed_results));
    ASSERT_TRUE(water_level_exceed_results[0]);
    ASSERT_FALSE(water_level_exceed_results[1]);
    ASSERT_FALSE(water_level_exceed_results[2]);
    ASSERT_FALSE(water_level_exceed_results[3]);
    ASSERT_FALSE(water_level_exceed_results[4]);
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming07) {
    // instance 0 block byte size = 1024, key count = 1
    // instance 1 block byte size = 1024, key count = 1
    // (1024 * 1 + 1024 * 1) / 2048 < 1.2
    // should *not* trigger reclaiming by the used_percentage strategy

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(1.2);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                       ins_group->name(),
                                                       ins_group->quota(),
                                                       ins_group->cache_config()->reclaim_strategy(),
                                                       instance_infos,
                                                       water_level_exceed_results));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming08) {
    // instance 0 block byte size = 1024, key count = 1
    // instance 1 block byte size = 1024, key count = 1
    // instance 2 block byte size = 1024, key count = 1
    // (1024 * 1 + 1024 * 1 + 1024 * 1) / 2048 > 1.2
    // should trigger reclaiming by the used_percentage strategy

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
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_TRUE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                      ins_group->name(),
                                                      ins_group->quota(),
                                                      ins_group->cache_config()->reclaim_strategy(),
                                                      instance_infos,
                                                      water_level_exceed_results));
    ASSERT_TRUE(water_level_exceed_results[0]);
    ASSERT_FALSE(water_level_exceed_results[1]);
    ASSERT_FALSE(water_level_exceed_results[2]);
    ASSERT_FALSE(water_level_exceed_results[3]);
    ASSERT_FALSE(water_level_exceed_results[4]);
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming09) {
    // instance 0 block byte size = 1024, key count = 16, max key count = 32
    // instance 1 block byte size = 1024, key count = 16, max key count = 32
    // (16 + 16) / (32 + 32) < 0.8
    // should not trigger reclaiming by the used_percentage strategy
    key_count = 16;
    max_key_count = 32;

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto &ins_group = instance_groups.at(0);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                       ins_group->name(),
                                                       ins_group->quota(),
                                                       ins_group->cache_config()->reclaim_strategy(),
                                                       instance_infos,
                                                       water_level_exceed_results));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming10) {
    // instance 0 block byte size = 1024, key count = 32, max key count = 32
    // instance 1 block byte size = 1024, key count = 32, max key count = 32
    // (double)((32 + 32) / (32 + 32)) is very close to 1.0
    // should trigger reclaiming by the used_percentage strategy
    key_count = 32;
    max_key_count = 32;

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto ins_group = InstanceGroupFactory();
    ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(1.0);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_TRUE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                      ins_group->name(),
                                                      ins_group->quota(),
                                                      ins_group->cache_config()->reclaim_strategy(),
                                                      instance_infos,
                                                      water_level_exceed_results));
    ASSERT_TRUE(water_level_exceed_results[0]);
    ASSERT_FALSE(water_level_exceed_results[1]);
    ASSERT_FALSE(water_level_exceed_results[2]);
    ASSERT_FALSE(water_level_exceed_results[3]);
    ASSERT_FALSE(water_level_exceed_results[4]);
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming11) {
    // instance 0 block byte size = 1024, key count = 32, max key count = 32
    // instance 1 block byte size = 1024, key count = 32, max key count = 32
    // (double)((32 + 32) / (32 + 32)) > 0.8
    // should trigger reclaiming by the used_percentage strategy
    key_count = 32;
    max_key_count = 32;

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    const auto &ins_group = instance_groups.at(0);
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_TRUE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                      ins_group->name(),
                                                      ins_group->quota(),
                                                      ins_group->cache_config()->reclaim_strategy(),
                                                      instance_infos,
                                                      water_level_exceed_results));
    ASSERT_TRUE(water_level_exceed_results[0]);
    ASSERT_FALSE(water_level_exceed_results[1]);
    ASSERT_FALSE(water_level_exceed_results[2]);
    ASSERT_FALSE(water_level_exceed_results[3]);
    ASSERT_FALSE(water_level_exceed_results[4]);
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming15) {
    // test empty instance info list
    // should *not* trigger reclaiming

    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_infos.clear();
    std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
    cache_reclaimer_->job_state_flag_ = true;
    ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                       ins_group->name(),
                                                       ins_group->quota(),
                                                       ins_group->cache_config()->reclaim_strategy(),
                                                       instance_infos,
                                                       water_level_exceed_results));
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming16) {
    // test edge cases: divide by zero, negative quota

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    {
        // instance 0 block byte size = 1024, key count = 32, max key count = 0
        // instance 1 block byte size = 1024, key count = 32, max key count = 0
        // (double)((32 + 32) / (0 + 0)) = inf > 0.8
        // should trigger reclaiming when group_used_key_count > 0

        key_count = 32;
        max_key_count = 0;

        const auto &ins_group = instance_groups.at(0);
        std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
        cache_reclaimer_->job_state_flag_ = true;
        ASSERT_TRUE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                          ins_group->name(),
                                                          ins_group->quota(),
                                                          ins_group->cache_config()->reclaim_strategy(),
                                                          instance_infos,
                                                          water_level_exceed_results));
        ASSERT_TRUE(water_level_exceed_results[0]);
        ASSERT_FALSE(water_level_exceed_results[1]);
        ASSERT_FALSE(water_level_exceed_results[2]);
        ASSERT_FALSE(water_level_exceed_results[3]);
        ASSERT_FALSE(water_level_exceed_results[4]);
    }

    {
        // instance 0 block byte size = 1024, key count = 0, max key count = 0
        // instance 1 block byte size = 1024, key count = 0, max key count = 0
        // (double)((0 + 0) / (0 + 0))
        // should *not* trigger reclaiming when group_used_key_count = 0

        key_count = 0;
        max_key_count = 0;

        const auto &ins_group = instance_groups.at(0);
        std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
        cache_reclaimer_->job_state_flag_ = true;
        ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                           ins_group->name(),
                                                           ins_group->quota(),
                                                           ins_group->cache_config()->reclaim_strategy(),
                                                           instance_infos,
                                                           water_level_exceed_results));
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
        std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
        cache_reclaimer_->job_state_flag_ = true;
        ASSERT_TRUE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                          ins_group->name(),
                                                          ins_group->quota(),
                                                          ins_group->cache_config()->reclaim_strategy(),
                                                          instance_infos,
                                                          water_level_exceed_results));
        ASSERT_TRUE(water_level_exceed_results[0]);
        ASSERT_FALSE(water_level_exceed_results[1]);
        ASSERT_FALSE(water_level_exceed_results[2]);
        ASSERT_FALSE(water_level_exceed_results[3]);
        ASSERT_FALSE(water_level_exceed_results[4]);
    }

    {
        // instance 0 block byte size = 1024, key count = 0, max key count = 32
        // instance 1 block byte size = 1024, key count = 0, max key count = 32
        // group quota capacity set to zero
        // should *not* trigger reclaiming when group_used_byte_size = 0

        key_count = 0;
        max_key_count = 32;

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(0);
        std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
        cache_reclaimer_->job_state_flag_ = true;
        ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                           ins_group->name(),
                                                           ins_group->quota(),
                                                           ins_group->cache_config()->reclaim_strategy(),
                                                           instance_infos,
                                                           water_level_exceed_results));
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
        std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
        cache_reclaimer_->job_state_flag_ = true;
        ASSERT_TRUE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                          ins_group->name(),
                                                          ins_group->quota(),
                                                          ins_group->cache_config()->reclaim_strategy(),
                                                          instance_infos,
                                                          water_level_exceed_results));
        ASSERT_TRUE(water_level_exceed_results[0]);
        ASSERT_FALSE(water_level_exceed_results[1]);
        ASSERT_FALSE(water_level_exceed_results[2]);
        ASSERT_FALSE(water_level_exceed_results[3]);
        ASSERT_FALSE(water_level_exceed_results[4]);
    }

    {
        // instance 0 block byte size = 1024, key count = 0, max key count = 32
        // instance 1 block byte size = 1024, key count = 0, max key count = 32
        // group quota capacity set to -1
        // should *not* trigger reclaiming when group_used_byte_size = 0

        key_count = 0;
        max_key_count = 32;

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(-1); // means no capacity, same as 0
        std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
        cache_reclaimer_->job_state_flag_ = true;
        ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                           ins_group->name(),
                                                           ins_group->quota(),
                                                           ins_group->cache_config()->reclaim_strategy(),
                                                           instance_infos,
                                                           water_level_exceed_results));
    }
}

TEST_F(CacheReclaimerTest, TestTriggerReclaiming17) {
    key_count = 2;

    // use instance 0 from setup()

    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    {
        // instance 0 block byte size = 1024, key count = 2
        // instance 1 block byte size = 1024, key count = 2
        // (1024 * 2 + 1024 * 2) / 5120 < 0.9, total waterlevel not exceed
        // (512 + 512) / 1024 > 0.9, waterlevel exceed

        dummy_meta_indexer->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_HF3FS)] =
            512;

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(5120);
        QuotaConfig qc(1024, DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        ins_group->quota_.set_quota_config({qc});
        ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.9);
        std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
        cache_reclaimer_->job_state_flag_ = true;
        ASSERT_TRUE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                          ins_group->name(),
                                                          ins_group->quota(),
                                                          ins_group->cache_config()->reclaim_strategy(),
                                                          instance_infos,
                                                          water_level_exceed_results));
        ASSERT_FALSE(water_level_exceed_results[0]);
        ASSERT_TRUE(water_level_exceed_results[1]);
        ASSERT_FALSE(water_level_exceed_results[2]);
        ASSERT_FALSE(water_level_exceed_results[3]);
        ASSERT_FALSE(water_level_exceed_results[4]);
    }

    {
        // instance 0 block byte size = 1024, key count = 2
        // instance 1 block byte size = 1024, key count = 2
        // (1024 * 2 + 1024 * 2) / 5120 < 0.9, total waterlevel not exceed
        // (128 + 128) / 1024 < 0.9, waterlevel not exceed

        dummy_meta_indexer->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_HF3FS)] =
            128;

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(5120);
        QuotaConfig qc(1024, DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        ins_group->quota_.set_quota_config({qc});
        ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.9);
        std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
        cache_reclaimer_->job_state_flag_ = true;
        ASSERT_FALSE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                           ins_group->name(),
                                                           ins_group->quota(),
                                                           ins_group->cache_config()->reclaim_strategy(),
                                                           instance_infos,
                                                           water_level_exceed_results));
    }

    {
        // instance 0 block byte size = 1024, key count = 2
        // instance 1 block byte size = 1024, key count = 2
        // instance 2 block byte size = 1024, key count = 2
        // (1024 * 2 + 1024 * 2 + 1024 * 2) / 5120 > 0.9, total waterlevel exceed
        // (512 + 512) / 1024 > 0.9, waterlevel exceed

        // construct another instance
        const auto ins_info3 = InstanceInfoFactory();
        ins_info3->set_instance_id("test_instance_id_3");
        instance_infos.emplace_back(ins_info3);

        dummy_meta_indexer->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_HF3FS)] =
            512;

        const auto ins_group = InstanceGroupFactory();
        ins_group->quota_.set_capacity(5120);
        QuotaConfig qc(1024, DataStorageType::DATA_STORAGE_TYPE_HF3FS);
        ins_group->quota_.set_quota_config({qc});
        ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.9);
        std::array<bool, 5> water_level_exceed_results{false, false, false, false, false};
        cache_reclaimer_->job_state_flag_ = true;
        ASSERT_TRUE(cache_reclaimer_->IsTriggerReclaiming(request_context_.get(),
                                                          ins_group->name(),
                                                          ins_group->quota(),
                                                          ins_group->cache_config()->reclaim_strategy(),
                                                          instance_infos,
                                                          water_level_exceed_results));
        ASSERT_TRUE(water_level_exceed_results[0]);
        ASSERT_TRUE(water_level_exceed_results[1]);
        ASSERT_FALSE(water_level_exceed_results[2]);
        ASSERT_FALSE(water_level_exceed_results[3]);
        ASSERT_FALSE(water_level_exceed_results[4]);
    }
}

TEST_F(CacheReclaimerTest, TestInsufficientSampledKeys) {
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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
    batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    // main thread sleeps for 10ms to ensure the worker thread do
    // reclaiming at least once (not 100% but should have reasonable
    // high probability)
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();
    // all blocks should be submitted when sampled keys are insufficient
    ASSERT_FALSE(submitted_del_requests.empty());
    const auto &req = submitted_del_requests.back();
    ASSERT_EQ(10, req.block_keys.size());
    for (std::int64_t i = 0; i != 10; ++i) {
        ASSERT_TRUE(VecContains(req.block_keys, i));
    }
    ASSERT_EQ(req.block_keys.size(), req.location_ids.size());
}

TEST_F(CacheReclaimerTest, TestReclaimByLRU00) {
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 2));
    batch_get_loc_out_maps =
        std::vector<CacheLocationMap>(cache_reclaimer_->GetBatchingSize(request_context_.get()), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    ASSERT_FALSE(submitted_del_requests.empty());
    const auto &req = submitted_del_requests.back();
    ASSERT_EQ(2, req.block_keys.size());
    ASSERT_TRUE(VecContains(req.block_keys, 0));
    ASSERT_TRUE(VecContains(req.block_keys, 1));
}

TEST_F(CacheReclaimerTest, TestReclaimByLRU01) {
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 3));
    batch_get_loc_out_maps =
        std::vector<CacheLocationMap>(cache_reclaimer_->GetBatchingSize(request_context_.get()), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    ASSERT_FALSE(submitted_del_requests.empty());
    const auto &req = submitted_del_requests.back();
    ASSERT_EQ(3, req.block_keys.size());
    // the 3 keys with minimal time point should be included
    ASSERT_TRUE(VecContains(req.block_keys, 1)); // time point -> 2
    ASSERT_TRUE(VecContains(req.block_keys, 5)); // time point -> 4
    ASSERT_TRUE(VecContains(req.block_keys, 6)); // time point -> 5
}

TEST_F(CacheReclaimerTest, TestReclaimByLRU02) {
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 3));
    batch_get_loc_out_maps =
        std::vector<CacheLocationMap>(cache_reclaimer_->GetBatchingSize(request_context_.get()), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    ASSERT_FALSE(submitted_del_requests.empty());
    const auto &req = submitted_del_requests.back();
    ASSERT_EQ(3, req.block_keys.size());
    // the 3 keys with minimal time point should be included
    ASSERT_TRUE(VecContains(req.block_keys, 9)); // time point -> 2
    ASSERT_TRUE(VecContains(req.block_keys, 5)); // time point -> 4
    ASSERT_TRUE(VecContains(req.block_keys, 6)); // time point -> 5
}

TEST_F(CacheReclaimerTest, TestReclaimByLRU03) {
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 0));
    batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    ASSERT_TRUE(submitted_del_requests.empty());
}

TEST_F(CacheReclaimerTest, TestReclaimByLRU04) {
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));
    batch_get_loc_out_maps =
        std::vector<CacheLocationMap>(cache_reclaimer_->GetBatchingSize(request_context_.get()), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    ASSERT_FALSE(submitted_del_requests.empty());
    const auto &req = submitted_del_requests.back();
    ASSERT_EQ(1, req.block_keys.size());
    // the 1 keys with minimal time point should be included
    ASSERT_TRUE(VecContains(req.block_keys, 9)); // time point -> 2
}

TEST_F(CacheReclaimerTest, TestMetaIndexerGetPropertiesFailure) {
    // set up test data
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

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

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted when GetProperties fails
    ASSERT_TRUE(submitted_del_requests.empty());
}

TEST_F(CacheReclaimerTest, TestMetaIndexerRandomSampleFailure) {
    // configure the RandomSample stub to return an error
    random_sample_result = ErrorCode::EC_ERROR;

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

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted when RandomSample fails
    ASSERT_TRUE(submitted_del_requests.empty());
}

TEST_F(CacheReclaimerTest, TestMetaIndexerSampleKeys00) {
    // test case that sampled keys size and properties size not match
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // size is 10
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

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted
    ASSERT_TRUE(submitted_del_requests.empty());
}

TEST_F(CacheReclaimerTest, TestMetaIndexerSampleKeys01) {
    // test case that properties size match but has wrong field
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // size is 10
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

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted
    ASSERT_TRUE(submitted_del_requests.empty());
}

TEST_F(CacheReclaimerTest, TestSchedulePlanExecutorDelFailure) {
    // set up test data
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();
}

TEST_F(CacheReclaimerTest, TestEmptyInstanceGroups) {
    // clear all instance groups
    instance_groups.clear();

    // the mocking sample keys are set but should never be accessed
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted when there are no instance groups
    ASSERT_TRUE(submitted_del_requests.empty());
}

TEST_F(CacheReclaimerTest, TestEmptyInstanceInfos) {
    // clear all instance infos
    instance_infos.clear();

    // the mocking sample keys are set but should never be accessed
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // no deletion requests should be submitted when there are no instance infos
    ASSERT_TRUE(submitted_del_requests.empty());
}

TEST_F(CacheReclaimerTest, TestMultipleInstanceGroups) {
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
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 5));
    batch_get_loc_out_maps =
        std::vector<CacheLocationMap>(cache_reclaimer_->GetBatchingSize(request_context_.get()), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

    cache_reclaimer_->Stop();

    // deletion requests should be submitted for both instance groups
    ASSERT_FALSE(submitted_del_requests.empty());

    // check that we have requests for both instances
    bool found_instance_1 = false;
    bool found_instance_2 = false;

    for (const auto &req : submitted_del_requests) {
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
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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
        // test with zero key count
        key_count = 0;
        max_key_count = 100;

        // update the trigger strategy to trigger the reclaiming based on percentage
        instance_groups.clear();
        const auto &ins_group = InstanceGroupFactory();
        ins_group->cache_config_->reclaim_strategy_->trigger_strategy_.set_used_percentage(0.01); // 1%
        instance_groups.emplace_back(ins_group);

        batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

        cache_reclaimer_->Stop();

        // with zero key count, the percentage usage would be 0%, so no reclaiming should happen
        ASSERT_TRUE(submitted_del_requests.empty());
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
        submitted_del_requests.clear();

        batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

        cache_reclaimer_->Stop();

        // with 100% key usage and trigger at 90%, reclaiming should happen
        ASSERT_FALSE(submitted_del_requests.empty());
    }

    {
        // test with zero max key count (divide by zero)
        key_count = 100;
        max_key_count = 0;

        // clear requests from previous test
        submitted_del_requests.clear();

        // update the trigger strategy to trigger at 90%
        instance_groups.clear();
        const auto &ins_group = InstanceGroupFactory();
        instance_groups.emplace_back(ins_group);

        batch_get_loc_out_maps = std::vector<CacheLocationMap>(random_sample_keys.size(), CacheLocationMap{});
        ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        ASSERT_TRUE(cache_reclaimer_->IsRunning()); // the worker thread should still be running

        cache_reclaimer_->Stop();

        // reclaiming should happen since group used key count > 0 and used size > 16
        ASSERT_FALSE(submitted_del_requests.empty());
    }
}

TEST_F(CacheReclaimerTest, TestCronJobAdaptiveSleepInterval) {
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    // use instance 0 from setup()
    // construct instance 1
    const auto ins_info = InstanceInfoFactory();
    ins_info->set_instance_id("test_instance_id_2");
    instance_infos.emplace_back(ins_info);

    instance_groups.clear();
    const auto ins_group = InstanceGroupFactory();
    ins_group->quota_.set_capacity(2048);
    instance_groups.emplace_back(ins_group);

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));
    batch_get_loc_out_maps =
        std::vector<CacheLocationMap>(cache_reclaimer_->GetBatchingSize(request_context_.get()), CacheLocationMap{});
    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->Start());

    // worker thread should first sleep for 10ms then do the reclaiming
    // multiple times with 0 sleep interval in between.
    // here we set the wait time to be 16ms to verify the worker thread
    // sleep interval is set to 0, or the reclaiming round would
    // otherwise equal to 1 (see below for explanation).
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
    cache_reclaimer_->Stop(); // join the worker thread

    // the worker thread is synchronised by join(),
    // so the count should be 1+1 if sleep interval reduction is not
    // working:
    //
    //    [start]
    //       |
    //       |
    //   sleep 10ms
    //       |
    //       |
    //       V
    // [1st triggered]
    //       |
    //       |
    // sleep 6ms (stopping requested by main thread)
    //       |
    //       |
    // working thread signaled and wake up immediately
    //       |
    //       |
    //       V
    //    [finish]
    //
    // but when the sleep interval reduction is working as expected,
    // what's going on would be:
    //
    //    [start]
    //       |
    //       |
    //   sleep 10ms
    //       |
    //       |
    //       V
    // [1st triggered]
    // <sleep interval becomes 0>
    //       |
    //       |
    // [2nd triggered]
    // [3rd triggered]
    // [   ......    ]
    // [nth triggered]
    //       |
    //       |
    //  (stopping requested)
    //       |
    //       |
    //       V
    //    [finish]
    ASSERT_LT(1, list_ins_group_call_counter);
    ASSERT_LT(1, submitted_del_requests.size());
}

TEST_F(CacheReclaimerTest, TestCronJobAdaptiveSleepIntervalRecovery) {
    random_sample_keys = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
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

    ASSERT_EQ(ErrorCode::EC_OK, cache_reclaimer_->SetBatchingSize(request_context_.get(), 1));
    batch_get_loc_out_maps =
        std::vector<CacheLocationMap>(cache_reclaimer_->GetBatchingSize(request_context_.get()), CacheLocationMap{});
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
    ASSERT_LE(2, list_ins_group_call_counter);
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
    cache_reclaimer_->delete_handlers_.emplace_front(
        request_context_, "test_instance", "test_instance_group", 2, 3, std::move(fut));

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
        cache_reclaimer_->delete_handlers_.emplace_front(request_context_,
                                                         "test_instance" + std::to_string(i),
                                                         "test_instance_group" + std::to_string(i),
                                                         0,
                                                         0,
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
    cache_reclaimer_->delete_handlers_.emplace_front(
        request_context_, "test_instance", "test_instance_group", 0, 0, std::move(fut));

    try {
        throw std::runtime_error("test exception");
    } catch (...) {
        promise->set_exception(std::current_exception());
    }

    cache_reclaimer_->HandleDelRes();
    ASSERT_TRUE(cache_reclaimer_->delete_handlers_.empty());
}

TEST_F(CacheReclaimerTest, TestHandleDelRes04) {
    // test invalid future
    const auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    auto fut = promise->get_future();

    promise->set_value(PlanExecuteResult{ErrorCode::EC_OK, "ok"});
    fut.get(); // fut is not valid anymore

    cache_reclaimer_->delete_handlers_.clear();
    cache_reclaimer_->delete_handlers_.emplace_front(
        request_context_, "test_instance", "test_instance_group", 0, 0, std::move(fut));

    cache_reclaimer_->HandleDelRes();
    ASSERT_TRUE(cache_reclaimer_->delete_handlers_.empty());
}
