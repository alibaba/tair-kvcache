#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/request_context.h"
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
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/hf3fs_backend.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/data_storage_selector.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"

using namespace kv_cache_manager;

ErrorCode ec_g;

std::shared_ptr<InstanceGroup> InstanceGroupFactory() {
    const auto instance_group = std::make_shared<InstanceGroup>();

    // set basic instance group properties
    instance_group->set_name("default_test_group");
    instance_group->set_storage_candidates({"nfs_storage_00", "3fs_storage_01"}); // <--------
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
    cache_config->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS); // <--------
    cache_config->set_reclaim_strategy(reclaim_strategy);
    cache_config->set_meta_indexer_config(meta_indexer_config);

    instance_group->set_cache_config(cache_config);
    return instance_group;
}

std::shared_ptr<DataStorageManager> data_storage_manager_g;

std::shared_ptr<DataStorageManager> RegistryManager_data_storage_manager_stub(void *obj) {
    return data_storage_manager_g;
}

std::shared_ptr<InstanceGroup> instance_group_g;

std::pair<ErrorCode, std::shared_ptr<const InstanceGroup>>
RegistryManager_GetInstanceGroup_stub(void *obj, RequestContext *rc, const std::string &ig) {
    return {ec_g, instance_group_g};
}

using ins_info_ptr_vec = std::vector<std::shared_ptr<const InstanceInfo>>;
ins_info_ptr_vec instance_infos_g;

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
    for (const auto &i : instance_infos_g) {
        if (!i // nullptr is reserved for testing purpose
            || i->instance_group_name() == ig) {
            iv.emplace_back(i);
        }
    }
    return {ec_g, iv};
}

std::vector<std::shared_ptr<DataStorageBackend>> avail_backends_g;

std::vector<std::shared_ptr<DataStorageBackend>> DataStorageManager_GetAvailableStorages_stub(void *obj) {
    // simulate the behavior of the real method that only available backends are returned
    for (auto it = avail_backends_g.begin(); it != avail_backends_g.end(); /* iterator is updated inside */) {
        if (!*it // reserve the nullptr for testing purpose
            || (*it)->Available()) {
            ++it;
        } else {
            it = avail_backends_g.erase(it);
        }
    }
    return avail_backends_g;
}

std::shared_ptr<MetaIndexer> meta_indexer_g;

std::shared_ptr<MetaIndexer> MetaIndexerManager_GetMetaIndexer_stub(void *obj, const std::string &ins_id) {
    return meta_indexer_g;
}

std::size_t key_count_g;

std::size_t MetaIndexer_GetKeyCount_stub(void *obj) { return key_count_g; }
void MetaIndexer_PersistMetaData_stub(void *obj) {}

class DataStorageSelectorTest : public TESTBASE {
public:
    void SetUp() override {
        stub_.set(ADDR(RegistryManager, data_storage_manager), RegistryManager_data_storage_manager_stub);
        stub_.set(ADDR(RegistryManager, GetInstanceGroup), RegistryManager_GetInstanceGroup_stub);
        stub_.set(ADDR(RegistryManager, ListInstanceInfo), RegistryManager_ListInstanceInfo_stub);
        stub_.set(ADDR(DataStorageManager, GetAvailableStorages), DataStorageManager_GetAvailableStorages_stub);
        stub_.set(ADDR(MetaIndexerManager, GetMetaIndexer), MetaIndexerManager_GetMetaIndexer_stub);
        stub_.set(ADDR(MetaIndexer, GetKeyCount), MetaIndexer_GetKeyCount_stub);
        stub_.set(ADDR(MetaIndexer, PersistMetaData), MetaIndexer_PersistMetaData_stub);

        metrics_registry_ = std::make_shared<MetricsRegistry>();
        data_storage_manager_g = std::make_shared<DataStorageManager>(metrics_registry_);
        ec_g = ErrorCode::EC_OK;
        instance_group_g = InstanceGroupFactory();
        instance_infos_g.emplace_back(InstanceInfoFactory());
        meta_indexer_g = std::make_shared<MetaIndexer>();
        key_count_g = 1;
        request_context_ = std::make_unique<RequestContext>("foo_trace_id");
        data_storage_selector_ = std::make_unique<DataStorageSelector>(
            std::make_shared<MetaIndexerManager>(), std::make_shared<RegistryManager>("foo", metrics_registry_));

        // initially we have 2 backends, both opened and available
        const auto sb0 = std::make_shared<NfsBackend>(metrics_registry_);
        sb0->config_.set_global_unique_name("nfs_storage_00");
        sb0->is_open_ = true;
        sb0->is_available_ = true;
        avail_backends_g.emplace_back(sb0);

        const auto sb1 = std::make_shared<Hf3fsBackend>(metrics_registry_);
        sb1->config_.set_global_unique_name("3fs_storage_01");
        sb1->is_open_ = true;
        sb1->is_available_ = true;
        avail_backends_g.emplace_back(sb1);
    }

    void TearDown() override {
        data_storage_manager_g.reset();
        instance_group_g.reset();
        instance_infos_g.clear();
        meta_indexer_g.reset();
        key_count_g = 0;
        avail_backends_g.clear();

        stub_.reset(ADDR(RegistryManager, data_storage_manager));
        stub_.reset(ADDR(RegistryManager, GetInstanceGroup));
        stub_.reset(ADDR(RegistryManager, ListInstanceInfo));
        stub_.reset(ADDR(DataStorageManager, GetAvailableStorages));
        stub_.reset(ADDR(MetaIndexerManager, GetMetaIndexer));
        stub_.reset(ADDR(MetaIndexer, GetKeyCount));
        stub_.reset(ADDR(MetaIndexer, PersistMetaData));
    }

    Stub stub_;
    std::unique_ptr<RequestContext> request_context_;
    std::unique_ptr<DataStorageSelector> data_storage_selector_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

TEST_F(DataStorageSelectorTest, TestNullptr) {
    stub_.reset(ADDR(RegistryManager, data_storage_manager));
    stub_.reset(ADDR(RegistryManager, GetInstanceGroup));
    stub_.reset(ADDR(DataStorageManager, GetAvailableStorages));

    {
        // should work with null request context
        const auto result = data_storage_selector_->SelectCacheWriteDataStorageBackend(nullptr, "default_test_group");
        ASSERT_EQ(ErrorCode::EC_BADARGS, result.ec);
    }

    {
        // should work with nullptrs
        data_storage_selector_ = std::make_unique<DataStorageSelector>(nullptr, nullptr);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        // should work with null meta indexer manager
        data_storage_selector_ =
            std::make_unique<DataStorageSelector>(nullptr, std::make_shared<RegistryManager>("foo", metrics_registry_));
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        // should work with null registry manager
        data_storage_selector_ = std::make_unique<DataStorageSelector>(std::make_shared<MetaIndexerManager>(), nullptr);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }
}

TEST_F(DataStorageSelectorTest, TestCopyControl) {
    ASSERT_FALSE(std::is_default_constructible<DataStorageSelector>::value);
    ASSERT_TRUE(std::is_copy_constructible<DataStorageSelector>::value);
    ASSERT_TRUE(std::is_copy_assignable<DataStorageSelector>::value);
    ASSERT_TRUE(std::is_move_constructible<DataStorageSelector>::value);
    ASSERT_TRUE(std::is_move_assignable<DataStorageSelector>::value);
    ASSERT_TRUE(std::is_swappable<DataStorageSelector>::value);
}

TEST_F(DataStorageSelectorTest, TestSelectCacheWriteStorageBackendAbnormal00) {
    // should work with null data storage manager
    data_storage_manager_g.reset();
    const auto result =
        data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
    ASSERT_NE(ErrorCode::EC_OK, result.ec);
}

TEST_F(DataStorageSelectorTest, TestSelectCacheWriteStorageBackendAbnormal01) {
    {
        // should work with null data storage backend(s)
        avail_backends_g.clear();
        avail_backends_g.emplace_back(nullptr);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        // should work on empty available data storage backend list
        avail_backends_g.clear();
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }
}

TEST_F(DataStorageSelectorTest, TestSelectCacheWriteStorageBackendAbnormal02) {
    {
        // should work with unexpected error code when get the instance group
        ec_g = ErrorCode::EC_ERROR;
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        // Should work with null cache config of the instance group even
        // the error code is OK.
        // The cache config is used to get the preference strategy
        // config; if it is unavailable, we should fall back to the
        // default preference, which allows data storage backend
        // selection fallback, so the select result should be OK.
        ec_g = ErrorCode::EC_OK;
        instance_group_g->cache_config_.reset();
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
    }

    {
        // should work with null instance group even the error code is OK
        ec_g = ErrorCode::EC_OK;
        instance_group_g.reset();
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }
}

TEST_F(DataStorageSelectorTest, TestSelectCacheWriteStorageBackendNormal00) {
    {
        // verify empty storage candidate config
        instance_group_g->set_storage_candidates({});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        // verify the basic selecting logic with preference strategy
        instance_group_g->set_storage_candidates({"nfs_storage_00", "3fs_storage_01"});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS);

        // only 3fs shall be selected out, honor the preference strategy
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_HF3FS, result.type);
        ASSERT_EQ("3fs_storage_01", result.name);
    }

    {
        // verify the basic selecting logic with preference strategy of a non-existing backend
        // with the preference strategy, fallback is allowed
        instance_group_g->set_storage_candidates(
            {"nfs_storage_00", "3fs_storage_01", "tair_mempool_02", "mooncake_03"});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_TAIR_MEMPOOL);

        // only nfs or 3fs shall be selected out; we currently do not care which is the final result
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
        if (result.type != DataStorageType::DATA_STORAGE_TYPE_HF3FS) {
            ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, result.type);
            ASSERT_EQ("nfs_storage_00", result.name);
        } else {
            ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_HF3FS, result.type);
            ASSERT_EQ("3fs_storage_01", result.name);
        }
    }

    {
        // verify the basic selecting logic with preference strategy of a non-existing backend
        // but the fallback option is narrowed
        instance_group_g->set_storage_candidates({"3fs_storage_01", "tair_mempool_02", "mooncake_03"});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_TAIR_MEMPOOL);

        // only 3fs shall be selected out, since the nfs is not in the candidate list
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_HF3FS, result.type);
        ASSERT_EQ("3fs_storage_01", result.name);
    }

    {
        // verify the basic selecting logic with preference strategy of a non-existing backend
        // but the fallback option is narrowed further, to which non is possible
        instance_group_g->set_storage_candidates({"tair_mempool_02", "mooncake_03"});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_TAIR_MEMPOOL);

        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        // verify the basic selecting logic with preference strategy of an existing backend
        // but not in the candidate list
        instance_group_g->set_storage_candidates({"tair_mempool_02", "mooncake_03"});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS);

        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }
}

TEST_F(DataStorageSelectorTest, TestSelectCacheWriteStorageBackendNormal01) {
    {
        // verify the basic selecting logic with "always" strategy
        instance_group_g->set_storage_candidates({"nfs_storage_00", "3fs_storage_01"});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_ALWAYS_3FS);

        // only 3fs shall be selected out, honor the "always" strategy
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_HF3FS, result.type);
        ASSERT_EQ("3fs_storage_01", result.name);
    }

    {
        // verify the basic selecting logic with "always" strategy, where fallback is prohibited
        instance_group_g->set_storage_candidates(
            {"nfs_storage_00", "3fs_storage_01", "tair_mempool_02", "mooncake_03"});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_ALWAYS_TAIR_MEMPOOL);

        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        instance_group_g->set_storage_candidates({"3fs_storage_01", "tair_mempool_02", "mooncake_03"});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_ALWAYS_TAIR_MEMPOOL);

        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        instance_group_g->set_storage_candidates({"tair_mempool_02", "mooncake_03"});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_ALWAYS_TAIR_MEMPOOL);

        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        instance_group_g->set_storage_candidates({"tair_mempool_02", "mooncake_03"});
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_ALWAYS_3FS);

        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }
}

TEST_F(DataStorageSelectorTest, TestSelectCacheWriteStorageBackendNormal02) {
    // the candidate list is the same as the available backends
    instance_group_g->set_storage_candidates({"nfs_storage_00", "3fs_storage_01"});

    {
        // only 3fs shall be selected out, honor the preference strategy
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_HF3FS, result.type);
        ASSERT_EQ("3fs_storage_01", result.name);
    }

    // the only 3fs backend is now unavailable
    avail_backends_g.back()->is_available_ = false;

    {
        // unavailable backend should not be selected; with the preference strategy, fallback is allowed
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, result.type);
        ASSERT_EQ("nfs_storage_00", result.name);
    }

    {
        // with "always" strategy, fallback is prohibited
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_ALWAYS_3FS);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    // another 3fs backend is added, opened and available
    const auto sb2 = std::make_shared<Hf3fsBackend>(metrics_registry_);
    sb2->config_.set_global_unique_name("3fs_storage_02");
    sb2->is_open_ = true;
    sb2->is_available_ = true;
    avail_backends_g.emplace_back(sb2);

    // update the candidate list
    instance_group_g->set_storage_candidates({"nfs_storage_00", "3fs_storage_01", "3fs_storage_02"});

    {
        // the new 3fs backend should be selected out since it is preferred
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_HF3FS, result.type);
        ASSERT_EQ("3fs_storage_02", result.name);
    }

    {
        // with "always" strategy, result should be the same
        instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_ALWAYS_3FS);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_HF3FS, result.type);
        ASSERT_EQ("3fs_storage_02", result.name);
    }
}

TEST_F(DataStorageSelectorTest, TestSelectCacheWriteStorageBackendQuota00) {
    instance_group_g->set_storage_candidates({"nfs_storage_00", "3fs_storage_01"});
    instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS);

    {
        // test quota_capacity = 0
        InstanceGroupQuota quota;
        quota.set_capacity(0);
        instance_group_g->set_quota(quota);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        // test quota_capacity < 0
        InstanceGroupQuota quota;
        quota.set_capacity(-1);
        instance_group_g->set_quota(quota);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        // test quota_capacity < group_used_size
        InstanceGroupQuota quota;
        quota.set_capacity(1);
        instance_group_g->set_quota(quota);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        // test quota_capacity == group_used_size
        InstanceGroupQuota quota;
        quota.set_capacity(1024);
        instance_group_g->set_quota(quota);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_NE(ErrorCode::EC_OK, result.ec);
    }

    {
        // test quota_capacity > group_used_size
        InstanceGroupQuota quota;
        quota.set_capacity(1025);
        instance_group_g->set_quota(quota);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_HF3FS, result.type);
        ASSERT_EQ("3fs_storage_01", result.name);
    }
}

TEST_F(DataStorageSelectorTest, TestSelectCacheWriteStorageBackendQuota01) {
    instance_group_g->set_storage_candidates({"nfs_storage_00", "3fs_storage_01"});
    instance_group_g->cache_config_->set_cache_prefer_strategy(CachePreferStrategy::CPS_PREFER_3FS);

    {
        meta_indexer_g->storage_usage_array_[static_cast<std::uint8_t>(DataStorageType::DATA_STORAGE_TYPE_HF3FS)] = 512;

        InstanceGroupQuota quota;
        quota.set_capacity(1025); // quota_capacity > group_used_size
        QuotaConfig qc(511, DataStorageType::DATA_STORAGE_TYPE_HF3FS); // hf3fs quota 511 < 512
        quota.set_quota_config({qc});
        instance_group_g->set_quota(quota);
        const auto result =
            data_storage_selector_->SelectCacheWriteDataStorageBackend(request_context_.get(), "default_test_group");
        ASSERT_EQ(ErrorCode::EC_OK, result.ec);
        ASSERT_EQ(DataStorageType::DATA_STORAGE_TYPE_NFS, result.type);
        ASSERT_EQ("nfs_storage_00", result.name);
    }
}
