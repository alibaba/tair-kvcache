#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/meta_searcher.h"
#include "kv_cache_manager/manager/migration_manager.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/service/admin_service_impl.h"

using namespace kv_cache_manager;

class AdminServiceImplTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);

        auto instance_group = std::make_shared<InstanceGroup>();
        auto meta_indexer_config = std::make_shared<MetaIndexerConfig>();
        instance_group->cache_config_ = std::make_shared<CacheConfig>();
        instance_group->cache_config_->meta_indexer_config_ = meta_indexer_config;
        instance_group->cache_config_->cache_prefer_strategy_ = CachePreferStrategy::CPS_PREFER_3FS;
        auto backend_config = std::make_shared<MetaStorageBackendConfig>();
        backend_config->storage_type_ = META_LOCAL_BACKEND_TYPE_STR;
        auto cache_policy_config = std::make_shared<MetaCachePolicyConfig>();
        meta_indexer_config->meta_storage_backend_config_ = backend_config;
        meta_indexer_config->meta_cache_policy_config_ = cache_policy_config;

        std::vector<LocationSpecInfo> spec_infos = {LocationSpecInfo("tp0", 512)};
        ModelDeployment md;
        md.set_model_name("m");
        md.set_tp_size(1);
        auto instance_info =
            std::make_shared<InstanceInfo>("test_quota_group", "default", kInstance, 64, spec_infos, md);
        registry_manager_->instance_group_configs_["test_group"] = instance_group;
        registry_manager_->instance_infos_[kInstance] = instance_info;
        registry_manager_->Init();

        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        ASSERT_TRUE(cache_manager_->Init());
        StartupConfigLoader loader;
        loader.Init(registry_manager_);
        loader.Load("");
        ASSERT_EQ(EC_OK, cache_manager_->DoRecover());
        cache_manager_->migration_manager()->DebugEnableCopySubmissionsForTest();

        ASSERT_TRUE(RegisterDummyStorage("hot_01"));
        ASSERT_TRUE(RegisterDummyStorage("cold_01"));

        admin_ = std::make_shared<AdminServiceImpl>(
            cache_manager_, /*reporter*/ nullptr, metrics_registry_, registry_manager_, /*leader*/ nullptr);
        admin_->EnableLeaderOnlyRequests();
    }

    void TearDown() override {}

    bool RegisterDummyStorage(const std::string &name) {
        auto spec = std::make_shared<DummyStorageSpec>();
        spec->set_root_path(GetPrivateTestRuntimeDataPath() + name + "/");
        spec->set_key_count_per_file(1);
        StorageConfig config;
        config.set_type(DataStorageType::DATA_STORAGE_TYPE_DUMMY);
        config.set_global_unique_name(name);
        config.set_storage_spec(spec);
        auto rc = std::make_shared<RequestContext>("reg_storage");
        return registry_manager_->data_storage_manager()->RegisterStorage(rc.get(), name, config) == EC_OK;
    }

    // 在 meta 中为 block_key 直接登记一个位于 hot_01 的 SERVING 源 location
    void SeedServingSource(int64_t block_key) {
        auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(kInstance);
        ASSERT_NE(nullptr, indexer);
        MetaSearcher meta_searcher(indexer);
        auto rc = std::make_shared<RequestContext>("seed");
        std::string uri = "dummy://hot_01/blk_" + StringUtil::Uint64ToHex(block_key) + "?size=16";
        auto loc = std::make_shared<CacheLocation>(
            DataStorageType::DATA_STORAGE_TYPE_DUMMY, 1, std::vector<LocationSpec>{LocationSpec("tp0", uri)});
        std::vector<std::string> ids;
        ASSERT_EQ(EC_OK, meta_searcher.BatchAddLocation(rc.get(), {block_key}, {loc}, ids));
        ASSERT_EQ(1u, ids.size());
        std::vector<std::vector<MetaSearcher::LocationCASTask>> cas{
            {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
        std::vector<std::vector<ErrorCode>> cas_results;
        ASSERT_EQ(EC_OK, meta_searcher.BatchCASLocationStatus(rc.get(), {block_key}, cas, cas_results));
    }

    // 在 meta 中为 block_key 直接添加一个位于指定 storage 的 CacheLocation。
    // status=CLS_SERVING 时 CAS WRITING->SERVING；status=CLS_WRITING 则保持。
    void SeedLocationOnStorage(int64_t block_key,
                               const std::string &storage_name,
                               CacheLocationStatus status) {
        auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(kInstance);
        ASSERT_NE(nullptr, indexer);
        MetaSearcher meta_searcher(indexer);
        auto rc = std::make_shared<RequestContext>("seed_dst");
        std::string uri = "dummy://" + storage_name + "/blk_" + StringUtil::Uint64ToHex(block_key) + "?size=16";
        auto loc = std::make_shared<CacheLocation>(
            DataStorageType::DATA_STORAGE_TYPE_DUMMY, 1, std::vector<LocationSpec>{LocationSpec("tp0", uri)});
        std::vector<std::string> ids;
        ASSERT_EQ(EC_OK, meta_searcher.BatchAddLocation(rc.get(), {block_key}, {loc}, ids));
        ASSERT_EQ(1u, ids.size());
        if (status == CLS_SERVING) {
            std::vector<std::vector<MetaSearcher::LocationCASTask>> cas{
                {MetaSearcher::LocationCASTask{ids[0], CLS_WRITING, CLS_SERVING}}};
            std::vector<std::vector<ErrorCode>> cas_results;
            ASSERT_EQ(EC_OK, meta_searcher.BatchCASLocationStatus(rc.get(), {block_key}, cas, cas_results));
        }
    }

    proto::admin::MigrateCacheRequest MakeReq(proto::admin::MigrationMethod method,
                                              const std::vector<int64_t> &block_keys) {
        proto::admin::MigrateCacheRequest req;
        req.set_trace_id("t");
        req.set_instance_id(kInstance);
        req.set_source_storage_name("hot_01");
        req.set_target_storage_name("cold_01");
        req.set_method(method);
        for (auto k : block_keys) {
            req.add_block_keys(k);
        }
        return req;
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<AdminServiceImpl> admin_;
    const std::string kInstance = "test_instance";
};

TEST_F(AdminServiceImplTest, TestInvalidArgs) {
    auto rc = std::make_shared<RequestContext>("t");

    {
        auto req = MakeReq(proto::admin::MIGRATION_METHOD_COPY, {1});
        req.clear_instance_id();
        proto::admin::MigrateCacheResponse resp;
        admin_->MigrateCache(rc.get(), &req, &resp);
        ASSERT_EQ(proto::admin::INVALID_ARGUMENT, resp.header().status().code());
    }
    {
        auto req = MakeReq(proto::admin::MIGRATION_METHOD_COPY, {1});
        req.clear_source_storage_name();
        proto::admin::MigrateCacheResponse resp;
        admin_->MigrateCache(rc.get(), &req, &resp);
        ASSERT_EQ(proto::admin::INVALID_ARGUMENT, resp.header().status().code());
    }
    {
        auto req = MakeReq(proto::admin::MIGRATION_METHOD_COPY, {1});
        req.clear_target_storage_name();
        proto::admin::MigrateCacheResponse resp;
        admin_->MigrateCache(rc.get(), &req, &resp);
        ASSERT_EQ(proto::admin::INVALID_ARGUMENT, resp.header().status().code());
    }
    {
        auto req = MakeReq(proto::admin::MIGRATION_METHOD_UNSPECIFIED, {1});
        proto::admin::MigrateCacheResponse resp;
        admin_->MigrateCache(rc.get(), &req, &resp);
        ASSERT_EQ(proto::admin::INVALID_ARGUMENT, resp.header().status().code());
    }
    {
        auto req = MakeReq(static_cast<proto::admin::MigrationMethod>(999), {1});
        proto::admin::MigrateCacheResponse resp;
        admin_->MigrateCache(rc.get(), &req, &resp);
        ASSERT_EQ(proto::admin::INVALID_ARGUMENT, resp.header().status().code());
    }
}

TEST_F(AdminServiceImplTest, TestExplicitBlockKeysCopy) {
    SeedServingSource(101);
    SeedServingSource(102);
    // 103 不 seed（不在源 storage） -> 应被拒绝

    auto rc = std::make_shared<RequestContext>("t");
    auto req = MakeReq(proto::admin::MIGRATION_METHOD_COPY, {101, 102, 103});
    proto::admin::MigrateCacheResponse resp;
    admin_->MigrateCache(rc.get(), &req, &resp);

    ASSERT_EQ(proto::admin::OK, resp.header().status().code());
    ASSERT_EQ(2, resp.accepted());
    ASSERT_EQ(1, resp.rejected());
    ASSERT_TRUE(cache_manager_->migration_manager()->HasMigrationTask(kInstance, 101));
    ASSERT_TRUE(cache_manager_->migration_manager()->HasMigrationTask(kInstance, 102));
    ASSERT_FALSE(cache_manager_->migration_manager()->HasMigrationTask(kInstance, 103));
}

TEST_F(AdminServiceImplTest, TestExplicitBlockKeysMark) {
    SeedServingSource(201);
    SeedServingSource(202);

    auto rc = std::make_shared<RequestContext>("t");
    auto req = MakeReq(proto::admin::MIGRATION_METHOD_MARK, {201, 202, 203});
    proto::admin::MigrateCacheResponse resp;
    admin_->MigrateCache(rc.get(), &req, &resp);

    ASSERT_EQ(proto::admin::OK, resp.header().status().code());
    ASSERT_EQ(2, resp.accepted());
    ASSERT_EQ(1, resp.rejected());
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite(kInstance, 201));
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite(kInstance, 202));
    ASSERT_FALSE(cache_manager_->migration_manager()->IsMarkedForTieredWrite(kInstance, 203));
    ASSERT_EQ("cold_01", cache_manager_->migration_manager()->GetTieredWriteTarget(kInstance, 201));
}

TEST_F(AdminServiceImplTest, TestExplicitBlockKeysBothPrefersCopy) {
    SeedServingSource(251);
    SeedServingSource(252);

    auto rc = std::make_shared<RequestContext>("t");
    auto req = MakeReq(proto::admin::MIGRATION_METHOD_BOTH, {251, 252});
    proto::admin::MigrateCacheResponse resp;
    admin_->MigrateCache(rc.get(), &req, &resp);

    ASSERT_EQ(proto::admin::OK, resp.header().status().code());
    ASSERT_EQ(2, resp.accepted());
    ASSERT_EQ(0, resp.rejected());
    ASSERT_TRUE(cache_manager_->migration_manager()->HasMigrationTask(kInstance, 251));
    ASSERT_TRUE(cache_manager_->migration_manager()->HasMigrationTask(kInstance, 252));
    ASSERT_FALSE(cache_manager_->migration_manager()->IsMarkedForTieredWrite(kInstance, 251));
    ASSERT_FALSE(cache_manager_->migration_manager()->IsMarkedForTieredWrite(kInstance, 252));
}

TEST_F(AdminServiceImplTest, TestRuleSampling) {
    SeedServingSource(301);
    SeedServingSource(302);

    auto rc = std::make_shared<RequestContext>("t");
    // block_keys 为空 -> 走 rule 采样
    auto req = MakeReq(proto::admin::MIGRATION_METHOD_MARK, {});
    req.mutable_rule()->set_sample_count(100);
    proto::admin::MigrateCacheResponse resp;
    admin_->MigrateCache(rc.get(), &req, &resp);

    ASSERT_EQ(proto::admin::OK, resp.header().status().code());
    // 采样到的 seeded block 都在 hot_01 上 -> 应被接受
    ASSERT_GE(resp.accepted(), 2);
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite(kInstance, 301));
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite(kInstance, 302));
}

TEST_F(AdminServiceImplTest, TestRuleSamplingFailureReturnsError) {
    auto indexer = cache_manager_->meta_indexer_manager()->GetMetaIndexer(kInstance);
    ASSERT_NE(nullptr, indexer);
    ASSERT_EQ(EC_OK, indexer->backend_manager_->Close());

    auto rc = std::make_shared<RequestContext>("t");
    auto req = MakeReq(proto::admin::MIGRATION_METHOD_MARK, {});
    req.mutable_rule()->set_sample_count(100);
    proto::admin::MigrateCacheResponse resp;
    admin_->MigrateCache(rc.get(), &req, &resp);

    ASSERT_EQ(proto::admin::INTERNAL_ERROR, resp.header().status().code());
    ASSERT_EQ(0, resp.accepted());
    ASSERT_EQ(0, resp.rejected());
}

TEST_F(AdminServiceImplTest, TestInstanceNotFound) {
    auto rc = std::make_shared<RequestContext>("t");
    auto req = MakeReq(proto::admin::MIGRATION_METHOD_COPY, {1});
    req.set_instance_id("no_such_instance");
    proto::admin::MigrateCacheResponse resp;
    admin_->MigrateCache(rc.get(), &req, &resp);
    ASSERT_EQ(proto::admin::INSTANCE_NOT_EXIST, resp.header().status().code());
}

// BOTH 方法下，Copy 提交失败的 block 应回落到 Mark。
// 通过把 target_storage_name 指为未注册的 storage 触发 PrepareCopyTask 返回 EC_NOENT。
TEST_F(AdminServiceImplTest, TestBothFallbackToMarkOnCopyFailure) {
    SeedServingSource(401);
    SeedServingSource(402);

    auto rc = std::make_shared<RequestContext>("t");
    auto req = MakeReq(proto::admin::MIGRATION_METHOD_BOTH, {401, 402});
    req.set_target_storage_name("nonexistent_cold"); // 未注册 -> Copy 必败

    proto::admin::MigrateCacheResponse resp;
    admin_->MigrateCache(rc.get(), &req, &resp);

    ASSERT_EQ(proto::admin::OK, resp.header().status().code());
    // Copy 全失败 -> 全部回落 Mark -> accepted = 2
    ASSERT_EQ(2, resp.accepted());
    ASSERT_EQ(0, resp.rejected());
    ASSERT_FALSE(cache_manager_->migration_manager()->HasMigrationTask(kInstance, 401));
    ASSERT_FALSE(cache_manager_->migration_manager()->HasMigrationTask(kInstance, 402));
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite(kInstance, 401));
    ASSERT_TRUE(cache_manager_->migration_manager()->IsMarkedForTieredWrite(kInstance, 402));
    ASSERT_EQ("nonexistent_cold", cache_manager_->migration_manager()->GetTieredWriteTarget(kInstance, 401));
}

// 目标 storage 上已存在 SERVING / WRITING 副本时，API 应拒绝该 block（与 CacheReclaimer 一致，
// 由 MigrationManager::CheckCopyAdmission 统一裁决）。
TEST_F(AdminServiceImplTest, TestExplicitBlockKeysRejectsDstAlreadyHasCopy) {
    // 701: 源 SERVING + 目标 SERVING -> kTargetServingExists -> reject
    SeedServingSource(701);
    SeedLocationOnStorage(701, "cold_01", CLS_SERVING);
    // 702: 源 SERVING + 目标 WRITING -> kTargetWritingExists -> reject（旧 admin 实现遗漏）
    SeedServingSource(702);
    SeedLocationOnStorage(702, "cold_01", CLS_WRITING);
    // 703: 仅源 SERVING -> kAccept -> 应被接受
    SeedServingSource(703);

    auto rc = std::make_shared<RequestContext>("t");
    auto req = MakeReq(proto::admin::MIGRATION_METHOD_COPY, {701, 702, 703});
    proto::admin::MigrateCacheResponse resp;
    admin_->MigrateCache(rc.get(), &req, &resp);

    ASSERT_EQ(proto::admin::OK, resp.header().status().code());
    ASSERT_EQ(1, resp.accepted());
    ASSERT_EQ(2, resp.rejected());
    ASSERT_FALSE(cache_manager_->migration_manager()->HasMigrationTask(kInstance, 701));
    ASSERT_FALSE(cache_manager_->migration_manager()->HasMigrationTask(kInstance, 702));
    ASSERT_TRUE(cache_manager_->migration_manager()->HasMigrationTask(kInstance, 703));
}
