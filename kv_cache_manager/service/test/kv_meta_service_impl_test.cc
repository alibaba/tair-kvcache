#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/cache_config.h"
#include "kv_cache_manager/config/cache_reclaim_strategy.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/data_storage/nfs_backend.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/kv_meta_instance.h"
#include "kv_cache_manager/manager/kv_meta_manager.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.pb.h"
#include "kv_cache_manager/service/admin_service_impl.h"
#include "kv_cache_manager/service/kv_meta_service_impl.h"
#include "kv_cache_manager/service/meta_service_impl.h"

namespace kv_cache_manager {
namespace {

class FailingAbortDeleteNfsBackend : public NfsBackend {
public:
    explicit FailingAbortDeleteNfsBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : NfsBackend(std::move(metrics_registry)) {}

    std::vector<ErrorCode>
    Delete(const std::vector<DataStorageUri> &storage_uris, const std::string &, std::function<void()> cb) override {
        ++delete_attempts;
        if (cb) {
            cb();
        }
        return std::vector<ErrorCode>(storage_uris.size(), EC_IO_ERROR);
    }

    std::size_t delete_attempts{0};
};

class KvMetaServiceImplTest : public TESTBASE {
protected:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        ASSERT_TRUE(registry_manager_->Init());

        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        ASSERT_TRUE(cache_manager_->Init());

        StartupConfigLoader loader;
        ASSERT_TRUE(loader.Init(registry_manager_));
        ASSERT_TRUE(loader.Load(""));

        const auto nfs_backend = registry_manager_->data_storage_manager()->GetDataStorageBackend("nfs_01");
        ASSERT_TRUE(nfs_backend);
        const auto nfs_spec = std::dynamic_pointer_cast<NfsStorageSpec>(nfs_backend->GetStorageConfig().storage_spec());
        ASSERT_TRUE(nfs_spec);
        std::error_code nfs_root_ec;
        std::filesystem::create_directories(nfs_spec->root_path(), nfs_root_ec);
        ASSERT_FALSE(nfs_root_ec) << nfs_root_ec.message();

        kv_meta_manager_ = std::make_shared<KvMetaManager>(cache_manager_, registry_manager_);
        ASSERT_TRUE(kv_meta_manager_->Init());
        ASSERT_EQ(EC_OK,
                  kv_meta_manager_->RegisterInstance(&setup_context_, "default", kInstanceId, "service-test").first);
        service_ = std::make_unique<KvMetaServiceImpl>(cache_manager_, kv_meta_manager_, nullptr);
        legacy_meta_service_ = std::make_unique<MetaServiceImpl>(cache_manager_, nullptr, nullptr);
        legacy_meta_service_->EnableLeaderOnlyRequests();
        admin_service_ =
            std::make_unique<AdminServiceImpl>(cache_manager_, nullptr, metrics_registry_, registry_manager_, nullptr);
        admin_service_->EnableLeaderOnlyRequests();
    }

    void TearDown() override {
        admin_service_.reset();
        legacy_meta_service_.reset();
        service_.reset();
        kv_meta_manager_->Shutdown();
        kv_meta_manager_.reset();
        cache_manager_.reset();
        registry_manager_.reset();
        metrics_registry_.reset();
    }

    static constexpr const char *kInstanceId = "embedding-service-instance";
    RequestContext setup_context_{"kv_meta_service_setup"};
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<KvMetaManager> kv_meta_manager_;
    std::unique_ptr<KvMetaServiceImpl> service_;
    std::unique_ptr<MetaServiceImpl> legacy_meta_service_;
    std::unique_ptr<AdminServiceImpl> admin_service_;
};

TEST_F(KvMetaServiceImplTest, DynamicSizeProtocolIsAlignedAndFinishFailsClosed) {
    proto::kv_meta::PutStartRequest start_request;
    start_request.set_trace_id("put-start");
    start_request.set_instance_id(kInstanceId);
    start_request.add_keys("emb-17");
    start_request.add_keys("emb-33");
    start_request.add_value_sizes(17);
    start_request.add_value_sizes(33);
    start_request.set_write_timeout_seconds(30);
    proto::kv_meta::PutStartResponse start_response;
    RequestContext start_context(start_request.trace_id());
    service_->PutStart(&start_context, &start_request, &start_response);

    ASSERT_EQ(proto::kv_meta::OK, start_response.header().status().code());
    ASSERT_EQ(2, start_response.key_mask().values_size());
    EXPECT_FALSE(start_response.key_mask().values(0));
    EXPECT_FALSE(start_response.key_mask().values(1));
    ASSERT_EQ(2, start_response.locations_size());
    EXPECT_EQ(17, start_response.locations(0).value_size());
    EXPECT_EQ(33, start_response.locations(1).value_size());
    ASSERT_FALSE(start_response.write_session_id().empty());

    proto::kv_meta::PutFinishRequest finish_request;
    finish_request.set_trace_id("put-finish");
    finish_request.set_instance_id(kInstanceId);
    finish_request.set_write_session_id(start_response.write_session_id());

    // An absent mask cannot accidentally commit the session.
    proto::kv_meta::CommonResponse absent_mask_response;
    RequestContext absent_mask_context("finish-absent-mask");
    service_->PutFinish(&absent_mask_context, &finish_request, &absent_mask_response);
    EXPECT_EQ(proto::kv_meta::INVALID_ARGUMENT, absent_mask_response.header().status().code());

    // A present but misaligned mask is rejected without consuming the session.
    finish_request.mutable_success_keys()->add_values(true);
    proto::kv_meta::CommonResponse short_mask_response;
    RequestContext short_mask_context("finish-short-mask");
    service_->PutFinish(&short_mask_context, &finish_request, &short_mask_response);
    EXPECT_EQ(proto::kv_meta::SIZE_MISMATCH, short_mask_response.header().status().code());

    finish_request.mutable_success_keys()->add_values(true);
    proto::kv_meta::CommonResponse finish_response;
    RequestContext finish_context(finish_request.trace_id());
    service_->PutFinish(&finish_context, &finish_request, &finish_response);
    ASSERT_EQ(proto::kv_meta::OK, finish_response.header().status().code());

    proto::kv_meta::GetRequest get_request;
    get_request.set_trace_id("get");
    get_request.set_instance_id(kInstanceId);
    get_request.set_query_type(proto::kv_meta::QT_BATCH_GET);
    get_request.add_keys("emb-17");
    get_request.add_keys("missing");
    get_request.add_keys("emb-33");
    proto::kv_meta::GetResponse get_response;
    RequestContext get_context(get_request.trace_id());
    service_->Get(&get_context, &get_request, &get_response);

    ASSERT_EQ(proto::kv_meta::OK, get_response.header().status().code());
    ASSERT_EQ(3, get_response.locations_size());
    ASSERT_EQ(3, get_response.hit_mask().values_size());
    EXPECT_TRUE(get_response.hit_mask().values(0));
    EXPECT_FALSE(get_response.hit_mask().values(1));
    EXPECT_TRUE(get_response.hit_mask().values(2));
    EXPECT_EQ(17, get_response.locations(0).value_size());
    EXPECT_EQ(0, get_response.locations(1).value_size());
    EXPECT_EQ(33, get_response.locations(2).value_size());

    proto::kv_meta::PutStartRequest wrong_size_request;
    wrong_size_request.set_trace_id("put-existing-wrong-size");
    wrong_size_request.set_instance_id(kInstanceId);
    wrong_size_request.add_keys("emb-17");
    wrong_size_request.add_value_sizes(18);
    wrong_size_request.set_write_timeout_seconds(30);
    proto::kv_meta::PutStartResponse wrong_size_response;
    RequestContext wrong_size_context(wrong_size_request.trace_id());
    service_->PutStart(&wrong_size_context, &wrong_size_request, &wrong_size_response);
    EXPECT_EQ(proto::kv_meta::SIZE_MISMATCH, wrong_size_response.header().status().code());
    EXPECT_TRUE(wrong_size_response.write_session_id().empty());
    EXPECT_TRUE(wrong_size_response.locations().empty());
}

TEST_F(KvMetaServiceImplTest, MalformedPutStartAbortMustCompleteBeforeReportingInternalError) {
    auto [clean_start_ec, clean_start] =
        kv_meta_manager_->StartWrite(&setup_context_, kInstanceId, {"malformed-clean-abort"}, {17}, 30);
    ASSERT_EQ(EC_OK, clean_start_ec);
    ASSERT_FALSE(clean_start.write_session_id.empty());
    ASSERT_EQ(1, clean_start.session_item_count);
    EXPECT_EQ(EC_OK,
              service_->AbortMalformedPutStart(
                  &setup_context_, kInstanceId, clean_start.write_session_id, clean_start.session_item_count));

    auto [failed_start_ec, failed_start] =
        kv_meta_manager_->StartWrite(&setup_context_, kInstanceId, {"malformed-failed-abort"}, {19}, 30);
    ASSERT_EQ(EC_OK, failed_start_ec);
    ASSERT_FALSE(failed_start.write_session_id.empty());
    ASSERT_EQ(1, failed_start.session_item_count);

    auto storage_manager = registry_manager_->data_storage_manager();
    ASSERT_TRUE(storage_manager);
    auto original = storage_manager->GetDataStorageBackend("nfs_01");
    ASSERT_TRUE(original);
    auto failing = std::make_shared<FailingAbortDeleteNfsBackend>(metrics_registry_);
    ASSERT_EQ(EC_OK, failing->Open(original->GetStorageConfig(), setup_context_.trace_id()));
    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = failing;
    }

    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              service_->AbortMalformedPutStart(
                  &setup_context_, kInstanceId, failed_start.write_session_id, failed_start.session_item_count));
    EXPECT_EQ(1, failing->delete_attempts);
    EXPECT_EQ(EC_OUTCOME_UNKNOWN,
              service_->AbortMalformedPutStart(&setup_context_, kInstanceId, failed_start.write_session_id, 0));

    {
        std::unique_lock<std::shared_mutex> lock(storage_manager->rw_lock_);
        storage_manager->storage_map_["nfs_01"] = original;
    }
}

TEST_F(KvMetaServiceImplTest, IndependentLeaderGateRejectsRequests) {
    service_->DisableLeaderOnlyRequests();

    proto::kv_meta::GetRequest request;
    request.set_trace_id("standby-get");
    request.set_instance_id(kInstanceId);
    request.add_keys("key");
    proto::kv_meta::GetResponse response;
    RequestContext context(request.trace_id());
    service_->Get(&context, &request, &response);

    EXPECT_EQ(proto::kv_meta::SERVER_NOT_LEADER, response.header().status().code());
    EXPECT_TRUE(response.locations().empty());

    service_->WaitForAllLeaderOnlyRequestsToComplete();
}

TEST_F(KvMetaServiceImplTest, LegacyServicesRejectAndHideReservedKvMetaNamespace) {
    const auto [list_ec, instances] = registry_manager_->ListInstanceInfo(&setup_context_, "default");
    ASSERT_EQ(EC_OK, list_ec);
    const auto internal = std::find_if(instances.begin(), instances.end(), [](const auto &instance) {
        return instance && IsKvMetaInstance(*instance);
    });
    ASSERT_NE(instances.end(), internal);
    const std::string internal_instance_id = (*internal)->instance_id();

    {
        proto::meta::RegisterInstanceRequest request;
        request.set_instance_id(std::string(kKvMetaInternalInstancePrefix) + "future-format");
        proto::meta::RegisterInstanceResponse response;
        RequestContext context("legacy-meta-register-reserved");
        legacy_meta_service_->RegisterInstance(&context, &request, &response);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
    }
    {
        proto::meta::GetInstanceInfoRequest request;
        request.set_instance_id(internal_instance_id);
        proto::meta::GetInstanceInfoResponse response;
        RequestContext context("legacy-meta-get-instance-reserved");
        legacy_meta_service_->GetInstanceInfo(&context, &request, &response);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
    }
    {
        proto::meta::StartWriteCacheRequest request;
        request.set_instance_id(internal_instance_id);
        proto::meta::StartWriteCacheResponse response;
        RequestContext context("legacy-meta-start-reserved");
        legacy_meta_service_->StartWriteCache(&context, &request, &response);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
        EXPECT_TRUE(response.write_session_id().empty());
    }
    {
        proto::meta::FinishWriteCacheRequest request;
        request.set_instance_id(internal_instance_id);
        proto::meta::CommonResponse response;
        RequestContext context("legacy-meta-finish-reserved");
        legacy_meta_service_->FinishWriteCache(&context, &request, &response);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
    }
    {
        proto::meta::RemoveCacheRequest request;
        request.set_instance_id(internal_instance_id);
        proto::meta::CommonResponse response;
        RequestContext context("legacy-meta-remove-reserved");
        legacy_meta_service_->RemoveCache(&context, &request, &response);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
    }
    {
        proto::meta::TrimCacheRequest request;
        request.set_instance_id(internal_instance_id);
        proto::meta::CommonResponse response;
        RequestContext context("legacy-meta-trim-reserved");
        legacy_meta_service_->TrimCache(&context, &request, &response);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
    }
    {
        proto::meta::ReportEventRequest request;
        request.set_instance_id(internal_instance_id);
        proto::meta::ReportEventResponse response;
        RequestContext context("legacy-meta-report-reserved");
        legacy_meta_service_->ReportEvent(&context, &request, &response);
        EXPECT_EQ(proto::meta::INVALID_ARGUMENT, response.header().status().code());
    }
    {
        proto::admin::RegisterInstanceRequest request;
        request.set_instance_id(internal_instance_id);
        proto::admin::CommonResponse response;
        RequestContext context("legacy-admin-register-reserved");
        admin_service_->RegisterInstance(&context, &request, &response);
        EXPECT_EQ(proto::admin::INVALID_ARGUMENT, response.header().status().code());
    }
    {
        proto::admin::GetInstanceInfoRequest request;
        request.set_instance_id(internal_instance_id);
        proto::admin::GetInstanceInfoResponse response;
        RequestContext context("legacy-admin-get-instance-reserved");
        admin_service_->GetInstanceInfo(&context, &request, &response);
        EXPECT_EQ(proto::admin::INVALID_ARGUMENT, response.header().status().code());
    }
    {
        proto::admin::RemoveInstanceRequest request;
        request.set_instance_id(internal_instance_id);
        proto::admin::CommonResponse response;
        RequestContext context("legacy-admin-remove-instance-reserved");
        admin_service_->RemoveInstance(&context, &request, &response);
        EXPECT_EQ(proto::admin::INVALID_ARGUMENT, response.header().status().code());
    }
    {
        proto::admin::RemoveCacheRequest request;
        request.set_instance_id(internal_instance_id);
        proto::admin::CommonResponse response;
        RequestContext context("legacy-admin-remove-reserved");
        admin_service_->RemoveCache(&context, &request, &response);
        EXPECT_EQ(proto::admin::INVALID_ARGUMENT, response.header().status().code());
    }
    {
        proto::admin::MigrateCacheRequest request;
        request.set_instance_id(internal_instance_id);
        proto::admin::MigrateCacheResponse response;
        RequestContext context("legacy-admin-migrate-reserved");
        admin_service_->MigrateCache(&context, &request, &response);
        EXPECT_EQ(proto::admin::INVALID_ARGUMENT, response.header().status().code());
        EXPECT_EQ(0, response.accepted());
    }
    {
        proto::admin::ListInstanceInfoRequest request;
        request.set_instance_group_name("default");
        proto::admin::ListInstanceInfoResponse response;
        RequestContext context("legacy-admin-list-reserved");
        admin_service_->ListInstanceInfo(&context, &request, &response);
        ASSERT_EQ(proto::admin::OK, response.header().status().code());
        for (const auto &instance : response.instance_info()) {
            EXPECT_FALSE(HasKvMetaReservedInstancePrefix(instance.instance_id()));
        }
    }

    auto [get_ec, instance_info] = kv_meta_manager_->GetInstanceInfo(&setup_context_, kInstanceId);
    EXPECT_EQ(EC_OK, get_ec);
    ASSERT_TRUE(instance_info);
    EXPECT_EQ(kInstanceId, instance_info->instance_id());
}

TEST_F(KvMetaServiceImplTest, LegacyAdminCannotRemoveAGroupContainingKvMetaInstances) {
    proto::admin::RemoveInstanceGroupRequest request;
    request.set_name("default");
    proto::admin::CommonResponse response;
    RequestContext context("legacy-admin-remove-kvmeta-group");
    admin_service_->RemoveInstanceGroup(&context, &request, &response);

    EXPECT_EQ(proto::admin::INVALID_ARGUMENT, response.header().status().code());
    const auto [group_ec, group] = registry_manager_->GetInstanceGroup(&setup_context_, "default");
    EXPECT_EQ(EC_OK, group_ec);
    EXPECT_TRUE(group);
    const auto [instance_ec, instance] = kv_meta_manager_->GetInstanceInfo(&setup_context_, kInstanceId);
    EXPECT_EQ(EC_OK, instance_ec);
    EXPECT_TRUE(instance);
}

TEST_F(KvMetaServiceImplTest, UnsupportedFieldsAreRejectedWithoutManagerMutation) {
    proto::kv_meta::GetRequest get_request;
    get_request.set_trace_id("get-with-meta");
    get_request.set_instance_id(kInstanceId);
    get_request.add_keys("key");
    get_request.add_metas()->set_key("future-filter");
    proto::kv_meta::GetResponse get_response;
    RequestContext get_context(get_request.trace_id());
    service_->Get(&get_context, &get_request, &get_response);
    EXPECT_EQ(proto::kv_meta::UNSUPPORTED, get_response.header().status().code());

    proto::kv_meta::TrimRequest trim_request;
    trim_request.set_trace_id("timestamp-trim");
    trim_request.set_instance_id(kInstanceId);
    trim_request.set_strategy(proto::kv_meta::TS_TIMESTAMP);
    proto::kv_meta::CommonResponse trim_response;
    RequestContext trim_context(trim_request.trace_id());
    service_->Trim(&trim_context, &trim_request, &trim_response);
    EXPECT_EQ(proto::kv_meta::UNSUPPORTED, trim_response.header().status().code());
}

TEST_F(KvMetaServiceImplTest, OversizedRequestShapesAreRejectedAtTheRpcBoundary) {
    proto::kv_meta::GetRequest get_request;
    get_request.set_instance_id(kInstanceId);
    for (std::size_t i = 0; i <= kv_meta_manager_->limits().max_batch_items; ++i) {
        get_request.add_keys("key-" + std::to_string(i));
    }
    proto::kv_meta::GetResponse get_response;
    RequestContext get_context("oversized-get");
    service_->Get(&get_context, &get_request, &get_response);
    EXPECT_EQ(proto::kv_meta::INVALID_ARGUMENT, get_response.header().status().code());
    EXPECT_TRUE(get_response.locations().empty());

    proto::kv_meta::PutStartRequest start_request;
    start_request.set_instance_id(kInstanceId);
    start_request.add_keys("key");
    start_request.set_write_timeout_seconds(30);
    proto::kv_meta::PutStartResponse start_response;
    RequestContext start_context("misaligned-start");
    service_->PutStart(&start_context, &start_request, &start_response);
    EXPECT_EQ(proto::kv_meta::INVALID_ARGUMENT, start_response.header().status().code());
    EXPECT_TRUE(start_response.write_session_id().empty());

    proto::kv_meta::RemoveRequest remove_request;
    remove_request.set_instance_id(kInstanceId);
    remove_request.add_keys(std::string(kv_meta_manager_->limits().max_key_bytes + 1, 'k'));
    proto::kv_meta::CommonResponse remove_response;
    RequestContext remove_context("oversized-remove");
    service_->Remove(&remove_context, &remove_request, &remove_response);
    EXPECT_EQ(proto::kv_meta::INVALID_ARGUMENT, remove_response.header().status().code());

    proto::kv_meta::PutFinishRequest finish_request;
    finish_request.set_instance_id(kInstanceId);
    finish_request.set_write_session_id("not-looked-up");
    for (std::size_t i = 0; i <= kv_meta_manager_->limits().max_batch_items; ++i) {
        finish_request.mutable_success_keys()->add_values(true);
    }
    proto::kv_meta::CommonResponse finish_response;
    RequestContext finish_context("oversized-finish");
    service_->PutFinish(&finish_context, &finish_request, &finish_response);
    EXPECT_EQ(proto::kv_meta::INVALID_ARGUMENT, finish_response.header().status().code());

    finish_request.mutable_success_keys()->clear_values();
    finish_request.mutable_success_keys()->add_values(true);
    finish_request.set_write_session_id(std::string(kv_meta_manager_->limits().max_write_session_id_bytes + 1, 's'));
    proto::kv_meta::CommonResponse oversized_session_response;
    RequestContext oversized_session_context("oversized-finish-session");
    service_->PutFinish(&oversized_session_context, &finish_request, &oversized_session_response);
    EXPECT_EQ(proto::kv_meta::INVALID_ARGUMENT, oversized_session_response.header().status().code());
}

TEST_F(KvMetaServiceImplTest, RemoveReportsAnActiveWriterWithoutConsumingItsSession) {
    proto::kv_meta::PutStartRequest start_request;
    start_request.set_trace_id("active-remove-start");
    start_request.set_instance_id(kInstanceId);
    start_request.add_keys("active-remove");
    start_request.add_value_sizes(17);
    start_request.set_write_timeout_seconds(30);
    proto::kv_meta::PutStartResponse start_response;
    RequestContext start_context(start_request.trace_id());
    service_->PutStart(&start_context, &start_request, &start_response);
    ASSERT_EQ(proto::kv_meta::OK, start_response.header().status().code());

    proto::kv_meta::RemoveRequest remove_request;
    remove_request.set_trace_id("active-remove-request");
    remove_request.set_instance_id(kInstanceId);
    remove_request.add_keys("active-remove");
    proto::kv_meta::CommonResponse remove_response;
    RequestContext remove_context(remove_request.trace_id());
    service_->Remove(&remove_context, &remove_request, &remove_response);
    EXPECT_EQ(proto::kv_meta::WRITE_IN_PROGRESS, remove_response.header().status().code());

    proto::kv_meta::PutFinishRequest finish_request;
    finish_request.set_trace_id("active-remove-finish");
    finish_request.set_instance_id(kInstanceId);
    finish_request.set_write_session_id(start_response.write_session_id());
    finish_request.mutable_success_keys()->add_values(false);
    proto::kv_meta::CommonResponse finish_response;
    RequestContext finish_context(finish_request.trace_id());
    service_->PutFinish(&finish_context, &finish_request, &finish_response);
    EXPECT_EQ(proto::kv_meta::OK, finish_response.header().status().code());
}

TEST_F(KvMetaServiceImplTest, PutStartReportsAnActiveWriterWithoutClaimingACacheHit) {
    proto::kv_meta::PutStartRequest first_request;
    first_request.set_trace_id("active-put-start-first");
    first_request.set_instance_id(kInstanceId);
    first_request.add_keys("active-put-start");
    first_request.add_value_sizes(17);
    first_request.set_write_timeout_seconds(30);
    proto::kv_meta::PutStartResponse first_response;
    RequestContext first_context(first_request.trace_id());
    service_->PutStart(&first_context, &first_request, &first_response);
    ASSERT_EQ(proto::kv_meta::OK, first_response.header().status().code());
    ASSERT_FALSE(first_response.write_session_id().empty());

    proto::kv_meta::PutStartRequest second_request = first_request;
    second_request.set_trace_id("active-put-start-second");
    second_request.set_value_sizes(0, 99);
    proto::kv_meta::PutStartResponse second_response;
    RequestContext second_context(second_request.trace_id());
    service_->PutStart(&second_context, &second_request, &second_response);
    EXPECT_EQ(proto::kv_meta::WRITE_IN_PROGRESS, second_response.header().status().code());
    EXPECT_TRUE(second_response.write_session_id().empty());
    EXPECT_TRUE(second_response.key_mask().values().empty());
    EXPECT_TRUE(second_response.locations().empty());

    // The rejected contender must not consume or mutate the original writer's
    // session. It can still commit normally.
    proto::kv_meta::PutFinishRequest finish_request;
    finish_request.set_trace_id("active-put-start-finish");
    finish_request.set_instance_id(kInstanceId);
    finish_request.set_write_session_id(first_response.write_session_id());
    finish_request.mutable_success_keys()->add_values(true);
    proto::kv_meta::CommonResponse finish_response;
    RequestContext finish_context(finish_request.trace_id());
    service_->PutFinish(&finish_context, &finish_request, &finish_response);
    EXPECT_EQ(proto::kv_meta::OK, finish_response.header().status().code());

    second_request.set_value_sizes(0, 17);
    proto::kv_meta::PutStartResponse committed_response;
    RequestContext committed_context("active-put-start-committed-hit");
    service_->PutStart(&committed_context, &second_request, &committed_response);
    ASSERT_EQ(proto::kv_meta::OK, committed_response.header().status().code());
    ASSERT_EQ(1, committed_response.key_mask().values_size());
    EXPECT_TRUE(committed_response.key_mask().values(0));
    EXPECT_TRUE(committed_response.write_session_id().empty());
    EXPECT_TRUE(committed_response.locations().empty());
}

TEST_F(KvMetaServiceImplTest, TrimReportsAnActiveWriterWithoutConsumingItsSession) {
    proto::kv_meta::PutStartRequest start_request;
    start_request.set_trace_id("active-trim-start");
    start_request.set_instance_id(kInstanceId);
    start_request.add_keys("active-trim");
    start_request.add_value_sizes(17);
    start_request.set_write_timeout_seconds(30);
    proto::kv_meta::PutStartResponse start_response;
    RequestContext start_context(start_request.trace_id());
    service_->PutStart(&start_context, &start_request, &start_response);
    ASSERT_EQ(proto::kv_meta::OK, start_response.header().status().code());

    proto::kv_meta::TrimRequest trim_request;
    trim_request.set_trace_id("active-trim-request");
    trim_request.set_instance_id(kInstanceId);
    trim_request.set_strategy(proto::kv_meta::TS_REMOVE_ALL_CACHE);
    proto::kv_meta::CommonResponse trim_response;
    RequestContext trim_context(trim_request.trace_id());
    service_->Trim(&trim_context, &trim_request, &trim_response);
    EXPECT_EQ(proto::kv_meta::WRITE_IN_PROGRESS, trim_response.header().status().code());

    proto::kv_meta::PutFinishRequest finish_request;
    finish_request.set_trace_id("active-trim-finish");
    finish_request.set_instance_id(kInstanceId);
    finish_request.set_write_session_id(start_response.write_session_id());
    finish_request.mutable_success_keys()->add_values(false);
    proto::kv_meta::CommonResponse finish_response;
    RequestContext finish_context(finish_request.trace_id());
    service_->PutFinish(&finish_context, &finish_request, &finish_response);
    EXPECT_EQ(proto::kv_meta::OK, finish_response.header().status().code());
}

TEST_F(KvMetaServiceImplTest, InvalidReclaimConfigurationIsReportedAsServiceNotReady) {
    const auto [group_ec, current_group] = registry_manager_->GetInstanceGroup(&setup_context_, "default");
    ASSERT_EQ(EC_OK, group_ec);
    ASSERT_TRUE(current_group);
    ASSERT_TRUE(current_group->cache_config());
    ASSERT_TRUE(current_group->cache_config()->reclaim_strategy());

    auto cache_config = std::make_shared<CacheConfig>();
    ASSERT_TRUE(cache_config->FromJsonString(current_group->cache_config()->ToJsonString()));
    auto unsupported_strategy = std::make_shared<CacheReclaimStrategy>(*cache_config->reclaim_strategy());
    unsupported_strategy->set_reclaim_policy(ReclaimPolicy::POLICY_TTL);
    cache_config->set_reclaim_strategy(unsupported_strategy);
    InstanceGroup updated_group(*current_group);
    updated_group.set_cache_config(cache_config);
    updated_group.set_version(current_group->version() + 1);
    ASSERT_EQ(EC_OK, registry_manager_->UpdateInstanceGroup(&setup_context_, updated_group, current_group->version()));

    proto::kv_meta::PutStartRequest request;
    request.set_trace_id("invalid-reclaim-config");
    request.set_instance_id(kInstanceId);
    request.add_keys("new-object");
    request.add_value_sizes(17);
    request.set_write_timeout_seconds(30);
    proto::kv_meta::PutStartResponse response;
    RequestContext context(request.trace_id());
    service_->PutStart(&context, &request, &response);

    EXPECT_EQ(proto::kv_meta::SERVICE_NOT_READY, response.header().status().code());
    EXPECT_TRUE(response.write_session_id().empty());
    EXPECT_TRUE(response.locations().empty());
}

} // namespace
} // namespace kv_cache_manager
