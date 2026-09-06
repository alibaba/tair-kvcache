#include <memory>
#include <string>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/kv_meta_manager.h"
#include "kv_cache_manager/manager/startup_config_loader.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/kv_meta_service.pb.h"
#include "kv_cache_manager/service/kv_meta_service_impl.h"

namespace kv_cache_manager {
namespace {

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

        kv_meta_manager_ = std::make_shared<KvMetaManager>(cache_manager_, registry_manager_);
        ASSERT_TRUE(kv_meta_manager_->Init());
        ASSERT_EQ(EC_OK,
                  kv_meta_manager_->RegisterInstance(&setup_context_, "default", kInstanceId, "service-test").first);
        service_ = std::make_unique<KvMetaServiceImpl>(cache_manager_, kv_meta_manager_, nullptr);
    }

    void TearDown() override {
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
    EXPECT_EQ(proto::kv_meta::SIZE_MISMATCH,
              wrong_size_response.header().status().code());
    EXPECT_TRUE(wrong_size_response.write_session_id().empty());
    EXPECT_TRUE(wrong_size_response.locations().empty());
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

} // namespace
} // namespace kv_cache_manager
