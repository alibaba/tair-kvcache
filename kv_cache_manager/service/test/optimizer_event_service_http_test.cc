#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/event_publishers_config.h"
#include "kv_cache_manager/event/optimizer_event_publisher.h"
#include "kv_cache_manager/event/optimizer_stream/subscription_event_sink.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"
#include "kv_cache_manager/service/http_service/meta_service_http.h"
#include "kv_cache_manager/service/meta_service_impl.h"

namespace kv_cache_manager {

class OptimizerEventServiceHttpTest : public TESTBASE {
protected:
    void SetUp() override {
        EventPublishersConfig configs;
        ASSERT_TRUE(configs.FromJsonString(
            R"({"optimizer":{"queue_size":64,"max_subscribers":1,"subscriber_queue_size":16}})"));
        const auto &optimizer_config = configs.optimizer_event_publisher_config();
        sink_ = std::make_shared<SubscriptionEventSink>(optimizer_config);
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        registry_manager_ = std::make_shared<RegistryManager>("", metrics_registry_);
        ASSERT_TRUE(registry_manager_->Init());
        cache_manager_ = std::make_shared<CacheManager>(metrics_registry_, registry_manager_);
        ASSERT_TRUE(cache_manager_->Init());
        publisher_ = std::make_shared<OptimizerEventPublisher>(sink_, optimizer_config);
        ASSERT_TRUE(publisher_->Init(""));
        ASSERT_TRUE(cache_manager_->event_manager()->RegisterPublisher("optimizer_event_publisher", publisher_));

        meta_service_impl_ = std::make_shared<MetaServiceImpl>(cache_manager_, nullptr, nullptr);
        http_service_ = std::make_unique<MetaServiceHttp>(metrics_registry_, meta_service_impl_, registry_manager_);
        http_service_->Init();
        http_service_->RegisterHandler();

        RequestContext request_context("seed-http-report-instance");
        InstanceGroup instance_group;
        instance_group.set_name("http-report-group");
        ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context, instance_group));
        ASSERT_EQ(EC_OK,
                  registry_manager_->RegisterInstance(&request_context,
                                                      "http-report-group",
                                                      "http-report-instance",
                                                      2,
                                                      {LocationSpecInfo("tp0", 128)},
                                                      ModelDeployment(),
                                                      {LocationSpecGroup("full_cache", {"tp0"})}));
    }

    void TearDown() override {
        if (publisher_) {
            publisher_->Stop();
        }
    }

    std::shared_ptr<SubscriptionEventSink> sink_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<OptimizerEventPublisher> publisher_;
    std::shared_ptr<MetaServiceImpl> meta_service_impl_;
    std::unique_ptr<MetaServiceHttp> http_service_;
};

TEST_F(OptimizerEventServiceHttpTest, TestRegistersAndReportsOptimizerEvent) {
    EXPECT_EQ(1u, http_service_->post_handlers_.count("/api/reportOptimizerEvent"));
    auto subscription = sink_->Subscribe("http-report-test");
    ASSERT_NE(nullptr, subscription);

    proto::optimizer::TraceQueryRequest request;
    request.set_trace_id("http-report-trace");
    request.set_instance_id("http-report-instance");
    for (int64_t token_id = 1; token_id <= 5; ++token_id) {
        request.add_token_ids(token_id);
    }
    request.set_timestamp_ns(123456789000);
    request.add_location_spec_names("tp0");
    proto::optimizer::CommonResponse response;
    http_service_->ReportOptimizerEvent(nullptr, &request, &response);
    ASSERT_EQ(proto::optimizer::OK, response.header().status().code());

    proto::optimizer::TraceQueryRequest received;
    ASSERT_EQ(SubscriptionEventSink::Subscription::WaitResult::kEvent,
              subscription->WaitNext(&received, std::chrono::seconds(2)));
    EXPECT_EQ("http-report-trace", received.trace_id());
    EXPECT_EQ("http-report-instance", received.instance_id());
    EXPECT_EQ(2, received.block_keys_size());
    EXPECT_EQ(5, received.input_token_len());
    EXPECT_EQ(123456789000, received.timestamp_ns());
    EXPECT_EQ((std::vector<std::string>{"tp0"}),
              std::vector<std::string>(received.location_spec_names().begin(), received.location_spec_names().end()));
}

TEST_F(OptimizerEventServiceHttpTest, TestReportsValidationAndLeaderErrors) {
    proto::optimizer::TraceQueryRequest invalid_request;
    invalid_request.set_trace_id("invalid-http-report");
    invalid_request.set_instance_id("http-report-instance");
    proto::optimizer::CommonResponse invalid_response;
    http_service_->ReportOptimizerEvent(nullptr, &invalid_request, &invalid_response);
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, invalid_response.header().status().code());

    meta_service_impl_->DisableLeaderOnlyRequests();
    proto::optimizer::TraceQueryRequest follower_request;
    follower_request.set_trace_id("follower-http-report");
    follower_request.set_instance_id("http-report-instance");
    follower_request.add_token_ids(1);
    proto::optimizer::CommonResponse follower_response;
    http_service_->ReportOptimizerEvent(nullptr, &follower_request, &follower_response);
    EXPECT_EQ(proto::optimizer::SERVER_NOT_LEADER, follower_response.header().status().code());
}

} // namespace kv_cache_manager
