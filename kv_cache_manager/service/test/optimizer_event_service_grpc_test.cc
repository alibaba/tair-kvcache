#include <chrono>
#include <cstdint>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/instance_group.h"
#include "kv_cache_manager/config/instance_group_quota.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/config/leader_elector.h"
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/event_publisher.h"
#include "kv_cache_manager/event/optimizer_event_publisher.h"
#include "kv_cache_manager/event/optimizer_stream/subscription_event_sink.h"
#include "kv_cache_manager/event/spec_events/optimizer_event.h"
#include "kv_cache_manager/manager/cache_manager.h"
#include "kv_cache_manager/manager/meta_searcher_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"
#include "kv_cache_manager/service/grpc_service/optimizer_event_service_grpc.h"
#include "kv_cache_manager/service/server.h"

using namespace ::testing;

namespace kv_cache_manager {

namespace {

template <typename Predicate>
bool WaitUntil(Predicate predicate, int timeout_ms = 2000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return predicate();
}

std::shared_ptr<CacheGetEvent> MakeGetEvent() {
    auto event = std::make_shared<CacheGetEvent>("instance-a");
    event->SetEventTriggerTime();
    event->set_trace_id("trace-grpc-e2e");
    event->SetAddtionalArgs(
        "prefix_match", {11, 22, 33}, std::vector<std::int64_t>(700, 5), BlockMask(), 0, {"tp0", "tp1"});
    return event;
}

std::string EncodeTokenIdsLe64(const std::vector<int64_t> &token_ids) {
    std::string encoded;
    encoded.reserve(token_ids.size() * sizeof(int64_t));
    for (const auto token_id : token_ids) {
        const auto value = static_cast<uint64_t>(token_id);
        for (size_t byte_index = 0; byte_index < sizeof(int64_t); ++byte_index) {
            encoded.push_back(static_cast<char>((value >> (byte_index * 8)) & 0xff));
        }
    }
    return encoded;
}

class CaptureEventPublisher : public EventPublisher {
public:
    bool Init(const std::string & /*config*/) override { return true; }
    bool Publish(const std::shared_ptr<BaseEvent> &event) override {
        events.push_back(event);
        return true;
    }
    bool Stop() override { return true; }

    std::vector<std::shared_ptr<BaseEvent>> events;
};

} // namespace

class OptimizerEventServiceGRpcTest : public TESTBASE {
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
        capture_publisher_ = std::make_shared<CaptureEventPublisher>();
        ASSERT_TRUE(capture_publisher_->Init(""));
        ASSERT_TRUE(cache_manager_->event_manager()->RegisterPublisher("log_event_publisher", capture_publisher_));
        leader_elector_ = std::make_shared<LeaderElector>(nullptr, "test-lock", "test-node");
        service_ =
            std::make_unique<OptimizerEventServiceGRpc>(sink_, registry_manager_, leader_elector_, cache_manager_);
        SetServingState(RoleState::LEADER, true);

        grpc::ServerBuilder builder;
        Server::ConfigureRpcServerBuilder(&builder);
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        ASSERT_NE(nullptr, server_);
        ASSERT_GT(port_, 0);

        auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port_), grpc::InsecureChannelCredentials());
        stub_ = proto::optimizer::OptimizerEventStreamService::NewStub(channel);
    }

    void TearDown() override {
        if (publisher_) {
            publisher_->Stop();
        }
        if (server_) {
            server_->Shutdown();
        }
    }

    void SetServingState(RoleState role_state, bool recover_complete) {
        leader_elector_->role_state_ = role_state;
        leader_elector_->is_transitioning_ = false;
        registry_manager_->recover_complete_ = recover_complete;
        if (role_state == RoleState::LEADER && recover_complete) {
            service_->EnableSubscriptions();
        } else {
            service_->DisableSubscriptions();
        }
    }

    std::shared_ptr<SubscriptionEventSink> sink_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<CacheManager> cache_manager_;
    std::shared_ptr<LeaderElector> leader_elector_;
    std::unique_ptr<OptimizerEventServiceGRpc> service_;
    std::shared_ptr<OptimizerEventPublisher> publisher_;
    std::shared_ptr<CaptureEventPublisher> capture_publisher_;
    std::unique_ptr<proto::optimizer::OptimizerEventStreamService::Stub> stub_;
    std::unique_ptr<grpc::Server> server_;
    int port_ = 0;
};

TEST_F(OptimizerEventServiceGRpcTest, TestReportsUnavailableOnFollower) {
    SetServingState(RoleState::FOLLOWER, false);

    grpc::ClientContext configuration_context;
    proto::optimizer::KvcmConfigurationRequest configuration_request;
    proto::optimizer::KvcmConfigurationResponse configuration_response;
    ASSERT_TRUE(stub_->GetConfiguration(&configuration_context, configuration_request, &configuration_response).ok());
    EXPECT_EQ(proto::optimizer::SERVICE_NOT_READY, configuration_response.header().status().code());

    grpc::ClientContext subscription_context;
    proto::optimizer::OptimizerEventSubscriptionRequest subscription_request;
    auto reader = stub_->SubscribeEvents(&subscription_context, subscription_request);
    proto::optimizer::TraceQueryRequest event;
    EXPECT_FALSE(reader->Read(&event));
    EXPECT_EQ(grpc::StatusCode::UNAVAILABLE, reader->Finish().error_code());
    EXPECT_EQ(0u, sink_->SubscriberCount());
}

TEST_F(OptimizerEventServiceGRpcTest, TestReportsUnavailableWhileLeaderIsRecovering) {
    SetServingState(RoleState::LEADER, false);

    grpc::ClientContext configuration_context;
    proto::optimizer::KvcmConfigurationRequest configuration_request;
    proto::optimizer::KvcmConfigurationResponse configuration_response;
    ASSERT_TRUE(stub_->GetConfiguration(&configuration_context, configuration_request, &configuration_response).ok());
    EXPECT_EQ(proto::optimizer::SERVICE_NOT_READY, configuration_response.header().status().code());

    grpc::ClientContext subscription_context;
    proto::optimizer::OptimizerEventSubscriptionRequest subscription_request;
    auto reader = stub_->SubscribeEvents(&subscription_context, subscription_request);
    proto::optimizer::TraceQueryRequest event;
    EXPECT_FALSE(reader->Read(&event));
    EXPECT_EQ(grpc::StatusCode::UNAVAILABLE, reader->Finish().error_code());
    EXPECT_EQ(0u, sink_->SubscriberCount());
}

TEST_F(OptimizerEventServiceGRpcTest, TestDemotionClosesStreamAndReenableAcceptsSubscription) {
    grpc::ClientContext first_context;
    proto::optimizer::OptimizerEventSubscriptionRequest request;
    request.set_consumer_id("first-optimizer");
    auto first_reader = stub_->SubscribeEvents(&first_context, request);
    ASSERT_TRUE(WaitUntil([this] { return sink_->SubscriberCount() == 1; }));

    SetServingState(RoleState::FOLLOWER, false);
    proto::optimizer::TraceQueryRequest event;
    EXPECT_FALSE(first_reader->Read(&event));
    EXPECT_EQ(grpc::StatusCode::UNAVAILABLE, first_reader->Finish().error_code());
    EXPECT_EQ(0u, sink_->SubscriberCount());

    SetServingState(RoleState::LEADER, true);
    grpc::ClientContext second_context;
    second_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));
    request.set_consumer_id("second-optimizer");
    auto second_reader = stub_->SubscribeEvents(&second_context, request);
    ASSERT_TRUE(WaitUntil([this] { return sink_->SubscriberCount() == 1; }));
    ASSERT_TRUE(publisher_->Publish(MakeGetEvent()));
    EXPECT_TRUE(second_reader->Read(&event));
    second_context.TryCancel();
    EXPECT_FALSE(second_reader->Read(&event));
    EXPECT_EQ(grpc::StatusCode::CANCELLED, second_reader->Finish().error_code());
}

TEST_F(OptimizerEventServiceGRpcTest, TestPublishesCacheReadOverServerStream) {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));
    proto::optimizer::OptimizerEventSubscriptionRequest subscription;
    subscription.set_consumer_id("integration-optimizer");
    auto reader = stub_->SubscribeEvents(&context, subscription);
    ASSERT_TRUE(WaitUntil([this] { return sink_->SubscriberCount() == 1; }));

    auto source = MakeGetEvent();
    const auto event_time_us = source->event_trigger_time_us();
    ASSERT_TRUE(publisher_->Publish(source));

    proto::optimizer::TraceQueryRequest received;
    ASSERT_TRUE(reader->Read(&received));
    EXPECT_EQ("trace-grpc-e2e", received.trace_id());
    EXPECT_EQ("instance-a", received.instance_id());
    EXPECT_EQ((std::vector<std::int64_t>{11, 22, 33}),
              std::vector<std::int64_t>(received.block_keys().begin(), received.block_keys().end()));
    EXPECT_EQ(700, received.input_token_len());
    EXPECT_EQ(event_time_us * 1000, received.timestamp_ns());
    EXPECT_EQ(0, received.token_ids_size());
    EXPECT_EQ((std::vector<std::string>{"tp0", "tp1"}),
              std::vector<std::string>(received.location_spec_names().begin(), received.location_spec_names().end()));

    context.TryCancel();
    EXPECT_FALSE(reader->Read(&received));
    EXPECT_EQ(grpc::StatusCode::CANCELLED, reader->Finish().error_code());
}

TEST_F(OptimizerEventServiceGRpcTest, TestReportsOptimizerEventWithoutCacheLocationLookup) {
    RequestContext request_context("seed-report-instance");
    InstanceGroup instance_group;
    instance_group.set_name("report-group");
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context, instance_group));
    ASSERT_EQ(EC_OK,
              registry_manager_->RegisterInstance(&request_context,
                                                  "report-group",
                                                  "report-instance",
                                                  2,
                                                  {LocationSpecInfo("tp0", 128)},
                                                  ModelDeployment(),
                                                  {LocationSpecGroup("full_cache", {"tp0"})}));
    ASSERT_EQ(nullptr, cache_manager_->meta_searcher_manager_->GetMetaSearcher("report-instance"));

    grpc::ClientContext subscription_context;
    subscription_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));
    proto::optimizer::OptimizerEventSubscriptionRequest subscription;
    subscription.set_consumer_id("report-integration-optimizer");
    auto reader = stub_->SubscribeEvents(&subscription_context, subscription);
    ASSERT_TRUE(WaitUntil([this] { return sink_->SubscriberCount() == 1; }));

    grpc::ClientContext report_context;
    proto::optimizer::TraceQueryRequest request;
    request.set_trace_id("reported-trace");
    request.set_instance_id("report-instance");
    request.add_token_ids(1);
    request.add_token_ids(2);
    request.add_token_ids(3);
    request.add_token_ids(4);
    request.add_token_ids(5);
    request.set_timestamp_ns(123456789000);
    request.add_location_spec_names("tp0");
    proto::optimizer::CommonResponse response;
    ASSERT_TRUE(stub_->ReportOptimizerEvent(&report_context, request, &response).ok());
    ASSERT_EQ(proto::optimizer::OK, response.header().status().code());

    proto::optimizer::TraceQueryRequest received;
    ASSERT_TRUE(reader->Read(&received));
    EXPECT_EQ("reported-trace", received.trace_id());
    EXPECT_EQ("report-instance", received.instance_id());
    EXPECT_EQ(2, received.block_keys_size());
    EXPECT_EQ(5, received.input_token_len());
    EXPECT_EQ(123456789000, received.timestamp_ns());
    EXPECT_EQ((std::vector<std::string>{"tp0"}),
              std::vector<std::string>(received.location_spec_names().begin(), received.location_spec_names().end()));
    EXPECT_EQ(nullptr, cache_manager_->meta_searcher_manager_->GetMetaSearcher("report-instance"));

    ASSERT_EQ(1u, capture_publisher_->events.size());
    auto broadcast_event = std::dynamic_pointer_cast<CacheGetEvent>(capture_publisher_->events.front());
    ASSERT_NE(nullptr, broadcast_event);
    EXPECT_EQ("reported-trace", broadcast_event->trace_id());
    EXPECT_EQ("report-instance", broadcast_event->event_source());
    EXPECT_EQ(5, broadcast_event->input_token_len());
    EXPECT_EQ(123456789, broadcast_event->event_trigger_time_us());
    EXPECT_EQ((std::vector<std::string>{"tp0"}), broadcast_event->location_spec_names());

    grpc::ClientContext raw_report_context;
    proto::optimizer::TraceQueryRequest raw_request;
    raw_request.set_trace_id("reported-raw-trace");
    raw_request.set_instance_id("report-instance");
    raw_request.set_token_ids_le64(EncodeTokenIdsLe64({1, 2, 3, 4, 5}));
    proto::optimizer::CommonResponse raw_response;
    ASSERT_TRUE(stub_->ReportOptimizerEvent(&raw_report_context, raw_request, &raw_response).ok());
    ASSERT_EQ(proto::optimizer::OK, raw_response.header().status().code());

    proto::optimizer::TraceQueryRequest raw_received;
    ASSERT_TRUE(reader->Read(&raw_received));
    EXPECT_EQ("reported-raw-trace", raw_received.trace_id());
    EXPECT_EQ(std::vector<int64_t>(received.block_keys().begin(), received.block_keys().end()),
              std::vector<int64_t>(raw_received.block_keys().begin(), raw_received.block_keys().end()));
    EXPECT_EQ(5, raw_received.input_token_len());
    ASSERT_EQ(2u, capture_publisher_->events.size());
    auto raw_broadcast_event = std::dynamic_pointer_cast<CacheGetEvent>(capture_publisher_->events.back());
    ASSERT_NE(nullptr, raw_broadcast_event);
    EXPECT_EQ("reported-raw-trace", raw_broadcast_event->trace_id());

    subscription_context.TryCancel();
    EXPECT_FALSE(reader->Read(&received));
    EXPECT_EQ(grpc::StatusCode::CANCELLED, reader->Finish().error_code());
}

TEST_F(OptimizerEventServiceGRpcTest, TestAcceptsOptimizerEventAboveDefaultGrpcLimit) {
    RequestContext request_context("seed-large-report-instance");
    InstanceGroup instance_group;
    instance_group.set_name("large-report-group");
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context, instance_group));
    ASSERT_EQ(EC_OK,
              registry_manager_->RegisterInstance(&request_context,
                                                  "large-report-group",
                                                  "large-report-instance",
                                                  256,
                                                  {LocationSpecInfo("tp0", 128)},
                                                  ModelDeployment(),
                                                  {LocationSpecGroup("full_cache", {"tp0"})}));

    grpc::ClientContext report_context;
    proto::optimizer::TraceQueryRequest request;
    request.set_trace_id("large-raw-report");
    request.set_instance_id("large-report-instance");
    request.set_token_ids_le64(EncodeTokenIdsLe64(std::vector<int64_t>(600000, 1)));
    proto::optimizer::CommonResponse response;
    ASSERT_TRUE(stub_->ReportOptimizerEvent(&report_context, request, &response).ok());
    EXPECT_EQ(proto::optimizer::OK, response.header().status().code());
}

TEST_F(OptimizerEventServiceGRpcTest, TestRejectsInvalidOrUnknownOptimizerEvent) {
    grpc::ClientContext invalid_context;
    proto::optimizer::TraceQueryRequest invalid_request;
    invalid_request.set_trace_id("invalid-report");
    invalid_request.set_instance_id("missing-input");
    proto::optimizer::CommonResponse invalid_response;
    ASSERT_TRUE(stub_->ReportOptimizerEvent(&invalid_context, invalid_request, &invalid_response).ok());
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, invalid_response.header().status().code());

    grpc::ClientContext unknown_context;
    proto::optimizer::TraceQueryRequest unknown_request;
    unknown_request.set_trace_id("unknown-report");
    unknown_request.set_instance_id("unknown-instance");
    unknown_request.add_token_ids(1);
    proto::optimizer::CommonResponse unknown_response;
    ASSERT_TRUE(stub_->ReportOptimizerEvent(&unknown_context, unknown_request, &unknown_response).ok());
    EXPECT_EQ(proto::optimizer::INSTANCE_NOT_EXIST, unknown_response.header().status().code());

    grpc::ClientContext ambiguous_context;
    proto::optimizer::TraceQueryRequest ambiguous_request;
    ambiguous_request.set_trace_id("ambiguous-report");
    ambiguous_request.set_instance_id("unknown-instance");
    ambiguous_request.add_token_ids(1);
    ambiguous_request.set_token_ids_le64(EncodeTokenIdsLe64({1}));
    proto::optimizer::CommonResponse ambiguous_response;
    ASSERT_TRUE(stub_->ReportOptimizerEvent(&ambiguous_context, ambiguous_request, &ambiguous_response).ok());
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, ambiguous_response.header().status().code());

    grpc::ClientContext malformed_context;
    proto::optimizer::TraceQueryRequest malformed_request;
    malformed_request.set_trace_id("malformed-report");
    malformed_request.set_instance_id("unknown-instance");
    malformed_request.set_token_ids_le64("short");
    proto::optimizer::CommonResponse malformed_response;
    ASSERT_TRUE(stub_->ReportOptimizerEvent(&malformed_context, malformed_request, &malformed_response).ok());
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, malformed_response.header().status().code());
}

TEST_F(OptimizerEventServiceGRpcTest, TestReturnsInstanceConfigurationSnapshot) {
    RequestContext request_context("seed-config");
    InstanceGroup instance_group;
    instance_group.set_name("group-a");
    InstanceGroupQuota quota;
    quota.set_capacity(3LL * 1024 * 1024 * 1024);
    instance_group.set_quota(quota);
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context, instance_group));

    ASSERT_EQ(EC_OK,
              registry_manager_->RegisterInstance(&request_context,
                                                  "group-a",
                                                  "instance-a",
                                                  16,
                                                  {LocationSpecInfo("tp0", 128)},
                                                  ModelDeployment(),
                                                  {LocationSpecGroup("full_cache", {"tp0"})}));

    grpc::ClientContext context;
    proto::optimizer::KvcmConfigurationRequest request;
    request.set_trace_id("list-config");
    proto::optimizer::KvcmConfigurationResponse response;
    ASSERT_TRUE(stub_->GetConfiguration(&context, request, &response).ok());
    ASSERT_EQ(proto::optimizer::OK, response.header().status().code());
    ASSERT_EQ(1, response.instance_groups_size());
    EXPECT_EQ("group-a", response.instance_groups(0).name());
    EXPECT_EQ(3LL * 1024 * 1024 * 1024, response.instance_groups(0).capacity_bytes());
    ASSERT_EQ(1, response.instances_size());
    EXPECT_EQ("instance-a", response.instances(0).instance_id());
    EXPECT_EQ(16, response.instances(0).block_size());
    ASSERT_EQ(1, response.instances(0).location_spec_infos_size());
    EXPECT_EQ("tp0", response.instances(0).location_spec_infos(0).name());
    ASSERT_EQ(1, response.instances(0).location_spec_groups_size());
    EXPECT_EQ("full_cache", response.instances(0).location_spec_groups(0).name());
}

} // namespace kv_cache_manager
