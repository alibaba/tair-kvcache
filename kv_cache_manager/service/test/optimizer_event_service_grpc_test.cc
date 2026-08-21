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
#include "kv_cache_manager/config/model_deployment.h"
#include "kv_cache_manager/config/registry_manager.h"
#include "kv_cache_manager/event/optimizer_event_publisher.h"
#include "kv_cache_manager/event/optimizer_stream/subscription_event_sink.h"
#include "kv_cache_manager/event/spec_events/optimizer_event.h"
#include "kv_cache_manager/manager/hash_util.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"
#include "kv_cache_manager/service/grpc_service/optimizer_event_service_grpc.h"

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
        service_ = std::make_unique<OptimizerEventServiceGRpc>(sink_, registry_manager_, metrics_registry_);

        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port_);
        builder.RegisterService(service_.get());
        server_ = builder.BuildAndStart();
        ASSERT_NE(nullptr, server_);
        ASSERT_GT(port_, 0);

        auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(port_), grpc::InsecureChannelCredentials());
        stub_ = proto::optimizer::OptimizerEventStreamService::NewStub(channel);

        publisher_ = std::make_unique<OptimizerEventPublisher>(sink_, optimizer_config);
        ASSERT_TRUE(publisher_->Init(""));
    }

    void TearDown() override {
        if (publisher_) {
            publisher_->Stop();
        }
        if (server_) {
            server_->Shutdown();
        }
    }

    std::shared_ptr<SubscriptionEventSink> sink_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::unique_ptr<OptimizerEventServiceGRpc> service_;
    std::unique_ptr<OptimizerEventPublisher> publisher_;
    std::unique_ptr<proto::optimizer::OptimizerEventStreamService::Stub> stub_;
    std::unique_ptr<grpc::Server> server_;
    int port_ = 0;
};

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

TEST_F(OptimizerEventServiceGRpcTest, TestReportsOrderedTraceObservationBatch) {
    RequestContext request_context("seed-batch-config");
    InstanceGroup instance_group;
    instance_group.set_name("group-a");
    ASSERT_EQ(EC_OK, registry_manager_->CreateInstanceGroup(&request_context, instance_group));
    ASSERT_EQ(EC_OK,
              registry_manager_->RegisterInstance(&request_context,
                                                  "group-a",
                                                  "instance-a",
                                                  2,
                                                  {LocationSpecInfo("tp0", 128)},
                                                  ModelDeployment(),
                                                  {LocationSpecGroup("full_cache", {"tp0"})}));

    grpc::ClientContext stream_context;
    stream_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));
    proto::optimizer::OptimizerEventSubscriptionRequest subscription;
    subscription.set_consumer_id("batch-optimizer");
    auto reader = stub_->SubscribeEvents(&stream_context, subscription);
    ASSERT_TRUE(WaitUntil([this] { return sink_->SubscriberCount() == 1; }));

    proto::optimizer::TraceObservationBatchRequest request;
    request.set_producer_id("dashtrace-process-a");
    for (std::uint64_t sequence = 7; sequence < 9; ++sequence) {
        auto *observation = request.add_observations();
        observation->set_sequence(sequence);
        observation->set_trace_id("trace-" + std::to_string(sequence));
        observation->set_instance_id("instance-a");
        observation->set_timestamp_ns(1000 + sequence);
        observation->add_token_ids(1);
        observation->add_token_ids(2);
        observation->add_token_ids(3);
        observation->add_token_ids(4 + sequence);
    }

    grpc::ClientContext report_context;
    proto::optimizer::TraceObservationBatchResponse response;
    ASSERT_TRUE(stub_->ReportTraceBatch(&report_context, request, &response).ok());
    ASSERT_EQ(proto::optimizer::OK, response.header().status().code());
    EXPECT_EQ(2, response.accepted_count());
    EXPECT_EQ(8, response.last_accepted_sequence());

    for (std::uint64_t expected_sequence = 7; expected_sequence < 9; ++expected_sequence) {
        proto::optimizer::TraceQueryRequest received;
        ASSERT_TRUE(reader->Read(&received));
        EXPECT_EQ("dashtrace-process-a", received.producer_id());
        EXPECT_EQ(expected_sequence, received.source_sequence());
        EXPECT_EQ("trace-" + std::to_string(expected_sequence), received.trace_id());
        EXPECT_EQ("instance-a", received.instance_id());
        EXPECT_EQ(4, received.input_token_len());
        EXPECT_EQ(2, received.block_keys_size());
        const std::vector<std::int64_t> tokens = {1, 2, 3, static_cast<std::int64_t>(4 + expected_sequence)};
        const auto first_key = hashInt64Array(0, tokens.data(), tokens.data() + 2);
        const auto second_key = hashInt64Array(first_key, tokens.data() + 2, tokens.data() + 4);
        EXPECT_EQ(first_key, received.block_keys(0));
        EXPECT_EQ(second_key, received.block_keys(1));
        EXPECT_EQ(1000 + expected_sequence, received.timestamp_ns());
    }
    ASSERT_TRUE(WaitUntil([this] {
        return metrics_registry_->GetGauge("optimizer_event_stream.queue_size").Get() == 0;
    }));

    stream_context.TryCancel();
    proto::optimizer::TraceQueryRequest ignored;
    EXPECT_FALSE(reader->Read(&ignored));
    EXPECT_EQ(grpc::StatusCode::CANCELLED, reader->Finish().error_code());
}

TEST_F(OptimizerEventServiceGRpcTest, TestRejectsNonIncreasingBatchSequence) {
    proto::optimizer::TraceObservationBatchRequest request;
    request.set_producer_id("dashtrace-process-a");
    for (const std::uint64_t sequence : {4, 2}) {
        auto *observation = request.add_observations();
        observation->set_sequence(sequence);
        observation->set_trace_id("trace");
        observation->set_instance_id("instance-a");
        observation->add_token_ids(1);
    }

    grpc::ClientContext context;
    proto::optimizer::TraceObservationBatchResponse response;
    ASSERT_TRUE(stub_->ReportTraceBatch(&context, request, &response).ok());
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, response.header().status().code());
    EXPECT_EQ(0, response.accepted_count());
}

} // namespace kv_cache_manager
