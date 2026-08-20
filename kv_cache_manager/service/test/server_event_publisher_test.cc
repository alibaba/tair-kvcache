#include <chrono>
#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/log_event_publisher.h"
#include "kv_cache_manager/event/optimizer_event_publisher.h"
#include "kv_cache_manager/event/optimizer_stream/subscription_event_sink.h"
#include "kv_cache_manager/service/grpc_service/optimizer_event_service_grpc.h"
#include "kv_cache_manager/service/server.h"

using namespace ::testing;

namespace kv_cache_manager {

namespace {

std::shared_ptr<EventManager> MakeEventManager() {
    auto manager = std::make_shared<EventManager>();
    EXPECT_TRUE(manager->Init());
    return manager;
}

void EnableOptimizerPublisher(Server *server) {
    server->config_.event_publishers_configs_ =
        R"({"optimizer":{"enable":true,"queue_size":123,"max_subscribers":2,"subscriber_queue_size":4096}})";
}

} // namespace

class ServerEventPublisherTest : public TESTBASE {};

TEST_F(ServerEventPublisherTest, TestOptimizerPublisherIsAbsentByDefault) {
    Server server;
    auto event_manager = MakeEventManager();
    server.RegisterEventPublishers(event_manager);

    EXPECT_TRUE(event_manager->HasPublisher("log_event_publisher"));
    EXPECT_FALSE(event_manager->HasPublisher("optimizer_event_publisher"));
    EXPECT_EQ(nullptr, server.optimizer_event_service_);
    event_manager->Stop();
}

TEST_F(ServerEventPublisherTest, TestConfiguredOptimizerPublisherIsRegistered) {
    Server server;
    EnableOptimizerPublisher(&server);
    auto event_manager = MakeEventManager();
    server.RegisterEventPublishers(event_manager);

    auto base = event_manager->GetPublisher("optimizer_event_publisher");
    auto publisher = std::dynamic_pointer_cast<OptimizerEventPublisher>(base);
    ASSERT_NE(nullptr, publisher);
    EXPECT_EQ(123u, publisher->config_.queue_size());

    auto sink = std::dynamic_pointer_cast<SubscriptionEventSink>(publisher->sink_);
    ASSERT_NE(nullptr, sink);
    EXPECT_EQ(2u, sink->config_.max_subscribers());
    EXPECT_EQ(4096u, sink->config_.subscriber_queue_size());
    EXPECT_NE(nullptr, server.optimizer_event_service_);
    event_manager->Stop();
}

TEST_F(ServerEventPublisherTest, TestLogPublisherCanBeDisabled) {
    Server server;
    server.config_.event_publishers_configs_ = R"({"log":{"enable":false}})";
    auto event_manager = MakeEventManager();
    server.RegisterEventPublishers(event_manager);

    EXPECT_FALSE(event_manager->HasPublisher("log_event_publisher"));
    EXPECT_FALSE(event_manager->HasPublisher("optimizer_event_publisher"));
    event_manager->Stop();
}

TEST_F(ServerEventPublisherTest, TestLogRegistrationFailureDoesNotBlockOptimizerPublisher) {
    Server server;
    EnableOptimizerPublisher(&server);
    auto event_manager = MakeEventManager();
    auto existing_log_publisher = std::make_shared<LogEventPublisher>();
    ASSERT_TRUE(existing_log_publisher->Init(""));
    ASSERT_TRUE(event_manager->RegisterPublisher("log_event_publisher", existing_log_publisher));

    server.RegisterEventPublishers(event_manager);

    EXPECT_EQ(existing_log_publisher, event_manager->GetPublisher("log_event_publisher"));
    EXPECT_TRUE(event_manager->HasPublisher("optimizer_event_publisher"));
    EXPECT_NE(nullptr, server.optimizer_event_service_);
    event_manager->Stop();
}

TEST_F(ServerEventPublisherTest, TestStopClosesOptimizerSubscriptions) {
    Server server;
    EnableOptimizerPublisher(&server);
    auto event_manager = MakeEventManager();
    server.RegisterEventPublishers(event_manager);

    auto publisher =
        std::dynamic_pointer_cast<OptimizerEventPublisher>(event_manager->GetPublisher("optimizer_event_publisher"));
    ASSERT_NE(nullptr, publisher);
    auto sink = std::dynamic_pointer_cast<SubscriptionEventSink>(publisher->sink_);
    ASSERT_NE(nullptr, sink);
    server.optimizer_event_service_->EnableSubscriptions();
    auto subscription = sink->Subscribe("optimizer");
    ASSERT_NE(nullptr, subscription);

    server.Stop();
    proto::optimizer::TraceQueryRequest event;
    EXPECT_EQ(SubscriptionEventSink::Subscription::WaitResult::kClosed,
              subscription->WaitNext(&event, std::chrono::milliseconds(10)));
    EXPECT_EQ(nullptr, sink->Subscribe("late"));
    event_manager->Stop();
}

} // namespace kv_cache_manager
