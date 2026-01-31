#include <chrono>
#include <fstream>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/base_event.h"
#include "kv_cache_manager/event/log_event_publisher.h"

using namespace ::testing;

namespace kv_cache_manager {

class LogEventPublisherTest : public TESTBASE {
protected:
    void SetUp() override {
        publisher_ = std::make_unique<LogEventPublisher>();

        log_config_ = GetWorkspacePath() + "/logs/log_publisher.log";
    }

    void TearDown() override {
        if (publisher_) {
            publisher_->Stop();
        }
        publisher_.reset();
    }

    std::unique_ptr<LogEventPublisher> publisher_;
    std::string log_config_;
};

// Test constructor
TEST_F(LogEventPublisherTest, TestConstructor) { EXPECT_NE(publisher_, nullptr); }

// Test initialization
TEST_F(LogEventPublisherTest, TestInit) {
    bool result = publisher_->Init(log_config_);
    EXPECT_TRUE(result);
}

// Test Publish
TEST_F(LogEventPublisherTest, TestPublish) {
    ASSERT_TRUE(publisher_->Init(log_config_));

    auto event = std::make_shared<BaseEvent>("test_source", "test_component", "test_type");
    event->SetEventTriggerTime();

    bool result = publisher_->Publish(event);
    EXPECT_TRUE(result);

    // Give some time for async processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

// Test Stop
TEST_F(LogEventPublisherTest, TestStop) {
    ASSERT_TRUE(publisher_->Init(log_config_));

    bool result = publisher_->Stop();
    EXPECT_TRUE(result);
}

// Test multiple events
TEST_F(LogEventPublisherTest, TestMultipleEvents) {
    ASSERT_TRUE(publisher_->Init(log_config_));

    // Publish multiple events
    for (int i = 0; i < 5; ++i) {
        auto event = std::make_shared<BaseEvent>("source", "component", "type");
        event->SetEventTriggerTime();

        bool result = publisher_->Publish(event);
        EXPECT_TRUE(result);
    }
}

// Test inheritance relationship
TEST_F(LogEventPublisherTest, TestInheritance) {
    // Ensure LogEventPublisher inherits from EventPublisher
    EXPECT_TRUE(dynamic_cast<EventPublisher *>(publisher_.get()) != nullptr);
}

} // namespace kv_cache_manager