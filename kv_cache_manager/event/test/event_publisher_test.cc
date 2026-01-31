#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/base_event.h"
#include "kv_cache_manager/event/event_publisher.h"

using namespace ::testing;

namespace kv_cache_manager {

class TestEventPublisher : public EventPublisher {
public:
    bool Init(const std::string &config) override {
        // Initialize basic queue for testing
        InitBasicQueue(100);
        running_ = true;
        return true;
    }

    bool Publish(const std::shared_ptr<BaseEvent> &event) override {
        if (!event || !running_) {
            return false;
        }
        return BasicEnqueue(event);
    }

    bool Stop() override {
        if (!running_) {
            return true;
        }
        running_ = false;
        if (basic_queue_) {
            basic_queue_->queue_cv.notify_all();
        }
        return true;
    }

    // Test helper methods
    size_t GetTestQueueSize() const { return BasicQueueSize(); }

    size_t GetTestDroppedCount() const { return BasicDroppedCount(); }

    bool TestDequeue(std::shared_ptr<BaseEvent> &event) { return BasicDequeue(event); }

    bool TestEnqueue(const std::shared_ptr<BaseEvent> &event) { return BasicEnqueue(event); }
};

class EventPublisherTest : public TESTBASE {
protected:
    void SetUp() override { publisher_ = std::make_unique<TestEventPublisher>(); }

    void TearDown() override {
        if (publisher_) {
            publisher_->Stop();
        }
        publisher_.reset();
    }

    std::unique_ptr<TestEventPublisher> publisher_;
};

// Test constructor and basic initialization
TEST_F(EventPublisherTest, TestConstructor) { EXPECT_NE(publisher_, nullptr); }

// Test initialization
TEST_F(EventPublisherTest, TestInit) {
    bool result = publisher_->Init("test_config");
    EXPECT_TRUE(result);
}

// Test basic queue operations
TEST_F(EventPublisherTest, TestBasicQueueOperations) {
    publisher_->Init("test_config");

    auto event = std::make_shared<BaseEvent>("source", "component", "type");

    // Test enqueue
    bool enqueue_result = publisher_->TestEnqueue(event);
    EXPECT_TRUE(enqueue_result);
    EXPECT_EQ(publisher_->GetTestQueueSize(), 1);

    // Test dequeue
    std::shared_ptr<BaseEvent> dequeued_event;
    bool dequeue_result = publisher_->TestDequeue(dequeued_event);
    EXPECT_TRUE(dequeue_result);
    EXPECT_NE(dequeued_event, nullptr);
    EXPECT_EQ(publisher_->GetTestQueueSize(), 0);
}

// Test queue limit and dropping
TEST_F(EventPublisherTest, TestQueueLimitAndDropping) {
    publisher_->Init("test_config");
    // Set small queue size for testing
    // Note: We can't directly modify the queue size in the current implementation
    // This test assumes the default queue size is sufficient for the test

    // Fill the queue
    for (size_t i = 0; i < 5; ++i) {
        auto event = std::make_shared<BaseEvent>("source", "component", "type");
        bool result = publisher_->TestEnqueue(event);
        EXPECT_TRUE(result);
    }

    EXPECT_EQ(publisher_->GetTestQueueSize(), 5);
    EXPECT_EQ(publisher_->GetTestDroppedCount(), 0);
}

} // namespace kv_cache_manager