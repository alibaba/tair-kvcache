#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/base_event.h"
#include "kv_cache_manager/event/event_manager.h"
#include "kv_cache_manager/event/log_event_publisher.h"

namespace kv_cache_manager {

class EventManagerTest : public TESTBASE {
protected:
    void SetUp() override {
        manager_ = std::make_unique<EventManager>();
        // Reset manager state
        manager_->Stop();
        manager_->ClearPublishers();
    }

    void TearDown() override {
        // Clean up
        manager_->Stop();
        manager_->ClearPublishers();
    }

    std::unique_ptr<EventManager> manager_;
};

// Test initialization
TEST_F(EventManagerTest, TestInit) {
    bool result = manager_->Init();
    EXPECT_TRUE(result);

    // Test double initialization
    result = manager_->Init();
    EXPECT_TRUE(result); // Should still return true
}

// Test publisher registration
TEST_F(EventManagerTest, TestPublisherRegistration) {
    manager_->Init();

    auto publisher = std::make_shared<LogEventPublisher>();
    std::string publisher_name = "test_publisher";

    // Test registration
    bool result = manager_->RegisterPublisher(publisher_name, publisher);
    EXPECT_TRUE(result);

    // Test duplicate registration
    result = manager_->RegisterPublisher(publisher_name, publisher);
    EXPECT_FALSE(result); // Should fail

    // Test has publisher
    EXPECT_TRUE(manager_->HasPublisher(publisher_name));

    // Test get publisher
    auto retrieved_publisher = manager_->GetPublisher(publisher_name);
    EXPECT_EQ(retrieved_publisher, publisher);

    // Test list publishers
    auto publisher_list = manager_->ListPublishers();
    EXPECT_EQ(publisher_list.size(), 1);
    EXPECT_EQ(publisher_list[0], publisher_name);
}

// Test publisher removal
TEST_F(EventManagerTest, TestPublisherRemoval) {
    manager_->Init();

    auto publisher = std::make_shared<LogEventPublisher>();
    std::string publisher_name = "test_publisher";

    // Register publisher
    ASSERT_TRUE(manager_->RegisterPublisher(publisher_name, publisher));

    // Test removal
    bool result = manager_->RemovePublisher(publisher_name);
    EXPECT_TRUE(result);

    // Test has publisher after removal
    EXPECT_FALSE(manager_->HasPublisher(publisher_name));

    // Test get publisher after removal
    auto retrieved_publisher = manager_->GetPublisher(publisher_name);
    EXPECT_EQ(retrieved_publisher, nullptr);

    // Test remove non-existent publisher
    result = manager_->RemovePublisher("non_existent");
    EXPECT_FALSE(result);
}

// Test event publishing
TEST_F(EventManagerTest, TestEventPublishing) {
    manager_->Init();

    // Create and register a test publisher
    auto publisher = std::make_shared<LogEventPublisher>();
    std::string test_log_file = GetPrivateTestRuntimeDataPath() + "test_events.log";
    ASSERT_TRUE(publisher->Init(test_log_file));
    ASSERT_TRUE(manager_->RegisterPublisher("log_publisher", publisher));

    // Create and publish event
    auto event = std::make_shared<BaseEvent>("test_source", "test_component", "test_type");
    event->SetEventTriggerTime();

    bool result = manager_->Publish(event);
    EXPECT_TRUE(result);

    // Give some time for async processing
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Stop to ensure all events are processed
    manager_->Stop();
}

// Test ClearPublishers
TEST_F(EventManagerTest, TestClearPublishers) {
    manager_->Init();

    // Register multiple publishers
    auto publisher1 = std::make_shared<LogEventPublisher>();
    auto publisher2 = std::make_shared<LogEventPublisher>();

    ASSERT_TRUE(manager_->RegisterPublisher("publisher1", publisher1));
    ASSERT_TRUE(manager_->RegisterPublisher("publisher2", publisher2));

    EXPECT_EQ(manager_->ListPublishers().size(), 2);

    // Clear publishers
    manager_->ClearPublishers();

    EXPECT_EQ(manager_->ListPublishers().size(), 0);
}

// Test Stop
TEST_F(EventManagerTest, TestStop) {
    manager_->Init();

    bool result = manager_->Stop();
    EXPECT_TRUE(result);

    // Test double stop
    result = manager_->Stop();
    EXPECT_TRUE(result); // Should still return true
}

} // namespace kv_cache_manager