#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/base_event.h"

namespace kv_cache_manager {

class BaseEventTest : public TESTBASE {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// Test basic constructor and getter methods
TEST_F(BaseEventTest, TestConstructorAndGetters) {
    std::string source = "TestSource";
    std::string component = "TestComponent";
    std::string type = "BaseEvent_1";

    BaseEvent event(source, component, type);

    EXPECT_EQ(event.event_source(), source);
    EXPECT_EQ(event.event_component(), component);
    EXPECT_EQ(event.event_type(), type);
    EXPECT_EQ(event.event_trigger_time_us(), 0);
}

// Test SetEventTriggerTime
TEST_F(BaseEventTest, TestSetEventTriggerTime) {
    BaseEvent event("source", "component", "type");

    // Initially should be 0
    EXPECT_EQ(event.event_trigger_time_us(), 0);

    // Set trigger time
    event.SetEventTriggerTime();

    // Should be greater than 0
    EXPECT_GT(event.event_trigger_time_us(), 0);
}

} // namespace kv_cache_manager