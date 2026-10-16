#include <gtest/gtest.h>
#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/event_publishers_config.h"

using namespace ::testing;

namespace kv_cache_manager {

class EventPublishersConfigTest : public TESTBASE {};

TEST_F(EventPublishersConfigTest, TestLogPublisherDefaultsAndOverrides) {
    EventPublishersConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({})"));
    ASSERT_TRUE(config.enable_log_event_publisher());
    EXPECT_EQ(10000u, config.log_event_publisher_config().queue_size());

    ASSERT_TRUE(config.FromJsonString(R"({"log":{"enable":false,"queue_size":123}})"));
    EXPECT_FALSE(config.enable_log_event_publisher());
    EXPECT_EQ(123u, config.log_event_publisher_config().queue_size());
}

TEST_F(EventPublishersConfigTest, TestOptimizerPublisherRequiresExplicitEnable) {
    EventPublishersConfig config;
    ASSERT_TRUE(config.FromJsonString(R"({})"));
    EXPECT_FALSE(config.enable_optimizer_event_publisher());

    ASSERT_TRUE(config.FromJsonString(R"({"optimizer":{}})"));
    EXPECT_FALSE(config.enable_optimizer_event_publisher());

    ASSERT_TRUE(config.FromJsonString(R"({"optimizer":{"enable":true}})"));
    ASSERT_TRUE(config.enable_optimizer_event_publisher());
    const auto &optimizer = config.optimizer_event_publisher_config();
    EXPECT_EQ(100000u, optimizer.queue_size());
    EXPECT_EQ(4u, optimizer.max_subscribers());
    EXPECT_EQ(10000u, optimizer.subscriber_queue_size());
}

TEST_F(EventPublishersConfigTest, TestOptimizerPublisherConfigOverridesDefaults) {
    EventPublishersConfig config;
    ASSERT_TRUE(config.FromJsonString(
        R"({"optimizer":{"enable":true,"queue_size":123,"max_subscribers":2,"subscriber_queue_size":4096}})"));
    ASSERT_TRUE(config.enable_optimizer_event_publisher());
    const auto &optimizer = config.optimizer_event_publisher_config();
    EXPECT_EQ(123u, optimizer.queue_size());
    EXPECT_EQ(2u, optimizer.max_subscribers());
    EXPECT_EQ(4096u, optimizer.subscriber_queue_size());
}

TEST_F(EventPublishersConfigTest, TestRejectsInvalidConfig) {
    for (const auto &value : {
             std::string("not-json"),
             std::string(R"({"log":[]})"),
             std::string(R"({"log":{"enable":"true"}})"),
             std::string(R"({"log":{"queue_size":0}})"),
             std::string(R"({"optimizer":[]})"),
             std::string(R"({"optimizer":{"enable":"true"}})"),
             std::string(R"({"optimizer":{"queue_size":0}})"),
             std::string(R"({"optimizer":{"max_subscribers":0}})"),
             std::string(R"({"optimizer":{"subscriber_queue_size":0}})"),
         }) {
        EventPublishersConfig config;
        EXPECT_FALSE(config.FromJsonString(value)) << value;
    }
}

} // namespace kv_cache_manager
