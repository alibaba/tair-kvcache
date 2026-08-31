#include <rapidjson/document.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/spec_events/optimizer_query_hit_event.h"

namespace kv_cache_manager {

class OptimizerQueryHitEventTest : public TESTBASE {};

TEST_F(OptimizerQueryHitEventTest, SerializesPerRequestHitResults) {
    OptimizerQueryHitEvent event("instance-1");
    event.SetEventTriggerTimeUs(123456);
    event.SetAdditionalArgs("request-id-1", 987654321, 12, 3);
    event.AddCapacityResult(1.0, 2, 2.0 / 3.0, 10);
    event.AddCapacityResult(2.0, 3, 1.0, 20);
    event.SetTheoreticalResult(3, 1.0, 8);

    rapidjson::Document doc;
    doc.Parse(event.ToJsonString().c_str());
    ASSERT_FALSE(doc.HasParseError());
    EXPECT_STREQ("instance-1", doc["source"].GetString());
    EXPECT_STREQ("optimizer", doc["component"].GetString());
    EXPECT_STREQ("OptimizerQueryHitEvent", doc["type"].GetString());
    EXPECT_STREQ("request-id-1", doc["trace_id"].GetString());
    EXPECT_EQ(987654321, doc["request_timestamp_ns"].GetInt64());
    EXPECT_EQ(12, doc["input_token_len"].GetInt64());
    EXPECT_EQ(3, doc["total_blocks"].GetInt64());

    const auto &capacity_results = doc["capacity_results"];
    ASSERT_TRUE(capacity_results.IsArray());
    ASSERT_EQ(2, capacity_results.Size());
    EXPECT_DOUBLE_EQ(1.0, capacity_results[0]["capacity_gb"].GetDouble());
    EXPECT_EQ(2, capacity_results[0]["cache_hit_count"].GetInt64());
    EXPECT_DOUBLE_EQ(2.0 / 3.0, capacity_results[0]["hit_rate"].GetDouble());
    EXPECT_EQ(10, capacity_results[0]["current_unique_keys"].GetInt64());

    const auto &theoretical_result = doc["theoretical_result"];
    ASSERT_TRUE(theoretical_result.IsObject());
    EXPECT_EQ(3, theoretical_result["max_hit_count"].GetInt64());
    EXPECT_DOUBLE_EQ(1.0, theoretical_result["hit_rate"].GetDouble());
    EXPECT_EQ(8, theoretical_result["current_unique_keys"].GetInt64());
}

} // namespace kv_cache_manager
