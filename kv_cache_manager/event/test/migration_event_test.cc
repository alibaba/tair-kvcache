#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/event/spec_events/migration_event.h"

#include <rapidjson/document.h>

namespace kv_cache_manager {

class MigrationEventTest : public TESTBASE {};

TEST_F(MigrationEventTest, TestMarkAddEventIncludesMethod) {
    MigrationMarkAddEvent event("instance_1");
    event.SetAdditionalArgs(123, "cold_01");

    rapidjson::Document doc;
    doc.Parse(event.ToJsonString().c_str());
    ASSERT_FALSE(doc.HasParseError());
    ASSERT_TRUE(doc.HasMember("type"));
    ASSERT_TRUE(doc.HasMember("method"));
    ASSERT_TRUE(doc.HasMember("dst_storage"));
    EXPECT_STREQ("MigrationMarkAdd", doc["type"].GetString());
    EXPECT_STREQ("mark", doc["method"].GetString());
    EXPECT_STREQ("cold_01", doc["dst_storage"].GetString());
}

TEST_F(MigrationEventTest, TestMarkConsumedEventIncludesMethod) {
    MigrationMarkConsumedEvent event("instance_1");
    event.SetAdditionalArgs(123, "cold_01");

    rapidjson::Document doc;
    doc.Parse(event.ToJsonString().c_str());
    ASSERT_FALSE(doc.HasParseError());
    ASSERT_TRUE(doc.HasMember("type"));
    ASSERT_TRUE(doc.HasMember("method"));
    ASSERT_TRUE(doc.HasMember("dst_storage"));
    EXPECT_STREQ("MigrationMarkConsumed", doc["type"].GetString());
    EXPECT_STREQ("mark", doc["method"].GetString());
    EXPECT_STREQ("cold_01", doc["dst_storage"].GetString());
}

} // namespace kv_cache_manager
