#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/trace_loader/standard_trace_loader.h"

using namespace kv_cache_manager;

class StandardTraceLoaderTest : public TESTBASE {};

TEST_F(StandardTraceLoaderTest, RejectsLegacyDialogTraceWithoutType) {
    const std::string path = GetTestTempRootPath() + "/legacy_dialog_trace.jsonl";
    std::ofstream out(path);
    out << R"({"instance_id":"instance-a","trace_id":"turn-1","timestamp_ns":1000,"keys":[1,2,3],"tokens":[],"input_len":48,"query_type":"prefix_match","block_mask":[],"total_keys":3,"output_len":16})"
        << "\n";
    out.close();

    try {
        (void)StandardTraceLoader::LoadFromFile(path);
        FAIL() << "expected legacy dialog-style trace to be rejected";
    } catch (const std::runtime_error &err) {
        EXPECT_THAT(std::string(err.what()), HasSubstr("legacy dialog-style trace without type is not supported"));
    }
}

TEST_F(StandardTraceLoaderTest, AllowsEmptyKeysWithPositiveInputLen) {
    const std::string path = GetTestTempRootPath() + "/empty_keys_trace.jsonl";
    std::ofstream out(path);
    out << R"({"type":"get","instance_id":"instance-a","trace_id":"short-read","timestamp_ns":1000,"keys":[],"tokens":[],"input_len":128,"query_type":"prefix_match","block_mask":[]})"
        << "\n";
    out << R"({"type":"write","instance_id":"instance-a","trace_id":"short-write","timestamp_ns":1001,"keys":[],"tokens":[],"input_len":128})"
        << "\n";
    out.close();

    const auto traces = StandardTraceLoader::LoadFromFile(path);
    ASSERT_EQ(traces.size(), 2);
    EXPECT_TRUE(traces[0]->keys().empty());
    EXPECT_EQ(traces[0]->input_len(), 128);
    EXPECT_TRUE(traces[1]->keys().empty());
    EXPECT_EQ(traces[1]->input_len(), 128);
}

TEST_F(StandardTraceLoaderTest, AllowsWriteTraceWithoutInputLen) {
    const std::string path = GetTestTempRootPath() + "/write_without_input_len.jsonl";
    std::ofstream out(path);
    out << R"({"type":"write","instance_id":"instance-a","trace_id":"write","timestamp_ns":1000,"keys":[1,2],"tokens":[]})"
        << "\n";
    out.close();

    const auto traces = StandardTraceLoader::LoadFromFile(path);
    ASSERT_EQ(traces.size(), 1);
    EXPECT_EQ(traces[0]->keys().size(), 2);
    EXPECT_EQ(traces[0]->input_len(), 2);
}

TEST_F(StandardTraceLoaderTest, MapsFullRangeUint64KeysToStableInt64Values) {
    const std::string path = GetTestTempRootPath() + "/uint64_keys_trace.jsonl";
    std::ofstream out(path);
    out << R"({"type":"get","instance_id":"instance-a","trace_id":"uint64-read","timestamp_ns":1000,"keys":[9223372036854775808,18446744073709551615],"tokens":[],"input_len":32,"query_type":"prefix_match","block_mask":[]})"
        << "\n";
    out.close();

    const auto traces = StandardTraceLoader::LoadFromFile(path);
    ASSERT_EQ(traces.size(), 1);
    ASSERT_EQ(traces[0]->keys().size(), 2);
    EXPECT_EQ(traces[0]->keys()[0], std::numeric_limits<int64_t>::min());
    EXPECT_EQ(traces[0]->keys()[1], -1);
}
