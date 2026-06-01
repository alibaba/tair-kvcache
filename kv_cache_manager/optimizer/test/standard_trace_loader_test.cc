#include <fstream>
#include <limits>
#include <memory>
#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/trace_loader/standard_trace_loader.h"

using namespace kv_cache_manager;

class StandardTraceLoaderTest : public TESTBASE {};

TEST_F(StandardTraceLoaderTest, ParsesStandardGetAndWriteRows) {
    const std::string path = GetTestTempRootPath() + "/standard_trace.jsonl";
    std::ofstream out(path);
    out << R"({"type":"get","instance_id":"instance-a","trace_id":"short-read","timestamp_ns":1000,"keys":[],"input_len":128,"query_type":"prefix_match","block_mask":[]})"
        << "\n";
    out << R"({"type":"write","instance_id":"instance-a","trace_id":"short-write","timestamp_ns":1001,"keys":[1,2],"input_len":"ignored"})"
        << "\n";
    out.close();

    const auto traces = StandardTraceLoader::LoadFromFile(path);
    ASSERT_EQ(traces.size(), 2);
    EXPECT_TRUE(traces[0]->keys().empty());
    EXPECT_EQ(traces[0]->input_len(), 128);
    EXPECT_EQ(traces[1]->keys().size(), 2);
    EXPECT_EQ(traces[1]->input_len(), -1);
}

TEST_F(StandardTraceLoaderTest, ParsesRequestRows) {
    const std::string path = GetTestTempRootPath() + "/request_trace.jsonl";
    std::ofstream out(path);
    out << R"({"type":"request","instance_id":"instance-a","trace_id":"request","timestamp_ns":1000,"keys":[1,2],"input_len":128,"query_type":"prefix_match","block_mask":[],"ttl_us":100})"
        << "\n";
    out.close();

    const auto traces = StandardTraceLoader::LoadFromFile(path);
    ASSERT_EQ(traces.size(), 1);
    auto request_trace = std::dynamic_pointer_cast<RequestSchemaTrace>(traces[0]);
    ASSERT_NE(request_trace, nullptr);
    EXPECT_EQ(request_trace->keys().size(), 2);
    EXPECT_EQ(request_trace->input_len(), 128);
    EXPECT_EQ(request_trace->ttl_us(), 100);
}

TEST_F(StandardTraceLoaderTest, MapsFullRangeUint64KeysToStableInt64Values) {
    const std::string path = GetTestTempRootPath() + "/uint64_keys_trace.jsonl";
    std::ofstream out(path);
    out << R"({"type":"get","instance_id":"instance-a","trace_id":"uint64-read","timestamp_ns":1000,"keys":[9223372036854775808,18446744073709551615],"input_len":32,"query_type":"prefix_match","block_mask":[]})"
        << "\n";
    out.close();

    const auto traces = StandardTraceLoader::LoadFromFile(path);
    ASSERT_EQ(traces.size(), 1);
    ASSERT_EQ(traces[0]->keys().size(), 2);
    EXPECT_EQ(traces[0]->keys()[0], std::numeric_limits<int64_t>::min());
    EXPECT_EQ(traces[0]->keys()[1], -1);
}
