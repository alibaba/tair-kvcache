#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "stub_source/kv_cache_manager/client/src/internal/sdk/tair_mempool_sdk.h"

namespace kv_cache_manager {
namespace {

TEST(TairMempoolRemoteItemTest, ParsesMaximumOffsetAndParametersExactly) {
    const auto item = TairMempoolRemoteItem::FromUri(
        DataStorageUri("pace:///18446744073709551615?media_type=4&node_id=5&range_id=6&size=4096"));

    EXPECT_EQ(std::numeric_limits<std::uint64_t>::max(), item.offset);
    EXPECT_EQ(4, item.media_type);
    EXPECT_EQ(5, item.node_id);
    EXPECT_EQ(6, item.range_id);
    EXPECT_EQ(4096, item.size);
}

TEST(TairMempoolRemoteItemTest, RejectsMalformedOffsetsWithoutThrowing) {
    const std::vector<std::string> invalid_offsets{
        "not-a-number",
        "-1",
        "+1",
        "12trailing",
        "18446744073709551616",
    };

    for (const auto &offset : invalid_offsets) {
        SCOPED_TRACE(offset);
        const auto item = TairMempoolRemoteItem::FromUri(
            DataStorageUri("pace:///" + offset + "?media_type=4&node_id=5&range_id=6&size=4096"));
        EXPECT_EQ(0, item.offset);
        EXPECT_EQ(4, item.media_type);
        EXPECT_EQ(5, item.node_id);
        EXPECT_EQ(6, item.range_id);
        EXPECT_EQ(4096, item.size);
    }
}

TEST(TairMempoolRemoteItemTest, DefaultsMissingPathAndParametersToZero) {
    const auto item = TairMempoolRemoteItem::FromUri(DataStorageUri(""));

    EXPECT_EQ(0, item.offset);
    EXPECT_EQ(0, item.media_type);
    EXPECT_EQ(0, item.node_id);
    EXPECT_EQ(0, item.range_id);
    EXPECT_EQ(0, item.size);
}

} // namespace
} // namespace kv_cache_manager
