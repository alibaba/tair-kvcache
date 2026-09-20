#include <chrono>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/test/mock_redis_client.h"
#include "kv_cache_manager/common/test/redis_test_base.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/redis_reclaim_sampler.h"

namespace kv_cache_manager {

class RedisReclaimSamplerTest : public RedisTestBase, public TESTBASE {
protected:
    void SetUp() override { EXPECT_CALL(client_, IsContextOk()).WillRepeatedly(Return(true)); }

    void ExpectScan(const std::string &cursor,
                    const std::string &next_cursor,
                    std::vector<std::optional<std::string>> keys,
                    const std::string &count = "128",
                    const std::string &pattern = "instance:*") {
        std::vector<ReplyUPtr> replies;
        replies.emplace_back(MakeFakeReplyScan(next_cursor, keys));
        EXPECT_CALL(client_,
                    TryExecPipeline(ElementsAre(ElementsAre(
                        StrEq("SCAN"), StrEq(cursor), StrEq("MATCH"), StrEq(pattern), StrEq("COUNT"), StrEq(count)))))
            .WillOnce(Return(ByMove(std::move(replies))));
    }

    ErrorCode Sample(const std::string &prefix, const int64_t count, KeyTypeVec &out_keys) {
        return sampler_.Sample(
            [this](const std::string &matching_prefix,
                   const std::string &cursor,
                   const int64_t scan_count,
                   std::string &out_next_cursor,
                   std::vector<std::string> &out_full_keys) {
                return client_.Scan(matching_prefix, cursor, scan_count, out_next_cursor, out_full_keys);
            },
            prefix,
            count,
            out_keys);
    }

    StandardUri uri_;
    MockRedisClient client_{uri_};
    RedisReclaimSampler sampler_;
};

TEST_F(RedisReclaimSamplerTest, TestNonPositiveCountDoesNotTouchRedis) {
    EXPECT_CALL(client_, TryExecPipeline(_)).Times(0);
    KeyTypeVec keys{999};
    EXPECT_EQ(EC_OK, Sample("instance:", 0, keys));
    EXPECT_TRUE(keys.empty());
    keys = {999};
    EXPECT_EQ(EC_OK, Sample("instance:", -1, keys));
    EXPECT_TRUE(keys.empty());
}

TEST_F(RedisReclaimSamplerTest, TestRejectsEmptyPrefixWithoutTouchingRedis) {
    EXPECT_CALL(client_, TryExecPipeline(_)).Times(0);
    KeyTypeVec keys{999};
    EXPECT_EQ(EC_BADARGS, Sample("", 1, keys));
    EXPECT_TRUE(keys.empty());
}

TEST_F(RedisReclaimSamplerTest, TestRejectsEmptyScanCallback) {
    RedisReclaimSampler::ScanCallback scan;
    KeyTypeVec keys{999};
    EXPECT_EQ(EC_BADARGS, sampler_.Sample(scan, "instance:", 1, keys));
    EXPECT_TRUE(keys.empty());
}

TEST_F(RedisReclaimSamplerTest, TestEscapesGlobCharactersInLiteralPrefix) {
    const std::string prefix = R"(tenant[*?\]:cache_)";
    ExpectScan("0", "0", {prefix + "42"}, "128", R"(tenant\[\*\?\\\]:cache_*)");
    KeyTypeVec keys;
    EXPECT_EQ(EC_OK, Sample(prefix, 1, keys));
    EXPECT_EQ((KeyTypeVec{42}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestBoundedEmptyScanContinuesFromSavedCursor) {
    for (int cursor = 0; cursor < 16; ++cursor) {
        ExpectScan(std::to_string(cursor), std::to_string(cursor + 1), {});
    }
    KeyTypeVec keys;
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_TRUE(keys.empty());

    ExpectScan("16", "0", {"instance:1"});
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{1}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestStopsAfterSoftTimeBudgetWithoutDiscardingCursor) {
    int scan_calls = 0;
    const RedisReclaimSampler::ScanCallback scan = [&](const std::string &,
                                                       const std::string &cursor,
                                                       const int64_t,
                                                       std::string &out_next_cursor,
                                                       std::vector<std::string> &out_keys) {
        ++scan_calls;
        EXPECT_EQ("0", cursor);
        std::this_thread::sleep_for(std::chrono::milliseconds(75));
        out_next_cursor = "7";
        out_keys.clear();
        return EC_OK;
    };
    KeyTypeVec keys;
    EXPECT_EQ(EC_OK, sampler_.Sample(scan, "instance:", 1, keys));
    EXPECT_TRUE(keys.empty());
    EXPECT_EQ(1, scan_calls);

    ExpectScan("7", "0", {"instance:8"});
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{8}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestDeduplicatesAndBuffersPageOverflow) {
    ExpectScan("0", "7", {"instance:1", "instance:1", "instance:2", "instance:3"});
    KeyTypeVec keys;
    EXPECT_EQ(EC_OK, Sample("instance:", 2, keys));
    EXPECT_EQ((KeyTypeVec{1, 2}), keys);

    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{3}), keys);

    ExpectScan("7", "0", {"instance:4"});
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{4}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestDeduplicatesScannedKeysAgainstBufferedKeys) {
    ExpectScan("0", "7", {"instance:1", "instance:2"});
    KeyTypeVec keys;
    ASSERT_EQ(EC_OK, Sample("instance:", 1, keys));
    ASSERT_EQ((KeyTypeVec{1}), keys);

    ExpectScan("7", "0", {"instance:2", "instance:3"});
    EXPECT_EQ(EC_OK, Sample("instance:", 2, keys));
    EXPECT_EQ((KeyTypeVec{2, 3}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestOverflowBufferHasHardLimit) {
    std::vector<std::optional<std::string>> page;
    page.reserve(5000);
    for (int key = 0; key < 5000; ++key) {
        page.emplace_back("instance:" + std::to_string(key));
    }
    ExpectScan("0", "9", std::move(page));
    KeyTypeVec keys;
    ASSERT_EQ(EC_OK, Sample("instance:", 1, keys));
    ASSERT_EQ((KeyTypeVec{0}), keys);

    ExpectScan("9", "0", {}, "904");
    ASSERT_EQ(EC_OK, Sample("instance:", 5000, keys));
    ASSERT_EQ(4096, keys.size());
    EXPECT_EQ(1, keys.front());
    EXPECT_EQ(4096, keys.back());
}

TEST_F(RedisReclaimSamplerTest, TestMalformedCacheKeyDoesNotBlockValidCandidates) {
    ExpectScan("0", "0", {"instance:not-an-int64", "instance:17"});
    KeyTypeVec keys;
    EXPECT_EQ(EC_OK, Sample("instance:", 2, keys));
    EXPECT_EQ((KeyTypeVec{17}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestCompletedCycleRestartsAtBaseCursor) {
    ExpectScan("0", "0", {});
    KeyTypeVec keys;
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_TRUE(keys.empty());

    ExpectScan("0", "0", {"instance:7"});
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{7}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestScanFailureClearsPartialResultAndRollsBackCursor) {
    ExpectScan("0", "5", {"instance:1"});
    EXPECT_CALL(client_,
                TryExecPipeline(ElementsAre(ElementsAre(
                    StrEq("SCAN"), StrEq("5"), StrEq("MATCH"), StrEq("instance:*"), StrEq("COUNT"), StrEq("128")))))
        .WillOnce(Return(ByMove(std::vector<ReplyUPtr>{})));
    KeyTypeVec keys;
    EXPECT_EQ(EC_ERROR, Sample("instance:", 2, keys));
    EXPECT_TRUE(keys.empty());

    ExpectScan("0", "0", {"instance:2"});
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{2}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestScanFailureRestoresDrainedOverflow) {
    ExpectScan("0", "9", {"instance:1", "instance:2"});
    KeyTypeVec keys;
    ASSERT_EQ(EC_OK, Sample("instance:", 1, keys));
    ASSERT_EQ((KeyTypeVec{1}), keys);

    EXPECT_CALL(client_,
                TryExecPipeline(ElementsAre(ElementsAre(
                    StrEq("SCAN"), StrEq("9"), StrEq("MATCH"), StrEq("instance:*"), StrEq("COUNT"), StrEq("128")))))
        .WillOnce(Return(ByMove(std::vector<ReplyUPtr>{})));
    EXPECT_EQ(EC_ERROR, Sample("instance:", 2, keys));
    EXPECT_TRUE(keys.empty());

    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{2}), keys);

    ExpectScan("9", "0", {"instance:3"});
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{3}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestUnexpectedPrefixRollsBackCursor) {
    ExpectScan("0", "5", {"other:1"});
    KeyTypeVec keys;
    EXPECT_EQ(EC_ERROR, Sample("instance:", 1, keys));
    EXPECT_TRUE(keys.empty());

    ExpectScan("0", "0", {"instance:1"});
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{1}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestRejectsMalformedEmptyCursor) {
    ExpectScan("0", "", {"instance:1"});
    KeyTypeVec keys;
    EXPECT_EQ(EC_ERROR, Sample("instance:", 1, keys));
    EXPECT_TRUE(keys.empty());
}

TEST_F(RedisReclaimSamplerTest, TestRejectsNonNumericCursorAndRollsBack) {
    ExpectScan("0", "not-a-cursor", {"instance:1"});
    KeyTypeVec keys;
    EXPECT_EQ(EC_ERROR, Sample("instance:", 1, keys));
    EXPECT_TRUE(keys.empty());

    ExpectScan("0", "0", {"instance:2"});
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{2}), keys);
}

TEST_F(RedisReclaimSamplerTest, TestRejectsCursorOutsideUint64Range) {
    ExpectScan("0", "18446744073709551616", {"instance:1"});
    KeyTypeVec keys;
    EXPECT_EQ(EC_ERROR, Sample("instance:", 1, keys));
    EXPECT_TRUE(keys.empty());
}

TEST_F(RedisReclaimSamplerTest, TestResetDropsCursorAndBufferedKeys) {
    ExpectScan("0", "9", {"instance:1", "instance:2"});
    KeyTypeVec keys;
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{1}), keys);

    sampler_.Reset();
    ExpectScan("0", "0", {"instance:9"});
    EXPECT_EQ(EC_OK, Sample("instance:", 1, keys));
    EXPECT_EQ((KeyTypeVec{9}), keys);
}

} // namespace kv_cache_manager
