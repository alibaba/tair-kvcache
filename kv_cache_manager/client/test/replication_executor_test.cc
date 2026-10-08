#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <gmock/gmock.h>
#include <thread>

#include "kv_cache_manager/client/include/meta_client.h"
#include "kv_cache_manager/client/include/transfer_client.h"
#include "kv_cache_manager/client/src/replication_executor.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;
using namespace std::chrono_literals;
using ::testing::_;
using ::testing::Return;

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

class MockMetaClient : public MetaClient {
public:
    MOCK_METHOD((std::pair<ClientErrorCode, Locations>),
                MatchLocation,
                (const std::string &,
                 QueryType,
                 const std::vector<int64_t> &,
                 const std::vector<int64_t> &,
                 const BlockMask &,
                 int32_t,
                 const std::vector<std::string> &,
                 std::vector<ClientReplicationHint> &),
                (override));
    MOCK_METHOD((std::pair<ClientErrorCode, WriteLocation>),
                StartWrite,
                (const std::string &,
                 const std::vector<int64_t> &,
                 const std::vector<int64_t> &,
                 const std::vector<std::string> &,
                 int64_t,
                 bool),
                (override));
    MOCK_METHOD(ClientErrorCode,
                FinishWrite,
                (const std::string &, const std::string &, const BlockMask &, const Locations &),
                (override));
    MOCK_METHOD(ClientErrorCode,
                ReplicateCache,
                (const std::string &, const ClientReplicationHint &, int32_t),
                (override));
    MOCK_METHOD(
        (std::pair<ClientErrorCode, Metas>),
        MatchMeta,
        (const std::string &, const std::vector<int64_t> &, const std::vector<int64_t> &, const BlockMask &, int32_t),
        (override));
    MOCK_METHOD((std::pair<ClientErrorCode, int64_t>),
                MatchLocationLen,
                (const std::string &, QueryType, const std::vector<int64_t> &, const std::vector<int64_t> &, int32_t),
                (override));
    MOCK_METHOD(ClientErrorCode,
                RemoveCache,
                (const std::string &, const std::vector<int64_t> &, const std::vector<int64_t> &, const BlockMask &),
                (override));
    MOCK_METHOD(const std::string &, GetStorageConfig, (), (const, override));
    MOCK_METHOD(std::string, GetCallerNode, (), (const, override));
    MOCK_METHOD(ClientErrorCode, Init, (const std::string &, const InitParams &), (override));
    MOCK_METHOD(void, Shutdown, (), (override));
};

class MockTransferClient : public TransferClient {
public:
    MOCK_METHOD(ClientErrorCode,
                LoadKvCaches,
                (const UriStrVec &, const BlockBuffers &, std::shared_ptr<TransferTraceInfo>),
                (override));
    MOCK_METHOD((std::pair<ClientErrorCode, UriStrVec>),
                SaveKvCaches,
                (const UriStrVec &, const BlockBuffers &, std::shared_ptr<TransferTraceInfo>),
                (override));
    MOCK_METHOD(ClientErrorCode, Init, (const std::string &, const InitParams &), (override));
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static WriteLocation MakeWriteLocation(const std::string &session_id = "sess_1",
                                       const std::string &spec_name = "spec0",
                                       const std::string &uri = "rdma://dest/block/0") {
    WriteLocation wl;
    wl.write_session_id = session_id;
    wl.block_mask = BlockMaskOffset(1);
    Location loc = {LocationSpecUnit{spec_name, uri}};
    wl.locations = {loc};
    return wl;
}

static WriteLocation MakeEmptyWriteLocation() {
    WriteLocation wl;
    wl.write_session_id = "sess_empty";
    wl.block_mask = BlockMaskOffset(0);
    return wl;
}

static ClientReplicationHint
MakeHint(int64_t block_key, const std::string &target, const std::string &source = "rdma://src/block/0?size=1024") {
    return ClientReplicationHint{block_key, source, target};
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class ReplicationExecutorTest : public TESTBASE {
protected:
    void SetUp() override {
        LoggerBroker::InitLoggerForClientOnce();
        ON_CALL(mock_meta_, ReplicateCache(_, _, _)).WillByDefault(Return(ER_SERVICE_UNSUPPORTED));
    }

    std::unique_ptr<ReplicationExecutor> MakeExecutor(int num_workers = 1) {
        return std::make_unique<ReplicationExecutor>(&mock_meta_, &mock_transfer_, num_workers);
    }

    void SetupSuccessfulWritePath() {
        ON_CALL(mock_meta_, StartWrite(_, _, _, _, _, _))
            .WillByDefault(Return(std::make_pair(ER_OK, MakeWriteLocation())));
        ON_CALL(mock_transfer_, SaveKvCaches(_, _, _)).WillByDefault(Return(std::make_pair(ER_OK, UriStrVec{})));
        ON_CALL(mock_meta_, FinishWrite(_, _, _, _)).WillByDefault(Return(ER_OK));
    }

    void SetupSuccessfulAsyncPath() {
        SetupSuccessfulWritePath();
        ON_CALL(mock_transfer_, LoadKvCaches(_, _, _)).WillByDefault(Return(ER_OK));
    }

    testing::NiceMock<MockMetaClient> mock_meta_;
    testing::NiceMock<MockTransferClient> mock_transfer_;
};

// ===========================================================================
// ReleaseGuard
// ===========================================================================

TEST_F(ReplicationExecutorTest, ReleaseGuardFiresOnDestruction) {
    bool released = false;
    {
        ReleaseGuard guard([&] { released = true; });
    }
    EXPECT_TRUE(released);
}

TEST_F(ReplicationExecutorTest, ReleaseGuardEmptyIsSafe) {
    { ReleaseGuard guard; }
}

TEST_F(ReplicationExecutorTest, ReleaseGuardMoveTransfersOwnership) {
    int count = 0;
    {
        ReleaseGuard g1([&] { ++count; });
        ReleaseGuard g2(std::move(g1));
        EXPECT_EQ(count, 0);
    }
    EXPECT_EQ(count, 1);
}

TEST_F(ReplicationExecutorTest, ReleaseGuardMoveAssignReleasesOld) {
    int count_a = 0, count_b = 0;
    {
        ReleaseGuard ga([&] { ++count_a; });
        ReleaseGuard gb([&] { ++count_b; });
        ga = std::move(gb);
        EXPECT_EQ(count_a, 1);
        EXPECT_EQ(count_b, 0);
    }
    EXPECT_EQ(count_b, 1);
}

// ===========================================================================
// Submit dedup
// ===========================================================================

TEST_F(ReplicationExecutorTest, SubmitDedupWithinSameCall) {
    std::atomic<int> start_write_count{0};
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).WillRepeatedly([&](auto...) {
        ++start_write_count;
        return std::make_pair(ER_OK, MakeEmptyWriteLocation());
    });

    auto executor = MakeExecutor(1);
    ClientReplicationHint hint = MakeHint(42, "nodeA");
    executor->Submit({hint, hint, hint});
    executor->Shutdown();

    EXPECT_EQ(start_write_count.load(), 1);
}

TEST_F(ReplicationExecutorTest, SubmitDedupWhileInflight) {
    std::atomic<bool> entered{false};
    std::promise<void> proceed;
    auto proceed_future = proceed.get_future().share();
    std::atomic<int> start_write_count{0};

    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).WillRepeatedly([&](auto...) {
        ++start_write_count;
        entered.store(true);
        proceed_future.wait();
        return std::make_pair(ER_OK, MakeEmptyWriteLocation());
    });

    auto executor = MakeExecutor(1);
    ClientReplicationHint hint = MakeHint(42, "nodeA");
    executor->Submit({hint});
    while (!entered.load()) {
        std::this_thread::yield();
    }

    executor->Submit({hint});

    proceed.set_value();
    executor->Shutdown();

    EXPECT_EQ(start_write_count.load(), 1);
}

TEST_F(ReplicationExecutorTest, SubmitDifferentKeysNotDeduped) {
    std::atomic<int> start_write_count{0};
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).WillRepeatedly([&](auto...) {
        ++start_write_count;
        return std::make_pair(ER_OK, MakeEmptyWriteLocation());
    });

    auto executor = MakeExecutor(2);
    executor->Submit({MakeHint(1, "nodeA"), MakeHint(2, "nodeB")});
    executor->Shutdown();

    EXPECT_EQ(start_write_count.load(), 2);
}

// ===========================================================================
// SubmitWithData buffer release
// ===========================================================================

TEST_F(ReplicationExecutorTest, SubmitWithDataWhenStoppedReleasesBuffer) {
    auto executor = MakeExecutor(1);
    executor->Shutdown();

    std::atomic<int> release_count{0};
    char buf[16] = {};
    executor->SubmitWithData(MakeHint(1, "nodeA"), buf, sizeof(buf), [&] { ++release_count; });

    EXPECT_EQ(release_count.load(), 1);
}

TEST_F(ReplicationExecutorTest, SubmitWithDataDedupReleasesBuffer) {
    std::atomic<bool> entered{false};
    std::promise<void> proceed;
    auto proceed_future = proceed.get_future().share();

    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).WillRepeatedly([&](auto...) {
        entered.store(true);
        proceed_future.wait();
        return std::make_pair(ER_OK, MakeEmptyWriteLocation());
    });

    auto executor = MakeExecutor(1);
    char buf[16] = {};

    std::atomic<int> release1{0}, release2{0};
    executor->SubmitWithData(MakeHint(1, "nodeA"), buf, sizeof(buf), [&] { ++release1; });
    while (!entered.load()) {
        std::this_thread::yield();
    }

    executor->SubmitWithData(MakeHint(1, "nodeA"), buf, sizeof(buf), [&] { ++release2; });

    EXPECT_EQ(release2.load(), 1);

    proceed.set_value();
    executor->Shutdown();
    EXPECT_EQ(release1.load(), 1);
}

TEST_F(ReplicationExecutorTest, SubmitWithDataUpgradesQueuedAutomaticCopy) {
    std::atomic<int> starts{0};
    std::atomic<bool> first_entered{false};
    std::promise<void> proceed;
    auto proceed_future = proceed.get_future().share();
    EXPECT_CALL(mock_meta_, ReplicateCache(_, _, _)).Times(0);
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).WillRepeatedly([&](auto...) {
        if (starts.fetch_add(1) == 0) {
            first_entered.store(true);
            proceed_future.wait();
        }
        return std::make_pair(ER_OK, MakeEmptyWriteLocation());
    });
    EXPECT_CALL(mock_meta_, FinishWrite(_, _, _, _)).WillRepeatedly(Return(ER_OK));

    auto executor = MakeExecutor(1);
    char first[16]{}, second[16]{};
    std::atomic<int> first_release{0}, second_release{0};
    executor->SubmitWithData(MakeHint(10, "nodeA"), first, sizeof(first), [&] { ++first_release; });
    while (!first_entered.load()) std::this_thread::yield();
    executor->Submit({MakeHint(11, "nodeA")});
    executor->SubmitWithData(MakeHint(11, "nodeA"), second, sizeof(second), [&] { ++second_release; });
    EXPECT_EQ(0, second_release.load());

    proceed.set_value();
    executor->Shutdown();
    EXPECT_EQ(2, starts.load());
    EXPECT_EQ(1, first_release.load());
    EXPECT_EQ(1, second_release.load());
}

TEST_F(ReplicationExecutorTest, SubmitWithDataQueueDepthLimitReleasesBuffer) {
    std::atomic<bool> entered{false};
    std::promise<void> proceed;
    auto proceed_future = proceed.get_future().share();

    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).WillRepeatedly([&](auto...) {
        entered.store(true);
        proceed_future.wait();
        return std::make_pair(ER_OK, MakeEmptyWriteLocation());
    });

    auto executor = MakeExecutor(1); // max_piggyback_queue_ = 2
    char buf[16] = {};
    std::atomic<int> accepted{0}, rejected{0};

    executor->Submit({MakeHint(0, "blocker")});
    while (!entered.load()) {
        std::this_thread::yield();
    }

    executor->SubmitWithData(MakeHint(1, "n1"), buf, sizeof(buf), [&] { ++accepted; });
    executor->SubmitWithData(MakeHint(2, "n2"), buf, sizeof(buf), [&] { ++accepted; });
    executor->SubmitWithData(MakeHint(3, "n3"), buf, sizeof(buf), [&] { ++rejected; });

    EXPECT_EQ(rejected.load(), 1);
    EXPECT_EQ(accepted.load(), 0);

    proceed.set_value();
    executor->Shutdown();
    EXPECT_EQ(accepted.load(), 2);
}

// ===========================================================================
// SubmitWithData push_front priority
// ===========================================================================

TEST_F(ReplicationExecutorTest, PiggybackTasksPreserveFifoAndDoNotStarveAsync) {
    std::atomic<bool> entered{false};
    std::promise<void> proceed;
    auto proceed_future = proceed.get_future().share();
    std::vector<int64_t> execution_order;
    std::mutex order_mu;

    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _))
        .WillRepeatedly([&](const std::string &,
                            const std::vector<int64_t> &keys,
                            auto...) -> std::pair<ClientErrorCode, WriteLocation> {
            if (!entered.load()) {
                entered.store(true);
                proceed_future.wait();
                return std::make_pair(ER_OK, MakeEmptyWriteLocation());
            }
            {
                std::lock_guard<std::mutex> lk(order_mu);
                if (!keys.empty()) {
                    execution_order.push_back(keys[0]);
                }
            }
            return std::make_pair(ER_OK, MakeEmptyWriteLocation());
        });

    auto executor = MakeExecutor(1);

    executor->Submit({MakeHint(0, "blocker")});
    while (!entered.load()) {
        std::this_thread::yield();
    }

    executor->Submit({MakeHint(100, "async_node")});
    char buf[16] = {};
    executor->SubmitWithData(MakeHint(200, "piggyback_node"), buf, sizeof(buf), [] {});

    proceed.set_value();
    executor->Shutdown();

    ASSERT_GE(execution_order.size(), 2u);
    EXPECT_EQ(execution_order[0], 100);
    EXPECT_EQ(execution_order[1], 200);
}

// ===========================================================================
// End-to-end: ExecuteWrite (piggyback path)
// ===========================================================================

TEST_F(ReplicationExecutorTest, ExecuteWriteEndToEnd) {
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true))
        .WillOnce(Return(std::make_pair(ER_OK, MakeWriteLocation("sess_pb", "spec0", "rdma://dest/0"))));
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).WillOnce(Return(std::make_pair(ER_OK, UriStrVec{})));
    EXPECT_CALL(mock_meta_, FinishWrite(_, "sess_pb", _, _)).WillOnce(Return(ER_OK));

    std::atomic<int> release_count{0};
    auto executor = MakeExecutor(1);
    char buf[64] = {};
    executor->SubmitWithData(MakeHint(42, "nodeA"), buf, sizeof(buf), [&] { ++release_count; });
    executor->Shutdown();

    EXPECT_EQ(release_count.load(), 1);
}

TEST_F(ReplicationExecutorTest, ExecuteWriteStartWriteFailsReleasesBuffer) {
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _))
        .WillRepeatedly(Return(std::make_pair(ER_CONNECT_FAIL, WriteLocation{})));

    std::atomic<int> release_count{0};
    auto executor = MakeExecutor(1);
    char buf[16] = {};
    executor->SubmitWithData(MakeHint(1, "nodeA"), buf, sizeof(buf), [&] { ++release_count; });
    executor->Shutdown();

    EXPECT_EQ(release_count.load(), 1);
}

TEST_F(ReplicationExecutorTest, ExecuteWriteSaveFailsReleasesBuffer) {
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _))
        .WillRepeatedly(Return(std::make_pair(ER_OK, MakeWriteLocation())));
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _))
        .WillRepeatedly(Return(std::make_pair(ER_SDKWRITE_ERROR, UriStrVec{})));

    std::atomic<int> release_count{0};
    auto executor = MakeExecutor(1);
    char buf[16] = {};
    executor->SubmitWithData(MakeHint(1, "nodeA"), buf, sizeof(buf), [&] { ++release_count; });
    executor->Shutdown();

    EXPECT_EQ(release_count.load(), 1);
}

// ===========================================================================
// End-to-end: ExecuteHintAsync (async path)
// ===========================================================================

TEST_F(ReplicationExecutorTest, AutomaticHintUsesServerSideCopyWithoutClientTransfer) {
    EXPECT_CALL(mock_meta_, ReplicateCache(_, _, 60)).WillOnce(Return(ER_OK));
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).Times(0);
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).Times(0);

    auto executor = MakeExecutor(1);
    executor->Submit({MakeHint(98, "nodeB", "rdma://src/block/0?size=1024")});
    executor->Shutdown();
    EXPECT_EQ(1u, executor->GetStats().succeeded);
    EXPECT_EQ(1024u, executor->GetStats().copied_bytes);
}

TEST_F(ReplicationExecutorTest, ServerSideCopyFailureDoesNotIssueSecondCopy) {
    EXPECT_CALL(mock_meta_, ReplicateCache(_, _, 60)).WillOnce(Return(ER_SERVICE_INTERNAL_ERROR));
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).Times(0);
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).Times(0);

    auto executor = MakeExecutor(1);
    executor->Submit({MakeHint(97, "nodeB", "rdma://src/block/0?size=1024")});
    executor->Shutdown();
    EXPECT_EQ(1u, executor->GetStats().failed);
}

TEST_F(ReplicationExecutorTest, ExecuteHintAsyncEndToEnd) {
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true))
        .WillOnce(Return(std::make_pair(ER_OK, MakeWriteLocation("sess_async", "spec0", "rdma://dest/0"))));
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).WillOnce(Return(ER_OK));
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).WillOnce(Return(std::make_pair(ER_OK, UriStrVec{})));
    EXPECT_CALL(mock_meta_, FinishWrite(_, "sess_async", _, _)).WillOnce(Return(ER_OK));

    auto executor = MakeExecutor(1);
    executor->Submit({MakeHint(99, "nodeB", "rdma://src/block/0?size=1024")});
    executor->Shutdown();
}

TEST_F(ReplicationExecutorTest, ExecuteHintAsyncNoSizeInUriAborts) {
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).WillOnce(Return(std::make_pair(ER_OK, MakeWriteLocation())));

    auto executor = MakeExecutor(1);
    executor->Submit({MakeHint(1, "nodeA", "rdma://src/block/0")});
    executor->Shutdown();
}

// ===========================================================================
// Shutdown releases remaining queued buffers
// ===========================================================================

TEST_F(ReplicationExecutorTest, ShutdownReleasesAllBuffers) {
    SetupSuccessfulWritePath();

    std::atomic<int> release_count{0};
    auto release_fn = [&] { ++release_count; };

    auto executor = MakeExecutor(2);
    char buf[16] = {};
    executor->SubmitWithData(MakeHint(1, "n1"), buf, sizeof(buf), release_fn);
    executor->SubmitWithData(MakeHint(2, "n2"), buf, sizeof(buf), release_fn);
    executor->SubmitWithData(MakeHint(3, "n3"), buf, sizeof(buf), release_fn);
    executor->Shutdown();

    EXPECT_EQ(release_count.load(), 3);
}

TEST_F(ReplicationExecutorTest, MultiSpecCopiesByNameAndPublishesAllActualUris) {
    auto location = MakeWriteLocation("multi", "kv", "rdma://dest/kv?size=4");
    location.locations[0].push_back({"state", "rdma://dest/state?size=4"});
    auto hint = MakeHint(42, "reader", "");
    hint.source_specs = {{"state", "rdma://src/state?size=4"}, {"kv", "rdma://src/kv?size=4"}};
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true)).WillOnce(Return(std::make_pair(ER_OK, location)));
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).Times(2).WillRepeatedly([](const auto &uris, const auto &bufs, auto) {
        const char value = uris[0].find("/state?") != std::string::npos ? 's' : 'k';
        EXPECT_EQ(4u, bufs[0].iovs[0].size);
        std::memset(bufs[0].iovs[0].base, value, 4);
        return ER_OK;
    });
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).WillOnce([](const auto &uris, const auto &bufs, auto) {
        EXPECT_EQ((UriStrVec{"rdma://dest/kv?size=4", "rdma://dest/state?size=4"}), uris);
        EXPECT_EQ("kkkk", std::string(static_cast<const char *>(bufs[0].iovs[0].base), 4));
        EXPECT_EQ("ssss", std::string(static_cast<const char *>(bufs[1].iovs[0].base), 4));
        return std::make_pair(ER_OK, UriStrVec{"rdma://actual/kv", "rdma://actual/state"});
    });
    EXPECT_CALL(mock_meta_, FinishWrite(_, "multi", _, _)).WillOnce([](auto, auto, const auto &mask, const auto &locs) {
        EXPECT_EQ(1u, std::get<BlockMaskOffset>(mask));
        EXPECT_EQ((Locations{{{"kv", "rdma://actual/kv"}, {"state", "rdma://actual/state"}}}), locs);
        return ER_OK;
    });
    auto executor = MakeExecutor();
    executor->Submit({hint});
    executor->Shutdown();
}

TEST_F(ReplicationExecutorTest, MultiSpecLoadFailureAbortsWholeSession) {
    auto location = MakeWriteLocation("multi", "kv", "rdma://dest/kv");
    location.locations[0].push_back({"state", "rdma://dest/state"});
    auto hint = MakeHint(42, "reader", "");
    hint.source_specs = {{"kv", "rdma://src/kv?size=4"}, {"state", "rdma://src/state?size=4"}};
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true)).WillOnce(Return(std::make_pair(ER_OK, location)));
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).WillOnce(Return(ER_OK)).WillOnce(Return(ER_SDKREAD_ERROR));
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(mock_meta_, FinishWrite(_, "multi", _, _)).WillOnce([](auto, auto, const auto &mask, const auto &locs) {
        EXPECT_EQ((BlockMaskVector{false}), std::get<BlockMaskVector>(mask));
        EXPECT_TRUE(locs.empty());
        return ER_OK;
    });
    auto executor = MakeExecutor();
    executor->Submit({hint});
    executor->Shutdown();
}

TEST_F(ReplicationExecutorTest, LegacySingleSourceCannotPublishMultipleSpecs) {
    auto location = MakeWriteLocation();
    location.locations[0].push_back({"state", "rdma://dest/state"});
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true)).WillOnce(Return(std::make_pair(ER_OK, location)));
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(mock_meta_, FinishWrite(_, "sess_1", _, _)).WillOnce([](auto, auto, const auto &mask, const auto &) {
        EXPECT_EQ((BlockMaskVector{false}), std::get<BlockMaskVector>(mask));
        return ER_OK;
    });
    auto executor = MakeExecutor();
    char buffer[16]{};
    int releases = 0;
    executor->SubmitWithData(MakeHint(42, "reader"), buffer, sizeof(buffer), [&] { ++releases; });
    executor->Shutdown();
    EXPECT_EQ(1, releases);
}

TEST_F(ReplicationExecutorTest, MultiSpecPiggybackFallsBackToCompleteNamedSources) {
    auto location = MakeWriteLocation("multi", "kv", "rdma://dest/kv");
    location.locations[0].push_back({"state", "rdma://dest/state"});
    auto hint = MakeHint(42, "reader", "");
    hint.source_specs = {{"kv", "rdma://src/kv?size=4"}, {"state", "rdma://src/state?size=4"}};
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true)).WillOnce(Return(std::make_pair(ER_OK, location)));
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).Times(2).WillRepeatedly(Return(ER_OK));
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).WillOnce([](const auto &uris, const auto &bufs, auto) {
        EXPECT_EQ(2u, uris.size());
        EXPECT_EQ(2u, bufs.size());
        return std::make_pair(ER_OK, UriStrVec{});
    });
    EXPECT_CALL(mock_meta_, FinishWrite(_, "multi", _, _)).WillOnce([](auto, auto, const auto &mask, const auto &locs) {
        EXPECT_EQ(1u, std::get<BlockMaskOffset>(mask));
        EXPECT_EQ(2u, locs[0].size());
        return ER_OK;
    });
    auto executor = MakeExecutor();
    char buffer[4]{};
    int releases = 0;
    executor->SubmitWithData(hint, buffer, sizeof(buffer), [&] { ++releases; });
    executor->Shutdown();
    EXPECT_EQ(1, releases);
}

TEST_F(ReplicationExecutorTest, MissingSourceSpecNeverPublishesPartialReplica) {
    auto location = MakeWriteLocation("multi", "kv", "rdma://dest/kv");
    location.locations[0].push_back({"state", "rdma://dest/state"});
    auto hint = MakeHint(42, "reader", "");
    hint.source_specs = {{"unrelated", "rdma://src/other?size=4"}};
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true)).WillOnce(Return(std::make_pair(ER_OK, location)));
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(mock_meta_, FinishWrite(_, "multi", _, _)).WillOnce([](auto, auto, const auto &mask, const auto &) {
        EXPECT_EQ((BlockMaskVector{false}), std::get<BlockMaskVector>(mask));
        return ER_OK;
    });
    auto executor = MakeExecutor();
    executor->Submit({hint});
    executor->Shutdown();
}

TEST_F(ReplicationExecutorTest, SaveFailureAbortsSessionWithoutPublishing) {
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true)).WillOnce(Return(std::make_pair(ER_OK, MakeWriteLocation())));
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).WillOnce(Return(std::make_pair(ER_SDKWRITE_ERROR, UriStrVec{})));
    EXPECT_CALL(mock_meta_, FinishWrite(_, "sess_1", _, _)).WillOnce([](auto, auto, const auto &mask, const auto &locs) {
        EXPECT_EQ((BlockMaskVector{false}), std::get<BlockMaskVector>(mask));
        EXPECT_TRUE(locs.empty());
        return ER_OK;
    });
    auto executor = MakeExecutor();
    char buffer[4]{};
    executor->SubmitWithData(MakeHint(42, "reader"), buffer, sizeof(buffer), [] {});
    executor->Shutdown();
}

TEST_F(ReplicationExecutorTest, FailedPublishAcknowledgementDoesNotSendContradictoryAbort) {
    SetupSuccessfulWritePath();
    EXPECT_CALL(mock_meta_, FinishWrite(_, "sess_1", _, _)).WillOnce([](auto, auto, const auto &mask, const auto &) {
        EXPECT_EQ(1u, std::get<BlockMaskOffset>(mask));
        return ER_CONNECT_FAIL;
    });
    auto executor = MakeExecutor();
    char buffer[4]{};
    executor->SubmitWithData(MakeHint(42, "reader"), buffer, sizeof(buffer), [] {});
    executor->Shutdown();
}

TEST_F(ReplicationExecutorTest, CallerChangeDropsStaleTargetHint) {
    EXPECT_CALL(mock_meta_, GetCallerNode()).WillOnce(Return("new_provider"));
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).Times(0);
    auto executor = MakeExecutor();
    executor->Submit({MakeHint(42, "old_provider")});
    executor->Shutdown();
}

TEST_F(ReplicationExecutorTest, AsyncQueueIsBoundedAndStoppedExecutorReleasesBuffers) {
    std::promise<void> entered, proceed;
    auto entered_future = entered.get_future();
    auto proceed_future = proceed.get_future().share();
    std::vector<int64_t> executed;
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).WillRepeatedly([&](auto, const auto &keys, auto, auto, auto, auto) {
        executed.push_back(keys[0]);
        if (keys[0] == 1) {
            entered.set_value();
            proceed_future.wait();
        }
        return std::make_pair(ER_OK, MakeEmptyWriteLocation());
    });
    auto executor = std::make_unique<ReplicationExecutor>(&mock_meta_, &mock_transfer_, 1, 2);
    executor->Submit({MakeHint(1, "reader")});
    const bool ready = entered_future.wait_for(5s) == std::future_status::ready;
    if (ready) {
        executor->Submit({MakeHint(2, "reader"), MakeHint(3, "reader"), MakeHint(4, "reader")});
    }
    proceed.set_value();
    executor->Shutdown();
    ASSERT_TRUE(ready);
    EXPECT_EQ((std::vector<int64_t>{1, 2, 3}), executed);
    int releases = 0;
    char buffer[4]{};
    executor->SubmitWithData(MakeHint(5, "reader"), buffer, sizeof(buffer), [&] { ++releases; });
    EXPECT_EQ(1, releases);
}

TEST_F(ReplicationExecutorTest, PartialActualUriResponseAbortsWholeReplica) {
    auto location = MakeWriteLocation("multi", "kv", "rdma://dest/kv");
    location.locations[0].push_back({"state", "rdma://dest/state"});
    auto hint = MakeHint(42, "reader", "");
    hint.source_specs = {{"kv", "rdma://src/kv?size=4"}, {"state", "rdma://src/state?size=4"}};
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true)).WillOnce(Return(std::make_pair(ER_OK, location)));
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).Times(2).WillRepeatedly(Return(ER_OK));
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _))
        .WillOnce(Return(std::make_pair(ER_OK, UriStrVec{"rdma://actual/kv"})));
    EXPECT_CALL(mock_meta_, FinishWrite(_, "multi", _, _)).WillOnce([](auto, auto, const auto &mask, const auto &locs) {
        EXPECT_EQ((BlockMaskVector{false}), std::get<BlockMaskVector>(mask));
        EXPECT_TRUE(locs.empty());
        return ER_OK;
    });
    auto executor = MakeExecutor();
    executor->Submit({hint});
    executor->Shutdown();
}

TEST_F(ReplicationExecutorTest, EmptyBuffersDoNotConsumePiggybackQueueCapacity) {
    std::promise<void> entered, proceed;
    auto entered_future = entered.get_future();
    auto proceed_future = proceed.get_future().share();
    std::vector<int64_t> executed;
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).WillRepeatedly([&](auto, const auto &keys, auto, auto, auto, auto) {
        executed.push_back(keys[0]);
        if (keys[0] == 1) {
            entered.set_value();
            proceed_future.wait();
        }
        return std::make_pair(ER_OK, MakeEmptyWriteLocation());
    });
    auto executor = MakeExecutor();
    executor->Submit({MakeHint(1, "reader")});
    const bool ready = entered_future.wait_for(5s) == std::future_status::ready;
    char buffer[4]{};
    std::atomic<int> releases{0};
    if (ready) {
        executor->SubmitWithData(MakeHint(2, "reader"), nullptr, 0, [&] { ++releases; });
        executor->SubmitWithData(MakeHint(3, "reader"), nullptr, 0, [&] { ++releases; });
        executor->SubmitWithData(MakeHint(4, "reader"), buffer, sizeof(buffer), [&] { ++releases; });
    }
    proceed.set_value();
    executor->Shutdown();
    ASSERT_TRUE(ready);
    EXPECT_EQ((std::vector<int64_t>{1, 2, 3, 4}), executed);
    EXPECT_EQ(3, releases.load());
}

TEST(ReplicationResourcesTest, EnforcesAggregateMemoryAndOversizeWithoutWaiting) {
    ReplicationResources resources;
    std::atomic<bool> stopped{false};
    const auto deadline = ReplicationResources::Clock::now() + 1s;
    EXPECT_TRUE(resources.TryRetain(6, 10));
    EXPECT_FALSE(resources.TryRetain(5, 10));
    EXPECT_FALSE(resources.Acquire("a", "node", 11, 10, 11, 0, deadline, stopped));
    EXPECT_TRUE(resources.Acquire("a", "node", 4, 10, 4, 0, deadline, stopped));
    resources.Release(10);
    EXPECT_TRUE(resources.TryRetain(10, 10));
    resources.Release(10);
}

TEST(ReplicationResourcesTest, RotatesInstancesWhilePreservingFifoWithinInstance) {
    ReplicationResources resources;
    std::atomic<bool> stopped{false};
    ASSERT_TRUE(resources.TryRetain(1, 1));
    std::mutex order_mutex;
    std::vector<std::string> order;
    auto launch = [&](std::string instance, std::string task) {
        return std::async(std::launch::async, [&, instance, task] {
            if (!resources.Acquire(instance, "node", 1, 1, 1, 0,
                                   ReplicationResources::Clock::now() + 2s, stopped)) return false;
            { std::lock_guard<std::mutex> lock(order_mutex); order.push_back(task); }
            resources.Release(1);
            return true;
        });
    };
    auto wait_count = [&](size_t count) {
        auto deadline = ReplicationResources::Clock::now() + 1s;
        while (ReplicationResources::Clock::now() < deadline) {
            { std::lock_guard<std::mutex> lock(resources.mu_); if (resources.waiters_.size() == count) return true; }
            std::this_thread::yield();
        }
        return false;
    };
    auto a1 = launch("a", "a1"); EXPECT_TRUE(wait_count(1));
    auto a2 = launch("a", "a2"); EXPECT_TRUE(wait_count(2));
    auto b1 = launch("b", "b1"); EXPECT_TRUE(wait_count(3));
    resources.Release(1);
    EXPECT_TRUE(a1.get()); EXPECT_TRUE(a2.get()); EXPECT_TRUE(b1.get());
    EXPECT_EQ((std::vector<std::string>{"a1", "b1", "a2"}), order);
}

TEST(ReplicationResourcesTest, PacesEachNodeAndCancelsExpiredWorkWithoutLeakingBytes) {
    ReplicationResources resources;
    std::atomic<bool> stopped{false};
    auto deadline = ReplicationResources::Clock::now() + 100ms;
    ASSERT_TRUE(resources.Acquire("a", "slow", 1, 2, 1000, 1, deadline, stopped));
    resources.Release(1);
    EXPECT_FALSE(resources.Acquire("b", "slow", 1, 2, 1, 1, deadline, stopped));
    EXPECT_TRUE(resources.Acquire("b", "other", 1, 2, 1, 1,
                                  ReplicationResources::Clock::now() + 1s, stopped));
    resources.Release(1);
    EXPECT_TRUE(resources.TryRetain(2, 2));
    resources.Release(2);
}

TEST_F(ReplicationExecutorTest, MetricsReportAcknowledgedSuccessAndExporterFailureDoesNotBreakWorker) {
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true))
        .WillOnce(Return(std::make_pair(ER_OK, MakeWriteLocation())));
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).WillOnce(Return(std::make_pair(ER_OK, UriStrVec{})));
    EXPECT_CALL(mock_meta_, FinishWrite(_, _, _, _)).WillOnce(Return(ER_OK));
    ReplicationOptions options;
    std::atomic<int> exported{0};
    options.metrics_callback = [&](const auto &stats) {
        EXPECT_EQ(1u, stats.succeeded);
        ++exported;
        throw std::runtime_error("export unavailable");
    };
    ReplicationExecutor executor(&mock_meta_, &mock_transfer_, 1, 1024, options);
    char data[16]{};
    executor.SubmitWithData(MakeHint(1, "node"), data, sizeof(data), [] {});
    executor.Shutdown();
    const auto stats = executor.GetStats();
    EXPECT_EQ(1u, stats.submitted); EXPECT_EQ(1u, stats.admitted);
    EXPECT_EQ(1u, stats.succeeded); EXPECT_EQ(16u, stats.copied_bytes);
    EXPECT_EQ(0u, stats.failed); EXPECT_EQ(0u, stats.active); EXPECT_EQ(0u, stats.queued);
    EXPECT_EQ(1, exported.load());
}

TEST_F(ReplicationExecutorTest, MetricsDistinguishFailedSkippedExpiredAndBudgetDrops) {
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _))
        .WillOnce(Return(std::make_pair(ER_CONNECT_FAIL, WriteLocation{})))
        .WillOnce(Return(std::make_pair(ER_OK, MakeEmptyWriteLocation())));
    ReplicationExecutor executor(&mock_meta_, &mock_transfer_, 1);
    executor.Submit({MakeHint(1, "node"), MakeHint(2, "node")});
    executor.Shutdown();
    EXPECT_EQ(1u, executor.GetStats().failed);
    EXPECT_EQ(1u, executor.GetStats().skipped);
    EXPECT_EQ(0u, executor.GetStats().copied_bytes);
    ReplicationOptions options;
    options.max_age_ms = 0;
    options.max_buffer_bytes = 8;
    ReplicationExecutor expired(&mock_meta_, &mock_transfer_, 1, 1024, options);
    expired.Submit({MakeHint(3, "node", "rdma://src/x?size=4"),
                    MakeHint(4, "node", "rdma://src/x?size=16")});
    expired.Shutdown();
    EXPECT_EQ(1u, expired.GetStats().expired);
    EXPECT_EQ(1u, expired.GetStats().dropped_budget);
    EXPECT_EQ(0u, expired.GetStats().failed);
}

TEST_F(ReplicationExecutorTest, NamedBuffersReuseAllSpecsByNameAndRetainOwnersThroughPublication) {
    auto location = MakeWriteLocation("named", "kv", "rdma://dest/kv?size=4");
    location.locations[0].push_back({"state", "rdma://dest/state?size=4"});
    auto hint = MakeHint(900, "reader", "");
    hint.source_specs = {{"state", "rdma://src/state?size=4"}, {"kv", "rdma://src/kv?size=4"}};
    std::atomic<int> released{0};
    std::shared_ptr<const void> owner(new char[8]{'k','k','k','k','s','s','s','s'},
        [&](const void *p) { delete[] static_cast<const char *>(p); ++released; });
    std::weak_ptr<const void> weak = owner;
    const auto *base = static_cast<const char *>(owner.get());
    std::vector<ClientReplicationBuffer> buffers{
        {"state", base + 4, 4, MemoryType::CPU, owner}, {"kv", base, 4, MemoryType::CPU, owner}};
    owner.reset();
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true)).WillOnce(Return(std::make_pair(ER_OK, location)));
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).WillOnce([&](const auto &, const auto &data, auto) {
        EXPECT_FALSE(weak.expired());
        EXPECT_EQ(base, data[0].iovs[0].base);
        EXPECT_EQ(base + 4, data[1].iovs[0].base);
        EXPECT_EQ("kkkk", std::string(static_cast<const char *>(data[0].iovs[0].base), 4));
        return std::make_pair(ER_OK, UriStrVec{});
    });
    EXPECT_CALL(mock_meta_, FinishWrite(_, "named", _, _)).WillOnce([&](auto, auto, const auto &mask, const auto &locs) {
        EXPECT_FALSE(weak.expired());
        EXPECT_EQ(1u, std::get<BlockMaskOffset>(mask));
        EXPECT_EQ(2u, locs[0].size());
        return ER_OK;
    });
    auto executor = MakeExecutor();
    EXPECT_TRUE(executor->SubmitWithBuffers(hint, std::move(buffers)));
    executor->Shutdown();
    EXPECT_TRUE(weak.expired()); EXPECT_EQ(1, released.load());
    EXPECT_EQ(8u, executor->GetStats().copied_bytes);
}

TEST_F(ReplicationExecutorTest, NamedPartialReuseLoadsOnlyMissingSpecAndPropagatesMemoryType) {
    auto location = MakeWriteLocation("named", "kv", "rdma://dest/kv?size=4");
    location.locations[0].push_back({"state", "rdma://dest/state?size=4"});
    auto hint = MakeHint(901, "reader", "");
    hint.source_specs = {{"state", "rdma://src/state?size=4"}, {"kv", "rdma://src/kv?size=4"}};
    auto owner = std::make_shared<std::vector<char>>(4, 'k');
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true)).WillOnce(Return(std::make_pair(ER_OK, location)));
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).WillOnce([](const auto &uris, const auto &data, auto) {
        EXPECT_EQ((UriStrVec{"rdma://src/state?size=4"}), uris);
        std::memset(data[0].iovs[0].base, 's', 4);
        return ER_OK;
    });
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).WillOnce([](const auto &, const auto &data, auto) {
        EXPECT_EQ(MemoryType::GPU, data[0].iovs[0].type);
        EXPECT_EQ(MemoryType::CPU, data[1].iovs[0].type);
        EXPECT_EQ("ssss", std::string(static_cast<const char *>(data[1].iovs[0].base), 4));
        return std::make_pair(ER_OK, UriStrVec{});
    });
    EXPECT_CALL(mock_meta_, FinishWrite(_, "named", _, _)).WillOnce(Return(ER_OK));
    auto executor = MakeExecutor();
    EXPECT_TRUE(executor->SubmitWithBuffers(hint, {{"kv", owner->data(), 4, MemoryType::GPU, owner}}));
    executor->Shutdown();
}

TEST_F(ReplicationExecutorTest, NamedInvalidBuffersRejectBeforeAllocationAndReleaseOwnership) {
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).Times(0);
    auto hint = MakeHint(902, "reader", "");
    hint.source_specs = {{"kv", "rdma://src/kv?size=4"}};
    auto owner = std::make_shared<std::vector<char>>(4);
    auto executor = MakeExecutor();
    ClientReplicationBuffer valid{"kv", owner->data(), 4, MemoryType::CPU, owner};
    EXPECT_FALSE(executor->SubmitWithBuffers(hint, {valid, valid}));
    auto invalid = valid; invalid.owner.reset();
    EXPECT_FALSE(executor->SubmitWithBuffers(hint, {invalid}));
    invalid = valid; invalid.spec_name = "unknown";
    EXPECT_FALSE(executor->SubmitWithBuffers(hint, {invalid}));
    invalid = valid; invalid.size = 3;
    EXPECT_FALSE(executor->SubmitWithBuffers(hint, {invalid}));
    EXPECT_EQ(4u, executor->GetStats().dropped_invalid);
    executor->Shutdown();
    EXPECT_FALSE(executor->SubmitWithBuffers(hint, {valid}));
    EXPECT_EQ(1u, executor->GetStats().dropped_queue);
}

TEST_F(ReplicationExecutorTest, NamedTransferExceptionAbortsWholeSessionAndReleasesOwners) {
    auto hint = MakeHint(903, "reader", "");
    hint.source_specs = {{"kv", "rdma://src/kv?size=4"}};
    auto owner = std::make_shared<std::vector<char>>(4);
    std::weak_ptr<std::vector<char>> weak = owner;
    std::vector<ClientReplicationBuffer> buffers{{"kv", owner->data(), 4, MemoryType::CPU, owner}};
    owner.reset();
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true)).WillOnce(
        Return(std::make_pair(ER_OK, MakeWriteLocation("named", "kv", "rdma://dest/kv?size=4"))));
    EXPECT_CALL(mock_transfer_, LoadKvCaches(_, _, _)).Times(0);
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _)).WillOnce(testing::Throw(std::runtime_error("transfer failure")));
    EXPECT_CALL(mock_meta_, FinishWrite(_, "named", _, _)).WillOnce([](auto, auto, const auto &mask, const auto &locs) {
        EXPECT_EQ((BlockMaskVector{false}), std::get<BlockMaskVector>(mask));
        EXPECT_TRUE(locs.empty());
        return ER_OK;
    });
    auto executor = MakeExecutor();
    EXPECT_TRUE(executor->SubmitWithBuffers(hint, std::move(buffers)));
    executor->Shutdown();
    EXPECT_TRUE(weak.expired()); EXPECT_EQ(1u, executor->GetStats().failed);
    EXPECT_EQ(0u, executor->GetStats().copied_bytes);
}

TEST_F(ReplicationExecutorTest, TransferAbortAndReleaseExceptionsDoNotTerminateWorker) {
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, true))
        .WillOnce(Return(std::make_pair(ER_OK, MakeWriteLocation())))
        .WillOnce(Return(std::make_pair(ER_OK, MakeEmptyWriteLocation())));
    EXPECT_CALL(mock_transfer_, SaveKvCaches(_, _, _))
        .WillOnce(testing::Throw(std::runtime_error("transfer failure")));
    EXPECT_CALL(mock_meta_, FinishWrite(_, _, _, _))
        .WillOnce(testing::Throw(std::runtime_error("abort failure")))
        .WillOnce(Return(ER_OK));
    char data[16]{};
    std::atomic<int> releases{0};
    auto executor = MakeExecutor();
    executor->SubmitWithData(MakeHint(904, "reader"), data, sizeof(data), [&] {
        ++releases;
        throw std::runtime_error("release failure");
    });
    executor->Submit({MakeHint(905, "reader")});
    executor->Shutdown();
    EXPECT_EQ(1, releases.load());
    EXPECT_EQ(1u, executor->GetStats().failed);
    EXPECT_EQ(1u, executor->GetStats().skipped);
    EXPECT_EQ(0u, executor->GetStats().active);
}

TEST(ReleaseGuardTest, MoveAssignmentReleasesBothCallbacksExactlyOnceWhenTheyThrow) {
    int releases = 0;
    {
        ReleaseGuard first([&] { ++releases; throw std::runtime_error("first"); });
        ReleaseGuard second([&] { ++releases; throw std::runtime_error("second"); });
        first = std::move(second);
        EXPECT_EQ(1, releases);
    }
    EXPECT_EQ(2, releases);
}

TEST_F(ReplicationExecutorTest, ServerCopyHonorsTargetRateAcrossAutomaticTasks) {
    EXPECT_CALL(mock_meta_, ReplicateCache(_, _, 60)).Times(1).WillOnce(Return(ER_OK));
    EXPECT_CALL(mock_meta_, StartWrite(_, _, _, _, _, _)).Times(0);
    ReplicationOptions options;
    options.node_bytes_per_second = 1;
    options.max_age_ms = 50;
    std::promise<void> completed;
    auto done = completed.get_future();
    options.metrics_callback = [&](const auto &stats) {
        if (stats.succeeded + stats.expired + stats.failed == 2) completed.set_value();
    };
    ReplicationExecutor executor(&mock_meta_, &mock_transfer_, 1, 1024, options);
    executor.Submit({MakeHint(910, "server_rate", "rdma://src/x?size=16"),
                     MakeHint(911, "server_rate", "rdma://src/y?size=16")});
    ASSERT_EQ(std::future_status::ready, done.wait_for(2s));
    executor.Shutdown();
    EXPECT_EQ(1u, executor.GetStats().succeeded);
    EXPECT_EQ(1u, executor.GetStats().expired);
    EXPECT_EQ(16u, executor.GetStats().copied_bytes);
}

TEST_F(ReplicationExecutorTest, UnsupportedServerCopyFallbackConsumesOnlyOneRateReservation) {
    EXPECT_CALL(mock_meta_, ReplicateCache(_, _, 60)).WillOnce(Return(ER_SERVICE_UNSUPPORTED));
    SetupSuccessfulAsyncPath();
    ReplicationOptions options;
    options.node_bytes_per_second = 1;
    options.max_age_ms = 100;
    ReplicationExecutor executor(&mock_meta_, &mock_transfer_, 1, 1024, options);
    executor.Submit({MakeHint(912, "fallback_rate", "rdma://src/x?size=16")});
    executor.Shutdown();
    EXPECT_EQ(1u, executor.GetStats().succeeded);
    EXPECT_EQ(1u, executor.GetStats().client_fallback);
    EXPECT_EQ(0u, executor.GetStats().expired);
}
