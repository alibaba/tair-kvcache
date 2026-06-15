#include <atomic>
#include <chrono>
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
    void SetUp() override { LoggerBroker::InitLoggerForClientOnce(); }

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

TEST_F(ReplicationExecutorTest, PiggybackTasksPrioritizedOverAsync) {
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
    EXPECT_EQ(execution_order[0], 200);
    EXPECT_EQ(execution_order[1], 100);
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
