#include "kv_cache_manager/config/raft_leader_elector.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/node_endpoint_info.h"
#include "kv_cache_manager/meta/raft/raft_coordinator.h"

namespace kv_cache_manager {

namespace {

int PickPort() {
    int base = 41000;
    int span = 5000;
    return base + static_cast<int>(::getpid() % span);
}

bool WaitFor(std::function<bool()> pred, std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return pred();
}

} // namespace

class RaftLeaderElectorTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        char tmpl[] = "/tmp/kvcm_raft_elector_XXXXXX";
        char *dir = ::mkdtemp(tmpl);
        ASSERT_NE(dir, nullptr);
        data_dir_ = dir;
    }

    raft_meta::RaftCoordinator::Config MakeSingleNodeConfig() {
        raft_meta::RaftCoordinator::Config cfg;
        cfg.server_id = 1;
        cfg.port = PickPort();
        cfg.self_endpoint = "127.0.0.1:" + std::to_string(cfg.port);

        NodeEndpointInfo info("node-1", "127.0.0.1", 8080, 8081, 9090, 9091, "test");
        cfg.self_aux = info.ToJsonString();

        cfg.data_dir = data_dir_;
        cfg.election_timeout_lower = 100;
        cfg.election_timeout_upper = 200;
        cfg.heart_beat_interval = 50;
        return cfg;
    }

    std::string data_dir_;
};

TEST_F(RaftLeaderElectorTest, StartRequiresHandlers) {
    raft_meta::RaftCoordinator coord;
    auto cfg = MakeSingleNodeConfig();
    ASSERT_EQ(EC_OK, coord.Start(cfg));

    RaftLeaderElector elector("node-1");
    elector.SetCoordinatorForTest(&coord);

    EXPECT_FALSE(elector.Start());

    elector.SetBecomeLeaderHandler([]() {});
    elector.SetNoLongerLeaderHandler([]() {});
    EXPECT_TRUE(elector.Start());

    coord.Stop();
}

TEST_F(RaftLeaderElectorTest, SingleNodeBecomesLeaderViaCallback) {
    raft_meta::RaftCoordinator coord;

    std::atomic<int> leader_count{0};
    RaftLeaderElector elector("node-1");
    elector.SetCoordinatorForTest(&coord);
    elector.SetBecomeLeaderHandler([&]() { leader_count.fetch_add(1); });
    elector.SetNoLongerLeaderHandler([]() {});
    ASSERT_TRUE(elector.Start());

    auto cfg = MakeSingleNodeConfig();
    ASSERT_EQ(EC_OK, coord.Start(cfg));

    ASSERT_TRUE(WaitFor([&] { return leader_count.load() > 0; }, std::chrono::seconds(5)))
        << "RaftLeaderElector never fired BecomeLeader callback";
    EXPECT_TRUE(elector.IsLeader());

    coord.Stop();
}

TEST_F(RaftLeaderElectorTest, GetSelfNodeIdAndInfo) {
    RaftLeaderElector elector("my-node");
    EXPECT_EQ("my-node", elector.GetSelfNodeID());

    NodeEndpointInfo info;
    EXPECT_EQ(EC_NOENT, elector.GetSelfNodeInfo(info));

    NodeEndpointInfo set_info("my-node", "10.0.0.1", 8080, 8081, 9090, 9091, "custom");
    EXPECT_EQ(EC_OK, elector.SetSelfNodeInfo(set_info));

    NodeEndpointInfo got;
    EXPECT_EQ(EC_OK, elector.GetSelfNodeInfo(got));
    EXPECT_EQ("my-node", got.node_id());
    EXPECT_EQ("10.0.0.1", got.host());
    EXPECT_EQ(8080, got.meta_rpc_port());
}

TEST_F(RaftLeaderElectorTest, GetLeaderNodeInfoSingleNode) {
    raft_meta::RaftCoordinator coord;

    NodeEndpointInfo self_info("node-1", "127.0.0.1", 8080, 8081, 9090, 9091, "");

    RaftLeaderElector elector("node-1");
    elector.SetCoordinatorForTest(&coord);
    elector.SetBecomeLeaderHandler([]() {});
    elector.SetNoLongerLeaderHandler([]() {});
    elector.SetSelfNodeInfo(self_info);
    ASSERT_TRUE(elector.Start());

    auto cfg = MakeSingleNodeConfig();
    ASSERT_EQ(EC_OK, coord.Start(cfg));

    ASSERT_TRUE(WaitFor([&] { return elector.IsLeader(); }, std::chrono::seconds(5)));

    NodeEndpointInfo leader_info;
    EXPECT_EQ(EC_OK, elector.GetLeaderNodeInfo(leader_info));
    EXPECT_EQ("node-1", leader_info.node_id());
    EXPECT_EQ("127.0.0.1", leader_info.host());

    coord.Stop();
}

TEST_F(RaftLeaderElectorTest, LeaseSpecificMethodsReturnDefaults) {
    RaftLeaderElector elector("n");
    EXPECT_EQ(-1, elector.GetLastLoopTimeUs());
    EXPECT_EQ(-1, elector.GetLeaseExpirationTime());
    EXPECT_EQ(0, elector.GetForbidCampaignLeaderTimeMs());
    elector.SetForbidCampaignLeaderTimeMs(1000);
    EXPECT_EQ(0, elector.GetForbidCampaignLeaderTimeMs());
}

} // namespace kv_cache_manager
