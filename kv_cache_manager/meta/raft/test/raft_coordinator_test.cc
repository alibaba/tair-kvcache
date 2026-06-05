#include "kv_cache_manager/meta/raft/raft_coordinator.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/raft/raft_log_codec.h"

namespace kv_cache_manager {
namespace raft_meta {

namespace {

// Pick a port in a high range, scrambled by pid so concurrent test
// processes don't trip over each other. Single-node raft binds the port
// but never accepts an outside connection, so collisions are the only
// real failure mode.
int PickPort() {
    int base = 40000;
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

CacheLocationConstPtr MakeLoc(const std::string &id) {
    auto loc = std::make_shared<CacheLocation>();
    loc->set_id(id);
    loc->set_spec_size(4096);
    return std::const_pointer_cast<const CacheLocation>(loc);
}

} // namespace

class RaftCoordinatorTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        char tmpl[] = "/tmp/kvcm_raft_coord_XXXXXX";
        char *dir = ::mkdtemp(tmpl);
        ASSERT_NE(dir, nullptr);
        data_dir_ = dir;
    }

    RaftCoordinator::Config MakeSingleNodeConfig() {
        RaftCoordinator::Config cfg;
        cfg.server_id = 1;
        cfg.port = PickPort();
        cfg.self_endpoint = "127.0.0.1:" + std::to_string(cfg.port);
        cfg.self_aux = R"({"node_id":"n1"})";
        cfg.data_dir = data_dir_;
        // Give election a short window so the single-node election
        // resolves quickly when skip_initial_election_timeout_ does its thing.
        cfg.election_timeout_lower = 100;
        cfg.election_timeout_upper = 200;
        cfg.heart_beat_interval = 50;
        return cfg;
    }

    std::string data_dir_;
};

TEST_F(RaftCoordinatorTest, StartRejectsInvalidConfig) {
    RaftCoordinator coord;
    RaftCoordinator::Config cfg;
    cfg.server_id = 0; // invalid
    cfg.self_endpoint = "127.0.0.1:9000";
    cfg.data_dir = data_dir_;
    EXPECT_NE(EC_OK, coord.Start(cfg));
    EXPECT_FALSE(coord.IsRunning());
}

TEST_F(RaftCoordinatorTest, SingleNodeBecomesLeaderAndAppliesCommit) {
    RaftCoordinator coord;
    auto cfg = MakeSingleNodeConfig();
    ASSERT_EQ(EC_OK, coord.Start(cfg));

    // Single-node skip_initial_election_timeout_ should make us leader fast.
    ASSERT_TRUE(WaitFor([&] { return coord.IsLeader(); }, std::chrono::seconds(5)))
        << "single-node coordinator did not become leader";
    EXPECT_EQ(1, coord.LeaderId());

    LogOp op;
    op.type = OpType::kPut;
    op.instance_id = "inst-A";
    op.key = 42;
    op.locations.emplace("loc-a", MakeLoc("loc-a"));
    EXPECT_EQ(EC_OK, coord.AppendAndWait(Encode(op)));

    auto backend = coord.GetOrCreateBackend("inst-A");
    ASSERT_NE(backend, nullptr);
    KeyVector keys{42};
    CacheLocationMapVector out_locs(1);
    PropertyMapVector out_props(1);
    auto rcs = backend->Get(nullptr, keys, out_locs, out_props);
    ASSERT_EQ(1u, rcs.size());
    EXPECT_EQ(EC_OK, rcs[0]);
    ASSERT_EQ(1u, out_locs[0].size());
    EXPECT_EQ("loc-a", out_locs[0].at("loc-a")->id());

    coord.Stop();
    EXPECT_FALSE(coord.IsRunning());
}

TEST_F(RaftCoordinatorTest, AppendBeforeStartFailsCleanly) {
    RaftCoordinator coord;
    LogOp op;
    op.type = OpType::kDelete;
    op.instance_id = "inst-X";
    op.key = 1;
    EXPECT_EQ(EC_ERROR, coord.AppendAndWait(Encode(op)));
}

TEST_F(RaftCoordinatorTest, LeadershipCallbackFiresOnBecomeLeader) {
    RaftCoordinator coord;
    std::atomic<int> leader_events{0};
    coord.SetLeadershipCallback([&](bool is_leader) {
        if (is_leader) {
            leader_events.fetch_add(1);
        }
    });
    auto cfg = MakeSingleNodeConfig();
    ASSERT_EQ(EC_OK, coord.Start(cfg));
    ASSERT_TRUE(WaitFor([&] { return leader_events.load() > 0; }, std::chrono::seconds(5)))
        << "leadership callback never observed BecomeLeader";
    coord.Stop();
}

TEST_F(RaftCoordinatorTest, SingletonRoundTrip) {
    RaftCoordinator coord;
    EXPECT_EQ(nullptr, RaftCoordinator::GetInstance());
    RaftCoordinator::SetInstance(&coord);
    EXPECT_EQ(&coord, RaftCoordinator::GetInstance());
    RaftCoordinator::SetInstance(nullptr);
    EXPECT_EQ(nullptr, RaftCoordinator::GetInstance());
}

} // namespace raft_meta
} // namespace kv_cache_manager
