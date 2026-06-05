#include "kv_cache_manager/meta/raft/raft_coordinator.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/raft/raft_log_codec.h"

namespace kv_cache_manager {
namespace raft_meta {

namespace {

constexpr int kNumNodes = 3;

int PickBasePort() {
    return 42000 + static_cast<int>(::getpid() % 3000);
}

bool WaitFor(std::function<bool()> pred, std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
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

class MultiNodeRaftTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        base_port_ = PickBasePort();
        for (int i = 0; i < kNumNodes; ++i) {
            char tmpl[] = "/tmp/kvcm_raft_mn_XXXXXX";
            char *dir = ::mkdtemp(tmpl);
            ASSERT_NE(dir, nullptr);
            data_dirs_.emplace_back(dir);
        }
    }

    RaftCoordinator::Config MakeConfig(int index) {
        RaftCoordinator::Config cfg;
        cfg.server_id = index + 1;
        cfg.port = base_port_ + index * 10;
        cfg.self_endpoint = "127.0.0.1:" + std::to_string(cfg.port);
        cfg.self_aux = R"({"node_id":"n)" + std::to_string(index + 1) + R"("})";
        cfg.data_dir = data_dirs_[index];
        cfg.election_timeout_lower = 200;
        cfg.election_timeout_upper = 400;
        cfg.heart_beat_interval = 80;

        for (int i = 0; i < kNumNodes; ++i) {
            RaftCoordinator::PeerSpec ps;
            ps.server_id = i + 1;
            ps.endpoint = "127.0.0.1:" + std::to_string(base_port_ + i * 10);
            cfg.peers.push_back(ps);
        }
        return cfg;
    }

    void StartAll() {
        for (int i = 0; i < kNumNodes; ++i) {
            nodes_.push_back(std::make_unique<RaftCoordinator>());
            auto cfg = MakeConfig(i);
            ASSERT_EQ(EC_OK, nodes_[i]->Start(cfg))
                << "node " << (i + 1) << " failed to start";
        }
    }

    void StopAll() {
        for (auto &n : nodes_) {
            if (n && n->IsRunning()) {
                n->Stop();
            }
        }
        nodes_.clear();
    }

    void TearDown() override {
        StopAll();
        TESTBASE::TearDown();
    }

    RaftCoordinator *FindLeader() {
        for (auto &n : nodes_) {
            if (n && n->IsRunning() && n->IsLeader()) {
                return n.get();
            }
        }
        return nullptr;
    }

    int CountLeaders() {
        int count = 0;
        for (auto &n : nodes_) {
            if (n && n->IsRunning() && n->IsLeader()) {
                ++count;
            }
        }
        return count;
    }

    int base_port_ = 0;
    std::vector<std::string> data_dirs_;
    std::vector<std::unique_ptr<RaftCoordinator>> nodes_;
};

TEST_F(MultiNodeRaftTest, ThreeNodeClusterElectsLeader) {
    StartAll();

    ASSERT_TRUE(WaitFor([this] { return FindLeader() != nullptr; },
                        std::chrono::seconds(10)))
        << "no leader elected in 3-node cluster";

    EXPECT_EQ(1, CountLeaders());

    int32_t leader_id = FindLeader()->SelfId();
    for (auto &n : nodes_) {
        EXPECT_EQ(leader_id, n->LeaderId())
            << "node " << n->SelfId() << " disagrees on leader";
    }
}

TEST_F(MultiNodeRaftTest, DataReplicationThroughLeader) {
    StartAll();
    ASSERT_TRUE(WaitFor([this] { return FindLeader() != nullptr; },
                        std::chrono::seconds(10)));

    auto *leader = FindLeader();
    ASSERT_NE(leader, nullptr);

    LogOp op;
    op.type = OpType::kPut;
    op.instance_id = "inst-multi";
    op.key = 100;
    op.locations.emplace("loc-1", MakeLoc("loc-1"));
    ASSERT_EQ(EC_OK, leader->AppendAndWait(Encode(op)));

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    for (auto &n : nodes_) {
        auto backend = n->GetOrCreateBackend("inst-multi");
        ASSERT_NE(backend, nullptr) << "node " << n->SelfId();
        KeyVector keys{100};
        CacheLocationMapVector out_locs(1);
        PropertyMapVector out_props(1);
        auto rcs = backend->Get(nullptr, keys, out_locs, out_props);
        ASSERT_EQ(1u, rcs.size()) << "node " << n->SelfId();
        EXPECT_EQ(EC_OK, rcs[0]) << "node " << n->SelfId();
        ASSERT_EQ(1u, out_locs[0].size()) << "node " << n->SelfId();
        EXPECT_EQ("loc-1", out_locs[0].at("loc-1")->id())
            << "node " << n->SelfId();
    }
}

TEST_F(MultiNodeRaftTest, FollowerRejectsWrite) {
    StartAll();
    ASSERT_TRUE(WaitFor([this] { return FindLeader() != nullptr; },
                        std::chrono::seconds(10)));

    for (auto &n : nodes_) {
        if (n->IsLeader()) continue;
        LogOp op;
        op.type = OpType::kDelete;
        op.instance_id = "inst-x";
        op.key = 1;
        EXPECT_EQ(EC_BADARGS, n->AppendAndWait(Encode(op)))
            << "follower " << n->SelfId() << " should reject write";
        break;
    }
}

TEST_F(MultiNodeRaftTest, LeaderFailoverAfterStop) {
    StartAll();
    ASSERT_TRUE(WaitFor([this] { return FindLeader() != nullptr; },
                        std::chrono::seconds(10)));

    int32_t old_leader_id = FindLeader()->SelfId();
    int old_index = old_leader_id - 1;

    nodes_[old_index]->Stop();

    ASSERT_TRUE(WaitFor(
        [this, old_index] {
            for (int i = 0; i < kNumNodes; ++i) {
                if (i == old_index) continue;
                if (nodes_[i]->IsLeader()) return true;
            }
            return false;
        },
        std::chrono::seconds(10)))
        << "no new leader elected after stopping node " << old_leader_id;

    auto *new_leader = FindLeader();
    ASSERT_NE(new_leader, nullptr);
    EXPECT_NE(old_leader_id, new_leader->SelfId());
}

} // namespace raft_meta
} // namespace kv_cache_manager
