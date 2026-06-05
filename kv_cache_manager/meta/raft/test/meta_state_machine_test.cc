#include "kv_cache_manager/meta/raft/meta_state_machine.h"

#include <libnuraft/buffer.hxx>
#include <libnuraft/buffer_serializer.hxx>
#include <libnuraft/snapshot.hxx>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <unordered_map>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/raft/raft_log_codec.h"

namespace kv_cache_manager {
namespace raft_meta {

class MetaStateMachineTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        char tmpl[] = "/tmp/kvcm_state_machine_XXXXXX";
        char *dir = ::mkdtemp(tmpl);
        ASSERT_NE(dir, nullptr);
        snapshot_dir_ = std::string(dir) + "/snapshot";

        sm_ = std::make_shared<MetaStateMachine>(
            [this](const std::string &iid) -> std::shared_ptr<MetaCacheBaseBackend> {
                std::lock_guard<std::mutex> g(backends_mu_);
                auto it = backends_.find(iid);
                if (it != backends_.end()) {
                    return it->second;
                }
                auto b = std::make_shared<MetaLocalBackend>();
                auto cfg = std::make_shared<MetaStorageBackendConfig>();
                EXPECT_EQ(EC_OK, b->Init(iid, cfg));
                EXPECT_EQ(EC_OK, b->Open());
                backends_.emplace(iid, b);
                return b;
            },
            snapshot_dir_);
    }

    void TearDown() override {
        std::lock_guard<std::mutex> g(backends_mu_);
        for (auto &[_, b] : backends_) {
            b->Close();
        }
        backends_.clear();
        TESTBASE::TearDown();
    }

    static CacheLocationConstPtr MakeLocation(const std::string &id) {
        auto loc = std::make_shared<CacheLocation>();
        loc->set_id(id);
        loc->set_spec_size(4096);
        return std::const_pointer_cast<const CacheLocation>(loc);
    }

    nuraft::ulong NextIdx() { return ++commit_idx_; }

    void Apply(const LogOp &op) {
        auto buf = Encode(op);
        sm_->commit(NextIdx(), *buf);
    }

    std::shared_ptr<MetaLocalBackend> BackendOf(const std::string &iid) {
        std::lock_guard<std::mutex> g(backends_mu_);
        auto it = backends_.find(iid);
        return it == backends_.end() ? nullptr : it->second;
    }

    std::mutex backends_mu_;
    std::unordered_map<std::string, std::shared_ptr<MetaLocalBackend>> backends_;
    std::shared_ptr<MetaStateMachine> sm_;
    std::string snapshot_dir_;
    nuraft::ulong commit_idx_ = 0;
};

TEST_F(MetaStateMachineTest, CommitPutThenGet) {
    LogOp op;
    op.type = OpType::kPut;
    op.instance_id = "inst-A";
    op.key = 1;
    op.locations.emplace("loc-a", MakeLocation("loc-a"));
    op.properties.emplace(PROPERTY_URI, "tair://x");

    Apply(op);

    auto backend = BackendOf("inst-A");
    ASSERT_NE(backend, nullptr);
    KeyVector keys{1};
    CacheLocationMapVector out_locs(1);
    PropertyMapVector out_props(1);
    auto rcs = backend->Get(nullptr, keys, out_locs, out_props);
    ASSERT_EQ(1u, rcs.size());
    EXPECT_EQ(EC_OK, rcs[0]);
    ASSERT_EQ(1u, out_locs[0].size());
    EXPECT_EQ("loc-a", out_locs[0].at("loc-a")->id());
    EXPECT_EQ("tair://x", out_props[0].at(PROPERTY_URI));
    EXPECT_EQ(1u, sm_->last_commit_index());
}

TEST_F(MetaStateMachineTest, CommitDelete) {
    LogOp put;
    put.type = OpType::kPut;
    put.instance_id = "inst-D";
    put.key = 42;
    put.locations.emplace("loc", MakeLocation("loc"));
    Apply(put);

    LogOp del;
    del.type = OpType::kDelete;
    del.instance_id = "inst-D";
    del.key = 42;
    Apply(del);

    auto backend = BackendOf("inst-D");
    ASSERT_NE(backend, nullptr);
    KeyVector keys{42};
    std::vector<bool> exists;
    auto rcs = backend->Exists(nullptr, keys, exists);
    ASSERT_EQ(1u, rcs.size());
    ASSERT_EQ(1u, exists.size());
    EXPECT_FALSE(exists[0]);
    EXPECT_EQ(2u, sm_->last_commit_index());
}

TEST_F(MetaStateMachineTest, CommitDeleteLocations) {
    LogOp put;
    put.type = OpType::kPut;
    put.instance_id = "inst-L";
    put.key = 7;
    put.locations.emplace("loc-keep", MakeLocation("loc-keep"));
    put.locations.emplace("loc-drop", MakeLocation("loc-drop"));
    Apply(put);

    LogOp del;
    del.type = OpType::kDeleteLocations;
    del.instance_id = "inst-L";
    del.key = 7;
    del.location_ids.push_back("loc-drop");
    Apply(del);

    auto backend = BackendOf("inst-L");
    ASSERT_NE(backend, nullptr);
    KeyVector keys{7};
    CacheLocationMapVector out_locs(1);
    auto rcs = backend->GetLocations(nullptr, keys, out_locs);
    ASSERT_EQ(1u, rcs.size());
    EXPECT_EQ(EC_OK, rcs[0]);
    ASSERT_EQ(1u, out_locs[0].size());
    EXPECT_EQ(1u, out_locs[0].count("loc-keep"));
    EXPECT_EQ(0u, out_locs[0].count("loc-drop"));
}

TEST_F(MetaStateMachineTest, CommitPutMetaDataAdvancesIndex) {
    // MetaLocalBackend::PutMetaData is a no-op, so we can only assert the
    // state machine dispatched the entry without throwing and bumped the
    // commit index. Backend-observable PutMetaData will land with a redis
    // (or future persistent) backend.
    LogOp op;
    op.type = OpType::kPutMetaData;
    op.instance_id = "inst-M";
    op.meta_fields.emplace("custom", "value");
    Apply(op);
    EXPECT_EQ(1u, sm_->last_commit_index());
}

TEST_F(MetaStateMachineTest, CorruptEntryStillAdvancesIndex) {
    nuraft::ptr<nuraft::buffer> buf = nuraft::buffer::alloc(4);
    nuraft::buffer_serializer bs(buf);
    bs.put_u8(99); // invalid version
    bs.put_u8(1);
    sm_->commit(7, *buf);
    EXPECT_EQ(7u, sm_->last_commit_index());
}

TEST_F(MetaStateMachineTest, RouteByInstanceIdIsolated) {
    // Same key in two Instances must not collide.
    LogOp a;
    a.type = OpType::kPut;
    a.instance_id = "inst-A";
    a.key = 100;
    a.locations.emplace("loc-a", MakeLocation("loc-a"));
    Apply(a);

    LogOp b;
    b.type = OpType::kPut;
    b.instance_id = "inst-B";
    b.key = 100;
    b.locations.emplace("loc-b", MakeLocation("loc-b"));
    Apply(b);

    auto backend_a = BackendOf("inst-A");
    auto backend_b = BackendOf("inst-B");
    ASSERT_NE(backend_a, nullptr);
    ASSERT_NE(backend_b, nullptr);
    EXPECT_NE(backend_a.get(), backend_b.get());

    KeyVector keys{100};
    {
        CacheLocationMapVector out_locs(1);
        backend_a->GetLocations(nullptr, keys, out_locs);
        ASSERT_EQ(1u, out_locs[0].size());
        EXPECT_EQ(1u, out_locs[0].count("loc-a"));
    }
    {
        CacheLocationMapVector out_locs(1);
        backend_b->GetLocations(nullptr, keys, out_locs);
        ASSERT_EQ(1u, out_locs[0].size());
        EXPECT_EQ(1u, out_locs[0].count("loc-b"));
    }
}

} // namespace raft_meta
} // namespace kv_cache_manager
