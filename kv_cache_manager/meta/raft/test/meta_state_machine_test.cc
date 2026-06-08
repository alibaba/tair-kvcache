#include "kv_cache_manager/meta/raft/meta_state_machine.h"

#include <libnuraft/buffer.hxx>
#include <libnuraft/buffer_serializer.hxx>
#include <libnuraft/cluster_config.hxx>
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

// --- Registry field-level operations ---

TEST_F(MetaStateMachineTest, RegistryFieldSaveMerge) {
    LogOp op1;
    op1.type = OpType::kRegistryFieldSave;
    op1.registry_key = "account";
    op1.registry_field_id = "alice";
    op1.registry_field_value = "{\"role\":\"admin\"}";
    Apply(op1);

    LogOp op2;
    op2.type = OpType::kRegistryFieldSave;
    op2.registry_key = "account";
    op2.registry_field_id = "bob";
    op2.registry_field_value = "{\"role\":\"reader\"}";
    Apply(op2);

    std::map<std::string, std::string> out;
    ASSERT_EQ(EC_OK, sm_->RegistryLoad("account", out));
    ASSERT_EQ(2u, out.size());
    EXPECT_EQ("{\"role\":\"admin\"}", out.at("alice"));
    EXPECT_EQ("{\"role\":\"reader\"}", out.at("bob"));
}

TEST_F(MetaStateMachineTest, RegistryFieldDeleteRemovesField) {
    LogOp save_a;
    save_a.type = OpType::kRegistryFieldSave;
    save_a.registry_key = "account";
    save_a.registry_field_id = "alice";
    save_a.registry_field_value = "v1";
    Apply(save_a);

    LogOp save_b;
    save_b.type = OpType::kRegistryFieldSave;
    save_b.registry_key = "account";
    save_b.registry_field_id = "bob";
    save_b.registry_field_value = "v2";
    Apply(save_b);

    LogOp del_a;
    del_a.type = OpType::kRegistryFieldDelete;
    del_a.registry_key = "account";
    del_a.registry_field_id = "alice";
    Apply(del_a);

    std::map<std::string, std::string> out;
    ASSERT_EQ(EC_OK, sm_->RegistryLoad("account", out));
    ASSERT_EQ(1u, out.size());
    EXPECT_EQ("v2", out.at("bob"));
}

TEST_F(MetaStateMachineTest, RegistryFieldDeleteLastFieldRemovesKey) {
    LogOp save;
    save.type = OpType::kRegistryFieldSave;
    save.registry_key = "storage";
    save.registry_field_id = "only-one";
    save.registry_field_value = "data";
    Apply(save);

    LogOp del;
    del.type = OpType::kRegistryFieldDelete;
    del.registry_key = "storage";
    del.registry_field_id = "only-one";
    Apply(del);

    std::map<std::string, std::string> out;
    sm_->RegistryLoad("storage", out);
    EXPECT_TRUE(out.empty());
}

TEST_F(MetaStateMachineTest, RegistryFieldSaveCallbackFired) {
    std::string cb_key;
    std::map<std::string, std::string> cb_fields;
    bool cb_is_save = false;
    int cb_count = 0;

    sm_->SetRegistryCommitCallback([&](bool is_save, const std::string &key,
                                       const std::map<std::string, std::string> &fields) {
        cb_is_save = is_save;
        cb_key = key;
        cb_fields = fields;
        ++cb_count;
    });

    LogOp op;
    op.type = OpType::kRegistryFieldSave;
    op.registry_key = "account";
    op.registry_field_id = "alice";
    op.registry_field_value = "v1";
    Apply(op);

    EXPECT_EQ(1, cb_count);
    EXPECT_TRUE(cb_is_save);
    EXPECT_EQ("account", cb_key);
    ASSERT_EQ(1u, cb_fields.size());
    EXPECT_EQ("v1", cb_fields.at("alice"));
}

// --- Snapshot round-trip ---

TEST_F(MetaStateMachineTest, SnapshotRoundTripMetaData) {
    LogOp op;
    op.type = OpType::kPut;
    op.instance_id = "inst-snap";
    op.key = 42;
    op.locations.emplace("loc-1", MakeLocation("loc-1"));
    op.properties.emplace(PROPERTY_URI, "tair://snap");
    Apply(op);

    // Create snapshot.
    auto snap = nuraft::cs_new<nuraft::snapshot>(commit_idx_, 1, nuraft::cs_new<nuraft::cluster_config>());
    bool snap_ok = false;
    nuraft::async_result<bool>::handler_type handler =
        [&](bool result, nuraft::ptr<std::exception> &) { snap_ok = result; };
    sm_->create_snapshot(*snap, handler);
    ASSERT_TRUE(snap_ok);

    // Build a new state machine and apply the snapshot.
    auto sm2 = std::make_shared<MetaStateMachine>(
        [this](const std::string &iid) -> std::shared_ptr<MetaCacheBaseBackend> {
            std::lock_guard<std::mutex> g(backends_mu_);
            auto b = std::make_shared<MetaLocalBackend>();
            auto cfg = std::make_shared<MetaStorageBackendConfig>();
            EXPECT_EQ(EC_OK, b->Init(iid, cfg));
            EXPECT_EQ(EC_OK, b->Open());
            backends_.emplace(iid, b);
            return b;
        },
        snapshot_dir_);
    ASSERT_TRUE(sm2->apply_snapshot(*snap));

    auto backend2 = sm2->GetBackend("inst-snap");
    ASSERT_NE(backend2, nullptr);
    KeyVector keys{42};
    CacheLocationMapVector out_locs(1);
    PropertyMapVector out_props(1);
    auto rcs = backend2->Get(nullptr, keys, out_locs, out_props);
    ASSERT_EQ(1u, rcs.size());
    EXPECT_EQ(EC_OK, rcs[0]);
    ASSERT_EQ(1u, out_locs[0].size());
    EXPECT_EQ("loc-1", out_locs[0].at("loc-1")->id());
    EXPECT_EQ("tair://snap", out_props[0].at(PROPERTY_URI));
}

TEST_F(MetaStateMachineTest, SnapshotRoundTripRegistry) {
    LogOp op;
    op.type = OpType::kRegistrySave;
    op.registry_key = "storage";
    op.registry_fields = {{"backend-a", "{\"type\":\"nfs\"}"}, {"backend-b", "{\"type\":\"dummy\"}"}};
    Apply(op);

    auto snap = nuraft::cs_new<nuraft::snapshot>(commit_idx_, 1, nuraft::cs_new<nuraft::cluster_config>());
    bool snap_ok = false;
    nuraft::async_result<bool>::handler_type handler =
        [&](bool result, nuraft::ptr<std::exception> &) { snap_ok = result; };
    sm_->create_snapshot(*snap, handler);
    ASSERT_TRUE(snap_ok);

    auto sm2 = std::make_shared<MetaStateMachine>(
        [](const std::string &) -> std::shared_ptr<MetaCacheBaseBackend> { return nullptr; },
        snapshot_dir_);
    ASSERT_TRUE(sm2->apply_snapshot(*snap));

    std::map<std::string, std::string> out;
    ASSERT_EQ(EC_OK, sm2->RegistryLoad("storage", out));
    ASSERT_EQ(2u, out.size());
    EXPECT_EQ("{\"type\":\"nfs\"}", out.at("backend-a"));
    EXPECT_EQ("{\"type\":\"dummy\"}", out.at("backend-b"));
}

TEST_F(MetaStateMachineTest, SnapshotRoundTripFieldOps) {
    LogOp op1;
    op1.type = OpType::kRegistryFieldSave;
    op1.registry_key = "account";
    op1.registry_field_id = "alice";
    op1.registry_field_value = "v1";
    Apply(op1);

    LogOp op2;
    op2.type = OpType::kRegistryFieldSave;
    op2.registry_key = "account";
    op2.registry_field_id = "bob";
    op2.registry_field_value = "v2";
    Apply(op2);

    LogOp del;
    del.type = OpType::kRegistryFieldDelete;
    del.registry_key = "account";
    del.registry_field_id = "alice";
    Apply(del);

    auto snap = nuraft::cs_new<nuraft::snapshot>(commit_idx_, 1, nuraft::cs_new<nuraft::cluster_config>());
    bool snap_ok = false;
    nuraft::async_result<bool>::handler_type handler =
        [&](bool result, nuraft::ptr<std::exception> &) { snap_ok = result; };
    sm_->create_snapshot(*snap, handler);
    ASSERT_TRUE(snap_ok);

    auto sm2 = std::make_shared<MetaStateMachine>(
        [](const std::string &) -> std::shared_ptr<MetaCacheBaseBackend> { return nullptr; },
        snapshot_dir_);
    ASSERT_TRUE(sm2->apply_snapshot(*snap));

    std::map<std::string, std::string> out;
    ASSERT_EQ(EC_OK, sm2->RegistryLoad("account", out));
    ASSERT_EQ(1u, out.size());
    EXPECT_EQ("v2", out.at("bob"));
}

TEST_F(MetaStateMachineTest, SnapshotPreservesLastCommitIndex) {
    for (int i = 0; i < 5; ++i) {
        LogOp op;
        op.type = OpType::kRegistryFieldSave;
        op.registry_key = "k";
        op.registry_field_id = "f" + std::to_string(i);
        op.registry_field_value = "v";
        Apply(op);
    }
    EXPECT_EQ(5u, sm_->last_commit_index());

    auto snap = nuraft::cs_new<nuraft::snapshot>(commit_idx_, 1, nuraft::cs_new<nuraft::cluster_config>());
    bool snap_ok = false;
    nuraft::async_result<bool>::handler_type handler =
        [&](bool result, nuraft::ptr<std::exception> &) { snap_ok = result; };
    sm_->create_snapshot(*snap, handler);
    ASSERT_TRUE(snap_ok);

    auto sm2 = std::make_shared<MetaStateMachine>(
        [](const std::string &) -> std::shared_ptr<MetaCacheBaseBackend> { return nullptr; },
        snapshot_dir_);
    ASSERT_TRUE(sm2->apply_snapshot(*snap));
    EXPECT_EQ(5u, sm2->last_commit_index());
}

TEST_F(MetaStateMachineTest, NoOpAdvancesIndex) {
    LogOp op;
    op.type = OpType::kNoOp;
    Apply(op);
    EXPECT_EQ(1u, sm_->last_commit_index());
}

} // namespace raft_meta
} // namespace kv_cache_manager
