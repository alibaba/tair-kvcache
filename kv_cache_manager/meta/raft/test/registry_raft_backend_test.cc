#include "kv_cache_manager/meta/raft/meta_state_machine.h"

#include <libnuraft/buffer.hxx>
#include <libnuraft/buffer_serializer.hxx>
#include <libnuraft/cluster_config.hxx>
#include <libnuraft/snapshot.hxx>

#include <cstdlib>
#include <map>
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

class RegistryRaftBackendTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        char tmpl[] = "/tmp/kvcm_registry_sm_XXXXXX";
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

    std::mutex backends_mu_;
    std::unordered_map<std::string, std::shared_ptr<MetaLocalBackend>> backends_;
    std::shared_ptr<MetaStateMachine> sm_;
    std::string snapshot_dir_;
    nuraft::ulong commit_idx_ = 0;
};

TEST_F(RegistryRaftBackendTest, SaveAndLoad) {
    LogOp op;
    op.type = OpType::kRegistrySave;
    op.registry_key = "instance_group";
    op.registry_fields = {{"group-a", "{\"name\":\"a\"}"}, {"group-b", "{\"name\":\"b\"}"}};
    Apply(op);

    std::map<std::string, std::string> out;
    EXPECT_EQ(EC_OK, sm_->RegistryLoad("instance_group", out));
    ASSERT_EQ(2u, out.size());
    EXPECT_EQ("{\"name\":\"a\"}", out.at("group-a"));
    EXPECT_EQ("{\"name\":\"b\"}", out.at("group-b"));
}

TEST_F(RegistryRaftBackendTest, DeleteThenLoad) {
    LogOp save;
    save.type = OpType::kRegistrySave;
    save.registry_key = "storage";
    save.registry_fields = {{"s1", "{}"}};
    Apply(save);

    std::map<std::string, std::string> out;
    EXPECT_EQ(EC_OK, sm_->RegistryLoad("storage", out));
    ASSERT_EQ(1u, out.size());

    LogOp del;
    del.type = OpType::kRegistryDelete;
    del.registry_key = "storage";
    Apply(del);

    out.clear();
    EXPECT_EQ(EC_NOENT, sm_->RegistryLoad("storage", out));
    EXPECT_TRUE(out.empty());
}

TEST_F(RegistryRaftBackendTest, OverwriteSave) {
    LogOp first;
    first.type = OpType::kRegistrySave;
    first.registry_key = "account";
    first.registry_fields = {{"acct-1", "old"}};
    Apply(first);

    LogOp second;
    second.type = OpType::kRegistrySave;
    second.registry_key = "account";
    second.registry_fields = {{"acct-1", "new"}, {"acct-2", "extra"}};
    Apply(second);

    std::map<std::string, std::string> out;
    EXPECT_EQ(EC_OK, sm_->RegistryLoad("account", out));
    ASSERT_EQ(2u, out.size());
    EXPECT_EQ("new", out.at("acct-1"));
    EXPECT_EQ("extra", out.at("acct-2"));
}

TEST_F(RegistryRaftBackendTest, SnapshotRoundTrip) {
    LogOp op1;
    op1.type = OpType::kRegistrySave;
    op1.registry_key = "instance_group";
    op1.registry_fields = {{"g1", "v1"}, {"g2", "v2"}};
    Apply(op1);

    LogOp op2;
    op2.type = OpType::kRegistrySave;
    op2.registry_key = "storage";
    op2.registry_fields = {{"s1", "cfg1"}};
    Apply(op2);

    auto cluster_cfg = nuraft::cs_new<nuraft::cluster_config>();
    auto snp = nuraft::cs_new<nuraft::snapshot>(commit_idx_, 1, cluster_cfg);
    bool snap_done = false;
    nuraft::async_result<bool>::handler_type handler =
        [&snap_done](bool &result, nuraft::ptr<std::exception> &) {
            snap_done = result;
        };
    sm_->create_snapshot(*snp, handler);
    ASSERT_TRUE(snap_done);

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

    ASSERT_TRUE(sm2->apply_snapshot(*snp));

    std::map<std::string, std::string> out;
    EXPECT_EQ(EC_OK, sm2->RegistryLoad("instance_group", out));
    ASSERT_EQ(2u, out.size());
    EXPECT_EQ("v1", out.at("g1"));
    EXPECT_EQ("v2", out.at("g2"));

    out.clear();
    EXPECT_EQ(EC_OK, sm2->RegistryLoad("storage", out));
    ASSERT_EQ(1u, out.size());
    EXPECT_EQ("cfg1", out.at("s1"));
}

TEST_F(RegistryRaftBackendTest, RegistryIsolatedFromMeta) {
    LogOp meta_put;
    meta_put.type = OpType::kPut;
    meta_put.instance_id = "inst-X";
    meta_put.key = 100;
    meta_put.locations.emplace("loc-a", MakeLocation("loc-a"));
    Apply(meta_put);

    LogOp reg_save;
    reg_save.type = OpType::kRegistrySave;
    reg_save.registry_key = "instance";
    reg_save.registry_fields = {{"inst-X", "{}"}};
    Apply(reg_save);

    {
        std::lock_guard<std::mutex> g(backends_mu_);
        auto it = backends_.find("inst-X");
        ASSERT_NE(it, backends_.end());
        KeyVector keys{100};
        CacheLocationMapVector out_locs(1);
        auto rcs = it->second->GetLocations(nullptr, keys, out_locs);
        ASSERT_EQ(1u, rcs.size());
        EXPECT_EQ(EC_OK, rcs[0]);
        ASSERT_EQ(1u, out_locs[0].size());
        EXPECT_EQ("loc-a", out_locs[0].at("loc-a")->id());
    }

    std::map<std::string, std::string> reg_out;
    EXPECT_EQ(EC_OK, sm_->RegistryLoad("instance", reg_out));
    ASSERT_EQ(1u, reg_out.size());
    EXPECT_EQ("{}", reg_out.at("inst-X"));

    LogOp reg_del;
    reg_del.type = OpType::kRegistryDelete;
    reg_del.registry_key = "instance";
    Apply(reg_del);

    EXPECT_EQ(EC_NOENT, sm_->RegistryLoad("instance", reg_out));

    {
        std::lock_guard<std::mutex> g(backends_mu_);
        auto it = backends_.find("inst-X");
        ASSERT_NE(it, backends_.end());
        KeyVector keys{100};
        CacheLocationMapVector out_locs(1);
        auto rcs = it->second->GetLocations(nullptr, keys, out_locs);
        ASSERT_EQ(1u, rcs.size());
        EXPECT_EQ(EC_OK, rcs[0]);
        ASSERT_EQ(1u, out_locs[0].size());
        EXPECT_EQ("loc-a", out_locs[0].at("loc-a")->id());
    }
}

TEST_F(RegistryRaftBackendTest, CommitIndexAdvancesOnRegistryOps) {
    EXPECT_EQ(0u, sm_->last_commit_index());

    LogOp op;
    op.type = OpType::kRegistrySave;
    op.registry_key = "test";
    op.registry_fields = {{"k", "v"}};
    Apply(op);
    EXPECT_EQ(1u, sm_->last_commit_index());

    LogOp del;
    del.type = OpType::kRegistryDelete;
    del.registry_key = "test";
    Apply(del);
    EXPECT_EQ(2u, sm_->last_commit_index());
}

TEST_F(RegistryRaftBackendTest, LoadNonexistentKeyReturnsNoent) {
    std::map<std::string, std::string> out;
    EXPECT_EQ(EC_NOENT, sm_->RegistryLoad("does_not_exist", out));
    EXPECT_TRUE(out.empty());
}

TEST_F(RegistryRaftBackendTest, RegistryClearRemovesAllEntries) {
    LogOp op1;
    op1.type = OpType::kRegistrySave;
    op1.registry_key = "key1";
    op1.registry_fields = {{"a", "1"}};
    Apply(op1);

    LogOp op2;
    op2.type = OpType::kRegistrySave;
    op2.registry_key = "key2";
    op2.registry_fields = {{"b", "2"}};
    Apply(op2);

    sm_->RegistryClear();

    std::map<std::string, std::string> out;
    EXPECT_EQ(EC_NOENT, sm_->RegistryLoad("key1", out));
    EXPECT_EQ(EC_NOENT, sm_->RegistryLoad("key2", out));
}

} // namespace raft_meta
} // namespace kv_cache_manager
