#include <cstring>
#include <fstream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

#include <grpcpp/grpcpp.h>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/online_optimizer/server/online_optimizer_server.h"
#include "kv_cache_manager/protocol/protobuf/admin_service.pb.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

namespace kv_cache_manager {

namespace {

int AllocateFreePort() {
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) return -1;

    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    if (bind(sockfd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) == -1) {
        close(sockfd);
        return -1;
    }

    socklen_t len = sizeof(addr);
    if (getsockname(sockfd, reinterpret_cast<struct sockaddr *>(&addr), &len) == -1) {
        close(sockfd);
        return -1;
    }

    int port = ntohs(addr.sin_port);
    close(sockfd);
    return port;
}

} // namespace

class OnlineOptimizerIntegrationTest : public TESTBASE {
protected:
    static void SetUpTestSuite() {
        rpc_port_ = AllocateFreePort();
        http_port_ = AllocateFreePort();
        ASSERT_GT(rpc_port_, 0);
        ASSERT_GT(http_port_, 0);

        const char *tmpdir = std::getenv("TEST_TMPDIR");
        config_path_ = std::string(tmpdir ? tmpdir : "/tmp") +
                        "/optimizer_integ_config.json";
        {
            std::ofstream ofs(config_path_);
            ASSERT_TRUE(ofs.is_open()) << "Cannot create config file: " << config_path_;
            ofs << "{\n"
                << "  \"rpc_port\": " << rpc_port_ << ",\n"
                << "  \"http_port\": " << http_port_ << ",\n"
                << "  \"registry_storage_uri\": \"\",\n"
                << "  \"metrics_report_interval_ms\": 0,\n"
                << "  \"enable_prometheus\": false,\n"
                << "  \"io_thread_num\": 2\n"
                << "}";
        }

        server_ = std::make_unique<OnlineOptimizerServer>();
        ASSERT_TRUE(server_->Init(config_path_));
        ASSERT_TRUE(server_->Start());

        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        auto channel = grpc::CreateChannel(
            "127.0.0.1:" + std::to_string(rpc_port_),
            grpc::InsecureChannelCredentials());
        stub_ = proto::optimizer::OptimizerService::NewStub(channel);
        ASSERT_NE(nullptr, stub_);
    }

    static void TearDownTestSuite() {
        stub_.reset();
        if (server_) {
            server_->Stop();
            server_.reset();
        }
        if (!config_path_.empty()) {
            std::remove(config_path_.c_str());
        }
    }

    void CreateTestGroup(const std::string &group_name,
                         double capacity_gb = 1.0,
                         int32_t linear_step = 1) {
        proto::optimizer::CreateInstanceGroupRequest req;
        req.set_trace_id("setup-" + group_name);
        auto *g = req.mutable_instance_group();
        g->set_name(group_name);
        g->set_enabled(true);
        g->set_max_stats(false);
        g->add_capacity_gb(capacity_gb);
        g->set_primary_capacity_index(0);
        g->set_linear_step(linear_step);
        g->set_full_group_name(group_name);

        proto::admin::CommonResponse resp;
        grpc::ClientContext ctx;
        auto status = stub_->CreateInstanceGroup(&ctx, req, &resp);
        ASSERT_TRUE(status.ok()) << status.error_message();
    }

    proto::optimizer::OptimizerRegisterInstanceResponse RegisterTestInstance(
        const std::string &group, const std::string &instance_id,
        int32_t block_size = 1024, double capacity_gb = 1.0,
        int32_t linear_step = 1) {
        CreateTestGroup(group, capacity_gb, linear_step);

        proto::optimizer::OptimizerRegisterInstanceRequest req;
        req.set_trace_id("integ-" + instance_id);
        req.set_instance_group(group);
        req.set_instance_id(instance_id);
        req.set_block_size(block_size);

        auto *spec = req.add_location_spec_infos();
        spec->set_name("full");
        spec->set_size(block_size);

        proto::optimizer::OptimizerRegisterInstanceResponse resp;
        grpc::ClientContext ctx;
        auto status = stub_->RegisterInstance(&ctx, req, &resp);
        EXPECT_TRUE(status.ok()) << status.error_message();
        return resp;
    }

    static int rpc_port_;
    static int http_port_;
    static std::string config_path_;
    static std::unique_ptr<OnlineOptimizerServer> server_;
    static std::unique_ptr<proto::optimizer::OptimizerService::Stub> stub_;
};

int OnlineOptimizerIntegrationTest::rpc_port_ = 0;
int OnlineOptimizerIntegrationTest::http_port_ = 0;
std::string OnlineOptimizerIntegrationTest::config_path_;
std::unique_ptr<OnlineOptimizerServer> OnlineOptimizerIntegrationTest::server_;
std::unique_ptr<proto::optimizer::OptimizerService::Stub> OnlineOptimizerIntegrationTest::stub_;

TEST_F(OnlineOptimizerIntegrationTest, InstanceGroupCRUD) {
    // Create
    {
        proto::optimizer::CreateInstanceGroupRequest req;
        req.set_trace_id("crud-create");
        auto *g = req.mutable_instance_group();
        g->set_name("crud_grp");
        g->set_enabled(true);
        g->add_capacity_gb(2.0);
        g->set_primary_capacity_index(0);
        g->set_linear_step(1);
        g->set_full_group_name("crud_grp");

        proto::admin::CommonResponse resp;
        grpc::ClientContext ctx;
        auto status = stub_->CreateInstanceGroup(&ctx, req, &resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, resp.header().status().code());
    }

    // Get
    {
        proto::optimizer::GetInstanceGroupRequest req;
        req.set_trace_id("crud-get");
        req.set_name("crud_grp");
        proto::optimizer::GetInstanceGroupResponse resp;
        grpc::ClientContext ctx;
        auto status = stub_->GetInstanceGroup(&ctx, req, &resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, resp.header().status().code());
        EXPECT_EQ("crud_grp", resp.instance_group().name());
        EXPECT_DOUBLE_EQ(2.0, resp.instance_group().capacity_gb(0));
    }

    // Update
    {
        proto::optimizer::UpdateInstanceGroupRequest req;
        req.set_trace_id("crud-update");
        auto *g = req.mutable_instance_group();
        g->set_name("crud_grp");
        g->set_enabled(true);
        g->add_capacity_gb(4.0);
        g->set_primary_capacity_index(0);
        g->set_linear_step(2);
        g->set_full_group_name("crud_grp");

        proto::admin::CommonResponse resp;
        grpc::ClientContext ctx;
        auto status = stub_->UpdateInstanceGroup(&ctx, req, &resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, resp.header().status().code());
    }

    // Verify update
    {
        proto::optimizer::GetInstanceGroupRequest req;
        req.set_trace_id("crud-get2");
        req.set_name("crud_grp");
        proto::optimizer::GetInstanceGroupResponse resp;
        grpc::ClientContext ctx;
        auto status = stub_->GetInstanceGroup(&ctx, req, &resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(2, resp.instance_group().linear_step());
    }

    // List
    {
        proto::optimizer::ListInstanceGroupsRequest req;
        req.set_trace_id("crud-list");
        proto::optimizer::ListInstanceGroupsResponse resp;
        grpc::ClientContext ctx;
        auto status = stub_->ListInstanceGroups(&ctx, req, &resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, resp.header().status().code());
        EXPECT_GE(resp.instance_groups_size(), 1);
    }

    // Remove
    {
        proto::optimizer::RemoveInstanceGroupRequest req;
        req.set_trace_id("crud-rm");
        req.set_name("crud_grp");
        proto::admin::CommonResponse resp;
        grpc::ClientContext ctx;
        auto status = stub_->RemoveInstanceGroup(&ctx, req, &resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, resp.header().status().code());
    }

    // Verify removed
    {
        proto::optimizer::GetInstanceGroupRequest req;
        req.set_trace_id("crud-get3");
        req.set_name("crud_grp");
        proto::optimizer::GetInstanceGroupResponse resp;
        grpc::ClientContext ctx;
        auto status = stub_->GetInstanceGroup(&ctx, req, &resp);
        ASSERT_TRUE(status.ok());
        EXPECT_NE(proto::admin::OK, resp.header().status().code());
    }
}

TEST_F(OnlineOptimizerIntegrationTest, RegisterWithoutGroupFails) {
    proto::optimizer::OptimizerRegisterInstanceRequest req;
    req.set_trace_id("integ-no-group");
    req.set_instance_group("nonexistent_group_xyz");
    req.set_instance_id("inst_fail");
    req.set_block_size(1024);
    auto *spec = req.add_location_spec_infos();
    spec->set_name("full");
    spec->set_size(1024);

    proto::optimizer::OptimizerRegisterInstanceResponse resp;
    grpc::ClientContext ctx;
    auto status = stub_->RegisterInstance(&ctx, req, &resp);
    ASSERT_TRUE(status.ok());
    EXPECT_NE(proto::admin::OK, resp.header().status().code());
}

TEST_F(OnlineOptimizerIntegrationTest, RegisterAndListInstances) {
    auto resp = RegisterTestInstance("integ_grp1", "integ_reg_list_1");
    EXPECT_EQ(proto::admin::OK, resp.header().status().code());
    EXPECT_GT(resp.capacity_blocks_size(), 0);
    EXPECT_GT(resp.capacity_blocks(0), 0);

    proto::optimizer::OptimizerListInstancesRequest list_req;
    list_req.set_trace_id("integ-list-1");
    list_req.set_instance_group("integ_grp1");
    proto::optimizer::OptimizerListInstancesResponse list_resp;
    grpc::ClientContext list_ctx;
    auto status = stub_->ListInstances(&list_ctx, list_req, &list_resp);
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(proto::admin::OK, list_resp.header().status().code());

    bool found = false;
    for (const auto &inst : list_resp.instances()) {
        if (inst.instance_id() == "integ_reg_list_1") {
            found = true;
            EXPECT_EQ("integ_grp1", inst.instance_group());
            EXPECT_EQ(1024, inst.block_size());
        }
    }
    EXPECT_TRUE(found) << "Registered instance not found in list";
}

TEST_F(OnlineOptimizerIntegrationTest, TraceQueryMissAndHit) {
    RegisterTestInstance("integ_grp_tq", "integ_tq_1");

    {
        proto::optimizer::TraceQueryRequest tq_req;
        tq_req.set_trace_id("integ-tq-miss");
        tq_req.set_instance_id("integ_tq_1");
        tq_req.add_block_keys(100);
        tq_req.add_block_keys(200);
        tq_req.add_block_keys(300);
        proto::optimizer::TraceQueryResponse tq_resp;
        grpc::ClientContext ctx;
        auto status = stub_->TraceQuery(&ctx, tq_req, &tq_resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, tq_resp.header().status().code());
        EXPECT_EQ(0, tq_resp.cache_hit_count());
        EXPECT_EQ(3, tq_resp.total_blocks());
    }

    {
        proto::optimizer::TraceQueryRequest tq_req;
        tq_req.set_trace_id("integ-tq-hit");
        tq_req.set_instance_id("integ_tq_1");
        tq_req.add_block_keys(100);
        tq_req.add_block_keys(200);
        tq_req.add_block_keys(300);
        proto::optimizer::TraceQueryResponse tq_resp;
        grpc::ClientContext ctx;
        auto status = stub_->TraceQuery(&ctx, tq_req, &tq_resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, tq_resp.header().status().code());
        EXPECT_EQ(3, tq_resp.cache_hit_count());
        EXPECT_EQ(3, tq_resp.total_blocks());
    }

    {
        proto::optimizer::TraceQueryRequest tq_req;
        tq_req.set_trace_id("integ-tq-mixed");
        tq_req.set_instance_id("integ_tq_1");
        tq_req.add_block_keys(100);
        tq_req.add_block_keys(400);
        tq_req.add_block_keys(500);
        proto::optimizer::TraceQueryResponse tq_resp;
        grpc::ClientContext ctx;
        auto status = stub_->TraceQuery(&ctx, tq_req, &tq_resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, tq_resp.header().status().code());
        EXPECT_EQ(1, tq_resp.cache_hit_count());
        EXPECT_EQ(3, tq_resp.total_blocks());
    }
}

TEST_F(OnlineOptimizerIntegrationTest, ResetStats) {
    RegisterTestInstance("integ_grp_reset", "integ_reset_1");

    {
        proto::optimizer::TraceQueryRequest tq_req;
        tq_req.set_trace_id("integ-reset-q1");
        tq_req.set_instance_id("integ_reset_1");
        tq_req.add_block_keys(1);
        tq_req.add_block_keys(2);
        proto::optimizer::TraceQueryResponse tq_resp;
        grpc::ClientContext ctx;
        stub_->TraceQuery(&ctx, tq_req, &tq_resp);
    }

    {
        proto::optimizer::OptimizerResetStatsRequest reset_req;
        reset_req.set_trace_id("integ-reset");
        reset_req.set_instance_id("integ_reset_1");
        proto::optimizer::OptimizerResetStatsResponse reset_resp;
        grpc::ClientContext ctx;
        auto status = stub_->ResetStats(&ctx, reset_req, &reset_resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, reset_resp.header().status().code());
    }

    {
        proto::optimizer::OptimizerListInstancesRequest list_req;
        list_req.set_trace_id("integ-reset-verify");
        list_req.set_instance_group("integ_grp_reset");
        proto::optimizer::OptimizerListInstancesResponse list_resp;
        grpc::ClientContext ctx;
        stub_->ListInstances(&ctx, list_req, &list_resp);
        ASSERT_EQ(1, list_resp.instances_size());
        EXPECT_EQ(0, list_resp.instances(0).total_queries());
        EXPECT_EQ(0, list_resp.instances(0).total_blocks_queried());
    }
}

TEST_F(OnlineOptimizerIntegrationTest, RemoveInstance) {
    RegisterTestInstance("integ_grp_rm", "integ_rm_1");

    {
        proto::optimizer::OptimizerRemoveInstanceRequest rm_req;
        rm_req.set_trace_id("integ-rm");
        rm_req.set_instance_id("integ_rm_1");
        proto::optimizer::OptimizerRemoveInstanceResponse rm_resp;
        grpc::ClientContext ctx;
        auto status = stub_->RemoveInstance(&ctx, rm_req, &rm_resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, rm_resp.header().status().code());
    }

    {
        proto::optimizer::OptimizerListInstancesRequest list_req;
        list_req.set_trace_id("integ-rm-verify");
        list_req.set_instance_group("integ_grp_rm");
        proto::optimizer::OptimizerListInstancesResponse list_resp;
        grpc::ClientContext ctx;
        stub_->ListInstances(&ctx, list_req, &list_resp);
        for (const auto &inst : list_resp.instances()) {
            EXPECT_NE("integ_rm_1", inst.instance_id());
        }
    }
}

TEST_F(OnlineOptimizerIntegrationTest, RemoveNonExistent) {
    proto::optimizer::OptimizerRemoveInstanceRequest rm_req;
    rm_req.set_trace_id("integ-rm-ne");
    rm_req.set_instance_id("integ_nonexistent_xyz");
    proto::optimizer::OptimizerRemoveInstanceResponse rm_resp;
    grpc::ClientContext ctx;
    auto status = stub_->RemoveInstance(&ctx, rm_req, &rm_resp);
    ASSERT_TRUE(status.ok());
    EXPECT_NE(proto::admin::OK, rm_resp.header().status().code());
}

TEST_F(OnlineOptimizerIntegrationTest, TraceQueryNonExistentInstance) {
    proto::optimizer::TraceQueryRequest tq_req;
    tq_req.set_trace_id("integ-tq-ne");
    tq_req.set_instance_id("integ_nonexistent_abc");
    tq_req.add_block_keys(1);
    proto::optimizer::TraceQueryResponse tq_resp;
    grpc::ClientContext ctx;
    auto status = stub_->TraceQuery(&ctx, tq_req, &tq_resp);
    ASSERT_TRUE(status.ok());
    EXPECT_NE(proto::admin::OK, tq_resp.header().status().code());
}

TEST_F(OnlineOptimizerIntegrationTest, ResetStatsNonExistent) {
    proto::optimizer::OptimizerResetStatsRequest reset_req;
    reset_req.set_trace_id("integ-reset-ne");
    reset_req.set_instance_id("integ_nonexistent_reset");
    proto::optimizer::OptimizerResetStatsResponse reset_resp;
    grpc::ClientContext ctx;
    auto status = stub_->ResetStats(&ctx, reset_req, &reset_resp);
    ASSERT_TRUE(status.ok());
    EXPECT_NE(proto::admin::OK, reset_resp.header().status().code());
}

TEST_F(OnlineOptimizerIntegrationTest, MultiGroupIsolation) {
    RegisterTestInstance("integ_iso_grpA", "integ_iso_A1");
    RegisterTestInstance("integ_iso_grpB", "integ_iso_B1");

    {
        proto::optimizer::OptimizerListInstancesRequest list_req;
        list_req.set_trace_id("integ-iso-listA");
        list_req.set_instance_group("integ_iso_grpA");
        proto::optimizer::OptimizerListInstancesResponse list_resp;
        grpc::ClientContext ctx;
        stub_->ListInstances(&ctx, list_req, &list_resp);

        bool found_a1 = false;
        for (const auto &inst : list_resp.instances()) {
            EXPECT_EQ("integ_iso_grpA", inst.instance_group());
            if (inst.instance_id() == "integ_iso_A1") found_a1 = true;
        }
        EXPECT_TRUE(found_a1);
    }

    {
        proto::optimizer::OptimizerListInstancesRequest list_req;
        list_req.set_trace_id("integ-iso-listB");
        list_req.set_instance_group("integ_iso_grpB");
        proto::optimizer::OptimizerListInstancesResponse list_resp;
        grpc::ClientContext ctx;
        stub_->ListInstances(&ctx, list_req, &list_resp);

        bool found_b1 = false;
        for (const auto &inst : list_resp.instances()) {
            EXPECT_EQ("integ_iso_grpB", inst.instance_group());
            if (inst.instance_id() == "integ_iso_B1") found_b1 = true;
        }
        EXPECT_TRUE(found_b1);
    }

    {
        proto::optimizer::OptimizerListInstancesRequest list_req;
        list_req.set_trace_id("integ-iso-listAll");
        list_req.set_instance_group("");
        proto::optimizer::OptimizerListInstancesResponse list_resp;
        grpc::ClientContext ctx;
        stub_->ListInstances(&ctx, list_req, &list_resp);

        bool found_a = false, found_b = false;
        for (const auto &inst : list_resp.instances()) {
            if (inst.instance_id() == "integ_iso_A1") found_a = true;
            if (inst.instance_id() == "integ_iso_B1") found_b = true;
        }
        EXPECT_TRUE(found_a);
        EXPECT_TRUE(found_b);
    }
}

TEST_F(OnlineOptimizerIntegrationTest, RegisterWithLinearStep) {
    auto resp = RegisterTestInstance("integ_grp_ls", "integ_ls_1", 1024, 1.0, 4);
    EXPECT_EQ(proto::admin::OK, resp.header().status().code());

    proto::optimizer::OptimizerListInstancesRequest list_req;
    list_req.set_trace_id("integ-ls-list");
    list_req.set_instance_group("integ_grp_ls");
    proto::optimizer::OptimizerListInstancesResponse list_resp;
    grpc::ClientContext ctx;
    stub_->ListInstances(&ctx, list_req, &list_resp);

    bool found = false;
    for (const auto &inst : list_resp.instances()) {
        if (inst.instance_id() == "integ_ls_1") {
            found = true;
            EXPECT_EQ(4, inst.linear_step());
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(OnlineOptimizerIntegrationTest, RegisterDuplicateOverwrites) {
    RegisterTestInstance("integ_grp_dup", "integ_dup_1", 1024, 1.0);

    {
        proto::optimizer::TraceQueryRequest tq_req;
        tq_req.set_trace_id("integ-dup-q");
        tq_req.set_instance_id("integ_dup_1");
        tq_req.add_block_keys(1);
        proto::optimizer::TraceQueryResponse tq_resp;
        grpc::ClientContext ctx;
        stub_->TraceQuery(&ctx, tq_req, &tq_resp);
    }

    // Re-register same instance (group already exists from first RegisterTestInstance)
    {
        proto::optimizer::OptimizerRegisterInstanceRequest req;
        req.set_trace_id("integ-dup-2");
        req.set_instance_group("integ_grp_dup");
        req.set_instance_id("integ_dup_1");
        req.set_block_size(1024);
        auto *spec = req.add_location_spec_infos();
        spec->set_name("full");
        spec->set_size(1024);

        proto::optimizer::OptimizerRegisterInstanceResponse resp;
        grpc::ClientContext ctx;
        auto status = stub_->RegisterInstance(&ctx, req, &resp);
        ASSERT_TRUE(status.ok());
        EXPECT_EQ(proto::admin::OK, resp.header().status().code());
    }

    {
        proto::optimizer::OptimizerListInstancesRequest list_req;
        list_req.set_trace_id("integ-dup-list");
        list_req.set_instance_group("integ_grp_dup");
        proto::optimizer::OptimizerListInstancesResponse list_resp;
        grpc::ClientContext ctx;
        stub_->ListInstances(&ctx, list_req, &list_resp);

        int count = 0;
        for (const auto &inst : list_resp.instances()) {
            if (inst.instance_id() == "integ_dup_1") count++;
        }
        EXPECT_EQ(1, count);
    }
}

TEST_F(OnlineOptimizerIntegrationTest, FullLifecycle) {
    std::string group = "integ_lifecycle_grp";
    std::string inst_id = "integ_lifecycle_1";

    // 1. Register (creates group + instance)
    auto reg_resp = RegisterTestInstance(group, inst_id);
    ASSERT_EQ(proto::admin::OK, reg_resp.header().status().code());

    // 2. Trace query (miss)
    {
        proto::optimizer::TraceQueryRequest tq_req;
        tq_req.set_trace_id("integ-lc-q1");
        tq_req.set_instance_id(inst_id);
        for (int64_t k = 1; k <= 5; ++k) tq_req.add_block_keys(k);
        proto::optimizer::TraceQueryResponse tq_resp;
        grpc::ClientContext ctx;
        stub_->TraceQuery(&ctx, tq_req, &tq_resp);
        EXPECT_EQ(0, tq_resp.cache_hit_count());
        EXPECT_EQ(5, tq_resp.total_blocks());
    }

    // 3. Trace query (hit)
    {
        proto::optimizer::TraceQueryRequest tq_req;
        tq_req.set_trace_id("integ-lc-q2");
        tq_req.set_instance_id(inst_id);
        for (int64_t k = 1; k <= 5; ++k) tq_req.add_block_keys(k);
        proto::optimizer::TraceQueryResponse tq_resp;
        grpc::ClientContext ctx;
        stub_->TraceQuery(&ctx, tq_req, &tq_resp);
        EXPECT_EQ(5, tq_resp.cache_hit_count());
        EXPECT_EQ(5, tq_resp.total_blocks());
    }

    // 4. List - verify stats
    {
        proto::optimizer::OptimizerListInstancesRequest list_req;
        list_req.set_trace_id("integ-lc-list");
        list_req.set_instance_group(group);
        proto::optimizer::OptimizerListInstancesResponse list_resp;
        grpc::ClientContext ctx;
        stub_->ListInstances(&ctx, list_req, &list_resp);
        ASSERT_GE(list_resp.instances_size(), 1);

        for (const auto &inst : list_resp.instances()) {
            if (inst.instance_id() == inst_id) {
                EXPECT_EQ(2, inst.total_queries());
                EXPECT_EQ(10, inst.total_blocks_queried());
                EXPECT_EQ(5, inst.total_hits());
                EXPECT_GT(inst.hit_rate(), 0.0);
                EXPECT_EQ(5, inst.unique_keys());
            }
        }
    }

    // 5. Reset stats
    {
        proto::optimizer::OptimizerResetStatsRequest reset_req;
        reset_req.set_trace_id("integ-lc-reset");
        reset_req.set_instance_id(inst_id);
        proto::optimizer::OptimizerResetStatsResponse reset_resp;
        grpc::ClientContext ctx;
        stub_->ResetStats(&ctx, reset_req, &reset_resp);
        EXPECT_EQ(proto::admin::OK, reset_resp.header().status().code());
    }

    // 6. Verify stats cleared
    {
        proto::optimizer::OptimizerListInstancesRequest list_req;
        list_req.set_trace_id("integ-lc-list2");
        list_req.set_instance_group(group);
        proto::optimizer::OptimizerListInstancesResponse list_resp;
        grpc::ClientContext ctx;
        stub_->ListInstances(&ctx, list_req, &list_resp);
        for (const auto &inst : list_resp.instances()) {
            if (inst.instance_id() == inst_id) {
                EXPECT_EQ(0, inst.total_queries());
            }
        }
    }

    // 7. Remove
    {
        proto::optimizer::OptimizerRemoveInstanceRequest rm_req;
        rm_req.set_trace_id("integ-lc-rm");
        rm_req.set_instance_id(inst_id);
        proto::optimizer::OptimizerRemoveInstanceResponse rm_resp;
        grpc::ClientContext ctx;
        stub_->RemoveInstance(&ctx, rm_req, &rm_resp);
        EXPECT_EQ(proto::admin::OK, rm_resp.header().status().code());
    }

    // 8. Verify removed - query should fail
    {
        proto::optimizer::TraceQueryRequest tq_req;
        tq_req.set_trace_id("integ-lc-q3");
        tq_req.set_instance_id(inst_id);
        tq_req.add_block_keys(1);
        proto::optimizer::TraceQueryResponse tq_resp;
        grpc::ClientContext ctx;
        stub_->TraceQuery(&ctx, tq_req, &tq_resp);
        EXPECT_NE(proto::admin::OK, tq_resp.header().status().code());
    }
}

} // namespace kv_cache_manager
