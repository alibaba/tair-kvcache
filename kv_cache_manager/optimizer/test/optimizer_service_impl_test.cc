#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/optimizer/config/optimizer_registry_manager.h"
#include "kv_cache_manager/optimizer/online_runtime/online_optimizer_manager.h"
#include "kv_cache_manager/optimizer/service/optimizer_service_impl.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

namespace kv_cache_manager {

class OptimizerServiceImplTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        registry_ = std::make_shared<OptimizerRegistryManager>("");
        registry_->Init();
        manager_ = std::make_shared<OnlineOptimizerManager>(registry_);
        service_ = std::make_shared<OptimizerServiceImpl>(manager_, nullptr);
    }

    void CreateTestGroup(const std::string &group_name, double capacity_gb = 1.0) {
        proto::optimizer::CreateInstanceGroupRequest req;
        req.set_trace_id("setup");
        auto *g = req.mutable_instance_group();
        g->set_name(group_name);
        g->add_capacity_gb(capacity_gb);
        g->set_eviction_policy(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU);

        proto::optimizer::CommonResponse resp;
        RequestContext ctx("setup", nullptr);
        service_->CreateInstanceGroup(&ctx, &req, &resp);
    }

    proto::optimizer::OptimizerRegisterInstanceRequest MakeRegisterRequest(const std::string &group,
                                                                           const std::string &instance_id,
                                                                           int32_t block_size,
                                                                           int32_t linear_step = 1,
                                                                           int64_t extra_spec_size = 0) {
        proto::optimizer::OptimizerRegisterInstanceRequest req;
        req.set_trace_id("test-trace");
        req.set_instance_group(group);
        req.set_instance_id(instance_id);
        req.set_block_size(block_size);
        req.set_linear_step(linear_step);

        auto *spec = req.add_location_spec_infos();
        spec->set_name("tp0");
        spec->set_size(block_size);

        auto *full_group = req.add_location_spec_groups();
        full_group->set_name("full");
        full_group->add_spec_names("tp0");

        auto *state_info = req.mutable_optimizer_state_info();
        state_info->set_full_location_spec_group_name("full");

        if (extra_spec_size > 0) {
            auto *extra_spec = req.add_location_spec_infos();
            extra_spec->set_name("tp0_linear");
            extra_spec->set_size(extra_spec_size);

            auto *linear_group = req.add_location_spec_groups();
            linear_group->set_name("linear");
            linear_group->add_spec_names("tp0_linear");
            state_info->set_linear_location_spec_group_name("linear");
        }

        return req;
    }

    std::shared_ptr<OptimizerRegistryManager> registry_;
    std::shared_ptr<OnlineOptimizerManager> manager_;
    std::shared_ptr<OptimizerServiceImpl> service_;
};

// InstanceGroup CRUD tests

TEST_F(OptimizerServiceImplTest, CreateAndGetInstanceGroup) {
    proto::optimizer::CreateInstanceGroupRequest create_req;
    create_req.set_trace_id("t1");
    auto *g = create_req.mutable_instance_group();
    g->set_name("test_grp");
    g->add_capacity_gb(2.0);
    g->set_eviction_policy(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU);

    proto::optimizer::CommonResponse create_resp;
    RequestContext ctx("t1", nullptr);
    service_->CreateInstanceGroup(&ctx, &create_req, &create_resp);
    EXPECT_EQ(proto::optimizer::OK, create_resp.header().status().code());

    proto::optimizer::GetInstanceGroupRequest get_req;
    get_req.set_trace_id("t2");
    get_req.set_name("test_grp");
    proto::optimizer::GetInstanceGroupResponse get_resp;
    service_->GetInstanceGroup(&ctx, &get_req, &get_resp);
    EXPECT_EQ(proto::optimizer::OK, get_resp.header().status().code());
    EXPECT_EQ("test_grp", get_resp.instance_group().name());
    EXPECT_EQ(1, get_resp.instance_group().capacity_gb_size());
    EXPECT_DOUBLE_EQ(2.0, get_resp.instance_group().capacity_gb(0));
    EXPECT_EQ(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU, get_resp.instance_group().eviction_policy());
}

TEST_F(OptimizerServiceImplTest, CreateDuplicateGroupFails) {
    CreateTestGroup("dup_grp");

    proto::optimizer::CreateInstanceGroupRequest req;
    req.set_trace_id("t1");
    auto *g = req.mutable_instance_group();
    g->set_name("dup_grp");
    g->add_capacity_gb(1.0);
    g->set_eviction_policy(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU);

    proto::optimizer::CommonResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->CreateInstanceGroup(&ctx, &req, &resp);
    EXPECT_NE(proto::optimizer::OK, resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, CreateInstanceGroupUnspecifiedPolicyFails) {
    proto::optimizer::CreateInstanceGroupRequest req;
    req.set_trace_id("t1");
    auto *g = req.mutable_instance_group();
    g->set_name("bad_policy_grp");
    g->add_capacity_gb(1.0);

    proto::optimizer::CommonResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->CreateInstanceGroup(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, UpdateInstanceGroup) {
    CreateTestGroup("upd_grp", 1.0);

    proto::optimizer::UpdateInstanceGroupRequest req;
    req.set_trace_id("t1");
    auto *g = req.mutable_instance_group();
    g->set_name("upd_grp");
    g->add_capacity_gb(4.0);
    g->set_eviction_policy(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU);
    g->set_ttl_seconds(5000);
    g->set_enable_theoretical_max_cache(true);

    proto::optimizer::CommonResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->UpdateInstanceGroup(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::OK, resp.header().status().code());

    proto::optimizer::GetInstanceGroupRequest get_req;
    get_req.set_trace_id("t2");
    get_req.set_name("upd_grp");
    proto::optimizer::GetInstanceGroupResponse get_resp;
    service_->GetInstanceGroup(&ctx, &get_req, &get_resp);
    ASSERT_EQ(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU, get_resp.instance_group().eviction_policy());
    EXPECT_EQ(5000, get_resp.instance_group().ttl_seconds());
    EXPECT_DOUBLE_EQ(4.0, get_resp.instance_group().capacity_gb(0));
    EXPECT_TRUE(get_resp.instance_group().enable_theoretical_max_cache());
}

TEST_F(OptimizerServiceImplTest, RemoveInstanceGroup) {
    CreateTestGroup("rm_grp");

    proto::optimizer::RemoveInstanceGroupRequest req;
    req.set_trace_id("t1");
    req.set_name("rm_grp");
    proto::optimizer::CommonResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->RemoveInstanceGroup(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::OK, resp.header().status().code());

    proto::optimizer::GetInstanceGroupRequest get_req;
    get_req.set_trace_id("t2");
    get_req.set_name("rm_grp");
    proto::optimizer::GetInstanceGroupResponse get_resp;
    service_->GetInstanceGroup(&ctx, &get_req, &get_resp);
    EXPECT_NE(proto::optimizer::OK, get_resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, RemoveInstanceGroupWithRegisteredInstanceFails) {
    CreateTestGroup("grp_with_inst");

    auto reg_req = MakeRegisterRequest("grp_with_inst", "inst1", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse reg_resp;
    RequestContext ctx("trace1", nullptr);
    service_->RegisterInstance(&ctx, &reg_req, &reg_resp);
    ASSERT_EQ(proto::optimizer::OK, reg_resp.header().status().code());

    proto::optimizer::RemoveInstanceGroupRequest req;
    req.set_trace_id("t1");
    req.set_name("grp_with_inst");
    proto::optimizer::CommonResponse resp;
    service_->RemoveInstanceGroup(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, resp.header().status().code());

    proto::optimizer::GetInstanceGroupRequest get_req;
    get_req.set_trace_id("t2");
    get_req.set_name("grp_with_inst");
    proto::optimizer::GetInstanceGroupResponse get_resp;
    service_->GetInstanceGroup(&ctx, &get_req, &get_resp);
    EXPECT_EQ(proto::optimizer::OK, get_resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, RemoveInstanceGroupWithActiveInstanceOnlyFails) {
    CreateTestGroup("grp_active_only");

    auto reg_req = MakeRegisterRequest("grp_active_only", "inst1", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse reg_resp;
    RequestContext ctx("trace1", nullptr);
    service_->RegisterInstance(&ctx, &reg_req, &reg_resp);
    ASSERT_EQ(proto::optimizer::OK, reg_resp.header().status().code());
    ASSERT_EQ(EC_OK, registry_->DeleteInstanceInfo("inst1"));

    proto::optimizer::RemoveInstanceGroupRequest req;
    req.set_trace_id("t1");
    req.set_name("grp_active_only");
    proto::optimizer::CommonResponse resp;
    service_->RemoveInstanceGroup(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, UpdateInstanceGroupWithRegisteredInstanceFails) {
    CreateTestGroup("grp_update_with_inst");

    auto reg_req = MakeRegisterRequest("grp_update_with_inst", "inst1", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse reg_resp;
    RequestContext ctx("trace1", nullptr);
    service_->RegisterInstance(&ctx, &reg_req, &reg_resp);
    ASSERT_EQ(proto::optimizer::OK, reg_resp.header().status().code());

    proto::optimizer::UpdateInstanceGroupRequest req;
    req.set_trace_id("t1");
    auto *g = req.mutable_instance_group();
    g->set_name("grp_update_with_inst");
    g->add_capacity_gb(4.0);
    g->set_eviction_policy(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU);

    proto::optimizer::CommonResponse resp;
    service_->UpdateInstanceGroup(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, ListInstanceGroups) {
    CreateTestGroup("list_g1");
    CreateTestGroup("list_g2");

    proto::optimizer::ListInstanceGroupsRequest req;
    req.set_trace_id("t1");
    proto::optimizer::ListInstanceGroupsResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->ListInstanceGroups(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::OK, resp.header().status().code());
    EXPECT_GE(resp.instance_groups_size(), 2);
}

TEST_F(OptimizerServiceImplTest, GetNonExistentGroupFails) {
    proto::optimizer::GetInstanceGroupRequest req;
    req.set_trace_id("t1");
    req.set_name("nonexistent");
    proto::optimizer::GetInstanceGroupResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->GetInstanceGroup(&ctx, &req, &resp);
    EXPECT_NE(proto::optimizer::OK, resp.header().status().code());
}

// Instance registration tests

TEST_F(OptimizerServiceImplTest, RegisterInstanceRequiresGroup) {
    auto req = MakeRegisterRequest("no_such_group", "inst1", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->RegisterInstance(&ctx, &req, &resp);
    EXPECT_NE(proto::optimizer::OK, resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, RegisterAndListInstances) {
    CreateTestGroup("grp1");

    auto req = MakeRegisterRequest("grp1", "inst1", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse resp;
    RequestContext ctx("trace1", nullptr);

    service_->RegisterInstance(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::OK, resp.header().status().code());
    EXPECT_EQ(1, resp.estimated_capacity_blocks_size());
    EXPECT_GT(resp.estimated_capacity_blocks(0), 0);

    proto::optimizer::OptimizerListInstancesRequest list_req;
    list_req.set_trace_id("trace2");
    list_req.set_instance_group("grp1");
    proto::optimizer::OptimizerListInstancesResponse list_resp;

    service_->ListInstances(&ctx, &list_req, &list_resp);
    EXPECT_EQ(proto::optimizer::OK, list_resp.header().status().code());
    EXPECT_EQ(1, list_resp.instances_size());
    EXPECT_EQ("inst1", list_resp.instances(0).instance_id());
    EXPECT_EQ("grp1", list_resp.instances(0).instance_group());
}

TEST_F(OptimizerServiceImplTest, RegisterDuplicateOverwrites) {
    CreateTestGroup("grp1");

    auto req = MakeRegisterRequest("grp1", "inst1", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse resp1, resp2;
    RequestContext ctx("trace1", nullptr);

    service_->RegisterInstance(&ctx, &req, &resp1);
    EXPECT_EQ(proto::optimizer::OK, resp1.header().status().code());

    service_->RegisterInstance(&ctx, &req, &resp2);
    EXPECT_EQ(proto::optimizer::OK, resp2.header().status().code());

    proto::optimizer::OptimizerListInstancesRequest list_req;
    list_req.set_instance_group("grp1");
    proto::optimizer::OptimizerListInstancesResponse list_resp;
    service_->ListInstances(&ctx, &list_req, &list_resp);
    EXPECT_EQ(1, list_resp.instances_size());
}

TEST_F(OptimizerServiceImplTest, RemoveInstance) {
    CreateTestGroup("grp1");

    auto req = MakeRegisterRequest("grp1", "inst1", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse reg_resp;
    RequestContext ctx("trace1", nullptr);
    service_->RegisterInstance(&ctx, &req, &reg_resp);

    proto::optimizer::OptimizerRemoveInstanceRequest rm_req;
    rm_req.set_trace_id("trace2");
    rm_req.set_instance_id("inst1");
    proto::optimizer::OptimizerRemoveInstanceResponse rm_resp;

    service_->RemoveInstance(&ctx, &rm_req, &rm_resp);
    EXPECT_EQ(proto::optimizer::OK, rm_resp.header().status().code());

    proto::optimizer::OptimizerListInstancesRequest list_req;
    list_req.set_instance_group("grp1");
    proto::optimizer::OptimizerListInstancesResponse list_resp;
    service_->ListInstances(&ctx, &list_req, &list_resp);
    EXPECT_EQ(0, list_resp.instances_size());
}

TEST_F(OptimizerServiceImplTest, RemoveNonExistent) {
    proto::optimizer::OptimizerRemoveInstanceRequest rm_req;
    rm_req.set_trace_id("trace1");
    rm_req.set_instance_id("nonexistent");
    proto::optimizer::OptimizerRemoveInstanceResponse rm_resp;
    RequestContext ctx("trace1", nullptr);

    service_->RemoveInstance(&ctx, &rm_req, &rm_resp);
    EXPECT_NE(proto::optimizer::OK, rm_resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, TraceQuery) {
    CreateTestGroup("grp1");

    auto req = MakeRegisterRequest("grp1", "inst1", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse reg_resp;
    RequestContext ctx("trace1", nullptr);
    service_->RegisterInstance(&ctx, &req, &reg_resp);

    proto::optimizer::TraceQueryRequest tq_req;
    tq_req.set_trace_id("trace2");
    tq_req.set_instance_id("inst1");
    tq_req.add_block_keys(1);
    tq_req.add_block_keys(2);
    tq_req.add_block_keys(3);
    proto::optimizer::TraceQueryResponse tq_resp;

    service_->TraceQuery(&ctx, &tq_req, &tq_resp);
    EXPECT_EQ(proto::optimizer::OK, tq_resp.header().status().code());
    EXPECT_EQ(3, tq_resp.total_blocks());
    ASSERT_EQ(1, tq_resp.capacity_results_size());
    EXPECT_EQ(0, tq_resp.capacity_results(0).cache_hit_count());
    EXPECT_EQ(3, tq_resp.capacity_results(0).current_unique_keys());

    proto::optimizer::TraceQueryResponse tq_resp2;
    service_->TraceQuery(&ctx, &tq_req, &tq_resp2);
    EXPECT_EQ(proto::optimizer::OK, tq_resp2.header().status().code());
    EXPECT_EQ(3, tq_resp2.total_blocks());
    ASSERT_EQ(1, tq_resp2.capacity_results_size());
    EXPECT_EQ(3, tq_resp2.capacity_results(0).cache_hit_count());
    EXPECT_EQ(3, tq_resp2.capacity_results(0).current_unique_keys());
}

TEST_F(OptimizerServiceImplTest, TraceQueryNonExistentInstance) {
    proto::optimizer::TraceQueryRequest tq_req;
    tq_req.set_trace_id("trace1");
    tq_req.set_instance_id("nonexistent");
    tq_req.add_block_keys(1);
    proto::optimizer::TraceQueryResponse tq_resp;
    RequestContext ctx("trace1", nullptr);

    service_->TraceQuery(&ctx, &tq_req, &tq_resp);
    EXPECT_NE(proto::optimizer::OK, tq_resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, ResetStats) {
    CreateTestGroup("grp1");

    auto req = MakeRegisterRequest("grp1", "inst1", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse reg_resp;
    RequestContext ctx("trace1", nullptr);
    service_->RegisterInstance(&ctx, &req, &reg_resp);

    proto::optimizer::TraceQueryRequest tq_req;
    tq_req.set_trace_id("trace2");
    tq_req.set_instance_id("inst1");
    tq_req.add_block_keys(1);
    proto::optimizer::TraceQueryResponse tq_resp;
    service_->TraceQuery(&ctx, &tq_req, &tq_resp);

    proto::optimizer::OptimizerResetStatsRequest reset_req;
    reset_req.set_trace_id("trace3");
    reset_req.set_instance_id("inst1");
    proto::optimizer::OptimizerResetStatsResponse reset_resp;

    service_->ResetStats(&ctx, &reset_req, &reset_resp);
    EXPECT_EQ(proto::optimizer::OK, reset_resp.header().status().code());

    proto::optimizer::OptimizerListInstancesRequest list_req;
    list_req.set_instance_group("grp1");
    proto::optimizer::OptimizerListInstancesResponse list_resp;
    service_->ListInstances(&ctx, &list_req, &list_resp);
    EXPECT_EQ(1, list_resp.instances_size());
    EXPECT_EQ(0, list_resp.instances(0).total_queries());
}

TEST_F(OptimizerServiceImplTest, ListInstancesFilterByGroup) {
    CreateTestGroup("grp1");
    CreateTestGroup("grp2");
    RequestContext ctx("trace1", nullptr);

    auto req1 = MakeRegisterRequest("grp1", "inst1", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse resp1;
    service_->RegisterInstance(&ctx, &req1, &resp1);

    auto req2 = MakeRegisterRequest("grp2", "inst2", 1024);
    proto::optimizer::OptimizerRegisterInstanceResponse resp2;
    service_->RegisterInstance(&ctx, &req2, &resp2);

    proto::optimizer::OptimizerListInstancesRequest list_req;
    list_req.set_instance_group("grp1");
    proto::optimizer::OptimizerListInstancesResponse list_resp;
    service_->ListInstances(&ctx, &list_req, &list_resp);
    EXPECT_EQ(1, list_resp.instances_size());
    EXPECT_EQ("inst1", list_resp.instances(0).instance_id());

    proto::optimizer::OptimizerListInstancesRequest list_all_req;
    list_all_req.set_instance_group("");
    proto::optimizer::OptimizerListInstancesResponse list_all_resp;
    service_->ListInstances(&ctx, &list_all_req, &list_all_resp);
    EXPECT_EQ(2, list_all_resp.instances_size());
}

TEST_F(OptimizerServiceImplTest, ResetStatsNonExistent) {
    proto::optimizer::OptimizerResetStatsRequest reset_req;
    reset_req.set_trace_id("trace1");
    reset_req.set_instance_id("nonexistent");
    proto::optimizer::OptimizerResetStatsResponse reset_resp;
    RequestContext ctx("trace1", nullptr);

    service_->ResetStats(&ctx, &reset_req, &reset_resp);
    EXPECT_NE(proto::optimizer::OK, reset_resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, RegisterWithLinearStep) {
    CreateTestGroup("grp_ls", 1.0);

    auto req = MakeRegisterRequest("grp_ls", "inst1", 1024, 4, 256);

    proto::optimizer::OptimizerRegisterInstanceResponse resp;
    RequestContext ctx("trace1", nullptr);
    service_->RegisterInstance(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::OK, resp.header().status().code());
    EXPECT_EQ(1024, resp.size_full_only());
    EXPECT_EQ(1280, resp.size_full_linear());
}

TEST_F(OptimizerServiceImplTest, GetInstanceSuccess) {
    CreateTestGroup("grp1");

    auto reg_req = MakeRegisterRequest("grp1", "inst1", 128, 2, 64);

    proto::optimizer::OptimizerRegisterInstanceResponse reg_resp;
    RequestContext ctx1("t1", nullptr);
    service_->RegisterInstance(&ctx1, &reg_req, &reg_resp);
    ASSERT_EQ(proto::optimizer::OK, reg_resp.header().status().code());

    proto::optimizer::OptimizerGetInstanceRequest get_req;
    get_req.set_trace_id("t2");
    get_req.set_instance_id("inst1");
    proto::optimizer::OptimizerGetInstanceResponse get_resp;
    RequestContext ctx2("t2", nullptr);
    service_->GetInstance(&ctx2, &get_req, &get_resp);

    EXPECT_EQ(proto::optimizer::OK, get_resp.header().status().code());
    EXPECT_EQ("grp1", get_resp.instance_group());
    EXPECT_EQ("inst1", get_resp.instance_id());
    EXPECT_EQ(128, get_resp.block_size());
    EXPECT_EQ(2, get_resp.linear_step());

    ASSERT_EQ(2, get_resp.location_spec_infos_size());
    EXPECT_EQ("tp0", get_resp.location_spec_infos(0).name());
    EXPECT_EQ(128, get_resp.location_spec_infos(0).size());
    EXPECT_EQ("tp0_linear", get_resp.location_spec_infos(1).name());
    EXPECT_EQ(64, get_resp.location_spec_infos(1).size());

    ASSERT_EQ(2, get_resp.location_spec_groups_size());
    EXPECT_EQ("full", get_resp.location_spec_groups(0).name());
    ASSERT_EQ(1, get_resp.location_spec_groups(0).spec_names_size());
    EXPECT_EQ("tp0", get_resp.location_spec_groups(0).spec_names(0));
    EXPECT_EQ("linear", get_resp.location_spec_groups(1).name());
    ASSERT_EQ(1, get_resp.location_spec_groups(1).spec_names_size());
    EXPECT_EQ("tp0_linear", get_resp.location_spec_groups(1).spec_names(0));
    EXPECT_EQ("full", get_resp.optimizer_state_info().full_location_spec_group_name());
    EXPECT_EQ("linear", get_resp.optimizer_state_info().linear_location_spec_group_name());
}

TEST_F(OptimizerServiceImplTest, GetInstanceNotExist) {
    proto::optimizer::OptimizerGetInstanceRequest req;
    req.set_trace_id("t1");
    req.set_instance_id("nonexistent");
    proto::optimizer::OptimizerGetInstanceResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->GetInstance(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::INSTANCE_NOT_EXIST, resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, CreateInstanceGroupNegativeTtlFails) {
    proto::optimizer::CreateInstanceGroupRequest req;
    req.set_trace_id("t1");
    auto *g = req.mutable_instance_group();
    g->set_name("bad_grp");
    g->add_capacity_gb(1.0);
    g->set_eviction_policy(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU);
    g->set_ttl_seconds(-1);

    proto::optimizer::CommonResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->CreateInstanceGroup(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, UpdateInstanceGroupNegativeTtlFails) {
    CreateTestGroup("update_grp");

    proto::optimizer::UpdateInstanceGroupRequest req;
    req.set_trace_id("t1");
    auto *g = req.mutable_instance_group();
    g->set_name("update_grp");
    g->add_capacity_gb(1.0);
    g->set_eviction_policy(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU);
    g->set_ttl_seconds(-1);

    proto::optimizer::CommonResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->UpdateInstanceGroup(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, CreateInstanceGroupEmptyName) {
    proto::optimizer::CreateInstanceGroupRequest req;
    req.set_trace_id("t1");
    auto *g = req.mutable_instance_group();
    g->set_name("");
    g->add_capacity_gb(1.0);
    g->set_eviction_policy(proto::optimizer::OPTIMIZER_EVICTION_POLICY_LRU);

    proto::optimizer::CommonResponse resp;
    RequestContext ctx("t1", nullptr);
    service_->CreateInstanceGroup(&ctx, &req, &resp);
    EXPECT_EQ(proto::optimizer::INVALID_ARGUMENT, resp.header().status().code());
}

TEST_F(OptimizerServiceImplTest, RegisterInstanceUnsupportedIndexerType) {
    // Create a group with valid type, then update it to unsupported type
    // to bypass service validation and test the manager's defense-in-depth check
    CreateTestGroup("idx_grp");

    // Directly register via manager with a bad group in registry to test factory nullptr path.
    OptimizerInstanceInfo info("idx_grp",
                               "inst_bad",
                               1024,
                               {LocationSpecInfo("full", 1024)},
                               {LocationSpecGroup("full_group", {"full"})},
                               1,
                               OptimizerStateInfo("full_group", ""));
    OptimizerInstanceGroup bad_group;
    bad_group.set_name("idx_grp");
    bad_group.set_capacity_gb({1.0});
    bad_group.set_eviction_policy("nonexistent_indexer");
    ASSERT_EQ(EC_OK, registry_->UpdateInstanceGroup(bad_group));

    RegisterInstanceResult result;
    ErrorCode ec = manager_->RegisterInstance(info, result);
    EXPECT_EQ(EC_BADARGS, ec);
}

} // namespace kv_cache_manager
