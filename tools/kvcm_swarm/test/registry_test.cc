// Registry and extension-boundary tests.
#include <gtest/gtest.h>

#include "async_simple/coro/SyncAwait.h"
#include "tools/kvcm_swarm/clients/registry.h"
#include "tools/kvcm_swarm/test/fake_behavior.h"

namespace kvcm_swarm {
namespace {

TEST(RegistryTest, DefaultRegistryExposesExactlyTheShippedBehaviors) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    const std::vector<std::string> types = registry.Types();
    ASSERT_EQ(types.size(), 2u);
    EXPECT_EQ(types[0], "health_probe");
    EXPECT_EQ(types[1], "v6d_deployment");
    EXPECT_NE(registry.Find("v6d_deployment"), nullptr);
    EXPECT_NE(registry.Find("health_probe"), nullptr);
    // event_reporter will arrive as its own top-level behavior, not as a mode.
    EXPECT_EQ(registry.Find("event_reporter"), nullptr);
}

// A behavior that only knows RuntimeServices can be registered and driven,
// which proves the common runtime carries no V6D domain state.
TEST(RegistryTest, FakeBehaviorRunsThroughTheCommonRuntimeOnly) {
    BehaviorRegistry registry;
    registry.Register("fake_behavior", MakeFakeBehaviorFactory());
    const BehaviorFactory *factory = registry.Find("fake_behavior");
    ASSERT_NE(factory, nullptr);

    BehaviorSpec spec;
    spec.id = "fake-a";
    spec.type = "fake_behavior";
    spec.transport = TransportKind::kHttp;
    spec.config_json = "{\"tick_interval\": \"5ms\"}";
    EXPECT_TRUE(factory->Validate(spec).ok);
    const BehaviorIdentityClaims claims = factory->Claims(spec);
    ASSERT_EQ(claims.exclusive_names.size(), 1u);
    EXPECT_EQ(claims.exclusive_names[0], "fake_behavior:fake-a");
    EXPECT_TRUE(claims.required_instance_groups.empty());

    SwarmExecutor executor(2);
    AdmissionController admission(executor, AdmissionLimits{});
    EvidenceSink evidence;
    PhaseSource phase;
    StopSource stop;
    SeedDeriver seeds(1);
    EndpointSet endpoints;
    endpoints.meta_http = "http://127.0.0.1:1";
    endpoints.meta_grpc = "127.0.0.1:2";
    endpoints.admin_http = "http://127.0.0.1:3";
    endpoints.admin_grpc = "127.0.0.1:2";
    TransportProvider transports(executor, admission, evidence, phase, endpoints, TransportLimits{}, 1, 1);
    RuntimeServices services{executor, admission, transports, evidence, seeds, stop.Token(), phase};

    std::unique_ptr<ClientBehavior> behavior = factory->Create(spec, services);
    ASSERT_NE(behavior, nullptr);
    EXPECT_EQ(behavior->TypeName(), "fake_behavior");
    ASSERT_TRUE(
        async_simple::coro::syncAwait(std::move(behavior->Initialize(Now() + std::chrono::seconds(1))).via(&executor)));
    behavior->StartTraffic();
    async_simple::coro::syncAwait(
        std::move(SleepFor(executor, std::chrono::milliseconds(60), StopToken())).via(&executor));
    async_simple::coro::syncAwait(std::move(behavior->Drain(Now() + std::chrono::seconds(1))).via(&executor));
    EXPECT_TRUE(behavior->Quiesced());
    const auto invariants = behavior->Invariants();
    ASSERT_EQ(invariants.size(), 1u);
    EXPECT_EQ(invariants[0].status, CheckStatus::kPass);
    EXPECT_GT(invariants[0].checked, 0u);
    // Optional report sections default to "nothing to add" for a behavior with
    // no cache and no workload of its own.
    JsonWriter probe(false);
    EXPECT_FALSE(behavior->WriteCacheReport(probe));
    EXPECT_FALSE(behavior->WriteWorkloadShape(probe));
    EXPECT_FALSE(behavior->WriteCleanupReport(probe));
    transports.Shutdown();
    executor.Shutdown();
}

} // namespace
} // namespace kvcm_swarm
