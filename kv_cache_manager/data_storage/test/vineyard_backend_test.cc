#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/data_storage/vineyard_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;
using namespace std::chrono_literals;

class VineyardBackendTest : public TESTBASE {
public:
    void SetUp() override { metrics_registry_ = std::make_shared<MetricsRegistry>(); }

    // Build a spec with very tight liveness intervals so the LivenessCheckerLoop
    // tests can converge in seconds instead of the default 30s timeout / 5min
    // cleanup grace period.
    static StorageConfig
    MakeConfig(int64_t hb_timeout_ms = 200, int64_t cleanup_grace_ms = 400, int64_t check_interval_ms = 50) {
        auto spec = std::make_shared<VineyardStorageSpec>();
        spec->set_cluster_name("v6d_cluster_test");
        spec->set_heartbeat_timeout_ms(hb_timeout_ms);
        spec->set_cleanup_grace_ms(cleanup_grace_ms);
        spec->set_liveness_check_interval_ms(check_interval_ms);
        return StorageConfig(DataStorageType::DATA_STORAGE_TYPE_VINEYARD, "v6d_test", spec);
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

// (1) GetType / Available / Create-Delete EC_UNIMPLEMENTED / GetStorageUsageRatio=1.0
TEST_F(VineyardBackendTest, BasicAccessors) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(backend.GetType(), DataStorageType::DATA_STORAGE_TYPE_VINEYARD);
    ASSERT_FALSE(backend.Available());

    ASSERT_DOUBLE_EQ(1.0, backend.GetStorageUsageRatio("trace"));

    // Create / Delete must surface EC_UNIMPLEMENTED (V8 §7.2): KVCM never
    // allocates storage on V6D's behalf.
    auto create_res = backend.Create({"k1", "k2"}, 64, "trace", []() {});
    ASSERT_EQ(create_res.size(), 2u);
    for (const auto &[ec, uri] : create_res) {
        ASSERT_EQ(ec, ErrorCode::EC_UNIMPLEMENTED);
    }
    DataStorageUri u;
    auto del_res = backend.Delete({u, u}, "trace", []() {});
    ASSERT_EQ(del_res.size(), 2u);
    for (auto ec : del_res) {
        ASSERT_EQ(ec, ErrorCode::EC_UNIMPLEMENTED);
    }
}

TEST_F(VineyardBackendTest, OpenWithWrongSpecTypeFails) {
    VineyardBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_root_path("/tmp");
    StorageConfig cfg(DataStorageType::DATA_STORAGE_TYPE_VINEYARD, "v6d_test", spec);
    ASSERT_NE(EC_OK, backend.Open(cfg, "trace"));
    ASSERT_FALSE(backend.Available());
}

TEST_F(VineyardBackendTest, OpenStartsLivenessLoopAndCloseStops) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(), "trace"));
    ASSERT_TRUE(backend.Available());
    ASSERT_TRUE(backend.liveness_checker_running_.load());
    ASSERT_TRUE(backend.liveness_checker_thread_.joinable());

    ASSERT_EQ(EC_OK, backend.Close());
    ASSERT_FALSE(backend.Available());
    ASSERT_FALSE(backend.liveness_checker_running_.load());
    // Close() is supposed to join() the loop thread; a second call must not
    // hang or crash.
    ASSERT_EQ(EC_OK, backend.Close());
}

// (2) RegisterNode / UnregisterNode
TEST_F(VineyardBackendTest, RegisterNodeWithMediums) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(), "trace"));

    ASSERT_EQ(EC_BADARGS, backend.RegisterNode("", {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.1:8080", {"mem", "disk"}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.1:8080"));

    // Idempotent: same host, additional medium -- merge.
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.1:8080", {"disk", "ssd"}));
    {
        auto it = backend.nodes_.find("10.0.0.1:8080");
        ASSERT_NE(it, backend.nodes_.end());
        ASSERT_EQ(it->second->mediums.size(), 3u); // mem + disk + ssd
    }

    // Re-registration after a node was flagged unavailable resurrects it.
    backend.SetNodeUnavailable("10.0.0.1:8080");
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.1:8080"));
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.1:8080", {"mem"}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.1:8080"));

    ASSERT_EQ(EC_OK, backend.UnregisterNode("10.0.0.1:8080"));
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.1:8080"));
    ASSERT_EQ(EC_NOENT, backend.UnregisterNode("10.0.0.1:8080"));
    ASSERT_EQ(EC_OK, backend.Close());
}

// (3) IsLocationAvailable: location_id parsing covers the host slice
TEST_F(VineyardBackendTest, IsLocationAvailableParsesHostFromLocationId) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(), "trace"));
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.2:8080", {"mem"}));

    // V8 §2.1.4 layout: kvs#v6d#{medium}#{ip:port}
    EXPECT_TRUE(backend.IsLocationAvailable("kvs#v6d#mem#10.0.0.2:8080"));
    EXPECT_TRUE(backend.IsLocationAvailable("kvs#v6d#disk#10.0.0.2:8080"));
    // Unregistered host
    EXPECT_FALSE(backend.IsLocationAvailable("kvs#v6d#mem#192.168.99.99:8080"));
    // Malformed: no '#' suffix
    EXPECT_FALSE(backend.IsLocationAvailable("not_a_location_id"));
    // Malformed: trailing '#' with empty host
    EXPECT_FALSE(backend.IsLocationAvailable("kvs#v6d#mem#"));

    ASSERT_EQ(EC_OK, backend.Close());
}

// (4) OnHeartbeat: first / refresh / wake-from-unavailable / unregistered
TEST_F(VineyardBackendTest, OnHeartbeatRefreshesAndRevivesNode) {
    VineyardBackend backend(metrics_registry_);
    // Tight thresholds so we can validate within a few hundred ms.
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 200, /*grace*/ 5000, /*tick*/ 50), "trace"));
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.3:8080", {"mem"}));

    int64_t initial_hb = 0;
    {
        auto it = backend.nodes_.find("10.0.0.3:8080");
        ASSERT_NE(it, backend.nodes_.end());
        initial_hb = it->second->last_heartbeat_ms.load();
        ASSERT_GT(initial_hb, 0);
    }

    // Refresh path: heartbeat after a small delay must bump last_heartbeat_ms.
    std::this_thread::sleep_for(20ms);
    backend.OnHeartbeat("10.0.0.3:8080", {{"version", "v6d-0.18"}});
    {
        auto it = backend.nodes_.find("10.0.0.3:8080");
        ASSERT_GT(it->second->last_heartbeat_ms.load(), initial_hb);
        ASSERT_EQ(it->second->last_system_status.at("version"), "v6d-0.18");
    }

    // Revive path: explicitly mark unavailable, then a fresh heartbeat must
    // flip available=true and clear unavailable_since_ms.
    backend.SetNodeUnavailable("10.0.0.3:8080");
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.3:8080"));
    backend.OnHeartbeat("10.0.0.3:8080", {});
    {
        auto it = backend.nodes_.find("10.0.0.3:8080");
        ASSERT_TRUE(it->second->available.load());
        ASSERT_EQ(it->second->unavailable_since_ms.load(), 0);
    }

    // Unregistered host: silently ignored, no crash, no entry created.
    backend.OnHeartbeat("99.99.99.99:8080", {{"x", "y"}});
    ASSERT_EQ(backend.nodes_.count("99.99.99.99:8080"), 0u);

    ASSERT_EQ(EC_OK, backend.Close());
}

// (5) LivenessCheckerLoop three-stage flow: healthy -> unavailable (lazy) ->
//     dead (cleanup callback fires + node erased from nodes_)
TEST_F(VineyardBackendTest, LivenessLoopHealthyToUnavailableToCleanup) {
    VineyardBackend backend(metrics_registry_);
    // hb=100ms, grace=200ms, tick=20ms -> total observable window ~300ms.
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 100, /*grace*/ 200, /*tick*/ 20), "trace"));

    std::atomic<int> cleanup_calls{0};
    std::string cleanup_host;
    backend.SetCleanupCallback([&](const std::string &host, uint64_t /*gen*/) {
        ++cleanup_calls;
        cleanup_host = host;
    });

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.4:8080", {"mem"}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.4:8080"));

    // Stage 1 -> 2: wait past hb_timeout, the loop should mark unavailable
    // but NOT yet trigger cleanup.
    std::this_thread::sleep_for(160ms);
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.4:8080"));
    EXPECT_EQ(cleanup_calls.load(), 0);

    // Stage 2 -> 3: wait past cleanup_grace_ms; cleanup callback should fire
    // exactly for this host. Allow generous slack for slow CI machines.
    for (int i = 0; i < 50 && cleanup_calls.load() == 0; ++i) {
        std::this_thread::sleep_for(20ms);
    }
    EXPECT_GE(cleanup_calls.load(), 1);
    EXPECT_EQ(cleanup_host, "10.0.0.4:8080");

    // After cleanup, the node must be removed from nodes_ (no zombie entry).
    EXPECT_EQ(backend.nodes_.count("10.0.0.4:8080"), 0u);

    ASSERT_EQ(EC_OK, backend.Close());
}

// (6) Heartbeat within the grace window restores availability before cleanup
//     fires (V8 §2.4.6 grace-period recovery).
TEST_F(VineyardBackendTest, HeartbeatWithinGraceWindowRecovers) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 80, /*grace*/ 5000, /*tick*/ 20), "trace"));

    std::atomic<int> cleanup_calls{0};
    backend.SetCleanupCallback([&](const std::string &, uint64_t /*gen*/) { ++cleanup_calls; });

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.5:8080", {"mem"}));
    // Wait long enough to pass hb_timeout but well within grace.
    std::this_thread::sleep_for(140ms);
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.5:8080"));

    // Reviving heartbeat.
    backend.OnHeartbeat("10.0.0.5:8080", {});
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.5:8080"));

    // Hold a bit more and confirm cleanup never fires (5s grace easily
    // covers the delay; we only sleep 60ms more).
    std::this_thread::sleep_for(60ms);
    EXPECT_EQ(cleanup_calls.load(), 0);

    ASSERT_EQ(EC_OK, backend.Close());
}

// (7) After cleanup fires the node is erased; re-registration creates a fresh
//     entry so V6D can rejoin the cluster without restart.
TEST_F(VineyardBackendTest, RegisterAfterCleanupCreatesNewEntry) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 80, /*grace*/ 120, /*tick*/ 20), "trace"));

    std::atomic<int> cleanup_calls{0};
    backend.SetCleanupCallback([&](const std::string &, uint64_t /*gen*/) { ++cleanup_calls; });

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.6:8080", {"mem"}));
    for (int i = 0; i < 80 && cleanup_calls.load() == 0; ++i) {
        std::this_thread::sleep_for(20ms);
    }
    ASSERT_GE(cleanup_calls.load(), 1);

    // Node must have been erased from the table after cleanup.
    EXPECT_EQ(backend.nodes_.count("10.0.0.6:8080"), 0u);

    // Re-register: creates a fresh NodeInfo (new entry branch, not merge).
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.6:8080", {"mem", "disk"}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.6:8080"));
    {
        auto it = backend.nodes_.find("10.0.0.6:8080");
        ASSERT_NE(it, backend.nodes_.end());
        EXPECT_EQ(it->second->mediums.size(), 2u);
    }

    ASSERT_EQ(EC_OK, backend.Close());
}

// (8) Simulate EVENT_HOST_DOWN path: SetNodeUnavailable + UnregisterNode
//     removes the node from nodes_ immediately; cleanup callback is NOT
//     triggered by the LivenessCheckerLoop afterwards.
TEST_F(VineyardBackendTest, HostDownRemovesNodeFromTable) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 200, /*grace*/ 400, /*tick*/ 50), "trace"));

    std::atomic<int> cleanup_calls{0};
    backend.SetCleanupCallback([&](const std::string &, uint64_t /*gen*/) { ++cleanup_calls; });

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.7:8080", {"mem"}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.7:8080"));

    // Simulate EVENT_HOST_DOWN: mark unavailable then unregister.
    backend.SetNodeUnavailable("10.0.0.7:8080");
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.7:8080"));
    ASSERT_EQ(EC_OK, backend.UnregisterNode("10.0.0.7:8080"));

    // Node must be gone.
    EXPECT_EQ(backend.nodes_.count("10.0.0.7:8080"), 0u);

    // Wait enough time to confirm LivenessCheckerLoop does NOT fire cleanup
    // for the already-removed node.
    std::this_thread::sleep_for(500ms);
    EXPECT_EQ(cleanup_calls.load(), 0);

    ASSERT_EQ(EC_OK, backend.Close());
}

// (9) Generation counter: re-registration bumps generation so stale cleanup
//     tasks can detect the race and bail out.
TEST_F(VineyardBackendTest, GenerationBumpsOnReRegistration) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 200, /*grace*/ 5000, /*tick*/ 50), "trace"));

    const std::string host = "10.0.0.8:8080";
    ASSERT_EQ(0u, backend.GetNodeGeneration(host));

    ASSERT_EQ(EC_OK, backend.RegisterNode(host, {"mem"}));
    ASSERT_EQ(1u, backend.GetNodeGeneration(host));

    // Simulate HOST_DOWN: unregister does NOT reset generation.
    backend.SetNodeUnavailable(host);
    ASSERT_EQ(EC_OK, backend.UnregisterNode(host));
    ASSERT_EQ(1u, backend.GetNodeGeneration(host));

    // Re-registration bumps the generation.
    ASSERT_EQ(EC_OK, backend.RegisterNode(host, {"mem", "disk"}));
    ASSERT_EQ(2u, backend.GetNodeGeneration(host));

    // Idempotent re-registration also bumps.
    ASSERT_EQ(EC_OK, backend.RegisterNode(host, {"ssd"}));
    ASSERT_EQ(3u, backend.GetNodeGeneration(host));

    ASSERT_EQ(EC_OK, backend.Close());
}

// (10) LivenessCheckerLoop passes the correct generation to the callback so
//      that CacheManager can fence stale cleanups.
TEST_F(VineyardBackendTest, LivenessLoopPassesGenerationToCallback) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 80, /*grace*/ 120, /*tick*/ 20), "trace"));

    std::atomic<uint64_t> received_gen{0};
    backend.SetCleanupCallback([&](const std::string &, uint64_t gen) { received_gen.store(gen); });

    const std::string host = "10.0.0.9:8080";
    ASSERT_EQ(EC_OK, backend.RegisterNode(host, {"mem"}));
    uint64_t expected_gen = backend.GetNodeGeneration(host);

    // Wait for cleanup to fire.
    for (int i = 0; i < 80 && received_gen.load() == 0; ++i) {
        std::this_thread::sleep_for(20ms);
    }
    EXPECT_EQ(received_gen.load(), expected_gen);

    ASSERT_EQ(EC_OK, backend.Close());
}
