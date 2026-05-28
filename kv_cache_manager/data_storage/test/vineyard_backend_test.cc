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
    ASSERT_EQ(EC_OK, backend.Close());
}

// (2) RegisterNode / UnregisterNode
TEST_F(VineyardBackendTest, RegisterNodeWithMediums) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(), "trace"));

    ASSERT_EQ(EC_BADARGS, backend.RegisterNode("", {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.1:8080", {"mem", "disk"}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.1:8080"));

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.1:8080", {"disk", "ssd"}));
    {
        auto it = backend.nodes_.find("10.0.0.1:8080");
        ASSERT_NE(it, backend.nodes_.end());
        ASSERT_EQ(it->second->mediums.size(), 3u); // mem + disk + ssd
    }

    backend.SetNodeUnavailable("10.0.0.1:8080");
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.1:8080"));
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.1:8080", {"mem"}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.1:8080"));

    ASSERT_EQ(EC_OK, backend.UnregisterNode("10.0.0.1:8080"));
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.1:8080"));
    ASSERT_EQ(EC_NOENT, backend.UnregisterNode("10.0.0.1:8080"));
    ASSERT_EQ(EC_OK, backend.Close());
}

// (3) OnHeartbeat
TEST_F(VineyardBackendTest, OnHeartbeatRefreshesAndRevivesNode) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 200, /*grace*/ 5000, /*tick*/ 50), "trace"));
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.3:8080", {"mem"}));

    int64_t initial_hb = 0;
    {
        auto it = backend.nodes_.find("10.0.0.3:8080");
        ASSERT_NE(it, backend.nodes_.end());
        initial_hb = it->second->last_heartbeat_ms.load();
        ASSERT_GT(initial_hb, 0);
    }

    std::this_thread::sleep_for(20ms);
    ASSERT_EQ(EC_OK, backend.OnHeartbeat("10.0.0.3:8080", {{"version", "v6d-0.18"}}));
    {
        auto it = backend.nodes_.find("10.0.0.3:8080");
        ASSERT_GT(it->second->last_heartbeat_ms.load(), initial_hb);
        ASSERT_EQ(it->second->last_system_status.at("version"), "v6d-0.18");
    }

    backend.SetNodeUnavailable("10.0.0.3:8080");
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.3:8080"));
    ASSERT_EQ(EC_OK, backend.OnHeartbeat("10.0.0.3:8080", {}));
    {
        auto it = backend.nodes_.find("10.0.0.3:8080");
        ASSERT_TRUE(it->second->available.load());
        ASSERT_EQ(it->second->unavailable_since_ms.load(), 0);
    }

    ASSERT_EQ(EC_NODE_NOT_REGISTERED, backend.OnHeartbeat("99.99.99.99:8080", {{"x", "y"}}));
    ASSERT_EQ(backend.nodes_.count("99.99.99.99:8080"), 0u);

    ASSERT_EQ(EC_OK, backend.Close());
}

// (5) LivenessCheckerLoop: healthy -> unavailable -> dead
TEST_F(VineyardBackendTest, LivenessLoopHealthyToUnavailableToCleanup) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 100, /*grace*/ 200, /*tick*/ 20), "trace"));

    std::atomic<int> cleanup_calls{0};
    std::string cleanup_host;
    backend.SetCleanupCallback([&](const std::string &host, uint64_t /*gen*/) {
        ++cleanup_calls;
        cleanup_host = host;
    });

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.4:8080", {"mem"}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.4:8080"));

    std::this_thread::sleep_for(160ms);
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.4:8080"));
    EXPECT_EQ(cleanup_calls.load(), 0);

    for (int i = 0; i < 50 && cleanup_calls.load() == 0; ++i) {
        std::this_thread::sleep_for(20ms);
    }
    EXPECT_GE(cleanup_calls.load(), 1);
    EXPECT_EQ(cleanup_host, "10.0.0.4:8080");

    EXPECT_EQ(backend.nodes_.count("10.0.0.4:8080"), 0u);

    ASSERT_EQ(EC_OK, backend.Close());
}

// (6) Grace-period recovery
TEST_F(VineyardBackendTest, HeartbeatWithinGraceWindowRecovers) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 80, /*grace*/ 5000, /*tick*/ 20), "trace"));

    std::atomic<int> cleanup_calls{0};
    backend.SetCleanupCallback([&](const std::string &, uint64_t /*gen*/) { ++cleanup_calls; });

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.5:8080", {"mem"}));
    std::this_thread::sleep_for(140ms);
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.5:8080"));

    ASSERT_EQ(EC_OK, backend.OnHeartbeat("10.0.0.5:8080", {}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.5:8080"));

    std::this_thread::sleep_for(60ms);
    EXPECT_EQ(cleanup_calls.load(), 0);

    ASSERT_EQ(EC_OK, backend.Close());
}

// (7) Re-registration after cleanup
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

    EXPECT_EQ(backend.nodes_.count("10.0.0.6:8080"), 0u);

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.6:8080", {"mem", "disk"}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.6:8080"));
    {
        auto it = backend.nodes_.find("10.0.0.6:8080");
        ASSERT_NE(it, backend.nodes_.end());
        EXPECT_EQ(it->second->mediums.size(), 2u);
    }

    ASSERT_EQ(EC_OK, backend.Close());
}

// (8) EVENT_HOST_DOWN: immediate removal, no cleanup callback
TEST_F(VineyardBackendTest, HostDownRemovesNodeFromTable) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 200, /*grace*/ 400, /*tick*/ 50), "trace"));

    std::atomic<int> cleanup_calls{0};
    backend.SetCleanupCallback([&](const std::string &, uint64_t /*gen*/) { ++cleanup_calls; });

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.7:8080", {"mem"}));
    ASSERT_TRUE(backend.IsNodeAvailable("10.0.0.7:8080"));

    backend.SetNodeUnavailable("10.0.0.7:8080");
    ASSERT_FALSE(backend.IsNodeAvailable("10.0.0.7:8080"));
    ASSERT_EQ(EC_OK, backend.UnregisterNode("10.0.0.7:8080"));

    EXPECT_EQ(backend.nodes_.count("10.0.0.7:8080"), 0u);

    std::this_thread::sleep_for(500ms);
    EXPECT_EQ(cleanup_calls.load(), 0);

    ASSERT_EQ(EC_OK, backend.Close());
}

// (9) Generation counter fences stale cleanup
TEST_F(VineyardBackendTest, GenerationBumpsOnReRegistration) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 200, /*grace*/ 5000, /*tick*/ 50), "trace"));

    const std::string host = "10.0.0.8:8080";
    ASSERT_EQ(0u, backend.GetNodeGeneration(host));

    ASSERT_EQ(EC_OK, backend.RegisterNode(host, {"mem"}));
    ASSERT_EQ(1u, backend.GetNodeGeneration(host));

    backend.SetNodeUnavailable(host);
    ASSERT_EQ(EC_OK, backend.UnregisterNode(host));
    ASSERT_EQ(1u, backend.GetNodeGeneration(host));

    ASSERT_EQ(EC_OK, backend.RegisterNode(host, {"mem", "disk"}));
    ASSERT_EQ(2u, backend.GetNodeGeneration(host));

    ASSERT_EQ(EC_OK, backend.RegisterNode(host, {"ssd"}));
    ASSERT_EQ(3u, backend.GetNodeGeneration(host));

    ASSERT_EQ(EC_OK, backend.Close());
}

// (10) Cleanup callback receives correct generation
TEST_F(VineyardBackendTest, LivenessLoopPassesGenerationToCallback) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 80, /*grace*/ 120, /*tick*/ 20), "trace"));

    std::atomic<uint64_t> received_gen{0};
    backend.SetCleanupCallback([&](const std::string &, uint64_t gen) { received_gen.store(gen); });

    const std::string host = "10.0.0.9:8080";
    ASSERT_EQ(EC_OK, backend.RegisterNode(host, {"mem"}));
    uint64_t expected_gen = backend.GetNodeGeneration(host);

    for (int i = 0; i < 80 && received_gen.load() == 0; ++i) {
        std::this_thread::sleep_for(20ms);
    }
    EXPECT_EQ(received_gen.load(), expected_gen);

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(VineyardBackendTest, OnHeartbeatPublishesMetricsGauges) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 50), "trace"));
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.10:9600", {"mem"}));

    backend.OnHeartbeat("10.0.0.10:9600", {
        {"hit_rate", "0.85"},
        {"active_leases", "5"},
        {"non_numeric_field", "BOTH_OK"},
    });

    auto hit_rate_data = metrics_registry_->GetMetricsData("v6d.hit_rate");
    ASSERT_NE(hit_rate_data, nullptr);
    MetricsTags expected_tags = {{"instance_id", "v6d_cluster_test"}, {"host", "10.0.0.10:9600"}};
    auto gauge = hit_rate_data->GetOrCreateGauge(expected_tags);
    ASSERT_DOUBLE_EQ(0.85, gauge.Get());

    auto leases_data = metrics_registry_->GetMetricsData("v6d.active_leases");
    ASSERT_NE(leases_data, nullptr);
    auto leases_gauge = leases_data->GetOrCreateGauge(expected_tags);
    ASSERT_DOUBLE_EQ(5.0, leases_gauge.Get());

    auto non_numeric = metrics_registry_->GetMetricsData("v6d.non_numeric_field");
    ASSERT_EQ(non_numeric, nullptr);

    backend.OnHeartbeat("10.0.0.10:9600", {
        {"hit_rate", "0.90"},
        {"brand_new_metric", "42"},
    });

    auto new_data = metrics_registry_->GetMetricsData("v6d.brand_new_metric");
    ASSERT_NE(new_data, nullptr);
    auto new_gauge = new_data->GetOrCreateGauge(expected_tags);
    ASSERT_DOUBLE_EQ(42.0, new_gauge.Get());
    ASSERT_DOUBLE_EQ(0.90, gauge.Get());

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(VineyardBackendTest, SetNodeUnavailableZerosGauges) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 50), "trace"));

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.30:9600", {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.31:9600", {"mem"}));

    backend.OnHeartbeat("10.0.0.30:9600", {{"hit_rate", "0.90"}, {"mem_used", "8192"}});
    backend.OnHeartbeat("10.0.0.31:9600", {{"hit_rate", "0.80"}, {"mem_used", "4096"}});

    MetricsTags tags_30 = {{"instance_id", "v6d_cluster_test"}, {"host", "10.0.0.30:9600"}};
    MetricsTags tags_31 = {{"instance_id", "v6d_cluster_test"}, {"host", "10.0.0.31:9600"}};

    auto hr_data = metrics_registry_->GetMetricsData("v6d.hit_rate");
    ASSERT_NE(hr_data, nullptr);
    ASSERT_DOUBLE_EQ(0.90, hr_data->GetOrCreateGauge(tags_30).Get());
    ASSERT_DOUBLE_EQ(0.80, hr_data->GetOrCreateGauge(tags_31).Get());

    backend.SetNodeUnavailable("10.0.0.30:9600");

    ASSERT_DOUBLE_EQ(0.0, hr_data->GetOrCreateGauge(tags_30).Get());
    auto mu_data = metrics_registry_->GetMetricsData("v6d.mem_used");
    ASSERT_DOUBLE_EQ(0.0, mu_data->GetOrCreateGauge(tags_30).Get());

    ASSERT_DOUBLE_EQ(0.80, hr_data->GetOrCreateGauge(tags_31).Get());
    ASSERT_DOUBLE_EQ(4096, mu_data->GetOrCreateGauge(tags_31).Get());

    backend.SetNodeUnavailable("10.0.0.30:9600");
    ASSERT_DOUBLE_EQ(0.0, hr_data->GetOrCreateGauge(tags_30).Get());

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(VineyardBackendTest, UnregisterNodeCleansUpGauges) {
    VineyardBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 50), "trace"));

    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.20:9600", {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode("10.0.0.21:9600", {"mem"}));

    backend.OnHeartbeat("10.0.0.20:9600", {{"hit_rate", "0.75"}, {"mem_used", "4096"}});
    backend.OnHeartbeat("10.0.0.21:9600", {{"hit_rate", "0.60"}, {"mem_used", "2048"}});

    MetricsTags tags_20 = {{"instance_id", "v6d_cluster_test"}, {"host", "10.0.0.20:9600"}};
    MetricsTags tags_21 = {{"instance_id", "v6d_cluster_test"}, {"host", "10.0.0.21:9600"}};

    auto hr_data = metrics_registry_->GetMetricsData("v6d.hit_rate");
    ASSERT_NE(hr_data, nullptr);
    ASSERT_DOUBLE_EQ(0.75, hr_data->GetOrCreateGauge(tags_20).Get());
    ASSERT_DOUBLE_EQ(0.60, hr_data->GetOrCreateGauge(tags_21).Get());

    ASSERT_EQ(EC_OK, backend.UnregisterNode("10.0.0.20:9600"));

    auto hr_values = hr_data->GetMetricsValues();
    for (const auto &[tags, val] : hr_values) {
        ASSERT_NE(tags, tags_20) << "node 20 gauge should have been removed";
    }
    auto mu_data = metrics_registry_->GetMetricsData("v6d.mem_used");
    auto mu_values = mu_data->GetMetricsValues();
    for (const auto &[tags, val] : mu_values) {
        ASSERT_NE(tags, tags_20) << "node 20 gauge should have been removed";
    }

    ASSERT_DOUBLE_EQ(0.60, hr_data->GetOrCreateGauge(tags_21).Get());
    ASSERT_DOUBLE_EQ(2048, mu_data->GetOrCreateGauge(tags_21).Get());

    ASSERT_EQ(EC_OK, backend.Close());
}
