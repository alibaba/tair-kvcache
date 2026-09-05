#include <atomic>
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/event_report_backend.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;
using namespace std::chrono_literals;

class EventReportBackendTest : public TESTBASE {
public:
    void SetUp() override { metrics_registry_ = std::make_shared<MetricsRegistry>(); }

    static StorageConfig
    MakeConfig(int64_t hb_timeout_ms = 200,
               int64_t cleanup_grace_ms = 400,
               int64_t check_interval_ms = 50,
               DataStorageType type = DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
               int64_t snapshot_min_interval_ms = EventReportStorageSpec::kDefaultSnapshotMinIntervalMs) {
        auto spec = std::make_shared<EventReportStorageSpec>();
        spec->set_heartbeat_timeout_ms(hb_timeout_ms);
        spec->set_cleanup_grace_ms(cleanup_grace_ms);
        spec->set_liveness_check_interval_ms(check_interval_ms);
        spec->set_snapshot_min_interval_ms(snapshot_min_interval_ms);
        return StorageConfig(type, "event_report_test_group", spec);
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

ErrorCode BeginSnapshotForRegisteredReporter(EventReportBackend &backend,
                                             const ReporterSnapshotKey &reporter_key,
                                             std::string &out_candidate_version,
                                             uint64_t &out_retry_after_ms) {
    if (!reporter_key.instance_id.empty() && !reporter_key.host_ip_port.empty() &&
        !backend.IsNodeRegistered(reporter_key.instance_id, reporter_key.host_ip_port)) {
        const ErrorCode register_ec =
            backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"});
        if (register_ec != EC_OK) {
            return register_ec;
        }
    }
    return backend.BeginSnapshot(reporter_key, out_candidate_version, out_retry_after_ms);
}

MetricsTags LockMetricTags(const std::string &lock_name, const std::string &operation) {
    return {{"lock_name", lock_name},
            {"operation", operation},
            {"type", "event_report_l1p5"},
            {"unique_name", "event_report_test_group"}};
}

Counter GetLockMetric(const std::shared_ptr<MetricsRegistry> &registry,
                      const std::string &metric_name,
                      const std::string &lock_name,
                      const std::string &operation) {
    return registry->GetCounter(metric_name, LockMetricTags(lock_name, operation));
}

// (1) GetType / Available / Create-Delete EC_UNIMPLEMENTED / GetStorageUsageRatio=1.0
TEST_F(EventReportBackendTest, BasicAccessors) {
    EventReportBackend backend(metrics_registry_);
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

    // After Open(), GetType() returns the configured type
    auto config = MakeConfig();
    std::dynamic_pointer_cast<EventReportStorageSpec>(config.storage_spec())->set_snapshot_delta_drain_timeout_ms(8765);
    ASSERT_EQ(EC_OK, backend.Open(config, "trace"));
    ASSERT_EQ(backend.GetType(), DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5);
    EXPECT_EQ(8765, backend.snapshot_delta_drain_timeout_ms_);
    ASSERT_TRUE(backend.Available());
    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(EventReportBackendTest, BuildLocationIdIncludesEventReportType) {
    EventReportBackend l1p5_backend(metrics_registry_);
    ASSERT_EQ(EC_OK, l1p5_backend.Open(MakeConfig(), "trace"));
    const std::string l1p5_location = l1p5_backend.BuildLocationId("mem", "10.0.0.1:8080");
    EXPECT_EQ("kvs#event_report_l1p5#mem#10.0.0.1:8080", l1p5_location);

    EventReportBackend l2_backend(metrics_registry_);
    ASSERT_EQ(EC_OK,
              l2_backend.Open(MakeConfig(200, 400, 50, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2), "trace"));
    const std::string l2_location = l2_backend.BuildLocationId("mem", "10.0.0.1:8080");
    EXPECT_EQ("kvs#event_report_l2#mem#10.0.0.1:8080", l2_location);

    std::string medium;
    std::string host;
    EXPECT_TRUE(l1p5_backend.ParseLocationId(l1p5_location, medium, host));
    EXPECT_EQ("mem", medium);
    EXPECT_EQ("10.0.0.1:8080", host);
    EXPECT_FALSE(l1p5_backend.ParseLocationId(l2_location, medium, host));
    EXPECT_TRUE(l2_backend.ParseLocationId(l2_location, medium, host));
    EXPECT_FALSE(l2_backend.ParseLocationId(l1p5_location, medium, host));
    EXPECT_TRUE(l1p5_backend.BuildLocationId("mem#bad", "10.0.0.1:8080").empty());
    EXPECT_TRUE(l1p5_backend.BuildLocationId("mem", "10.0.0.1#8080").empty());
    EXPECT_FALSE(SnapshotUriUtils::IsValidLocationIdComponent(""));
    EXPECT_FALSE(SnapshotUriUtils::IsValidLocationIdComponent("mem#bad"));
    EXPECT_TRUE(SnapshotUriUtils::IsValidLocationIdComponent("hbm-cache"));
}

TEST_F(EventReportBackendTest, OpenWithWrongSpecTypeFails) {
    EventReportBackend backend(metrics_registry_);
    auto spec = std::make_shared<NfsStorageSpec>();
    spec->set_root_path("/tmp");
    StorageConfig cfg(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5, "event_report_test", spec);
    ASSERT_NE(EC_OK, backend.Open(cfg, "trace"));
    ASSERT_FALSE(backend.Available());
}

TEST_F(EventReportBackendTest, OpenStartsLivenessLoopAndCloseStops) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(), "trace"));
    ASSERT_TRUE(backend.Available());
    ASSERT_TRUE(backend.liveness_checker_running_.load());
    ASSERT_TRUE(backend.liveness_checker_thread_.joinable());

    DataStorageBackend &backend_base = backend;
    backend_base.SetAvailable(false);
    ASSERT_FALSE(backend.Available());
    backend_base.SetAvailable(true);
    ASSERT_TRUE(backend.Available());
    EXPECT_NE(EC_OK, backend.Open(MakeConfig(), "duplicate_open"));
    EXPECT_TRUE(backend.Available());

    ASSERT_EQ(EC_OK, backend.Close());
    ASSERT_FALSE(backend.Available());
    EXPECT_NE(EC_OK, backend.Open(MakeConfig(), "reopen_retired_backend"));
    EXPECT_FALSE(backend.Available());
    ASSERT_FALSE(backend.liveness_checker_running_.load());
    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(EventReportBackendTest, CloseInterruptsLongLivenessWait) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 60000), "trace"));

    const auto close_begin = std::chrono::steady_clock::now();
    ASSERT_EQ(EC_OK, backend.Close());
    const auto close_elapsed = std::chrono::steady_clock::now() - close_begin;
    EXPECT_LT(close_elapsed, 2s);
}

TEST_F(EventReportBackendTest, RequestedLockPathsRecordWaitHoldAndAcquireCount) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 60000), "trace"));
    const ReporterSnapshotKey reporter_key{"test_inst", "10.0.0.1:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));

    auto acquire = [&](const std::string &lock_name, const std::string &operation) {
        return GetLockMetric(metrics_registry_, "event.lock_acquire_counter", lock_name, operation);
    };
    const auto begin_before = acquire("nodes_mutex", "begin").Get();
    const auto end_before = acquire("nodes_mutex", "end").Get();
    const auto snapshot_get_before = acquire("nodes_mutex", "snapshot_get").Get();
    const auto ensure_node_before = acquire("nodes_mutex", "ensure_node").Get();
    const auto lifecycle_before = acquire("lifecycle_fences_mutex", "access").Get();

    ASSERT_EQ(EC_OK, backend.EnsureNodeRegistered(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));
    EXPECT_TRUE(backend.GetSnapshotVersion(reporter_key).empty());
    std::string committed_version;
    uint64_t lifecycle_generation = 0;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, committed_version, &lifecycle_generation));
    backend.EndDeltaMutation(reporter_key, lifecycle_generation, committed_version);
    EventReportBackend::LifecycleMutationLease lifecycle_lease;
    ASSERT_EQ(EC_OK, backend.AcquireLifecycleMutationLease(reporter_key, lifecycle_generation, lifecycle_lease));
    lifecycle_lease.reset();

    EXPECT_EQ(begin_before + 1, acquire("nodes_mutex", "begin").Get());
    EXPECT_EQ(end_before + 1, acquire("nodes_mutex", "end").Get());
    EXPECT_EQ(snapshot_get_before + 1, acquire("nodes_mutex", "snapshot_get").Get());
    EXPECT_EQ(ensure_node_before + 1, acquire("nodes_mutex", "ensure_node").Get());
    EXPECT_EQ(lifecycle_before + 1, acquire("lifecycle_fences_mutex", "access").Get());

    for (const auto &[lock_name, operation] :
         std::vector<std::pair<std::string, std::string>>{{"nodes_mutex", "begin"},
                                                          {"nodes_mutex", "end"},
                                                          {"nodes_mutex", "snapshot_get"},
                                                          {"nodes_mutex", "ensure_node"},
                                                          {"lifecycle_fences_mutex", "access"}}) {
        const auto wait = GetLockMetric(metrics_registry_, "event.lock_wait_time_us_sum", lock_name, operation);
        const auto hold = GetLockMetric(metrics_registry_, "event.lock_hold_time_us_sum", lock_name, operation);
        ASSERT_NE(nullptr, wait.GetRaw());
        ASSERT_NE(nullptr, hold.GetRaw());
        EXPECT_TRUE(wait.GetRaw()->touched.load(std::memory_order_relaxed));
        EXPECT_TRUE(hold.GetRaw()->touched.load(std::memory_order_relaxed));
    }
}

TEST_F(EventReportBackendTest, ContendedGlobalLocksAccumulateWaitTime) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 60000), "trace"));
    const ReporterSnapshotKey reporter_key{"test_inst", "10.0.0.1:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));
    const uint64_t lifecycle_generation =
        backend.GetNodeGeneration(reporter_key.instance_id, reporter_key.host_ip_port);

    auto nodes_wait = GetLockMetric(metrics_registry_, "event.lock_wait_time_us_sum", "nodes_mutex", "snapshot_get");
    const auto nodes_wait_before = nodes_wait.Get();
    std::unique_lock<std::shared_mutex> nodes_guard(backend.nodes_mutex_);
    std::promise<void> nodes_started;
    auto nodes_started_future = nodes_started.get_future();
    auto snapshot_future = std::async(std::launch::async, [&] {
        nodes_started.set_value();
        return backend.GetSnapshotVersion(reporter_key);
    });
    nodes_started_future.wait();
    std::this_thread::sleep_for(50ms);
    nodes_guard.unlock();
    EXPECT_TRUE(snapshot_future.get().empty());
    EXPECT_GE(nodes_wait.Get() - nodes_wait_before, 20000u);

    auto lifecycle_wait =
        GetLockMetric(metrics_registry_, "event.lock_wait_time_us_sum", "lifecycle_fences_mutex", "access");
    const auto lifecycle_wait_before = lifecycle_wait.Get();
    std::unique_lock<std::mutex> lifecycle_guard(backend.lifecycle_fences_mutex_);
    std::promise<void> lifecycle_started;
    auto lifecycle_started_future = lifecycle_started.get_future();
    auto lease_future = std::async(std::launch::async, [&] {
        lifecycle_started.set_value();
        EventReportBackend::LifecycleMutationLease lease;
        return backend.AcquireLifecycleMutationLease(reporter_key, lifecycle_generation, lease);
    });
    lifecycle_started_future.wait();
    std::this_thread::sleep_for(50ms);
    lifecycle_guard.unlock();
    EXPECT_EQ(EC_OK, lease_future.get());
    EXPECT_GE(lifecycle_wait.Get() - lifecycle_wait_before, 20000u);
}

// (2) RegisterNode / UnregisterNode
TEST_F(EventReportBackendTest, RegisterNodeWithMediums) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(), "trace"));

    ASSERT_EQ(EC_BADARGS, backend.RegisterNode("test_inst", "", {"mem"}));
    ASSERT_EQ(EC_BADARGS, backend.RegisterNode("", "10.0.0.1:8080", {"mem"}));
    ASSERT_EQ(EC_BADARGS, backend.RegisterNode("test_inst", "10.0.0.1#8080", {"mem"}));
    ASSERT_EQ(EC_BADARGS, backend.RegisterNode("test_inst", "10.0.0.1:8080", {"mem#bad"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.1:8080", {"mem", "disk"}));
    ASSERT_TRUE(backend.IsNodeAvailable("test_inst", "10.0.0.1:8080"));

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.1:8080", {"disk", "ssd"}));
    {
        auto &host_map = backend.instance_nodes_["test_inst"];
        auto it = host_map.find("10.0.0.1:8080");
        ASSERT_NE(it, host_map.end());
        ASSERT_EQ(it->second->mediums.size(), 3u); // mem + disk + ssd
    }

    backend.SetNodeUnavailable("test_inst", "10.0.0.1:8080");
    ASSERT_FALSE(backend.IsNodeAvailable("test_inst", "10.0.0.1:8080"));
    ASSERT_EQ(EC_OK, backend.EnsureNodeRegistered("test_inst", "10.0.0.1:8080", {"hbm"}));
    ASSERT_FALSE(backend.IsNodeAvailable("test_inst", "10.0.0.1:8080"));
    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.1:8080", {"mem"}));
    ASSERT_TRUE(backend.IsNodeAvailable("test_inst", "10.0.0.1:8080"));

    ASSERT_EQ(EC_OK, backend.UnregisterNode("test_inst", "10.0.0.1:8080"));
    ASSERT_FALSE(backend.IsNodeAvailable("test_inst", "10.0.0.1:8080"));
    ASSERT_EQ(EC_NODE_NOT_REGISTERED, backend.EnsureNodeRegistered("test_inst", "10.0.0.1:8080", {"mem"}));
    ASSERT_EQ(EC_NODE_NOT_REGISTERED, backend.OnHeartbeat("test_inst", "10.0.0.1:8080", {}));
    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.1:8080", {"mem"}));
    ASSERT_TRUE(backend.IsNodeAvailable("test_inst", "10.0.0.1:8080"));
    ASSERT_EQ(EC_OK, backend.UnregisterNode("test_inst", "10.0.0.1:8080"));
    ASSERT_EQ(EC_NOENT, backend.UnregisterNode("test_inst", "10.0.0.1:8080"));
    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(EventReportBackendTest, MightExistTracksRegisteredNodeAvailability) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 50), "trace"));

    const std::string instance_id = "test_inst";
    const std::string available_host = "10.0.0.2:8080";
    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, available_host, {"mem"}));

    const ReporterSnapshotKey reporter_key{instance_id, available_host};
    std::string committed_token;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, committed_token, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, committed_token));

    std::string available_uri_string;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://10.0.0.2:8080/mem", committed_token, available_uri_string));
    const DataStorageUri available_uri(available_uri_string);

    std::string unknown_token_uri_string;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(
        "event_report://10.0.0.3:8080/mem", "ffffffffffffffffffffffffffffffff", unknown_token_uri_string));
    const DataStorageUri unknown_token_uri(unknown_token_uri_string);
    const DataStorageUri invalid_uri;

    auto result = backend.MightExist({available_uri, unknown_token_uri, invalid_uri});
    ASSERT_EQ(3u, result.size());
    EXPECT_TRUE(result[0]);
    EXPECT_FALSE(result[1]);
    EXPECT_FALSE(result[2]);

    backend.SetNodeUnavailable(instance_id, available_host);
    result = backend.MightExist({available_uri});
    ASSERT_EQ(1u, result.size());
    EXPECT_FALSE(result[0]);

    ASSERT_EQ(EC_OK, backend.OnHeartbeat(instance_id, available_host, {}));
    result = backend.MightExist({available_uri});
    ASSERT_EQ(1u, result.size());
    EXPECT_TRUE(result[0]);

    backend.SetAvailable(false);
    EXPECT_EQ((std::vector<bool>{false}), backend.MightExist({available_uri}));
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST, backend.EnsureNodeRegistered(instance_id, available_host, {"mem"}));
    backend.SetAvailable(true);
    EXPECT_EQ((std::vector<bool>{true}), backend.MightExist({available_uri}));

    ASSERT_EQ(EC_OK, backend.Close());
}

// (3) OnHeartbeat
TEST_F(EventReportBackendTest, OnHeartbeatRefreshesAndRevivesNode) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 200, /*grace*/ 5000, /*tick*/ 50), "trace"));
    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.3:8080", {"mem"}));
    const uint64_t registered_generation = backend.GetNodeGeneration("test_inst", "10.0.0.3:8080");

    int64_t initial_hb = 0;
    {
        auto &host_map = backend.instance_nodes_["test_inst"];
        auto it = host_map.find("10.0.0.3:8080");
        ASSERT_NE(it, host_map.end());
        initial_hb = it->second->last_heartbeat_ms.load();
        ASSERT_GT(initial_hb, 0);
    }

    std::this_thread::sleep_for(20ms);
    ASSERT_EQ(EC_OK, backend.OnHeartbeat("test_inst", "10.0.0.3:8080", {{"version", "er-0.18"}}));
    ASSERT_EQ(registered_generation, backend.GetNodeGeneration("test_inst", "10.0.0.3:8080"));
    {
        auto &host_map = backend.instance_nodes_["test_inst"];
        auto it = host_map.find("10.0.0.3:8080");
        ASSERT_GT(it->second->last_heartbeat_ms.load(), initial_hb);
        ASSERT_EQ(it->second->last_system_status.at("version"), "er-0.18");
    }

    backend.SetNodeUnavailable("test_inst", "10.0.0.3:8080");
    ASSERT_FALSE(backend.IsNodeAvailable("test_inst", "10.0.0.3:8080"));
    ASSERT_EQ(EC_OK, backend.OnHeartbeat("test_inst", "10.0.0.3:8080", {}));
    ASSERT_GT(backend.GetNodeGeneration("test_inst", "10.0.0.3:8080"), registered_generation);
    {
        auto &host_map = backend.instance_nodes_["test_inst"];
        auto it = host_map.find("10.0.0.3:8080");
        ASSERT_TRUE(it->second->available.load());
        ASSERT_EQ(it->second->unavailable_since_ms.load(), 0);
    }

    ASSERT_EQ(EC_OK, backend.OnHeartbeat("test_inst", "99.99.99.99:8080", {{"x", "y"}}));
    ASSERT_TRUE(backend.IsNodeRegistered("test_inst", "99.99.99.99:8080"));
    ASSERT_TRUE(backend.IsNodeAvailable("test_inst", "99.99.99.99:8080"));
    ASSERT_EQ(backend.instance_nodes_["test_inst"]["99.99.99.99:8080"]->last_system_status.at("x"), "y");

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(EventReportBackendTest, SteadyHeartbeatDoesNotRejectConcurrentLifecycleMutationLease) {
    EventReportBackend backend(metrics_registry_);
    const ReporterSnapshotKey reporter_key{"heartbeat-mutation-lease", "10.0.0.32:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));
    const uint64_t generation = backend.GetNodeGeneration(reporter_key.instance_id, reporter_key.host_ip_port);

    auto &node = *backend.instance_nodes_[reporter_key.instance_id][reporter_key.host_ip_port];
    node.last_heartbeat_ms.store(0, std::memory_order_relaxed);

    // Pin HEARTBEAT after it refreshes the timestamp but before it publishes
    // system status. At that point it still holds its lifecycle lease, so the
    // concurrent mutation below exercises the exact production overlap
    // without relying on scheduler timing or sleeps.
    std::unique_lock<std::mutex> status_gate(node.status_mutex);
    auto heartbeat = std::async(std::launch::async, [&] {
        return backend.OnHeartbeat(reporter_key.instance_id, reporter_key.host_ip_port, {{"load", "1"}});
    });

    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (node.last_heartbeat_ms.load(std::memory_order_acquire) == 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool heartbeat_reached_status_gate = node.last_heartbeat_ms.load(std::memory_order_acquire) != 0;
    if (!heartbeat_reached_status_gate) {
        status_gate.unlock();
        EXPECT_EQ(EC_OK, heartbeat.get());
        FAIL() << "heartbeat did not reach the status publication gate";
    }

    // A steady heartbeat is a lifecycle reader, not an unfenced operation:
    // lifecycle writers must still wait until its status publication ends.
    const auto lifecycle_fence = backend.GetOrCreateLifecycleFence(reporter_key);
    std::unique_lock<std::shared_mutex> lifecycle_writer(lifecycle_fence->mutex, std::try_to_lock);
    EXPECT_FALSE(lifecycle_writer.owns_lock());

    EventReportBackend::LifecycleMutationLease mutation_lease;
    const ErrorCode mutation_ec = backend.AcquireLifecycleMutationLease(reporter_key, generation, mutation_lease);

    mutation_lease.reset();
    status_gate.unlock();
    EXPECT_EQ(EC_OK, heartbeat.get());
    EXPECT_EQ(EC_OK, mutation_ec);
}

TEST_F(EventReportBackendTest, DataMutationsDoNotRefreshHeartbeat) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 50), "trace"));
    const ReporterSnapshotKey reporter_key{"test_inst", "10.0.0.31:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));

    int64_t registered_heartbeat_ms = 0;
    {
        std::shared_lock<std::shared_mutex> lock(backend.nodes_mutex_);
        const auto instance_it = backend.instance_nodes_.find(reporter_key.instance_id);
        ASSERT_NE(backend.instance_nodes_.end(), instance_it);
        const auto host_it = instance_it->second.find(reporter_key.host_ip_port);
        ASSERT_NE(instance_it->second.end(), host_it);
        registered_heartbeat_ms = host_it->second->last_heartbeat_ms.load();
    }

    std::string delta_version;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, delta_version));
    backend.EndDeltaMutation(reporter_key);

    std::string snapshot_version;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter_key, snapshot_version, retry_after_ms));
    backend.AbortSnapshotVersion(reporter_key, snapshot_version);

    {
        std::shared_lock<std::shared_mutex> lock(backend.nodes_mutex_);
        const auto &node = backend.instance_nodes_.at(reporter_key.instance_id).at(reporter_key.host_ip_port);
        EXPECT_EQ(registered_heartbeat_ms, node->last_heartbeat_ms.load());
    }
    ASSERT_EQ(EC_OK, backend.Close());
}

// (5) LivenessCheckerLoop: healthy -> unavailable -> dead
TEST_F(EventReportBackendTest, LivenessLoopHealthyToUnavailableToCleanup) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 100, /*grace*/ 200, /*tick*/ 20), "trace"));

    std::atomic<int> cleanup_calls{0};
    std::string cleanup_host;
    backend.SetCleanupCallback([&](const std::string & /*instance_id*/, const std::string &host, uint64_t /*gen*/) {
        cleanup_host = host;
        cleanup_calls.fetch_add(1, std::memory_order_release);
    });

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.4:8080", {"mem"}));
    ASSERT_TRUE(backend.IsNodeAvailable("test_inst", "10.0.0.4:8080"));

    const auto unavailable_deadline = std::chrono::steady_clock::now() + 1s;
    while (std::chrono::steady_clock::now() < unavailable_deadline &&
           backend.IsNodeAvailable("test_inst", "10.0.0.4:8080")) {
        std::this_thread::sleep_for(10ms);
    }
    ASSERT_FALSE(backend.IsNodeAvailable("test_inst", "10.0.0.4:8080"));
    EXPECT_EQ(cleanup_calls.load(), 0);

    const auto cleanup_deadline = std::chrono::steady_clock::now() + 1s;
    while ((backend.IsNodeRegistered("test_inst", "10.0.0.4:8080") ||
            cleanup_calls.load(std::memory_order_acquire) == 0) &&
           std::chrono::steady_clock::now() < cleanup_deadline) {
        std::this_thread::yield();
    }
    ASSERT_FALSE(backend.IsNodeRegistered("test_inst", "10.0.0.4:8080"));
    EXPECT_GE(cleanup_calls.load(std::memory_order_acquire), 1);
    EXPECT_EQ(cleanup_host, "10.0.0.4:8080");

    ASSERT_EQ(EC_OK, backend.Close());
}

// (6) Grace-period recovery
TEST_F(EventReportBackendTest, HeartbeatWithinGraceWindowRecovers) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 80, /*grace*/ 5000, /*tick*/ 20), "trace"));

    std::atomic<int> cleanup_calls{0};
    backend.SetCleanupCallback([&](const std::string &, const std::string &, uint64_t /*gen*/) { ++cleanup_calls; });

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.5:8080", {"mem"}));
    std::this_thread::sleep_for(140ms);
    ASSERT_FALSE(backend.IsNodeAvailable("test_inst", "10.0.0.5:8080"));

    ASSERT_EQ(EC_OK, backend.OnHeartbeat("test_inst", "10.0.0.5:8080", {}));
    ASSERT_TRUE(backend.IsNodeAvailable("test_inst", "10.0.0.5:8080"));

    std::this_thread::sleep_for(60ms);
    EXPECT_EQ(cleanup_calls.load(), 0);

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(EventReportBackendTest, LivenessUnregistersBeforeCleanupAndHeartbeatCannotReviveOldSnapshot) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 100, /*grace*/ 50, /*tick*/ 5), "trace"));
    backend.SetSnapshotMinIntervalMsForTest(0);

    std::promise<void> cleanup_entered;
    std::promise<void> release_cleanup;
    std::promise<void> cleanup_returned;
    auto release_future = release_cleanup.get_future().share();
    std::atomic<bool> first_cleanup{true};
    backend.SetCleanupCallback([&](const std::string &, const std::string &, uint64_t) {
        if (first_cleanup.exchange(false)) {
            cleanup_entered.set_value();
            release_future.wait();
            cleanup_returned.set_value();
        }
    });

    const std::string instance_id = "test_inst";
    const std::string host = "10.0.0.50:8080";
    const ReporterSnapshotKey reporter_key{instance_id, host};
    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"mem"}));
    const uint64_t initial_generation = backend.GetNodeGeneration(instance_id, host);

    std::string token;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, token, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, token));
    std::string uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri("event_report://physical-cache:9600/mem", token, uri));
    ASSERT_EQ((std::vector<bool>{true}), backend.MightExist({DataStorageUri(uri)}));

    auto cleanup_entered_future = cleanup_entered.get_future();
    ASSERT_EQ(std::future_status::ready, cleanup_entered_future.wait_for(2s));
    EXPECT_FALSE(backend.IsNodeAvailable(instance_id, host));
    EXPECT_EQ((std::vector<bool>{false}), backend.MightExist({DataStorageUri(uri)}));

    // Expiry is linearized before the callback can delete metadata. A heartbeat
    // arriving while cleanup is running must observe the tombstone instead of
    // reviving a committed version whose metadata may already be gone.
    EXPECT_FALSE(backend.IsNodeRegistered(instance_id, host));
    EXPECT_EQ(initial_generation, backend.GetNodeGeneration(instance_id, host));
    EXPECT_TRUE(backend.GetSnapshotVersion(reporter_key).empty());
    EXPECT_EQ(EC_NODE_NOT_REGISTERED, backend.OnHeartbeat(instance_id, host, {}));

    release_cleanup.set_value();
    ASSERT_EQ(std::future_status::ready, cleanup_returned.get_future().wait_for(1s));
    EXPECT_FALSE(backend.IsNodeRegistered(instance_id, host));
    EXPECT_EQ((std::vector<bool>{false}), backend.MightExist({DataStorageUri(uri)}));

    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"mem"}));
    EXPECT_GT(backend.GetNodeGeneration(instance_id, host), initial_generation);
    EXPECT_TRUE(backend.GetSnapshotVersion(reporter_key).empty());

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(EventReportBackendTest, ConditionalUnregisterCannotRemoveNewGeneration) {
    EventReportBackend backend(metrics_registry_);
    const std::string instance_id = "generation-race";
    const std::string host = "10.0.0.90:8080";
    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"mem"}));
    const uint64_t stale_generation = backend.GetNodeGeneration(instance_id, host);

    // Model the exact liveness window: it selected generation N, then an
    // explicit REGISTER recovered the same host before final unregister.
    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"disk"}));
    ASSERT_GT(backend.GetNodeGeneration(instance_id, host), stale_generation);
    EXPECT_EQ(EC_MISMATCH, backend.UnregisterNodeIfGeneration(instance_id, host, stale_generation));
    EXPECT_TRUE(backend.IsNodeRegistered(instance_id, host));

    const uint64_t current_generation = backend.GetNodeGeneration(instance_id, host);
    EXPECT_EQ(EC_OK, backend.UnregisterNodeIfGeneration(instance_id, host, current_generation));
    EXPECT_FALSE(backend.IsNodeRegistered(instance_id, host));
}

TEST_F(EventReportBackendTest, HostDownAtomicallyCapturesGenerationAndLeavesReregisteredNodeIntact) {
    EventReportBackend backend(metrics_registry_);
    const std::string instance_id = "host-down-generation";
    const std::string host = "10.0.0.91:8080";
    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"mem"}));
    const uint64_t original_generation = backend.GetNodeGeneration(instance_id, host);

    uint64_t host_down_generation = 0;
    ASSERT_EQ(EC_OK, backend.UnregisterNodeForHostDown(instance_id, host, host_down_generation));
    EXPECT_EQ(original_generation, host_down_generation);
    EXPECT_FALSE(backend.IsNodeRegistered(instance_id, host));
    uint64_t repeated_host_down_generation = 0;
    EXPECT_EQ(EC_OK, backend.UnregisterNodeForHostDown(instance_id, host, repeated_host_down_generation));
    EXPECT_EQ(host_down_generation, repeated_host_down_generation);

    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"disk"}));
    const uint64_t restored_generation = backend.GetNodeGeneration(instance_id, host);
    EXPECT_GT(restored_generation, host_down_generation);
    EXPECT_EQ(EC_MISMATCH, backend.UnregisterNodeIfGeneration(instance_id, host, host_down_generation));
    EXPECT_TRUE(backend.IsNodeRegistered(instance_id, host));
}

TEST_F(EventReportBackendTest, CleanupLeaseFencesReregisterThroughFinalDeleteStage) {
    EventReportBackend backend(metrics_registry_);
    const ReporterSnapshotKey reporter_key{"cleanup-generation-race", "10.0.0.92:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));
    const uint64_t cleanup_generation = backend.GetNodeGeneration(reporter_key.instance_id, reporter_key.host_ip_port);

    EventReportBackend::LifecycleMutationLease cleanup_lease;
    EXPECT_EQ(EC_MISMATCH, backend.AcquireLifecycleCleanupLease(reporter_key, cleanup_generation, cleanup_lease));
    uint64_t unregistered_generation = 0;
    ASSERT_EQ(EC_OK,
              backend.UnregisterNodeForHostDown(
                  reporter_key.instance_id, reporter_key.host_ip_port, unregistered_generation));
    ASSERT_EQ(cleanup_generation, unregistered_generation);
    ASSERT_EQ(EC_OK, backend.AcquireLifecycleCleanupLease(reporter_key, cleanup_generation, cleanup_lease));
    auto reregister = std::async(std::launch::async, [&] {
        return backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"disk"});
    });
    EXPECT_EQ(std::future_status::timeout, reregister.wait_for(std::chrono::milliseconds(20)));

    cleanup_lease.reset();
    ASSERT_EQ(std::future_status::ready, reregister.wait_for(std::chrono::seconds(1)));
    ASSERT_EQ(EC_OK, reregister.get());
    EventReportBackend::LifecycleMutationLease stale_cleanup_lease;
    EXPECT_EQ(EC_MISMATCH, backend.AcquireLifecycleCleanupLease(reporter_key, cleanup_generation, stale_cleanup_lease));
}

TEST_F(EventReportBackendTest, LifecycleCleanupLeaseDoesNotBlockUnrelatedReporter) {
    EventReportBackend backend(metrics_registry_);
    const ReporterSnapshotKey reporter_a{"lifecycle-isolation", "10.0.0.93:8080"};
    const ReporterSnapshotKey reporter_b{"lifecycle-isolation", "10.0.0.94:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_a.instance_id, reporter_a.host_ip_port, {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_b.instance_id, reporter_b.host_ip_port, {"mem"}));

    const uint64_t generation_a = backend.GetNodeGeneration(reporter_a.instance_id, reporter_a.host_ip_port);
    uint64_t unregistered_generation_a = 0;
    ASSERT_EQ(
        EC_OK,
        backend.UnregisterNodeForHostDown(reporter_a.instance_id, reporter_a.host_ip_port, unregistered_generation_a));
    ASSERT_EQ(generation_a, unregistered_generation_a);
    EventReportBackend::LifecycleMutationLease cleanup_lease_a;
    ASSERT_EQ(EC_OK, backend.AcquireLifecycleCleanupLease(reporter_a, generation_a, cleanup_lease_a));

    auto register_b = std::async(std::launch::async, [&] {
        return backend.RegisterNode(reporter_b.instance_id, reporter_b.host_ip_port, {"disk"});
    });
    ASSERT_EQ(std::future_status::ready, register_b.wait_for(std::chrono::seconds(1)));
    EXPECT_EQ(EC_OK, register_b.get());

    const uint64_t generation_b = backend.GetNodeGeneration(reporter_b.instance_id, reporter_b.host_ip_port);
    EventReportBackend::LifecycleMutationLease mutation_lease_b;
    EXPECT_EQ(EC_OK, backend.AcquireLifecycleMutationLease(reporter_b, generation_b, mutation_lease_b));
}

TEST_F(EventReportBackendTest, EnsureNodeRegisteredMergesNewMediums) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.EnsureNodeRegistered("medium-merge", "10.0.0.91:8080", {"mem"}));
    const uint64_t generation = backend.GetNodeGeneration("medium-merge", "10.0.0.91:8080");
    ASSERT_EQ(EC_OK, backend.EnsureNodeRegistered("medium-merge", "10.0.0.91:8080", {"disk", "mem"}));
    EXPECT_EQ(generation, backend.GetNodeGeneration("medium-merge", "10.0.0.91:8080"));
    ASSERT_EQ(2u, backend.instance_nodes_["medium-merge"]["10.0.0.91:8080"]->mediums.size());
}

TEST_F(EventReportBackendTest, EnsureNodeRegisteredHandlesConcurrentKnownAndNewMediums) {
    EventReportBackend backend(metrics_registry_);
    const std::string instance_id = "medium-concurrency";
    const std::string host = "10.0.0.95:8080";
    ASSERT_EQ(EC_OK, backend.EnsureNodeRegistered(instance_id, host, {"mem"}));
    const uint64_t generation = backend.GetNodeGeneration(instance_id, host);

    std::atomic<size_t> failures{0};
    std::vector<std::thread> workers;
    for (size_t worker = 0; worker < 12; ++worker) {
        workers.emplace_back([&, worker] {
            const std::string medium = worker % 2 == 0 ? "mem" : "disk";
            for (size_t iteration = 0; iteration < 200; ++iteration) {
                if (backend.EnsureNodeRegistered(instance_id, host, {medium}) != EC_OK) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto &worker : workers) {
        worker.join();
    }

    EXPECT_EQ(0u, failures.load(std::memory_order_relaxed));
    EXPECT_EQ(generation, backend.GetNodeGeneration(instance_id, host));
    const auto &mediums = backend.instance_nodes_[instance_id][host]->mediums;
    EXPECT_EQ((std::set<std::string>{"disk", "mem"}), (std::set<std::string>(mediums.begin(), mediums.end())));
}

// (7) Re-registration after cleanup
TEST_F(EventReportBackendTest, RegisterAfterCleanupCreatesNewEntry) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 80, /*grace*/ 120, /*tick*/ 20), "trace"));

    std::atomic<int> cleanup_calls{0};
    backend.SetCleanupCallback([&](const std::string &, const std::string &, uint64_t /*gen*/) { ++cleanup_calls; });

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.6:8080", {"mem"}));
    for (int i = 0; i < 80 && cleanup_calls.load() == 0; ++i) {
        std::this_thread::sleep_for(20ms);
    }
    ASSERT_GE(cleanup_calls.load(), 1);

    const auto cleanup_deadline = std::chrono::steady_clock::now() + 1s;
    while (backend.IsNodeRegistered("test_inst", "10.0.0.6:8080") &&
           std::chrono::steady_clock::now() < cleanup_deadline) {
        std::this_thread::yield();
    }
    ASSERT_FALSE(backend.IsNodeRegistered("test_inst", "10.0.0.6:8080"));

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.6:8080", {"mem", "disk"}));
    ASSERT_TRUE(backend.IsNodeAvailable("test_inst", "10.0.0.6:8080"));
    {
        std::shared_lock<std::shared_mutex> lock(backend.nodes_mutex_);
        auto instance_it = backend.instance_nodes_.find("test_inst");
        ASSERT_NE(instance_it, backend.instance_nodes_.end());
        auto &host_map = instance_it->second;
        auto it = host_map.find("10.0.0.6:8080");
        ASSERT_NE(it, host_map.end());
        EXPECT_EQ(it->second->mediums.size(), 2u);
    }

    ASSERT_EQ(EC_OK, backend.Close());
}

// (8) EVENT_HOST_DOWN: immediate removal, no cleanup callback
TEST_F(EventReportBackendTest, HostDownRemovesNodeFromTable) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 200, /*grace*/ 400, /*tick*/ 50), "trace"));

    std::atomic<int> cleanup_calls{0};
    backend.SetCleanupCallback([&](const std::string &, const std::string &, uint64_t /*gen*/) { ++cleanup_calls; });

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.7:8080", {"mem"}));
    ASSERT_TRUE(backend.IsNodeAvailable("test_inst", "10.0.0.7:8080"));

    backend.SetNodeUnavailable("test_inst", "10.0.0.7:8080");
    ASSERT_FALSE(backend.IsNodeAvailable("test_inst", "10.0.0.7:8080"));
    ASSERT_EQ(EC_OK, backend.UnregisterNode("test_inst", "10.0.0.7:8080"));

    EXPECT_EQ(backend.instance_nodes_["test_inst"].count("10.0.0.7:8080"), 0u);

    std::this_thread::sleep_for(500ms);
    EXPECT_EQ(cleanup_calls.load(), 0);

    ASSERT_EQ(EC_OK, backend.Close());
}

// (9) Generation counter fences stale cleanup
TEST_F(EventReportBackendTest, GenerationBumpsOnReRegistration) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 200, /*grace*/ 5000, /*tick*/ 50), "trace"));

    const std::string host = "10.0.0.8:8080";
    ASSERT_EQ(0u, backend.GetNodeGeneration("test_inst", host));

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", host, {"mem"}));
    ASSERT_EQ(1u, backend.GetNodeGeneration("test_inst", host));

    backend.SetNodeUnavailable("test_inst", host);
    ASSERT_EQ(EC_OK, backend.UnregisterNode("test_inst", host));
    ASSERT_EQ(1u, backend.GetNodeGeneration("test_inst", host));

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", host, {"mem", "disk"}));
    ASSERT_EQ(2u, backend.GetNodeGeneration("test_inst", host));

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", host, {"ssd"}));
    ASSERT_EQ(3u, backend.GetNodeGeneration("test_inst", host));

    ASSERT_EQ(EC_OK, backend.Close());
}

// (10) Cleanup callback receives correct generation
TEST_F(EventReportBackendTest, LivenessLoopPassesGenerationToCallback) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 80, /*grace*/ 120, /*tick*/ 20), "trace"));

    std::atomic<uint64_t> received_gen{0};
    backend.SetCleanupCallback(
        [&](const std::string &, const std::string &, uint64_t gen) { received_gen.store(gen); });

    const std::string host = "10.0.0.9:8080";
    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", host, {"mem"}));
    uint64_t expected_gen = backend.GetNodeGeneration("test_inst", host);

    for (int i = 0; i < 80 && received_gen.load() == 0; ++i) {
        std::this_thread::sleep_for(20ms);
    }
    EXPECT_EQ(received_gen.load(), expected_gen);

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(EventReportBackendTest, OnHeartbeatPublishesMetricsGauges) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 50), "trace"));
    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.10:9600", {"mem"}));

    backend.OnHeartbeat("test_inst",
                        "10.0.0.10:9600",
                        {
                            {"hit_rate", "0.85"},
                            {"active_leases", "5"},
                            {"non_numeric_field", "BOTH_OK"},
                        });

    auto hit_rate_data = metrics_registry_->GetMetricsData("event_report.hit_rate");
    ASSERT_NE(hit_rate_data, nullptr);
    MetricsTags expected_tags = {
        {"instance_id", "test_inst"}, {"host", "10.0.0.10:9600"}, {"type", "event_report_l1p5"}};
    auto gauge = hit_rate_data->GetOrCreateGauge(expected_tags);
    ASSERT_DOUBLE_EQ(0.85, gauge.Get());

    auto leases_data = metrics_registry_->GetMetricsData("event_report.active_leases");
    ASSERT_NE(leases_data, nullptr);
    auto leases_gauge = leases_data->GetOrCreateGauge(expected_tags);
    ASSERT_DOUBLE_EQ(5.0, leases_gauge.Get());

    auto non_numeric = metrics_registry_->GetMetricsData("event_report.non_numeric_field");
    ASSERT_EQ(non_numeric, nullptr);

    backend.OnHeartbeat("test_inst",
                        "10.0.0.10:9600",
                        {
                            {"hit_rate", "0.90"},
                            {"brand_new_metric", "42"},
                        });

    auto new_data = metrics_registry_->GetMetricsData("event_report.brand_new_metric");
    ASSERT_NE(new_data, nullptr);
    auto new_gauge = new_data->GetOrCreateGauge(expected_tags);
    ASSERT_DOUBLE_EQ(42.0, new_gauge.Get());
    ASSERT_DOUBLE_EQ(0.90, gauge.Get());
    EXPECT_FALSE(leases_data->GetGauge(expected_tags).has_value());

    // A non-numeric replacement is also a full-snapshot removal rather than
    // leaving the prior numeric sample visible forever.
    ASSERT_EQ(EC_OK, backend.OnHeartbeat("test_inst", "10.0.0.10:9600", {{"hit_rate", "unknown"}}));
    EXPECT_FALSE(hit_rate_data->GetGauge(expected_tags).has_value());

    // strtod accepts these spellings, but non-finite values are not valid
    // operational gauges and must not poison downstream metric aggregation.
    ASSERT_EQ(EC_OK,
              backend.OnHeartbeat("test_inst", "10.0.0.10:9600", {{"nan_metric", "nan"}, {"inf_metric", "inf"}}));
    EXPECT_EQ(nullptr, metrics_registry_->GetMetricsData("event_report.nan_metric"));
    EXPECT_EQ(nullptr, metrics_registry_->GetMetricsData("event_report.inf_metric"));

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(EventReportBackendTest, MetricsGaugesAreIsolatedByEventReportType) {
    EventReportBackend l1p5_backend(metrics_registry_);
    EventReportBackend l2_backend(metrics_registry_);
    ASSERT_EQ(EC_OK, l1p5_backend.Open(MakeConfig(), "trace"));
    ASSERT_EQ(
        EC_OK,
        l2_backend.Open(MakeConfig(5000, 10000, 50, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2), "trace"));

    const std::string instance_id = "test_inst";
    const std::string host = "10.0.0.10:9600";
    ASSERT_EQ(EC_OK, l1p5_backend.RegisterNode(instance_id, host, {"mem"}));
    ASSERT_EQ(EC_OK, l2_backend.RegisterNode(instance_id, host, {"mem"}));

    ASSERT_EQ(EC_OK, l1p5_backend.OnHeartbeat(instance_id, host, {{"used_bytes", "100"}}));
    ASSERT_EQ(EC_OK, l2_backend.OnHeartbeat(instance_id, host, {{"used_bytes", "200"}}));

    auto data = metrics_registry_->GetMetricsData("event_report.used_bytes");
    ASSERT_NE(data, nullptr);
    MetricsTags l1p5_tags = {{"instance_id", instance_id}, {"host", host}, {"type", "event_report_l1p5"}};
    MetricsTags l2_tags = {{"instance_id", instance_id}, {"host", host}, {"type", "event_report_l2"}};
    ASSERT_DOUBLE_EQ(100.0, data->GetOrCreateGauge(l1p5_tags).Get());
    ASSERT_DOUBLE_EQ(200.0, data->GetOrCreateGauge(l2_tags).Get());

    l1p5_backend.SetNodeUnavailable(instance_id, host);
    ASSERT_DOUBLE_EQ(0.0, data->GetOrCreateGauge(l1p5_tags).Get());
    ASSERT_DOUBLE_EQ(200.0, data->GetOrCreateGauge(l2_tags).Get());

    ASSERT_EQ(EC_OK, l2_backend.UnregisterNode(instance_id, host));
    auto values = data->GetMetricsValues();
    for (const auto &[tags, val] : values) {
        ASSERT_NE(tags, l2_tags) << "l2 gauge should have been removed";
    }
    ASSERT_DOUBLE_EQ(0.0, data->GetOrCreateGauge(l1p5_tags).Get());

    ASSERT_EQ(EC_OK, l1p5_backend.Close());
    ASSERT_EQ(EC_OK, l2_backend.Close());
}

TEST_F(EventReportBackendTest, SetNodeUnavailableZerosGauges) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 50), "trace"));

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.30:9600", {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.31:9600", {"mem"}));

    backend.OnHeartbeat("test_inst", "10.0.0.30:9600", {{"hit_rate", "0.90"}, {"mem_used", "8192"}});
    backend.OnHeartbeat("test_inst", "10.0.0.31:9600", {{"hit_rate", "0.80"}, {"mem_used", "4096"}});

    MetricsTags tags_30 = {{"instance_id", "test_inst"}, {"host", "10.0.0.30:9600"}, {"type", "event_report_l1p5"}};
    MetricsTags tags_31 = {{"instance_id", "test_inst"}, {"host", "10.0.0.31:9600"}, {"type", "event_report_l1p5"}};

    auto hr_data = metrics_registry_->GetMetricsData("event_report.hit_rate");
    ASSERT_NE(hr_data, nullptr);
    ASSERT_DOUBLE_EQ(0.90, hr_data->GetOrCreateGauge(tags_30).Get());
    ASSERT_DOUBLE_EQ(0.80, hr_data->GetOrCreateGauge(tags_31).Get());

    backend.SetNodeUnavailable("test_inst", "10.0.0.30:9600");

    ASSERT_DOUBLE_EQ(0.0, hr_data->GetOrCreateGauge(tags_30).Get());
    auto mu_data = metrics_registry_->GetMetricsData("event_report.mem_used");
    ASSERT_DOUBLE_EQ(0.0, mu_data->GetOrCreateGauge(tags_30).Get());

    ASSERT_DOUBLE_EQ(0.80, hr_data->GetOrCreateGauge(tags_31).Get());
    ASSERT_DOUBLE_EQ(4096, mu_data->GetOrCreateGauge(tags_31).Get());

    backend.SetNodeUnavailable("test_inst", "10.0.0.30:9600");
    ASSERT_DOUBLE_EQ(0.0, hr_data->GetOrCreateGauge(tags_30).Get());

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(EventReportBackendTest, UnregisterNodeCleansUpGauges) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 5000, /*grace*/ 10000, /*tick*/ 50), "trace"));

    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.20:9600", {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode("test_inst", "10.0.0.21:9600", {"mem"}));

    backend.OnHeartbeat("test_inst", "10.0.0.20:9600", {{"hit_rate", "0.75"}, {"mem_used", "4096"}});
    backend.OnHeartbeat("test_inst", "10.0.0.21:9600", {{"hit_rate", "0.60"}, {"mem_used", "2048"}});

    MetricsTags tags_20 = {{"instance_id", "test_inst"}, {"host", "10.0.0.20:9600"}, {"type", "event_report_l1p5"}};
    MetricsTags tags_21 = {{"instance_id", "test_inst"}, {"host", "10.0.0.21:9600"}, {"type", "event_report_l1p5"}};

    auto hr_data = metrics_registry_->GetMetricsData("event_report.hit_rate");
    ASSERT_NE(hr_data, nullptr);
    ASSERT_DOUBLE_EQ(0.75, hr_data->GetOrCreateGauge(tags_20).Get());
    ASSERT_DOUBLE_EQ(0.60, hr_data->GetOrCreateGauge(tags_21).Get());

    ASSERT_EQ(EC_OK, backend.UnregisterNode("test_inst", "10.0.0.20:9600"));

    auto hr_values = hr_data->GetMetricsValues();
    for (const auto &[tags, val] : hr_values) {
        ASSERT_NE(tags, tags_20) << "node 20 gauge should have been removed";
    }
    auto mu_data = metrics_registry_->GetMetricsData("event_report.mem_used");
    auto mu_values = mu_data->GetMetricsValues();
    for (const auto &[tags, val] : mu_values) {
        ASSERT_NE(tags, tags_20) << "node 20 gauge should have been removed";
    }

    ASSERT_DOUBLE_EQ(0.60, hr_data->GetOrCreateGauge(tags_21).Get());
    ASSERT_DOUBLE_EQ(2048, mu_data->GetOrCreateGauge(tags_21).Get());

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(EventReportBackendTest, TwoInstancesSameHostIsolated) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 200, /*grace*/ 5000, /*tick*/ 50), "trace"));

    const std::string host = "10.0.0.50:8080";
    const std::string inst_a = "instance_a";
    const std::string inst_b = "instance_b";

    ASSERT_EQ(EC_OK, backend.RegisterNode(inst_a, host, {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode(inst_b, host, {"mem", "disk"}));

    ASSERT_TRUE(backend.IsNodeAvailable(inst_a, host));
    ASSERT_TRUE(backend.IsNodeAvailable(inst_b, host));

    ASSERT_EQ(1u, backend.GetNodeGeneration(inst_a, host));
    ASSERT_EQ(1u, backend.GetNodeGeneration(inst_b, host));

    backend.SetNodeUnavailable(inst_a, host);
    ASSERT_FALSE(backend.IsNodeAvailable(inst_a, host));
    ASSERT_TRUE(backend.IsNodeAvailable(inst_b, host));

    ASSERT_EQ(EC_OK, backend.OnHeartbeat(inst_b, host, {{"metric", "42"}}));
    ASSERT_TRUE(backend.IsNodeAvailable(inst_b, host));

    ASSERT_EQ(EC_OK, backend.UnregisterNode(inst_a, host));
    ASSERT_EQ(EC_NOENT, backend.UnregisterNode(inst_a, host));
    ASSERT_TRUE(backend.IsNodeAvailable(inst_b, host));
    ASSERT_EQ(1u, backend.GetNodeGeneration(inst_a, host));

    ASSERT_EQ(EC_OK, backend.RegisterNode(inst_a, host, {"ssd"}));
    ASSERT_EQ(2u, backend.GetNodeGeneration(inst_a, host));
    ASSERT_EQ(1u, backend.GetNodeGeneration(inst_b, host));

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST(EventReportBackendSnapshotTest, UnregisteredReporterCannotCreateDeltaVersion) {
    EventReportBackend backend(nullptr);
    const ReporterSnapshotKey scope{"instance-a", "10.0.0.1:8080"};

    std::string committed;
    EXPECT_EQ(EC_SNAPSHOT_REQUIRED, backend.BeginDeltaMutation(scope, committed));
    EXPECT_TRUE(committed.empty());
    EXPECT_TRUE(backend.GetSnapshotVersion(scope).empty());
}

TEST(EventReportBackendSnapshotTest, RegisteredReporterFirstDeltaCreatesReusableVersion) {
    EventReportBackend backend(nullptr);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"hbm"}));

    std::string first;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, first));
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(first));
    EXPECT_EQ(first, backend.GetSnapshotVersion(reporter_key));
    backend.EndDeltaMutation(reporter_key);

    std::string second;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, second));
    EXPECT_EQ(first, second);
    backend.EndDeltaMutation(reporter_key);
}

TEST(EventReportBackendSnapshotTest, ConcurrentFirstDeltasPublishExactlyOneReusableVersion) {
    EventReportBackend backend(nullptr);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.2:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"hbm"}));

    constexpr size_t kThreadCount = 32;
    std::atomic<bool> start{false};
    std::vector<ErrorCode> ecs(kThreadCount, EC_ERROR);
    std::vector<std::string> versions(kThreadCount);
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (size_t i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&, i] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            ecs[i] = backend.BeginDeltaMutation(reporter_key, versions[i]);
            if (ecs[i] == EC_OK) {
                backend.EndDeltaMutation(reporter_key);
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thread : threads) {
        thread.join();
    }

    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(versions.front()));
    for (size_t i = 0; i < kThreadCount; ++i) {
        EXPECT_EQ(EC_OK, ecs[i]) << "thread=" << i;
        EXPECT_EQ(versions.front(), versions[i]) << "thread=" << i;
    }
    EXPECT_EQ(versions.front(), backend.GetSnapshotVersion(reporter_key));
}

TEST(EventReportBackendSnapshotTest, SnapshotCommitPublishesOpaqueToken) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey scope{"instance-a", "10.0.0.1:8080"};

    std::string candidate;
    uint64_t retry_after_ms = 123;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope, candidate, retry_after_ms));
    EXPECT_EQ(0u, retry_after_ms);
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(candidate));
    EXPECT_TRUE(backend.GetSnapshotVersion(scope).empty());

    EXPECT_TRUE(backend.CommitSnapshotVersion(scope, candidate));
    EXPECT_EQ(candidate, backend.GetSnapshotVersion(scope));

    std::string committed;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(scope, committed));
    EXPECT_EQ(candidate, committed);
    backend.EndDeltaMutation(scope);
}

TEST(EventReportBackendSnapshotTest, SnapshotCommitRejectsChangedLifecycleGeneration) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-commit-fence", "10.0.0.71:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));

    std::string candidate;
    uint64_t retry_after_ms = 0;
    uint64_t admitted_generation = 0;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter_key, candidate, retry_after_ms, &admitted_generation));
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(candidate));
    ASSERT_NE(0u, admitted_generation);

    // An explicit REGISTER is a lifecycle boundary. A snapshot admitted by
    // the previous lifecycle must not publish after that boundary even if its
    // metadata phase already completed.
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));
    ASSERT_NE(admitted_generation, backend.GetNodeGeneration(reporter_key.instance_id, reporter_key.host_ip_port));
    EXPECT_EQ(EC_NODE_NOT_REGISTERED,
              backend.CommitSnapshotVersionIfGeneration(reporter_key, candidate, admitted_generation));
    EXPECT_TRUE(backend.GetSnapshotVersion(reporter_key).empty());
    backend.AbortSnapshotVersion(reporter_key, candidate);
}

TEST(EventReportBackendSnapshotTest, SnapshotCleanupLeaseFencesLaterAttemptAdmission) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-cleanup-fence", "10.0.0.72:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));

    std::string committed;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter_key, committed, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, committed));
    const uint64_t cleanup_generation = backend.GetNodeGeneration(reporter_key.instance_id, reporter_key.host_ip_port);
    const uint64_t cleanup_attempt_epoch = backend.GetSnapshotAttemptEpoch(reporter_key);

    EventReportBackend::LifecycleMutationLease cleanup_lease;
    ASSERT_EQ(EC_OK,
              backend.AcquireSnapshotCleanupLease(
                  reporter_key, cleanup_generation, committed, cleanup_attempt_epoch, cleanup_lease));

    std::promise<void> attempt_started;
    std::string next_candidate;
    auto next_attempt = std::async(std::launch::async, [&] {
        attempt_started.set_value();
        uint64_t retry_ms = 0;
        return backend.BeginSnapshot(reporter_key, next_candidate, retry_ms);
    });
    attempt_started.get_future().wait();
    EXPECT_EQ(std::future_status::timeout, next_attempt.wait_for(20ms));

    // Releasing the old cleanup's final-delete lease lets the next attempt
    // publish its epoch. The same cleanup identity must then be rejected even
    // though the reporter lifecycle generation itself has not changed.
    cleanup_lease.reset();
    ASSERT_EQ(std::future_status::ready, next_attempt.wait_for(1s));
    ASSERT_EQ(EC_OK, next_attempt.get());
    ASSERT_GT(backend.GetSnapshotAttemptEpoch(reporter_key), cleanup_attempt_epoch);

    EventReportBackend::LifecycleMutationLease stale_cleanup_lease;
    EXPECT_EQ(EC_MISMATCH,
              backend.AcquireSnapshotCleanupLease(
                  reporter_key, cleanup_generation, committed, cleanup_attempt_epoch, stale_cleanup_lease));
    EXPECT_FALSE(stale_cleanup_lease);
    backend.AbortSnapshotVersion(reporter_key, next_candidate);
}

TEST(EventReportBackendSnapshotTest, SnapshotCommitAndCleanupWaitForTransientLifecycleWriter) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-transient-writer", "10.0.0.74:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));

    std::string candidate;
    uint64_t retry_after_ms = 0;
    uint64_t lifecycle_generation = 0;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter_key, candidate, retry_after_ms, &lifecycle_generation));
    const auto lifecycle_fence = backend.GetOrCreateLifecycleFence(reporter_key);
    ASSERT_TRUE(lifecycle_fence);

    {
        std::unique_lock<std::shared_mutex> transient_writer(lifecycle_fence->mutex);
        std::promise<void> call_started;
        auto commit = std::async(std::launch::async, [&] {
            call_started.set_value();
            return backend.CommitSnapshotVersionIfGeneration(reporter_key, candidate, lifecycle_generation);
        });
        call_started.get_future().wait();
        EXPECT_EQ(std::future_status::timeout, commit.wait_for(20ms));
        transient_writer.unlock();
        ASSERT_EQ(std::future_status::ready, commit.wait_for(1s));
        EXPECT_EQ(EC_OK, commit.get());
    }

    const uint64_t attempt_epoch = backend.GetSnapshotAttemptEpoch(reporter_key);
    {
        std::unique_lock<std::shared_mutex> transient_writer(lifecycle_fence->mutex);
        std::promise<void> call_started;
        auto cleanup = std::async(std::launch::async, [&] {
            call_started.set_value();
            EventReportBackend::LifecycleMutationLease lease;
            return backend.AcquireSnapshotCleanupLease(
                reporter_key, lifecycle_generation, candidate, attempt_epoch, lease);
        });
        call_started.get_future().wait();
        EXPECT_EQ(std::future_status::timeout, cleanup.wait_for(20ms));
        transient_writer.unlock();
        ASSERT_EQ(std::future_status::ready, cleanup.wait_for(1s));
        EXPECT_EQ(EC_OK, cleanup.get());
    }
}

TEST(EventReportBackendSnapshotTest, QueryVisibilityIsStrictOnlyAfterSuccessfulSnapshot) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));

    bool strict = true;
    std::string committed = "stale";
    ASSERT_TRUE(backend.GetQueryVisibilityState(reporter_key, strict, committed));
    EXPECT_FALSE(strict);
    EXPECT_TRUE(committed.empty());

    std::string first;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter_key, first, retry_after_ms));
    ASSERT_TRUE(backend.GetQueryVisibilityState(reporter_key, strict, committed));
    EXPECT_FALSE(strict);
    EXPECT_TRUE(committed.empty());
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, first));
    ASSERT_TRUE(backend.GetQueryVisibilityState(reporter_key, strict, committed));
    EXPECT_TRUE(strict);
    EXPECT_EQ(first, committed);

    std::string failed;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter_key, failed, retry_after_ms));
    ASSERT_TRUE(backend.GetQueryVisibilityState(reporter_key, strict, committed));
    EXPECT_TRUE(strict);
    EXPECT_EQ(first, committed);
    backend.AbortSnapshotVersion(reporter_key, failed);
    ASSERT_TRUE(backend.GetQueryVisibilityState(reporter_key, strict, committed));
    EXPECT_FALSE(strict);
    EXPECT_EQ(first, committed);

    std::string recovered;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter_key, recovered, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, recovered));
    ASSERT_TRUE(backend.GetQueryVisibilityState(reporter_key, strict, committed));
    EXPECT_TRUE(strict);
    EXPECT_EQ(recovered, committed);
}

TEST(EventReportBackendSnapshotTest, QueryVisibilitySnapshotIsInstanceScopedAndExcludesUnavailableReporters) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey soft_reporter{"instance-a", "10.0.0.1:8080"};
    const ReporterSnapshotKey strict_reporter{"instance-a", "10.0.0.2:8080"};
    const ReporterSnapshotKey other_instance{"instance-b", "10.0.0.3:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(soft_reporter.instance_id, soft_reporter.host_ip_port, {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode(strict_reporter.instance_id, strict_reporter.host_ip_port, {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode(other_instance.instance_id, other_instance.host_ip_port, {"mem"}));

    std::string soft_version;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(soft_reporter, soft_version));
    backend.EndDeltaMutation(soft_reporter);
    std::string strict_version;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(strict_reporter, strict_version, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(strict_reporter, strict_version));

    EventReportBackend::QueryVisibilitySnapshot snapshot;
    backend.GetQueryVisibilitySnapshot("instance-a", snapshot);
    ASSERT_EQ(2u, snapshot.size());
    EXPECT_FALSE(snapshot.at(soft_reporter.host_ip_port).strict);
    EXPECT_EQ(soft_version, snapshot.at(soft_reporter.host_ip_port).committed_version);
    EXPECT_TRUE(snapshot.at(strict_reporter.host_ip_port).strict);
    EXPECT_EQ(strict_version, snapshot.at(strict_reporter.host_ip_port).committed_version);
    EXPECT_EQ(0u, snapshot.count(other_instance.host_ip_port));

    backend.SetNodeUnavailable(soft_reporter.instance_id, soft_reporter.host_ip_port);
    backend.GetQueryVisibilitySnapshot("instance-a", snapshot);
    ASSERT_EQ(1u, snapshot.size());
    EXPECT_EQ(0u, snapshot.count(soft_reporter.host_ip_port));
    EXPECT_EQ(1u, snapshot.count(strict_reporter.host_ip_port));

    ASSERT_EQ(EC_OK, backend.UnregisterNode(strict_reporter.instance_id, strict_reporter.host_ip_port));
    backend.GetQueryVisibilitySnapshot("instance-a", snapshot);
    EXPECT_TRUE(snapshot.empty());
}

TEST(EventReportBackendSnapshotTest, SnapshotTokensAreNeverReusedAcrossAttempts) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};
    std::set<std::string> observed;

    for (size_t attempt = 0; attempt < 128; ++attempt) {
        std::string candidate;
        uint64_t retry_after_ms = 0;
        ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, candidate, retry_after_ms));
        ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(candidate));
        EXPECT_TRUE(observed.insert(candidate).second);
        if (attempt % 2 == 0) {
            ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, candidate));
        } else {
            backend.AbortSnapshotVersion(reporter_key, candidate);
        }
    }
    EXPECT_EQ(128u, observed.size());
}

TEST(EventReportBackendSnapshotTest, SnapshotAndDeltaWaitForEachOther) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};

    std::string first;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, first, retry_after_ms));

    std::string concurrent_snapshot;
    EXPECT_EQ(EC_SNAPSHOT_IN_PROGRESS,
              BeginSnapshotForRegisteredReporter(backend, reporter_key, concurrent_snapshot, retry_after_ms));

    std::promise<void> delta_started;
    std::promise<std::pair<ErrorCode, std::string>> delta_result;
    auto delta_result_future = delta_result.get_future();
    std::thread delta_thread([&] {
        delta_started.set_value();
        std::string committed;
        const ErrorCode ec = backend.BeginDeltaMutation(reporter_key, committed);
        delta_result.set_value({ec, committed});
    });
    delta_started.get_future().wait();
    EXPECT_EQ(std::future_status::timeout, delta_result_future.wait_for(20ms));
    EXPECT_TRUE(backend.CommitSnapshotVersion(reporter_key, first));
    const auto delta_status = delta_result_future.wait_for(1s);
    EXPECT_EQ(std::future_status::ready, delta_status);
    if (delta_status != std::future_status::ready) {
        backend.Close();
    }
    const auto [delta_ec, delta_version] = delta_result_future.get();
    EXPECT_EQ(EC_OK, delta_ec);
    EXPECT_EQ(first, delta_version);
    delta_thread.join();

    // Release the delta lease acquired by delta_thread only after a new
    // snapshot has closed the gate and started waiting for it.
    std::promise<void> snapshot_started;
    std::promise<std::tuple<ErrorCode, std::string, uint64_t>> snapshot_result;
    auto snapshot_result_future = snapshot_result.get_future();
    std::thread snapshot_thread([&] {
        snapshot_started.set_value();
        std::string candidate;
        uint64_t retry_ms = 0;
        const ErrorCode ec = BeginSnapshotForRegisteredReporter(backend, reporter_key, candidate, retry_ms);
        snapshot_result.set_value({ec, candidate, retry_ms});
    });
    snapshot_started.get_future().wait();
    EXPECT_EQ(std::future_status::timeout, snapshot_result_future.wait_for(20ms));
    backend.EndDeltaMutation(reporter_key);
    const auto snapshot_status = snapshot_result_future.wait_for(1s);
    EXPECT_EQ(std::future_status::ready, snapshot_status);
    if (snapshot_status != std::future_status::ready) {
        backend.Close();
    }
    const auto [snapshot_ec, candidate, retry_ms] = snapshot_result_future.get();
    EXPECT_EQ(EC_OK, snapshot_ec);
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(candidate));
    EXPECT_EQ(0u, retry_ms);
    snapshot_thread.join();
    backend.AbortSnapshotVersion(reporter_key, candidate);
}

TEST(EventReportBackendSnapshotTest, DeltaWaitTimeoutReturnsSnapshotInProgressWithoutAbortingSnapshot) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    backend.SetSnapshotDeltaDrainTimeoutMsForTest(20);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};

    std::string candidate;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, candidate, retry_after_ms));

    std::string committed = "stale";
    uint64_t lifecycle_generation = std::numeric_limits<uint64_t>::max();
    EXPECT_EQ(EC_SNAPSHOT_IN_PROGRESS, backend.BeginDeltaMutation(reporter_key, committed, &lifecycle_generation));
    EXPECT_TRUE(committed.empty());
    EXPECT_EQ(0u, lifecycle_generation);

    std::string observed_committed;
    std::string observed_in_flight;
    backend.GetSnapshotVersionTokens(reporter_key, observed_committed, observed_in_flight);
    EXPECT_TRUE(observed_committed.empty());
    EXPECT_EQ(candidate, observed_in_flight);

    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, candidate));
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, committed));
    EXPECT_EQ(candidate, committed);
    backend.EndDeltaMutation(reporter_key);
}

TEST(EventReportBackendSnapshotTest, AbortUnblocksWaitingDeltaWithoutPublishingCandidate) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};

    std::string first;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, first, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, first));

    std::string candidate;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, candidate, retry_after_ms));
    std::string observed_committed;
    std::string observed_in_flight;
    backend.GetSnapshotVersionTokens(reporter_key, observed_committed, observed_in_flight);
    EXPECT_EQ(first, observed_committed);
    EXPECT_EQ(candidate, observed_in_flight);
    std::promise<std::pair<ErrorCode, std::string>> delta_result;
    auto delta_result_future = delta_result.get_future();
    std::thread delta_thread([&] {
        std::string committed;
        const ErrorCode ec = backend.BeginDeltaMutation(reporter_key, committed);
        delta_result.set_value({ec, committed});
    });
    EXPECT_EQ(std::future_status::timeout, delta_result_future.wait_for(20ms));
    backend.AbortSnapshotVersion(reporter_key, candidate);
    backend.GetSnapshotVersionTokens(reporter_key, observed_committed, observed_in_flight);
    EXPECT_EQ(first, observed_committed);
    EXPECT_TRUE(observed_in_flight.empty());
    const auto delta_status = delta_result_future.wait_for(1s);
    EXPECT_EQ(std::future_status::ready, delta_status);
    if (delta_status != std::future_status::ready) {
        backend.Close();
    }
    const auto [delta_ec, committed] = delta_result_future.get();
    EXPECT_EQ(EC_OK, delta_ec);
    EXPECT_EQ(first, committed);
    delta_thread.join();
    backend.EndDeltaMutation(reporter_key);
}

TEST(EventReportBackendSnapshotTest, CommitUnblocksAllWaitingDeltasWithNewToken) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};

    std::string candidate;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, candidate, retry_after_ms));

    constexpr size_t kWaiterCount = 4;
    std::vector<std::future<std::pair<ErrorCode, std::string>>> futures;
    futures.reserve(kWaiterCount);
    for (size_t i = 0; i < kWaiterCount; ++i) {
        futures.push_back(std::async(std::launch::async, [&backend, &reporter_key] {
            std::string committed;
            const ErrorCode ec = backend.BeginDeltaMutation(reporter_key, committed);
            if (ec == EC_OK) {
                backend.EndDeltaMutation(reporter_key);
            }
            return std::make_pair(ec, committed);
        }));
    }
    for (auto &future : futures) {
        EXPECT_EQ(std::future_status::timeout, future.wait_for(20ms));
    }

    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, candidate));
    for (auto &future : futures) {
        ASSERT_EQ(std::future_status::ready, future.wait_for(1s));
        const auto [ec, committed] = future.get();
        EXPECT_EQ(EC_OK, ec);
        EXPECT_EQ(candidate, committed);
    }
}

TEST(EventReportBackendSnapshotTest, SnapshotWaitsUntilEveryAdmittedDeltaDrains) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};

    std::string first;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, first, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, first));

    constexpr size_t kActiveDeltaCount = 3;
    for (size_t i = 0; i < kActiveDeltaCount; ++i) {
        std::string committed;
        ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, committed));
        ASSERT_EQ(first, committed);
    }

    auto snapshot = std::async(std::launch::async, [&] {
        std::string candidate;
        uint64_t retry_ms = 0;
        const ErrorCode ec = BeginSnapshotForRegisteredReporter(backend, reporter_key, candidate, retry_ms);
        return std::make_tuple(ec, candidate, retry_ms);
    });
    EXPECT_EQ(std::future_status::timeout, snapshot.wait_for(20ms));

    backend.EndDeltaMutation(reporter_key);
    EXPECT_EQ(std::future_status::timeout, snapshot.wait_for(20ms));
    backend.EndDeltaMutation(reporter_key);
    EXPECT_EQ(std::future_status::timeout, snapshot.wait_for(20ms));
    backend.EndDeltaMutation(reporter_key);

    ASSERT_EQ(std::future_status::ready, snapshot.wait_for(1s));
    const auto [ec, candidate, retry_ms] = snapshot.get();
    EXPECT_EQ(EC_OK, ec);
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(candidate));
    EXPECT_EQ(0u, retry_ms);
    EXPECT_TRUE(backend.CommitSnapshotVersion(reporter_key, candidate));
}

TEST(EventReportBackendSnapshotTest, SnapshotDrainTimeoutAbortsCandidateAndReopensWriteGate) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    backend.SetSnapshotDeltaDrainTimeoutMsForTest(20);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};

    std::string committed;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, committed, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, committed));

    std::string delta_version;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, delta_version));
    ASSERT_EQ(committed, delta_version);

    auto timed_out_snapshot = std::async(std::launch::async, [&] {
        std::string candidate = "stale";
        uint64_t retry_ms = 99;
        const ErrorCode ec = backend.BeginSnapshot(reporter_key, candidate, retry_ms);
        return std::make_tuple(ec, candidate, retry_ms);
    });
    const auto timeout_status = timed_out_snapshot.wait_for(1s);
    if (timeout_status != std::future_status::ready) {
        backend.Close();
    }
    ASSERT_EQ(std::future_status::ready, timeout_status);
    const auto [snapshot_ec, candidate, retry_ms] = timed_out_snapshot.get();
    EXPECT_EQ(EC_SNAPSHOT_IN_PROGRESS, snapshot_ec);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(0u, retry_ms);

    std::string observed_committed;
    std::string observed_in_flight;
    backend.GetSnapshotVersionTokens(reporter_key, observed_committed, observed_in_flight);
    EXPECT_EQ(committed, observed_committed);
    EXPECT_TRUE(observed_in_flight.empty());

    // The timed-out snapshot must not leave the reporter write gate closed.
    std::string later_delta_version;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, later_delta_version));
    EXPECT_EQ(committed, later_delta_version);
    backend.EndDeltaMutation(reporter_key);
    backend.EndDeltaMutation(reporter_key);

    std::string retry_candidate;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter_key, retry_candidate, retry_after_ms));
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(retry_candidate));
    backend.AbortSnapshotVersion(reporter_key, retry_candidate);
}

TEST(EventReportBackendSnapshotTest, ConcurrentSnapshotsHaveExactlyOneWinner) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};
    constexpr size_t kContenderCount = 12;

    std::promise<void> start;
    auto start_signal = start.get_future().share();
    std::vector<std::future<std::pair<ErrorCode, std::string>>> contenders;
    contenders.reserve(kContenderCount);
    for (size_t i = 0; i < kContenderCount; ++i) {
        contenders.push_back(std::async(std::launch::async, [&] {
            start_signal.wait();
            std::string candidate;
            uint64_t retry_after_ms = 0;
            const ErrorCode ec = BeginSnapshotForRegisteredReporter(backend, reporter_key, candidate, retry_after_ms);
            return std::make_pair(ec, candidate);
        }));
    }
    start.set_value();

    size_t winner_count = 0;
    size_t busy_count = 0;
    std::string winning_token;
    for (auto &contender : contenders) {
        ASSERT_EQ(std::future_status::ready, contender.wait_for(1s));
        const auto [ec, candidate] = contender.get();
        if (ec == EC_OK) {
            ++winner_count;
            winning_token = candidate;
        } else {
            EXPECT_EQ(EC_SNAPSHOT_IN_PROGRESS, ec);
            EXPECT_TRUE(candidate.empty());
            ++busy_count;
        }
    }
    EXPECT_EQ(1u, winner_count);
    EXPECT_EQ(kContenderCount - 1, busy_count);
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(winning_token));
    EXPECT_TRUE(backend.CommitSnapshotVersion(reporter_key, winning_token));
}

TEST(EventReportBackendSnapshotTest, UnregisterCancelsSnapshotWaitingForActiveDelta) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const std::string instance_id = "instance-a";
    const std::string host = "10.0.0.1:8080";
    const ReporterSnapshotKey reporter_key{instance_id, host};
    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"hbm"}));

    std::string first;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, first, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, first));
    std::string committed;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, committed));

    auto snapshot = std::async(std::launch::async, [&] {
        std::string candidate = "stale";
        uint64_t retry_ms = 99;
        const ErrorCode ec = BeginSnapshotForRegisteredReporter(backend, reporter_key, candidate, retry_ms);
        return std::make_tuple(ec, candidate, retry_ms);
    });
    EXPECT_EQ(std::future_status::timeout, snapshot.wait_for(20ms));
    ASSERT_EQ(EC_OK, backend.UnregisterNode(instance_id, host));

    ASSERT_EQ(std::future_status::ready, snapshot.wait_for(1s));
    const auto [ec, candidate, retry_ms] = snapshot.get();
    EXPECT_EQ(EC_SNAPSHOT_REQUIRED, ec);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(0u, retry_ms);
    backend.EndDeltaMutation(reporter_key);
}

TEST_F(EventReportBackendTest, AutomaticLivenessCleanupCancelsSnapshotWaitingForActiveDelta) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 20, /*grace*/ 20, /*tick*/ 5), "liveness_snapshot_wait"));
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.60:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));

    std::string committed;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, committed));
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(committed));

    auto snapshot = std::async(std::launch::async, [&] {
        std::string candidate = "stale";
        uint64_t retry_after_ms = 99;
        const ErrorCode ec = backend.BeginSnapshot(reporter_key, candidate, retry_after_ms);
        return std::make_tuple(ec, candidate, retry_after_ms);
    });
    EXPECT_EQ(std::future_status::timeout, snapshot.wait_for(5ms));
    ASSERT_EQ(std::future_status::ready, snapshot.wait_for(2s));
    const auto [ec, candidate, retry_after_ms] = snapshot.get();
    EXPECT_EQ(EC_SNAPSHOT_REQUIRED, ec);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(0u, retry_after_ms);
    EXPECT_FALSE(backend.IsNodeRegistered(reporter_key.instance_id, reporter_key.host_ip_port));
    EXPECT_TRUE(backend.GetSnapshotVersion(reporter_key).empty());

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST(EventReportBackendSnapshotTest, UnregisterUnblocksWaitingDeltaAndRequiresNewSnapshot) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const std::string instance_id = "instance-a";
    const std::string host = "10.0.0.1:8080";
    const ReporterSnapshotKey reporter_key{instance_id, host};
    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"hbm"}));

    std::string first;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, first, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, first));
    std::string second;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, second, retry_after_ms));

    auto delta = std::async(std::launch::async, [&] {
        std::string committed = "stale";
        const ErrorCode ec = backend.BeginDeltaMutation(reporter_key, committed);
        return std::make_pair(ec, committed);
    });
    EXPECT_EQ(std::future_status::timeout, delta.wait_for(20ms));
    ASSERT_EQ(EC_OK, backend.UnregisterNode(instance_id, host));
    ASSERT_EQ(std::future_status::ready, delta.wait_for(1s));
    const auto [ec, committed] = delta.get();
    EXPECT_EQ(EC_SNAPSHOT_REQUIRED, ec);
    EXPECT_TRUE(committed.empty());
}

TEST(EventReportBackendSnapshotTest, OtherReporterIsNotBlockedBySnapshot) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_a{"instance-a", "10.0.0.1:8080"};
    const ReporterSnapshotKey reporter_b{"instance-a", "10.0.0.2:8080"};
    uint64_t retry_after_ms = 0;

    std::string token_b;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_b, token_b, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_b, token_b));

    std::string token_a;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_a, token_a, retry_after_ms));
    std::string committed_b;
    EXPECT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_b, committed_b));
    EXPECT_EQ(token_b, committed_b);
    backend.EndDeltaMutation(reporter_b);
    backend.AbortSnapshotVersion(reporter_a, token_a);
}

TEST(EventReportBackendSnapshotTest, AbortNeverPublishesAndWrongTokenCannotCommit) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey scope{"instance-a", "10.0.0.1:8080"};

    std::string candidate;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope, candidate, retry_after_ms));
    EXPECT_FALSE(backend.CommitSnapshotVersion(scope, std::string(32, 'f')));
    EXPECT_TRUE(backend.GetSnapshotVersion(scope).empty());

    backend.AbortSnapshotVersion(scope, std::string(32, 'e'));
    std::string still_blocked;
    EXPECT_EQ(EC_SNAPSHOT_IN_PROGRESS,
              BeginSnapshotForRegisteredReporter(backend, scope, still_blocked, retry_after_ms));

    backend.AbortSnapshotVersion(scope, candidate);
    EXPECT_TRUE(backend.GetSnapshotVersion(scope).empty());
    EXPECT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope, still_blocked, retry_after_ms));
    backend.AbortSnapshotVersion(scope, still_blocked);
}

TEST(EventReportBackendSnapshotTest, SnapshotRateLimitReturnsRetryDelay) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(30'000);
    const ReporterSnapshotKey scope{"instance-a", "10.0.0.1:8080"};

    std::string first;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope, first, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(scope, first));

    std::string second;
    EXPECT_EQ(EC_SNAPSHOT_RATE_LIMITED, BeginSnapshotForRegisteredReporter(backend, scope, second, retry_after_ms));
    EXPECT_GT(retry_after_ms, 0u);
    EXPECT_LE(retry_after_ms, 30'000u);
    EXPECT_TRUE(second.empty());

    const uint64_t first_retry_after_ms = retry_after_ms;
    std::this_thread::sleep_for(2ms);
    second = "stale";
    retry_after_ms = 0;
    EXPECT_EQ(EC_SNAPSHOT_RATE_LIMITED, BeginSnapshotForRegisteredReporter(backend, scope, second, retry_after_ms));
    EXPECT_TRUE(second.empty());
    EXPECT_GT(retry_after_ms, 0u);
    EXPECT_LE(retry_after_ms, first_retry_after_ms);

    backend.SetSnapshotMinIntervalMsForTest(0);
    EXPECT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope, second, retry_after_ms));
    backend.AbortSnapshotVersion(scope, second);
}

TEST_F(EventReportBackendTest, ConfiguredSnapshotRateLimitIsAppliedOnOpen) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK,
              backend.Open(MakeConfig(200, 400, 50, DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5, 120'000),
                           "configured_snapshot_rate_limit"));

    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};
    std::string first;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, first, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, first));

    std::string second = "stale";
    ASSERT_EQ(EC_SNAPSHOT_RATE_LIMITED,
              BeginSnapshotForRegisteredReporter(backend, reporter_key, second, retry_after_ms));
    EXPECT_TRUE(second.empty());
    EXPECT_GT(retry_after_ms, 100'000u);
    EXPECT_LE(retry_after_ms, 120'000u);
    ASSERT_EQ(EC_OK, backend.Close());
}

TEST(EventReportBackendSnapshotTest, ScopesAreIsolatedByInstanceAndReporterHost) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey scope_a{"instance-a", "10.0.0.1:8080"};
    const ReporterSnapshotKey scope_b{"instance-a", "10.0.0.2:8080"};
    const ReporterSnapshotKey scope_c{"instance-b", "10.0.0.1:8080"};

    std::string token_a;
    std::string token_b;
    std::string token_c;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope_a, token_a, retry_after_ms));
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope_b, token_b, retry_after_ms));
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope_c, token_c, retry_after_ms));
    EXPECT_NE(token_a, token_b);
    EXPECT_NE(token_a, token_c);
    EXPECT_NE(token_b, token_c);

    EXPECT_TRUE(backend.CommitSnapshotVersion(scope_a, token_a));
    EXPECT_TRUE(backend.CommitSnapshotVersion(scope_b, token_b));
    EXPECT_TRUE(backend.CommitSnapshotVersion(scope_c, token_c));
    EXPECT_EQ(token_a, backend.GetSnapshotVersion(scope_a));
    EXPECT_EQ(token_b, backend.GetSnapshotVersion(scope_b));
    EXPECT_EQ(token_c, backend.GetSnapshotVersion(scope_c));
}

TEST(EventReportBackendSnapshotTest, UnregisterForcesFullSnapshotAgain) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const std::string instance_id = "instance-a";
    const std::string host = "10.0.0.1:8080";
    const ReporterSnapshotKey scope{instance_id, host};

    std::string rejected_token;
    uint64_t retry_after_ms = 0;
    EXPECT_EQ(EC_SNAPSHOT_REQUIRED, backend.BeginSnapshot(scope, rejected_token, retry_after_ms));
    EXPECT_TRUE(rejected_token.empty());
    EXPECT_EQ(0u, backend.snapshot_versions_.count(scope));

    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"hbm", "dram"}));
    std::string token;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope, token, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(scope, token));
    ASSERT_EQ(token, backend.GetSnapshotVersion(scope));

    ASSERT_EQ(EC_OK, backend.UnregisterNode(instance_id, host));
    EXPECT_TRUE(backend.GetSnapshotVersion(scope).empty());
    EXPECT_EQ(EC_SNAPSHOT_REQUIRED, backend.BeginSnapshot(scope, rejected_token, retry_after_ms));
    EXPECT_TRUE(rejected_token.empty());
    EXPECT_EQ(0u, backend.snapshot_versions_.count(scope));
    std::string committed;
    EXPECT_EQ(EC_SNAPSHOT_REQUIRED, backend.BeginDeltaMutation(scope, committed));
}

TEST(EventReportBackendSnapshotTest, FailedDeltaClearsOutputAndDoesNotCreateACommit) {
    EventReportBackend backend(nullptr);
    std::string committed = "stale-token";

    EXPECT_EQ(EC_BADARGS, backend.BeginDeltaMutation({"", "10.0.0.1:8080"}, committed));
    EXPECT_TRUE(committed.empty());

    committed = "stale-token";
    const ReporterSnapshotKey scope{"instance-a", "10.0.0.1:8080"};
    EXPECT_EQ(EC_SNAPSHOT_REQUIRED, backend.BeginDeltaMutation(scope, committed));
    EXPECT_TRUE(committed.empty());
    EXPECT_TRUE(backend.GetSnapshotVersion(scope).empty());
}

TEST(EventReportBackendSnapshotTest, UnregisterThenReregisterLetsFirstDeltaCreateNewVersion) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const std::string instance_id = "instance-a";
    const std::string host = "10.0.0.1:8080";
    const ReporterSnapshotKey scope{instance_id, host};

    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"hbm", "dram"}));
    std::string first_token;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope, first_token, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(scope, first_token));

    ASSERT_EQ(EC_OK, backend.UnregisterNode(instance_id, host));
    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"hbm", "dram"}));
    EXPECT_TRUE(backend.GetSnapshotVersion(scope).empty());
    std::string committed = "stale-token";
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(scope, committed));
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(committed));
    EXPECT_NE(first_token, committed);
    backend.EndDeltaMutation(scope);

    std::string second_token;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, scope, second_token, retry_after_ms));
    EXPECT_NE(first_token, second_token);
    EXPECT_TRUE(backend.CommitSnapshotVersion(scope, second_token));
}

TEST(EventReportBackendSnapshotTest, StaleDeltaEndCannotDrainReregisteredLifecycle) {
    EventReportBackend backend(nullptr);
    const ReporterSnapshotKey reporter_key{"delta-incarnation", "10.0.0.73:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));

    std::string old_token;
    uint64_t old_generation = 0;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, old_token, &old_generation));
    ASSERT_EQ(1u, backend.snapshot_versions_[reporter_key].active_delta_mutations);

    ASSERT_EQ(EC_OK, backend.UnregisterNode(reporter_key.instance_id, reporter_key.host_ip_port));
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));
    std::string new_token;
    uint64_t new_generation = 0;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, new_token, &new_generation));
    ASSERT_NE(old_token, new_token);
    ASSERT_NE(old_generation, new_generation);
    ASSERT_EQ(1u, backend.snapshot_versions_[reporter_key].active_delta_mutations);

    backend.EndDeltaMutation(reporter_key, old_generation, old_token);
    EXPECT_EQ(1u, backend.snapshot_versions_[reporter_key].active_delta_mutations);

    backend.EndDeltaMutation(reporter_key, new_generation, new_token);
    EXPECT_EQ(0u, backend.snapshot_versions_[reporter_key].active_delta_mutations);
}

TEST(EventReportBackendSnapshotTest, StableLocationIdHasNoSnapshotGeneration) {
    EventReportBackend backend(nullptr);
    ASSERT_EQ(EC_OK, backend.Open(EventReportBackendTest::MakeConfig(), "snapshot_location_test"));
    const std::string location_id = backend.BuildLocationId("hbm", "10.0.0.1:8080");
    EXPECT_EQ("kvs#event_report_l1p5#hbm#10.0.0.1:8080", location_id);

    std::string medium;
    std::string host;
    EXPECT_TRUE(backend.ParseLocationId(location_id, medium, host));
    EXPECT_EQ("hbm", medium);
    EXPECT_EQ("10.0.0.1:8080", host);
    EXPECT_FALSE(backend.ParseLocationId("kvs#event_report#hbm#snapshot_v=7#10.0.0.1:8080", medium, host));
}

TEST(EventReportBackendSnapshotTest, RegisterAndHeartbeatPreserveCommittedToken) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const std::string instance_id = "instance-a";
    const std::string host = "10.0.0.1:8080";
    const ReporterSnapshotKey reporter_key{instance_id, host};

    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"hbm"}));
    std::string token;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, token, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, token));

    EXPECT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"memory"}));
    EXPECT_EQ(EC_OK, backend.OnHeartbeat(instance_id, host, {{"load", "1"}}));
    EXPECT_EQ(token, backend.GetSnapshotVersion(reporter_key));

    std::string committed;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, committed));
    EXPECT_EQ(token, committed);
    backend.EndDeltaMutation(reporter_key);
}

TEST(EventReportBackendSnapshotTest, InvalidInputsDoNotMutateOrReleaseSnapshotState) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.1:8080"};

    std::string candidate = "stale";
    uint64_t retry_after_ms = 99;
    EXPECT_EQ(EC_BADARGS,
              BeginSnapshotForRegisteredReporter(backend, {"", reporter_key.host_ip_port}, candidate, retry_after_ms));
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(0u, retry_after_ms);

    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, candidate, retry_after_ms));
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(candidate));
    EXPECT_FALSE(backend.CommitSnapshotVersion(reporter_key, ""));
    EXPECT_FALSE(backend.CommitSnapshotVersion(reporter_key, std::string(31, 'a')));
    EXPECT_FALSE(backend.CommitSnapshotVersion(reporter_key, std::string(32, 'g')));

    backend.AbortSnapshotVersion(reporter_key, std::string(32, 'f'));
    std::string blocked = "stale";
    retry_after_ms = 99;
    EXPECT_EQ(EC_SNAPSHOT_IN_PROGRESS,
              BeginSnapshotForRegisteredReporter(backend, reporter_key, blocked, retry_after_ms));
    EXPECT_TRUE(blocked.empty());
    EXPECT_EQ(0u, retry_after_ms);

    backend.AbortSnapshotVersion(reporter_key, candidate);
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, candidate, retry_after_ms));
    backend.AbortSnapshotVersion(reporter_key, candidate);
}

TEST(EventReportBackendSnapshotTest, LocationIdParserRejectsMalformedAndLegacyIds) {
    EventReportBackend backend(nullptr);
    ASSERT_EQ(EC_OK, backend.Open(EventReportBackendTest::MakeConfig(), "snapshot_parser_test"));
    std::string medium;
    std::string host;

    const std::string valid = backend.BuildLocationId("hbm-cache", "host.example:8080");
    ASSERT_TRUE(backend.ParseLocationId(valid, medium, host));
    EXPECT_EQ("hbm-cache", medium);
    EXPECT_EQ("host.example:8080", host);

    for (const std::string &invalid : {
             std::string{},
             std::string{"kvs#event_report#"},
             std::string{"kvs#event_report##host:8080"},
             std::string{"kvs#event_report#hbm#"},
             std::string{"kvs#event_report#hbm#host:8080#extra"},
             std::string{"kvs#event_report#hbm#snapshot_v=7#host:8080"},
             std::string{"other#event_report#hbm#host:8080"},
         }) {
        EXPECT_FALSE(backend.ParseLocationId(invalid, medium, host)) << invalid;
    }
}

TEST(EventReportBackendSnapshotTest, UriCarriesOnlyOpaqueSnapshotToken) {
    const std::string token = "00112233445566778899aabbccddeeff";
    const std::string reporter_uri = "vineyard://127.0.0.1:9600/object?size=1024";
    std::string versioned_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(reporter_uri, token, versioned_uri));
    EXPECT_NE(std::string::npos, versioned_uri.find("s_version=" + token));
    EXPECT_EQ(std::string::npos, versioned_uri.find("kvcm_"));

    SnapshotUriInfo info;
    ASSERT_TRUE(SnapshotUriUtils::ParseSnapshotUriInfo(versioned_uri, info));
    EXPECT_EQ(token, info.version);
    EXPECT_TRUE(SnapshotUriUtils::HasEventReportInternalUriMetadata(DataStorageUri(versioned_uri)));

    EXPECT_FALSE(SnapshotUriUtils::ParseSnapshotUriInfo(versioned_uri + "&s_version=" + token, info));
    EXPECT_FALSE(SnapshotUriUtils::AddSnapshotVersionToUri(versioned_uri, token, versioned_uri));
    EXPECT_FALSE(SnapshotUriUtils::IsValidSnapshotVersionToken(""));
    EXPECT_FALSE(SnapshotUriUtils::IsValidSnapshotVersionToken("7"));
    EXPECT_FALSE(SnapshotUriUtils::IsValidSnapshotVersionToken(std::string(32, 'g')));
}

TEST(EventReportBackendSnapshotTest, MightExistRequiresCurrentTokenAndAvailableReporter) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const std::string instance_id = "instance-a";
    const std::string reporter_host = "10.0.0.1:8080";
    const ReporterSnapshotKey reporter_key{instance_id, reporter_host};
    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, reporter_host, {"hbm"}));

    std::string token;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, token, retry_after_ms));

    // The URI endpoint may differ from the reporter identity. The opaque
    // token must resolve the owner; URI hostname must not be used for liveness.
    const std::string raw_uri = "event_report://physical-cache:9600/hbm";
    std::string versioned_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(raw_uri, token, versioned_uri));
    const DataStorageUri committed_uri(versioned_uri);
    ASSERT_TRUE(committed_uri.Valid());

    EXPECT_EQ((std::vector<bool>{false, false}), backend.MightExist({DataStorageUri(raw_uri), committed_uri}));

    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, token));
    EXPECT_EQ((std::vector<bool>{true}), backend.MightExist({committed_uri}));

    std::string unknown_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(raw_uri, "ffffffffffffffffffffffffffffffff", unknown_uri));
    EXPECT_EQ((std::vector<bool>{false}), backend.MightExist({DataStorageUri(unknown_uri)}));

    backend.SetNodeUnavailable(instance_id, reporter_host);
    EXPECT_EQ((std::vector<bool>{false}), backend.MightExist({committed_uri}));

    ASSERT_EQ(EC_OK, backend.OnHeartbeat(instance_id, reporter_host, {}));
    EXPECT_EQ((std::vector<bool>{true}), backend.MightExist({committed_uri}));

    std::string replacement_token;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, replacement_token, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, replacement_token));
    std::string replacement_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(raw_uri, replacement_token, replacement_uri));
    EXPECT_EQ((std::vector<bool>{false, true}), backend.MightExist({committed_uri, DataStorageUri(replacement_uri)}));

    ASSERT_EQ(EC_OK, backend.UnregisterNode(instance_id, reporter_host));
    EXPECT_EQ((std::vector<bool>{false}), backend.MightExist({DataStorageUri(replacement_uri)}));
}

TEST_F(EventReportBackendTest, MightExistFollowsAutomaticLivenessAndFullReporterLifecycle) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(/*hb*/ 80, /*grace*/ 250, /*tick*/ 10), "trace"));
    backend.SetSnapshotMinIntervalMsForTest(0);
    backend.SetCleanupCallback([](const std::string &, const std::string &, uint64_t) {});

    const std::string instance_id = "instance-lifecycle";
    const std::string host = "10.0.0.60:8080";
    const ReporterSnapshotKey reporter_key{instance_id, host};
    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"mem"}));

    std::string first_token;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, first_token, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, first_token));
    std::string first_uri;
    ASSERT_TRUE(
        SnapshotUriUtils::AddSnapshotVersionToUri("event_report://physical-cache:9600/mem", first_token, first_uri));
    ASSERT_EQ((std::vector<bool>{true}), backend.MightExist({DataStorageUri(first_uri)}));

    const auto wait_until = [](auto predicate, std::chrono::milliseconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (predicate()) {
                return true;
            }
            std::this_thread::sleep_for(5ms);
        }
        return predicate();
    };

    ASSERT_TRUE(wait_until([&] { return !backend.IsNodeAvailable(instance_id, host); }, 1s));
    EXPECT_TRUE(backend.IsNodeRegistered(instance_id, host));
    EXPECT_EQ((std::vector<bool>{false}), backend.MightExist({DataStorageUri(first_uri)}));

    ASSERT_EQ(EC_OK, backend.OnHeartbeat(instance_id, host, {}));
    EXPECT_EQ((std::vector<bool>{true}), backend.MightExist({DataStorageUri(first_uri)}));

    ASSERT_TRUE(wait_until([&] { return !backend.IsNodeRegistered(instance_id, host); }, 2s));
    EXPECT_EQ((std::vector<bool>{false}), backend.MightExist({DataStorageUri(first_uri)}));

    ASSERT_EQ(EC_OK, backend.RegisterNode(instance_id, host, {"mem"}));
    EXPECT_TRUE(backend.GetSnapshotVersion(reporter_key).empty());
    EXPECT_EQ((std::vector<bool>{false}), backend.MightExist({DataStorageUri(first_uri)}));
    std::string committed = "must-be-cleared";
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, committed));
    ASSERT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken(committed));
    EXPECT_NE(first_token, committed);
    backend.EndDeltaMutation(reporter_key);

    std::string second_token;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, second_token, retry_after_ms));
    ASSERT_NE(first_token, second_token);
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, second_token));
    std::string second_uri;
    ASSERT_TRUE(
        SnapshotUriUtils::AddSnapshotVersionToUri("event_report://physical-cache:9600/mem", second_token, second_uri));
    EXPECT_EQ((std::vector<bool>{false, true}),
              backend.MightExist({DataStorageUri(first_uri), DataStorageUri(second_uri)}));

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST(EventReportBackendSnapshotTest, MightExistBatchPreservesOrderAcrossTokenAndLivenessFailures) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_a{"instance-a", "10.0.0.80:8080"};
    const ReporterSnapshotKey reporter_b{"instance-a", "10.0.0.81:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_a.instance_id, reporter_a.host_ip_port, {"mem"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_b.instance_id, reporter_b.host_ip_port, {"mem"}));

    auto commit = [&](const ReporterSnapshotKey &reporter) {
        std::string token;
        uint64_t retry_after_ms = 0;
        EXPECT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter, token, retry_after_ms));
        EXPECT_TRUE(backend.CommitSnapshotVersion(reporter, token));
        return token;
    };
    const std::string old_a = commit(reporter_a);
    const std::string current_a = commit(reporter_a);
    const std::string current_b = commit(reporter_b);

    auto versioned = [](const std::string &token) {
        std::string uri;
        EXPECT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri("event_report://physical-cache:9600/mem", token, uri));
        return DataStorageUri(uri);
    };
    const DataStorageUri current_a_uri = versioned(current_a);
    const DataStorageUri old_a_uri = versioned(old_a);
    const DataStorageUri unknown_uri = versioned("ffffffffffffffffffffffffffffffff");
    const DataStorageUri current_b_uri = versioned(current_b);
    const DataStorageUri malformed_uri("event_report://physical-cache:9600/mem");
    ASSERT_TRUE(current_a_uri.Valid());
    ASSERT_TRUE(old_a_uri.Valid());
    ASSERT_TRUE(unknown_uri.Valid());
    ASSERT_TRUE(current_b_uri.Valid());
    ASSERT_TRUE(malformed_uri.Valid());

    backend.SetNodeUnavailable(reporter_b.instance_id, reporter_b.host_ip_port);
    EXPECT_EQ((std::vector<bool>{true, false, false, false, false, true}),
              backend.MightExist({current_a_uri, old_a_uri, unknown_uri, current_b_uri, malformed_uri, current_a_uri}));
}

TEST(EventReportBackendSnapshotTest, MightExistUsesTokenOwnerAcrossInstancesAndPreservesCommittedOnAbort) {
    EventReportBackend backend(nullptr);
    backend.SetSnapshotMinIntervalMsForTest(0);
    const std::string shared_reporter = "10.0.0.1:8080";
    const ReporterSnapshotKey reporter_a{"instance-a", shared_reporter};
    const ReporterSnapshotKey reporter_b{"instance-b", shared_reporter};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_a.instance_id, shared_reporter, {"hbm"}));
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_b.instance_id, shared_reporter, {"hbm"}));
    EXPECT_TRUE(backend.IsNodeRegistered(reporter_a.instance_id, shared_reporter));
    EXPECT_TRUE(backend.IsNodeRegistered(reporter_b.instance_id, shared_reporter));
    EXPECT_FALSE(backend.IsNodeRegistered("instance-c", shared_reporter));

    auto commit_snapshot = [&](const ReporterSnapshotKey &reporter_key) {
        std::string token;
        uint64_t retry_after_ms = 0;
        EXPECT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, token, retry_after_ms));
        EXPECT_TRUE(backend.CommitSnapshotVersion(reporter_key, token));
        return token;
    };
    const std::string token_a = commit_snapshot(reporter_a);
    const std::string token_b = commit_snapshot(reporter_b);
    ASSERT_NE(token_a, token_b);

    const std::string raw_uri = "event_report://physical-cache:9600/hbm";
    std::string uri_a;
    std::string uri_b;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(raw_uri, token_a, uri_a));
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(raw_uri, token_b, uri_b));
    EXPECT_EQ((std::vector<bool>{true, true}), backend.MightExist({DataStorageUri(uri_a), DataStorageUri(uri_b)}));

    std::string aborted_token;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_a, aborted_token, retry_after_ms));
    std::string aborted_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(raw_uri, aborted_token, aborted_uri));
    EXPECT_EQ((std::vector<bool>{true, false}),
              backend.MightExist({DataStorageUri(uri_a), DataStorageUri(aborted_uri)}));
    backend.AbortSnapshotVersion(reporter_a, aborted_token);
    EXPECT_EQ((std::vector<bool>{true}), backend.MightExist({DataStorageUri(uri_a)}));

    backend.SetNodeUnavailable(reporter_a.instance_id, shared_reporter);
    EXPECT_TRUE(backend.IsNodeRegistered(reporter_a.instance_id, shared_reporter));
    EXPECT_FALSE(backend.IsNodeAvailable(reporter_a.instance_id, shared_reporter));
    EXPECT_EQ((std::vector<bool>{false, true}), backend.MightExist({DataStorageUri(uri_a), DataStorageUri(uri_b)}));
    ASSERT_EQ(EC_OK, backend.OnHeartbeat(reporter_a.instance_id, shared_reporter, {}));
    EXPECT_EQ((std::vector<bool>{true, true}), backend.MightExist({DataStorageUri(uri_a), DataStorageUri(uri_b)}));

    ASSERT_EQ(EC_OK, backend.Close());
    EXPECT_FALSE(backend.IsNodeRegistered(reporter_a.instance_id, shared_reporter));
    EXPECT_EQ((std::vector<bool>{false, false}), backend.MightExist({DataStorageUri(uri_a), DataStorageUri(uri_b)}));
}

TEST(EventReportBackendSnapshotTest, CloseUnblocksSnapshotAndDeltaWaiters) {
    {
        EventReportBackend backend(nullptr);
        backend.SetSnapshotMinIntervalMsForTest(0);
        const ReporterSnapshotKey reporter_key{"instance-a", "10.0.0.70:8080"};
        ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));
        std::string token;
        uint64_t retry_after_ms = 0;
        ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, token, retry_after_ms));
        ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, token));
        std::string committed;
        ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, committed));

        auto waiting_snapshot = std::async(std::launch::async, [&] {
            std::string candidate;
            uint64_t retry_ms = 0;
            return backend.BeginSnapshot(reporter_key, candidate, retry_ms);
        });
        std::string observed_committed;
        std::string observed_in_flight;
        const auto in_flight_deadline = std::chrono::steady_clock::now() + 1s;
        do {
            backend.GetSnapshotVersionTokens(reporter_key, observed_committed, observed_in_flight);
            if (!observed_in_flight.empty()) {
                break;
            }
            std::this_thread::yield();
        } while (std::chrono::steady_clock::now() < in_flight_deadline);
        ASSERT_FALSE(observed_in_flight.empty());
        ASSERT_EQ(std::future_status::timeout, waiting_snapshot.wait_for(0ms));
        ASSERT_EQ(EC_OK, backend.Close());
        ASSERT_EQ(std::future_status::ready, waiting_snapshot.wait_for(1s));
        EXPECT_EQ(EC_INSTANCE_NOT_EXIST, waiting_snapshot.get());
        backend.EndDeltaMutation(reporter_key);
    }

    {
        EventReportBackend backend(nullptr);
        backend.SetSnapshotMinIntervalMsForTest(0);
        const ReporterSnapshotKey reporter_key{"instance-b", "10.0.0.71:8080"};
        ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));
        std::string first;
        uint64_t retry_after_ms = 0;
        ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, first, retry_after_ms));
        ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, first));
        std::string second;
        ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, second, retry_after_ms));

        std::promise<void> delta_call_started;
        auto delta_call_started_future = delta_call_started.get_future();
        auto waiting_delta = std::async(std::launch::async, [&] {
            delta_call_started.set_value();
            std::string committed;
            return backend.BeginDeltaMutation(reporter_key, committed);
        });
        ASSERT_EQ(std::future_status::ready, delta_call_started_future.wait_for(1s));
        ASSERT_EQ(std::future_status::timeout, waiting_delta.wait_for(20ms));
        ASSERT_EQ(EC_OK, backend.Close());
        ASSERT_EQ(std::future_status::ready, waiting_delta.wait_for(1s));
        EXPECT_EQ(EC_INSTANCE_NOT_EXIST, waiting_delta.get());
    }
}

TEST_F(EventReportBackendTest, DisableWhileSnapshotDrainsAbortsCandidateAndReopensGate) {
    EventReportBackend backend(metrics_registry_);
    ASSERT_EQ(EC_OK,
              backend.Open(MakeConfig(/*hb*/ 5000,
                                      /*grace*/ 10000,
                                      /*tick*/ 60000,
                                      DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5,
                                      /*snapshot_min_interval*/ 0),
                           "disable_while_snapshot_drains"));
    backend.SetSnapshotMinIntervalMsForTest(0);
    const ReporterSnapshotKey reporter_key{"instance-disable", "10.0.0.72:8080"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter_key.instance_id, reporter_key.host_ip_port, {"mem"}));
    std::string first;
    uint64_t retry_after_ms = 0;
    ASSERT_EQ(EC_OK, BeginSnapshotForRegisteredReporter(backend, reporter_key, first, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter_key, first));
    std::string committed;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, committed));

    std::string candidate;
    auto waiting_snapshot = std::async(std::launch::async, [&] {
        uint64_t retry_ms = 0;
        return backend.BeginSnapshot(reporter_key, candidate, retry_ms);
    });
    std::string observed_committed;
    std::string observed_in_flight;
    const auto in_flight_deadline = std::chrono::steady_clock::now() + 1s;
    do {
        backend.GetSnapshotVersionTokens(reporter_key, observed_committed, observed_in_flight);
        if (!observed_in_flight.empty()) {
            break;
        }
        std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < in_flight_deadline);
    ASSERT_FALSE(observed_in_flight.empty());

    DataStorageBackend &backend_base = backend;
    backend_base.SetAvailable(false);
    ASSERT_EQ(std::future_status::ready, waiting_snapshot.wait_for(1s));
    EXPECT_EQ(EC_INSTANCE_NOT_EXIST, waiting_snapshot.get());
    EXPECT_TRUE(candidate.empty());

    backend_base.SetAvailable(true);
    backend.EndDeltaMutation(reporter_key);
    std::string after_reenable;
    ASSERT_EQ(EC_OK, backend.BeginDeltaMutation(reporter_key, after_reenable));
    EXPECT_EQ(first, after_reenable);
    backend.EndDeltaMutation(reporter_key);
    ASSERT_EQ(EC_OK, backend.Close());
}

TEST(EventReportBackendSnapshotTest, SnapshotUriUtilitiesHandleExactParameterBoundaries) {
    const std::string token = "00112233445566778899aabbccddeeff";
    const std::string raw_uri = "event_report://10.0.0.1:8080/mem?size=7&user_s_version=kept&s_version_hint=kept";
    EXPECT_EQ(0u, SnapshotUriUtils::CountUriParam(raw_uri, SnapshotUriUtils::kSnapshotVersionParam));

    std::string versioned_uri;
    ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(raw_uri, token, versioned_uri));
    EXPECT_EQ(1u, SnapshotUriUtils::CountUriParam(versioned_uri, SnapshotUriUtils::kSnapshotVersionParam));
    const DataStorageUri parsed(versioned_uri);
    ASSERT_TRUE(parsed.Valid());
    EXPECT_EQ("7", parsed.GetParam("size"));
    EXPECT_EQ("kept", parsed.GetParam("user_s_version"));
    EXPECT_EQ("kept", parsed.GetParam("s_version_hint"));
    EXPECT_EQ(token, parsed.GetParam(SnapshotUriUtils::kSnapshotVersionParam));

    SnapshotUriInfo info;
    info.version = "stale";
    EXPECT_FALSE(SnapshotUriUtils::ParseSnapshotUriInfo(raw_uri, info));
    EXPECT_TRUE(info.version.empty());
    EXPECT_FALSE(SnapshotUriUtils::ParseSnapshotUriInfo(
        "event_report://10.0.0.1:8080/mem?s_version=" + std::string(31, 'a'), info));
    EXPECT_FALSE(SnapshotUriUtils::ParseSnapshotUriInfo(
        "event_report://10.0.0.1:8080/mem?s_version=" + std::string(32, 'g'), info));
    EXPECT_FALSE(SnapshotUriUtils::ParseSnapshotUriInfo(versioned_uri + "&s_version=" + token, info));
    versioned_uri = "stale";
    EXPECT_FALSE(SnapshotUriUtils::AddSnapshotVersionToUri("", token, versioned_uri));
    EXPECT_TRUE(versioned_uri.empty());
    versioned_uri = "stale";
    EXPECT_FALSE(SnapshotUriUtils::AddSnapshotVersionToUri(raw_uri, std::string(31, 'a'), versioned_uri));
    EXPECT_TRUE(versioned_uri.empty());

    std::string storage_type = "stale";
    std::string medium = "stale";
    std::string reporter = "stale";
    EXPECT_FALSE(
        SnapshotUriUtils::ParseEventReportLocationId("kvs#event_report_l2##host:8080", storage_type, medium, reporter));
    EXPECT_TRUE(storage_type.empty());
    EXPECT_TRUE(medium.empty());
    EXPECT_TRUE(reporter.empty());
}
