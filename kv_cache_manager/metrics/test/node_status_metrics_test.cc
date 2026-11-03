#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "kv_cache_manager/metrics/node_status_metrics.h"
#include "kv_cache_manager/metrics/prometheus_exporter.h"

using namespace kv_cache_manager;

namespace {

// Read and write capability are driven independently so a test can express the
// read-only degradation TairMempool reports, which NfsBackend cannot because
// its Writable() just forwards Available(). The data plane methods are never
// called by the metrics path; they only satisfy the abstract interface.
class CapabilityBackend : public DataStorageBackend {
public:
    explicit CapabilityBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : DataStorageBackend(std::move(metrics_registry)) {}
    ~CapabilityBackend() override = default;

    DataStorageType GetType() override { return DataStorageType::DATA_STORAGE_TYPE_DUMMY; }
    bool Available() override { return available_; }
    bool Writable() override { return writable_; }
    double GetStorageUsageRatio(const std::string & /*trace_id*/) const override { return 0.0; }

    void SetCapability(bool available, bool writable) {
        available_ = available;
        writable_ = writable;
    }

    ErrorCode DoOpen(const StorageConfig & /*config*/, const std::string & /*trace_id*/) override { return EC_OK; }
    ErrorCode Close() override { return EC_OK; }
    std::vector<std::pair<ErrorCode, DataStorageUri>> Create(const std::vector<std::string> &keys,
                                                             size_t /*size_per_key*/,
                                                             const std::string & /*trace_id*/,
                                                             std::function<void()> /*cb*/) override {
        return std::vector<std::pair<ErrorCode, DataStorageUri>>(keys.size());
    }
    std::vector<ErrorCode> Delete(const std::vector<DataStorageUri> &storage_uris,
                                  const std::string & /*trace_id*/,
                                  std::function<void()> /*cb*/) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }
    std::vector<bool> Exist(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<bool>(storage_uris.size(), true);
    }
    std::vector<ErrorCode> Lock(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }
    std::vector<ErrorCode> UnLock(const std::vector<DataStorageUri> &storage_uris) override {
        return std::vector<ErrorCode>(storage_uris.size(), EC_OK);
    }

private:
    bool available_ = false;
    bool writable_ = false;
};

} // namespace

class NodeStatusMetricsTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        data_storage_manager_ = std::make_shared<DataStorageManager>(metrics_registry_);
    }

    void TearDown() override {}

    std::shared_ptr<CapabilityBackend> AddBackend(const std::string &unique_name, bool available, bool writable) {
        auto backend = std::make_shared<CapabilityBackend>(metrics_registry_);
        backend->SetCapability(available, writable);
        data_storage_manager_->storage_map_.emplace(unique_name, backend);
        return backend;
    }

    double GetGaugeValue(const std::string &name) { return metrics_registry_->GetGauge(name).Get(); }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::shared_ptr<DataStorageManager> data_storage_manager_;
};

TEST_F(NodeStatusMetricsTest, TestComputeAccessModeWithoutStorage) {
    EXPECT_EQ(NodeAccessMode::NONE, NodeStatusMetrics::ComputeAccessMode(nullptr));
    EXPECT_EQ(NodeAccessMode::NONE, NodeStatusMetrics::ComputeAccessMode(data_storage_manager_));
}

TEST_F(NodeStatusMetricsTest, TestComputeAccessModeSingleBackend) {
    auto backend = AddBackend("ro_or_rw", false, false);
    EXPECT_EQ(NodeAccessMode::NONE, NodeStatusMetrics::ComputeAccessMode(data_storage_manager_));

    backend->SetCapability(true, true);
    EXPECT_EQ(NodeAccessMode::READ_WRITE, NodeStatusMetrics::ComputeAccessMode(data_storage_manager_));

    // read-only degradation: reads still served, writes rejected
    backend->SetCapability(true, false);
    EXPECT_EQ(NodeAccessMode::READ_ONLY, NodeStatusMetrics::ComputeAccessMode(data_storage_manager_));

    backend->SetCapability(false, false);
    EXPECT_EQ(NodeAccessMode::NONE, NodeStatusMetrics::ComputeAccessMode(data_storage_manager_));
}

TEST_F(NodeStatusMetricsTest, TestComputeAccessModeAcrossBackends) {
    // one healthy backend is enough to serve the node
    AddBackend("down", false, false);
    auto up = AddBackend("up", true, false);
    EXPECT_EQ(NodeAccessMode::READ_ONLY, NodeStatusMetrics::ComputeAccessMode(data_storage_manager_));

    up->SetCapability(true, true);
    EXPECT_EQ(NodeAccessMode::READ_WRITE, NodeStatusMetrics::ComputeAccessMode(data_storage_manager_));

    up->SetCapability(false, false);
    EXPECT_EQ(NodeAccessMode::NONE, NodeStatusMetrics::ComputeAccessMode(data_storage_manager_));
}

TEST_F(NodeStatusMetricsTest, TestPublishFollowerIgnoresStorage) {
    AddBackend("healthy", true, true);

    NodeStatusMetrics::Publish(metrics_registry_, false, data_storage_manager_);
    EXPECT_DOUBLE_EQ(0.0, GetGaugeValue(NodeStatusMetrics::kIsLeaderName));
    EXPECT_DOUBLE_EQ(0.0, GetGaugeValue(NodeStatusMetrics::kAccessModeName));
}

TEST_F(NodeStatusMetricsTest, TestPublishLeaderTracksStorage) {
    auto backend = AddBackend("storage", true, true);

    NodeStatusMetrics::Publish(metrics_registry_, true, data_storage_manager_);
    EXPECT_DOUBLE_EQ(1.0, GetGaugeValue(NodeStatusMetrics::kIsLeaderName));
    EXPECT_DOUBLE_EQ(2.0, GetGaugeValue(NodeStatusMetrics::kAccessModeName));

    backend->SetCapability(true, false);
    NodeStatusMetrics::Publish(metrics_registry_, true, data_storage_manager_);
    EXPECT_DOUBLE_EQ(1.0, GetGaugeValue(NodeStatusMetrics::kIsLeaderName));
    EXPECT_DOUBLE_EQ(1.0, GetGaugeValue(NodeStatusMetrics::kAccessModeName));

    backend->SetCapability(false, false);
    NodeStatusMetrics::Publish(metrics_registry_, true, data_storage_manager_);
    EXPECT_DOUBLE_EQ(1.0, GetGaugeValue(NodeStatusMetrics::kIsLeaderName));
    EXPECT_DOUBLE_EQ(0.0, GetGaugeValue(NodeStatusMetrics::kAccessModeName));
}

TEST_F(NodeStatusMetricsTest, TestPublishLeaderWithoutStorageManager) {
    NodeStatusMetrics::Publish(metrics_registry_, true, nullptr);
    EXPECT_DOUBLE_EQ(1.0, GetGaugeValue(NodeStatusMetrics::kIsLeaderName));
    EXPECT_DOUBLE_EQ(0.0, GetGaugeValue(NodeStatusMetrics::kAccessModeName));
}

TEST_F(NodeStatusMetricsTest, TestPublishWithNullRegistry) {
    EXPECT_NO_FATAL_FAILURE(NodeStatusMetrics::Publish(nullptr, true, data_storage_manager_));
}

TEST_F(NodeStatusMetricsTest, TestPrometheusExposure) {
    AddBackend("storage", true, true);
    NodeStatusMetrics::Publish(metrics_registry_, true, data_storage_manager_);

    const auto output = PrometheusExporter::Expose(*metrics_registry_);
    EXPECT_NE(std::string::npos, output.find("# TYPE kvcm_is_leader gauge"));
    EXPECT_NE(std::string::npos, output.find("kvcm_is_leader 1"));
    EXPECT_NE(std::string::npos, output.find("# TYPE kvcm_access_mode gauge"));
    EXPECT_NE(std::string::npos, output.find("kvcm_access_mode 2"));
}
