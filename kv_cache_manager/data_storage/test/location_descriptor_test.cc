// Verify DataStorageBackend default implementations:
//   1. CreateWithHints wraps legacy Create into vector<LocationDescriptor>
//      with empty node_id (non affinity-aware backend).
//   2. SnapshotPerNodeMetrics returns empty vector.
// Uses DummyBackend to avoid backend-specific configuration.

#include <filesystem>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_backend.h"
#include "kv_cache_manager/data_storage/dummy_backend.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

class LocationDescriptorTest : public TESTBASE {
public:
    void SetUp() override {
        metrics_registry_ = std::make_shared<MetricsRegistry>();
        test_root_ = GetPrivateTestRuntimeDataPath() + "loc_desc_root/";
    }
    void TearDown() override {}

    StorageConfig MakeConfig() {
        auto spec = std::make_shared<DummyStorageSpec>();
        spec->set_root_path(test_root_);
        spec->set_key_count_per_file(1);
        return StorageConfig(DataStorageType::DATA_STORAGE_TYPE_DUMMY, "test_dummy", spec);
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
    std::string test_root_;
};

TEST_F(LocationDescriptorTest, StructHasZeroDefaults) {
    LocationDescriptor d;
    EXPECT_EQ(EC_OK, d.ec);
    EXPECT_EQ("", d.node_id);
}

TEST_F(LocationDescriptorTest, DefaultCreateWithHintsWrapsLegacyCreate) {
    DummyBackend backend(metrics_registry_);
    auto config = MakeConfig();
    ASSERT_EQ(EC_OK, backend.Open(config, "trace_open"));

    std::vector<std::string> keys = {"alpha", "beta"};
    WriteHints hints;
    auto descs = backend.CreateWithHints(keys, 64, hints, /*strict=*/false, "trace_write", []() {});

    ASSERT_EQ(keys.size(), descs.size());
    for (auto &d : descs) {
        EXPECT_EQ(EC_OK, d.ec);
        // 老 backend 默认实现：node_id 保持空串
        EXPECT_EQ("", d.node_id);
        EXPECT_EQ("dummy", d.uri.GetProtocol());
    }
    EXPECT_NE(std::string::npos, descs[0].uri.ToUriString().find("alpha"));
    EXPECT_NE(std::string::npos, descs[1].uri.ToUriString().find("beta"));

    ASSERT_EQ(EC_OK, backend.Close());
}

TEST_F(LocationDescriptorTest, DefaultSnapshotPerNodeMetricsReturnsEmpty) {
    DummyBackend backend(metrics_registry_);
    auto snap = backend.SnapshotPerNodeMetrics();
    EXPECT_TRUE(snap.empty());
}

TEST_F(LocationDescriptorTest, SupportsAffinityDefaultsFalse) {
    DummyBackend backend(metrics_registry_);
    EXPECT_FALSE(backend.SupportsAffinity());
}

} // namespace kv_cache_manager
