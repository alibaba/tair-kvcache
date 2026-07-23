#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/event_report_backend.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

class EventReportBackendTest : public TESTBASE {
public:
    void SetUp() override { metrics_registry_ = std::make_shared<MetricsRegistry>(); }

    StorageConfig MakeConfig() {
        auto spec = std::make_shared<EventReportingStorageSpec>();
        return StorageConfig(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT, "event_report_test", std::move(spec));
    }

    std::shared_ptr<MetricsRegistry> metrics_registry_;
};

TEST_F(EventReportBackendTest, UsesDistinctMetadataOnlyIdentity) {
    EventReportBackend backend(metrics_registry_);
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT, backend.GetType());
    EXPECT_EQ(DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT, backend.GetStorageType());
    EXPECT_EQ("event-report", backend.GetProtocol());
    EXPECT_EQ("kvs#event#hbm#10.0.0.1:8080", backend.BuildLocationId("hbm", "10.0.0.1:8080"));

    ASSERT_EQ(EC_OK, backend.Open(MakeConfig(), "event_report_test"));
    ASSERT_EQ(EC_OK, backend.RegisterNode("instance", "10.0.0.1:8080", {"hbm"}));
    EXPECT_TRUE(backend.IsNodeAvailable("instance", "10.0.0.1:8080"));

    auto exists = backend.MightExist({DataStorageUri("event-report://10.0.0.1:8080/hbm")});
    ASSERT_EQ(1u, exists.size());
    EXPECT_TRUE(exists[0]);
    EXPECT_EQ(EC_OK, backend.UnregisterNode("instance", "10.0.0.1:8080"));
    EXPECT_FALSE(backend.IsNodeAvailable("instance", "10.0.0.1:8080"));
    EXPECT_EQ(EC_OK, backend.Close());
}
