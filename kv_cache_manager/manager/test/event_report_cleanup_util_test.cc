#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/event_report_backend.h"
#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"
#include "kv_cache_manager/data_storage/storage_config.h"
#include "kv_cache_manager/manager/event_report_cleanup_util.h"
#include "kv_cache_manager/meta/cache_location.h"

using namespace kv_cache_manager;

namespace {

StorageConfig MakeEventReportConfig() {
    auto spec = std::make_shared<EventReportStorageSpec>();
    spec->set_snapshot_min_interval_ms(0);
    return StorageConfig(
        DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L1P5, "event_report_cleanup_util_test", std::move(spec));
}

std::string VersionedUri(const std::string &version) {
    std::string uri;
    EXPECT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri("event_report://physical-cache:9600/mem", version, uri));
    return uri;
}

} // namespace

class EventReportCleanupUtilTest : public TESTBASE {};

TEST_F(EventReportCleanupUtilTest, MalformedSpecNeverAuthorizesLocationDeletion) {
    EventReportBackend backend(nullptr);
    ASSERT_EQ(EC_OK, backend.Open(MakeEventReportConfig(), "open"));

    const ReporterSnapshotKey reporter{"instance_a", "10.0.0.1:9000"};
    ASSERT_EQ(EC_OK, backend.RegisterNode(reporter.instance_id, reporter.host_ip_port, {"mem"}));

    uint64_t retry_after_ms = 0;
    std::string old_version;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter, old_version, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter, old_version));
    std::string committed_version;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter, committed_version, retry_after_ms));
    ASSERT_TRUE(backend.CommitSnapshotVersion(reporter, committed_version));

    const std::string committed_uri = VersionedUri(committed_version);
    const std::string malformed_uri = committed_uri + "&s_version=" + committed_version;
    const std::string location_id = backend.BuildLocationId("mem", reporter.host_ip_port);

    CacheLocation committed_then_malformed(
        location_id,
        CLS_SERVING,
        backend.GetStorageType(),
        2,
        {LocationSpec("current", committed_uri), LocationSpec("malformed", malformed_uri)});
    EXPECT_FALSE(IsSnapshotLocationStaleForCleanup(
        &backend, reporter.instance_id, committed_then_malformed, /*preserve_in_flight=*/true));

    CacheLocation malformed_only(
        location_id, CLS_SERVING, backend.GetStorageType(), 1, {LocationSpec("malformed", malformed_uri)});
    EXPECT_FALSE(
        IsSnapshotLocationStaleForCleanup(&backend, reporter.instance_id, malformed_only, /*preserve_in_flight=*/true));

    CacheLocation old_only(
        location_id, CLS_SERVING, backend.GetStorageType(), 1, {LocationSpec("old", VersionedUri(old_version))});
    EXPECT_TRUE(
        IsSnapshotLocationStaleForCleanup(&backend, reporter.instance_id, old_only, /*preserve_in_flight=*/true));

    std::string in_flight_version;
    ASSERT_EQ(EC_OK, backend.BeginSnapshot(reporter, in_flight_version, retry_after_ms));
    CacheLocation malformed_then_in_flight(
        location_id,
        CLS_SERVING,
        backend.GetStorageType(),
        2,
        {LocationSpec("malformed", malformed_uri), LocationSpec("in_flight", VersionedUri(in_flight_version))});
    EXPECT_FALSE(IsSnapshotLocationStaleForCleanup(
        &backend, reporter.instance_id, malformed_then_in_flight, /*preserve_in_flight=*/true));

    ASSERT_EQ(EC_OK, backend.Close());
}
