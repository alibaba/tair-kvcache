#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"

namespace kv_cache_manager {

class SnapshotUriUtilsTest : public ::testing::Test {};

TEST_F(SnapshotUriUtilsTest, InspectSnapshotUriForVisibilityReturnsBorrowedVersion) {
    constexpr std::string_view token = "0123456789abcdefABCDEF0123456789";
    const std::string uri = "event_report://10.0.0.1:8080/mem?rank=0&s_version=" + std::string(token) + "&size=4096";

    std::string_view version;
    ASSERT_TRUE(SnapshotUriUtils::InspectSnapshotUriForVisibility(uri, version));
    EXPECT_EQ(token, version);
    EXPECT_EQ(uri.data() + uri.find(token), version.data());
}

TEST_F(SnapshotUriUtilsTest, InspectSnapshotUriForVisibilityAcceptsLegacyAndUnrelatedParams) {
    const std::vector<std::string> uris = {
        "event_report://10.0.0.1:8080/mem",
        "event_report://10.0.0.1:8080/mem?rank=0&size=4096",
        "event_report://10.0.0.1:8080/mem?xs_version=ignored&s_version_suffix=ignored",
        "event_report://10.0.0.1:8080/mem?&rank=0&&",
    };

    for (const auto &uri : uris) {
        std::string_view version = "must be cleared";
        EXPECT_TRUE(SnapshotUriUtils::InspectSnapshotUriForVisibility(uri, version)) << uri;
        EXPECT_TRUE(version.empty()) << uri;
    }
}

TEST_F(SnapshotUriUtilsTest, InspectSnapshotUriForVisibilityRejectsMalformedUri) {
    const std::vector<std::string> uris = {
        "",
        "event_report:/10.0.0.1/mem",
        "://10.0.0.1/mem",
        "event?bad_report://10.0.0.1/mem",
    };

    for (const auto &uri : uris) {
        std::string_view version = "must be cleared";
        EXPECT_FALSE(SnapshotUriUtils::InspectSnapshotUriForVisibility(uri, version)) << uri;
        EXPECT_TRUE(version.empty()) << uri;
    }
}

TEST_F(SnapshotUriUtilsTest, InspectSnapshotUriForVisibilityRejectsInvalidOrDuplicateVersion) {
    constexpr const char *token = "0123456789abcdef0123456789abcdef";
    const std::vector<std::string> uris = {
        "event_report://host/mem?s_version",
        "event_report://host/mem?s_version=",
        "event_report://host/mem?s_version=0123456789abcdef0123456789abcde",
        "event_report://host/mem?s_version=0123456789abcdef0123456789abcdef0",
        "event_report://host/mem?s_version=0123456789abcdef0123456789abcdeg",
        std::string("event_report://host/mem?s_version=") + token + "&s_version=" + token,
        std::string("event_report://host/mem?s_version=") + token + "&s_version",
        std::string("event_report://host/mem?s_version&rank=0&s_version=") + token,
    };

    for (const auto &uri : uris) {
        std::string_view version = "must be cleared";
        EXPECT_FALSE(SnapshotUriUtils::InspectSnapshotUriForVisibility(uri, version)) << uri;
    }
}

TEST_F(SnapshotUriUtilsTest, InspectSnapshotUriForVisibilityAcceptsVersionAtParamBoundaries) {
    constexpr const char *token = "abcdef0123456789abcdef0123456789";
    const std::vector<std::string> uris = {
        std::string("event_report://host/mem?s_version=") + token,
        std::string("event_report://host/mem?&s_version=") + token,
        std::string("event_report://host/mem?rank=0&s_version=") + token + "&",
        std::string("event_report://host/mem?rank=0&s_version=") + token + "&size=1",
    };

    for (const auto &uri : uris) {
        std::string_view version;
        ASSERT_TRUE(SnapshotUriUtils::InspectSnapshotUriForVisibility(uri, version)) << uri;
        EXPECT_EQ(token, version) << uri;
    }
}

TEST_F(SnapshotUriUtilsTest, ParseEventReportLocationIdViewBorrowsComponents) {
    const std::string location_id = "kvs#event_report_l2#hbm-cache#10.0.0.1:8080";
    std::string_view storage_type;
    std::string_view medium;
    std::string_view host;
    ASSERT_TRUE(SnapshotUriUtils::ParseEventReportLocationIdView(location_id, storage_type, medium, host));
    EXPECT_EQ("event_report_l2", storage_type);
    EXPECT_EQ("hbm-cache", medium);
    EXPECT_EQ("10.0.0.1:8080", host);
    EXPECT_EQ(location_id.data() + location_id.find(storage_type), storage_type.data());
    EXPECT_EQ(location_id.data() + location_id.find(medium), medium.data());
    EXPECT_EQ(location_id.data() + location_id.find(host), host.data());
}

TEST_F(SnapshotUriUtilsTest, ParseEventReportLocationIdViewRejectsMalformedValuesAndClearsOutputs) {
    const std::vector<std::string> location_ids = {
        "",
        "kvs#",
        "bad#event_report_l2#mem#host:8080",
        "kvs#unknown#mem#host:8080",
        "kvs#event_report_l2##host:8080",
        "kvs#event_report_l2#mem#",
        "kvs#event_report_l2#mem#host:8080#extra",
    };
    for (const auto &location_id : location_ids) {
        std::string_view storage_type = "stale";
        std::string_view medium = "stale";
        std::string_view host = "stale";
        EXPECT_FALSE(SnapshotUriUtils::ParseEventReportLocationIdView(location_id, storage_type, medium, host))
            << location_id;
        EXPECT_TRUE(storage_type.empty()) << location_id;
        EXPECT_TRUE(medium.empty()) << location_id;
        EXPECT_TRUE(host.empty()) << location_id;
    }
}

} // namespace kv_cache_manager
