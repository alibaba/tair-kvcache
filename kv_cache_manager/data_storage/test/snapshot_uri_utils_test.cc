#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"

namespace kv_cache_manager {

class SnapshotUriUtilsTest : public ::testing::Test {};

TEST_F(SnapshotUriUtilsTest, SnapshotVersionTokenUsesStrictAsciiHex) {
    EXPECT_TRUE(SnapshotUriUtils::IsValidSnapshotVersionToken("0123456789abcdefABCDEF0123456789"));
    EXPECT_FALSE(SnapshotUriUtils::IsValidSnapshotVersionToken("0123456789abcdefABCDEF012345678g"));
    EXPECT_FALSE(
        SnapshotUriUtils::IsValidSnapshotVersionToken(std::string("0123456789abcdefABCDEF01234567") + "\xc3\xa9"));
    EXPECT_FALSE(SnapshotUriUtils::IsValidSnapshotVersionToken("0123456789abcdefABCDEF012345678"));
}

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
        "event_report://host:0/mem",
        "event_report://host:08080/mem",
        "event_report://host:9223372036854775807/mem",
        "event_report://user:secret@host:8080/mem",
        "event_report://host/path:with:colon?callback=http://peer:9000/path",
        "event_report://10.0.0.1:8080/mem?rank=0&size=4096",
        "event_report://10.0.0.1:8080/mem?xs_version=ignored&s_version_suffix=ignored",
        "event_report://10.0.0.1:8080/mem?&rank=0&&",
    };

    for (const auto &uri : uris) {
        std::string_view version = "must be cleared";
        EXPECT_TRUE(SnapshotUriUtils::InspectSnapshotUriForVisibility(uri, version)) << uri;
        EXPECT_TRUE(version.empty()) << uri;
        version = "must be cleared";
        EXPECT_TRUE(SnapshotUriUtils::InspectSnapshotUriForVisibility(uri, version, true)) << uri;
        EXPECT_TRUE(version.empty()) << uri;
    }
}

TEST_F(SnapshotUriUtilsTest, InspectSnapshotUriForVisibilityRejectsMalformedUri) {
    const std::vector<std::string> uris = {
        "",
        "event_report:/10.0.0.1/mem",
        "://10.0.0.1/mem",
        "event?bad_report://10.0.0.1/mem",
        "event_report://host:/mem",
        "event_report://host:-0/mem",
        "event_report://host:-1/mem",
        "event_report://host:+1/mem",
        "event_report://host:not-a-port/mem",
        "event_report://host:9223372036854775808/mem",
        "event_report://user:secret@host:not-a-port/mem",
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
        version = {};
        ASSERT_TRUE(SnapshotUriUtils::InspectSnapshotUriForVisibility(uri, version, true)) << uri;
        EXPECT_EQ(token, version) << uri;
    }
}

TEST_F(SnapshotUriUtilsTest, PrevalidatedStructureStillRejectsProtocolQueryAndBadVersionMetadata) {
    std::string_view version = "must be cleared";
    EXPECT_FALSE(SnapshotUriUtils::InspectSnapshotUriForVisibility("event?bad_report://host/mem", version, true));
    EXPECT_TRUE(version.empty());

    EXPECT_FALSE(SnapshotUriUtils::InspectSnapshotUriForVisibility(
        "event_report://host/mem?s_version=not-a-version", version, true));
}

TEST_F(SnapshotUriUtilsTest, AddSnapshotVersionRejectsInvalidPortWithoutCanonicalizingItAway) {
    constexpr const char *token = "0123456789abcdef0123456789abcdef";
    for (const std::string &invalid_uri : {
             "event_report://physical-cache:not-a-port/mem?size=1",
             "event_report://physical-cache:-1/mem?size=1",
             "event_report://physical-cache:9223372036854775808/mem?size=1",
         }) {
        std::string out_uri = "stale";
        EXPECT_FALSE(SnapshotUriUtils::AddSnapshotVersionToUri(invalid_uri, token, out_uri));
        EXPECT_TRUE(out_uri.empty());
    }
}

TEST_F(SnapshotUriUtilsTest, CanonicalSnapshotAppendMatchesStandardUriWithoutAllocatingParseState) {
    constexpr const char *token = "0123456789abcdef0123456789abcdef";
    const std::vector<std::pair<std::string, std::uint64_t>> cases = {
        {"event_report://host/mem", 0},
        {"event_report://user@host:8080/mem?a=1&size=4096&z=last", 4096},
        {"event_report://host:9223372036854775807/mem?size=18446744073709551615",
         std::numeric_limits<std::uint64_t>::max()},
        {"event_report://host/mem?a=1&s=before&size=7", 7},
        {"event_report://host/mem?size=invalid&z=last", 0},
        {"event_report://host/mem?size=18446744073709551616&z=last", 0},
        {"scheme://host?empty=&rank=0", 0},
    };

    for (const auto &[uri, expected_size] : cases) {
        CanonicalSnapshotUriAppendInfo info;
        ASSERT_TRUE(SnapshotUriUtils::ParseCanonicalUriForSnapshotAppend(uri, info)) << uri;
        EXPECT_EQ(expected_size, info.size) << uri;

        std::string fast_uri;
        ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToCanonicalUri(uri, info, token, fast_uri)) << uri;
        std::string prevalidated_uri;
        ASSERT_TRUE(SnapshotUriUtils::AddPrevalidatedSnapshotVersionToCanonicalUri(uri, info, token, prevalidated_uri))
            << uri;
        EXPECT_EQ(fast_uri, prevalidated_uri) << uri;
        DataStorageUri parsed(uri);
        ASSERT_TRUE(parsed.Valid()) << uri;
        std::string standard_uri;
        ASSERT_TRUE(SnapshotUriUtils::AddSnapshotVersionToUri(std::move(parsed), token, standard_uri)) << uri;
        EXPECT_EQ(standard_uri, fast_uri) << uri;
    }
}

TEST_F(SnapshotUriUtilsTest, CanonicalSnapshotAppendDefersNoncanonicalUrisToStandardParser) {
    const std::vector<std::string> noncanonical_uris = {
        "event_report://host:08080/mem?size=1",
        "event_report://host:0/mem?size=1",
        "event_report://@host/mem?size=1",
        "event_report://host/mem?",
        "event_report://host/mem?size=1&",
        "event_report://host/mem?size",
        "event_report://host/mem?z=1&a=2",
        "event_report://host/mem?a=1&a=2",
        "event_report://host/mem?s_version=0123456789abcdef0123456789abcdef",
        "event_report://host?query=before/path",
    };
    for (const auto &uri : noncanonical_uris) {
        CanonicalSnapshotUriAppendInfo info;
        EXPECT_FALSE(SnapshotUriUtils::ParseCanonicalUriForSnapshotAppend(uri, info)) << uri;
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
