#include <gtest/gtest.h>
#include <map>
#include <string>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/vineyard_uri.h"
#include "rapidjson/document.h"

using namespace kv_cache_manager;

class VineyardUriTest : public TESTBASE {};

// (1) Build: deterministic ordering of query parameters (std::map alphabetic).
TEST_F(VineyardUriTest, BuildEmittsScheme_Host_Medium_Query) {
    EXPECT_EQ(VineyardUri::Build("192.168.1.100:8080", "mem"), "vineyard://192.168.1.100:8080/mem");
    EXPECT_EQ(VineyardUri::Build("192.168.1.100:8080", "disk", {{"gpu", "A100"}}),
              "vineyard://192.168.1.100:8080/disk?gpu=A100");
    // Multiple params: alphabetic order, joined by '&'.
    EXPECT_EQ(VineyardUri::Build("h:1", "mem", {{"b", "2"}, {"a", "1"}}), "vineyard://h:1/mem?a=1&b=2");
    // Empty medium -> no path segment.
    EXPECT_EQ(VineyardUri::Build("h:1", ""), "vineyard://h:1");
}

// (2) Parse: full vineyard URI round-trip.
TEST_F(VineyardUriTest, ParseExtractsHostMediumAndQuery) {
    std::string host;
    std::string medium;
    std::map<std::string, std::string> params;

    ASSERT_TRUE(VineyardUri::Parse("vineyard://10.0.0.1:8080/mem?gpu=A100&expire_at=1735689600", host, medium, params));
    EXPECT_EQ(host, "10.0.0.1:8080");
    EXPECT_EQ(medium, "mem");
    ASSERT_EQ(params.size(), 2u);
    EXPECT_EQ(params.at("gpu"), "A100");
    EXPECT_EQ(params.at("expire_at"), "1735689600");
}

TEST_F(VineyardUriTest, ParseHandlesNoQueryAndNoMedium) {
    std::string host;
    std::string medium;
    std::map<std::string, std::string> params;

    ASSERT_TRUE(VineyardUri::Parse("vineyard://10.0.0.1:8080/disk", host, medium, params));
    EXPECT_EQ(host, "10.0.0.1:8080");
    EXPECT_EQ(medium, "disk");
    EXPECT_TRUE(params.empty());

    // path absent -> medium is empty
    ASSERT_TRUE(VineyardUri::Parse("vineyard://10.0.0.1:8080", host, medium, params));
    EXPECT_EQ(host, "10.0.0.1:8080");
    EXPECT_TRUE(medium.empty());
    EXPECT_TRUE(params.empty());
}

TEST_F(VineyardUriTest, ParseRejectsWrongSchemeOrInvalid) {
    std::string host;
    std::string medium;
    std::map<std::string, std::string> params;

    EXPECT_FALSE(VineyardUri::Parse("3fs://h:1/foo", host, medium, params));
    EXPECT_FALSE(VineyardUri::Parse("not-a-uri", host, medium, params));
    EXPECT_FALSE(VineyardUri::Parse("", host, medium, params));
}

TEST_F(VineyardUriTest, ParseSurfacesValuelessQueryKey) {
    std::string host;
    std::string medium;
    std::map<std::string, std::string> params;

    ASSERT_TRUE(VineyardUri::Parse("vineyard://h:1/mem?flag&gpu=A100", host, medium, params));
    EXPECT_EQ(params.at("flag"), "");
    EXPECT_EQ(params.at("gpu"), "A100");
}

// (3) FromJson: legacy V6D shape -> standard URI.
TEST_F(VineyardUriTest, FromJsonProducesStandardUri) {
    const std::string uri =
        VineyardUri::FromJson(R"({"addr":"10.0.0.1:8080","type":"mem","gpu":"A100","expire_at":"1735689600"})");
    ASSERT_FALSE(uri.empty());

    // Round-trip back through Parse to verify all fields landed correctly.
    std::string host;
    std::string medium;
    std::map<std::string, std::string> params;
    ASSERT_TRUE(VineyardUri::Parse(uri, host, medium, params));
    EXPECT_EQ(host, "10.0.0.1:8080");
    EXPECT_EQ(medium, "mem");
    EXPECT_EQ(params.at("gpu"), "A100");
    EXPECT_EQ(params.at("expire_at"), "1735689600");
}

TEST_F(VineyardUriTest, FromJsonDefaultsMissingTypeToMem) {
    const std::string uri = VineyardUri::FromJson(R"({"addr":"h:1"})");
    ASSERT_FALSE(uri.empty());
    std::string host;
    std::string medium;
    std::map<std::string, std::string> params;
    ASSERT_TRUE(VineyardUri::Parse(uri, host, medium, params));
    EXPECT_EQ(medium, "mem"); // default for legacy payloads w/o `type`
}

TEST_F(VineyardUriTest, FromJsonRejectsMalformedInput) {
    EXPECT_TRUE(VineyardUri::FromJson("").empty());
    EXPECT_TRUE(VineyardUri::FromJson("not-json").empty());
    // Missing mandatory `addr` -> empty result.
    EXPECT_TRUE(VineyardUri::FromJson(R"({"type":"mem"})").empty());
}

TEST_F(VineyardUriTest, FromJsonCoercesNumericAndBoolValues) {
    const std::string uri = VineyardUri::FromJson(R"({"addr":"h:1","type":"disk","expire_at":1735689600,"flag":true})");
    ASSERT_FALSE(uri.empty());
    std::string host;
    std::string medium;
    std::map<std::string, std::string> params;
    ASSERT_TRUE(VineyardUri::Parse(uri, host, medium, params));
    EXPECT_EQ(params.at("expire_at"), "1735689600");
    EXPECT_EQ(params.at("flag"), "true");
}

// (4) ToJson: standard URI -> legacy V6D shape.
TEST_F(VineyardUriTest, ToJsonRoundTripWithFromJson) {
    const std::string original_json = R"({"addr":"10.0.0.1:8080","type":"mem","gpu":"A100","expire_at":"1735689600"})";
    const std::string uri = VineyardUri::FromJson(original_json);
    const std::string back = VineyardUri::ToJson(uri);

    rapidjson::Document doc;
    ASSERT_FALSE(doc.Parse(back.c_str()).HasParseError());
    EXPECT_STREQ(doc["addr"].GetString(), "10.0.0.1:8080");
    EXPECT_STREQ(doc["type"].GetString(), "mem");
    EXPECT_STREQ(doc["gpu"].GetString(), "A100");
    EXPECT_STREQ(doc["expire_at"].GetString(), "1735689600");
}

TEST_F(VineyardUriTest, ToJsonRejectsNonVineyardUri) {
    EXPECT_TRUE(VineyardUri::ToJson("3fs://h:1/foo").empty());
    EXPECT_TRUE(VineyardUri::ToJson("").empty());
    EXPECT_TRUE(VineyardUri::ToJson("invalid").empty());
}

TEST_F(VineyardUriTest, ToJsonOmitsEmptyParams) {
    const std::string uri = VineyardUri::Build("h:1", "mem");
    const std::string json = VineyardUri::ToJson(uri);
    rapidjson::Document doc;
    ASSERT_FALSE(doc.Parse(json.c_str()).HasParseError());
    EXPECT_STREQ(doc["addr"].GetString(), "h:1");
    EXPECT_STREQ(doc["type"].GetString(), "mem");
    // No extra keys beyond addr/type.
    EXPECT_EQ(doc.MemberCount(), 2u);
}
