#include "client_c.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

constexpr const char *kDefaultMetadataUrl = "http://0.0.0.0:8090/metadata";
constexpr const char *kDefaultMasterEntry = "localhost:50051";
constexpr const char *kDefaultProtocol = "tcp";

std::string GetEnvOrDefault(const char *name, const char *default_value) {
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return value;
}

std::string BuildLocalHostname() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return "mooncake-client-c-integration-" + std::to_string(now);
}

class MooncakeClientCIntegrationTest : public ::testing::Test {
public:
    void SetUp() override {
        metadata_url_ = GetEnvOrDefault("MOONCAKE_TEST_METADATA_URL",
                                        kDefaultMetadataUrl);
        master_entry_ = GetEnvOrDefault("MOONCAKE_TEST_MASTER_ENTRY",
                                        kDefaultMasterEntry);
        protocol_ = GetEnvOrDefault("MOONCAKE_TEST_PROTOCOL", kDefaultProtocol);
    }

protected:
    std::string metadata_url_;
    std::string master_entry_;
    std::string protocol_;
};

TEST_F(MooncakeClientCIntegrationTest, GetStoreStatusFromRealMaster) {
    const std::string local_hostname = BuildLocalHostname();

    client_t client = mooncake_client_create(local_hostname.c_str(),
                                             metadata_url_.c_str(),
                                             protocol_.c_str(),
                                             "",
                                             master_entry_.c_str());
    ASSERT_NE(nullptr, client)
        << "failed to create mooncake client, metadata_url=" << metadata_url_
        << ", master_entry=" << master_entry_;

    MooncakeStoreStatus_t status{};
    const auto ec = mooncake_client_get_store_status(client, &status);

    mooncake_client_destroy(client);

    ASSERT_EQ(MOONCAKE_ERROR_OK, ec)
        << "failed to fetch store status from real master";
    EXPECT_TRUE(status.healthy);
    EXPECT_EQ(200, status.health_status_code);
    EXPECT_GE(status.total_capacity_bytes, status.allocated_bytes);
    EXPECT_GE(status.used_ratio, 0.0);
    EXPECT_LE(status.used_ratio, 1.0);
}

} // namespace
