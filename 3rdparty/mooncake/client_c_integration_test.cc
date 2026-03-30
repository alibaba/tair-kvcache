#include "client_c.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <string>
#include <thread>

namespace {

constexpr const char *kDefaultMetadataConnString = "P2PHANDSHAKE";
constexpr const char *kDefaultMasterEntry = "localhost:50051";
constexpr const char *kDefaultProtocol = "tcp";
constexpr const char *kRedisHaEntryEnv = "MOONCAKE_TEST_REDIS_HA_ENTRY";

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
                                        kDefaultMetadataConnString);
        master_entry_ = GetEnvOrDefault("MOONCAKE_TEST_MASTER_ENTRY",
                                        kDefaultMasterEntry);
        protocol_ = GetEnvOrDefault("MOONCAKE_TEST_PROTOCOL", kDefaultProtocol);
    }

protected:
    void ExpectStoreStatusCanBeFetched(const std::string &master_entry) {
        const std::string local_hostname = BuildLocalHostname();

        client_t client = mooncake_client_create(local_hostname.c_str(),
                                                 metadata_url_.c_str(),
                                                 protocol_.c_str(),
                                                 "",
                                                 master_entry.c_str());
        ASSERT_NE(nullptr, client)
            << "failed to create mooncake client, metadata_url="
            << metadata_url_ << ", master_entry=" << master_entry;

        MooncakeStoreStatus_t status{};
        const auto ec = mooncake_client_get_store_status(client, &status);

        mooncake_client_destroy(client);

        ASSERT_EQ(MOONCAKE_ERROR_OK, ec)
            << "failed to fetch store status from master_entry="
            << master_entry;
        EXPECT_TRUE(status.healthy);
        EXPECT_EQ(200, status.health_status_code);
        EXPECT_GE(status.total_capacity_bytes, status.allocated_bytes);
        EXPECT_GE(status.used_ratio, 0.0);
        EXPECT_LE(status.used_ratio, 1.0);
    }

    std::string metadata_url_;
    std::string master_entry_;
    std::string protocol_;
};

TEST_F(MooncakeClientCIntegrationTest, GetStoreStatusFromRealMaster) {
    ExpectStoreStatusCanBeFetched(master_entry_);
}

TEST_F(MooncakeClientCIntegrationTest, GetStoreStatusFromRedisHaMaster) {
    const char *redis_ha_entry = std::getenv(kRedisHaEntryEnv);
    if (redis_ha_entry == nullptr || redis_ha_entry[0] == '\0') {
        GTEST_SKIP() << kRedisHaEntryEnv
                     << " is not set; skipping Redis HA integration test";
    }

    ExpectStoreStatusCanBeFetched(redis_ha_entry);
}

} // namespace
