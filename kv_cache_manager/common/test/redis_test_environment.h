#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

namespace kv_cache_manager::redis_test {

inline constexpr int64_t kRedisClientDb = 0;
inline constexpr int64_t kRedisClientAlternateDb = 1;
inline constexpr int64_t kRegistryManagerDb = 2;
inline constexpr int64_t kRegistryManagerAlternateDb = 3;
inline constexpr int64_t kCoordinationDb = 4;
inline constexpr int64_t kCoordinationAlternateDb = 5;
inline constexpr int64_t kMetaRedisBackendDb = 6;
inline constexpr int64_t kMetaRedisBackendAlternateDb = 7;
inline constexpr int64_t kMetaIndexerDb = 8;
inline constexpr int64_t kMetaStorageBackendManagerDb = 9;
inline constexpr int64_t kMetaSearcherDb = 10;

inline std::string Host() {
    const char *host = std::getenv("KVCM_TEST_REDIS_HOST");
    return host != nullptr && host[0] != '\0' ? host : "localhost";
}

inline std::string Uri(const std::string &user_info, int64_t db, const std::string &additional_params = std::string()) {
    std::string uri = "redis://" + user_info + "@" + Host() + ":6379/?db=" + std::to_string(db);
    if (!additional_params.empty()) {
        uri += "&" + additional_params;
    }
    return uri;
}

} // namespace kv_cache_manager::redis_test
