#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/common/redis_client_ext.h"
#include "kv_cache_manager/config/test/distributed_lock_backend_test_base.h"

namespace kv_cache_manager {

const std::string kRedisUri = "redis://test_redis_user:test_redis_password@localhost:6379/"
                              "?timeout_ms=1000&retry_count=3&client_max_pool_size=2";

DistributedLockBackendTestConfig redis_backend_config{
    .get_test_uri = [](DistributedLockBackendTest *test_base) { return kRedisUri; },
    .set_up_ =
        [](DistributedLockBackendTest *test_base) {
            RedisClientExt client(StandardUri::FromUri(kRedisUri));
            client.Open();
            client.FlushAll();
            client.Close();
        },
    .tear_down_ =
        [](DistributedLockBackendTest *test_base) {
            RedisClientExt client(StandardUri::FromUri(kRedisUri));
            client.Open();
            client.FlushAll();
            client.Close();
        }};

INSTANTIATE_TEST_SUITE_P(DistributedLockBackendRedisTest,
                         DistributedLockBackendTest,
                         testing::Values(redis_backend_config));

} // namespace kv_cache_manager