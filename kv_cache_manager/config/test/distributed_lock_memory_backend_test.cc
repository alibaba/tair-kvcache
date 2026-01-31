#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/config/test/distributed_lock_backend_test_base.h"

namespace kv_cache_manager {

DistributedLockBackendTestConfig memory_backend_config{
    .get_test_uri = [](DistributedLockBackendTest *test_base) { return "memory://"; },
    .set_up_ = [](DistributedLockBackendTest *test_base) {},
    .tear_down_ = [](DistributedLockBackendTest *test_base) {}};

INSTANTIATE_TEST_SUITE_P(DistributedLockBackendMemoryTest,
                         DistributedLockBackendTest,
                         testing::Values(memory_backend_config));

} // namespace kv_cache_manager