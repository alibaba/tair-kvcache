#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/config/test/coordination_backend_test_base.h"

namespace kv_cache_manager {

CoordinationBackendTestConfig memory_backend_config{
    .get_test_uri = [](CoordinationBackendTest *test_base) { return "memory://"; },
    .set_up_ = [](CoordinationBackendTest *test_base) {},
    .tear_down_ = [](CoordinationBackendTest *test_base) {}};

INSTANTIATE_TEST_SUITE_P(CoordinationBackendMemoryTest,
                         CoordinationBackendTest,
                         testing::Values(memory_backend_config));

} // namespace kv_cache_manager