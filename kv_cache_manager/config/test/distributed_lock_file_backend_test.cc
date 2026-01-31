#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

#include "kv_cache_manager/config/test/distributed_lock_backend_test_base.h"

namespace kv_cache_manager {

DistributedLockBackendTestConfig file_backend_config{
    .get_test_uri =
        [](DistributedLockBackendTest *test_base) {
            return "file://" + test_base->GetPrivateTestRuntimeDataPath() + "distributed_lock_test";
        },
    .set_up_ =
        [](DistributedLockBackendTest *test_base) {
            // 创建测试目录
            std::string test_dir = test_base->GetPrivateTestRuntimeDataPath() + "distributed_lock_test";
            std::error_code ec;
            std::filesystem::create_directories(test_dir, ec);
        },
    .tear_down_ =
        [](DistributedLockBackendTest *test_base) {
            // 清理测试目录
            std::error_code ec;
            std::string test_dir = test_base->GetPrivateTestRuntimeDataPath() + "distributed_lock_test";
            std::filesystem::remove_all(test_dir, ec);
        }};

INSTANTIATE_TEST_SUITE_P(DistributedLockBackendLocalTest,
                         DistributedLockBackendTest,
                         testing::Values(file_backend_config));

} // namespace kv_cache_manager