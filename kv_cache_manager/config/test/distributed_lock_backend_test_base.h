#include <chrono>
#include <filesystem>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/config/distributed_lock_backend.h"
#include "kv_cache_manager/config/distributed_lock_backend_factory.h"

namespace kv_cache_manager {
class DistributedLockBackendTest;
struct DistributedLockBackendTestConfig {
    std::function<std::string(DistributedLockBackendTest *)> get_test_uri;
    std::function<void(DistributedLockBackendTest *test_base)> set_up_;
    std::function<void(DistributedLockBackendTest *test_base)> tear_down_;
};

class DistributedLockBackendTest : public TESTBASE,
                                   public testing::WithParamInterface<DistributedLockBackendTestConfig> {
protected:
    void SetUp() override {
        GetParam().set_up_(this);
        // 创建锁后端实例
        backend_ = DistributedLockBackendFactory::CreateAndInitDistributedLockBackend(GetTestUri());
        EXPECT_NE(backend_, nullptr);
    }

    void TearDown() override { GetParam().tear_down_(this); }

    // 获取测试用的URI
    std::string GetTestUri() { return GetParam().get_test_uri(this); }

protected:
    std::string test_dir_;
    std::unique_ptr<DistributedLockBackend> backend_;
};

// 测试初始化
TEST_P(DistributedLockBackendTest, TestInit) {
    // 测试重复初始化（应该也成功）
    StandardUri uri = StandardUri::FromUri(this->GetTestUri());
    ErrorCode ec = backend_->Init(uri);
    EXPECT_EQ(EC_OK, ec);
}

// 测试TryLock基本功能
TEST_P(DistributedLockBackendTest, TestTryLockBasic) {

    const std::string lock_key = "test_lock";
    const std::string lock_value = "value1";
    const int64_t ttl_ms = 1000; // 1秒

    // 第一次获取锁应该成功
    ErrorCode ec = backend_->TryLock(lock_key, lock_value, ttl_ms);
    EXPECT_EQ(EC_OK, ec);

    // 用相同的值再次获取锁（应该成功，锁已持有）
    ec = backend_->TryLock(lock_key, lock_value, ttl_ms);
    EXPECT_EQ(EC_OK, ec);

    // 用不同的值再次获取锁（应该失败，锁已存在）
    ec = backend_->TryLock(lock_key, "value2", ttl_ms);
    EXPECT_EQ(EC_EXIST, ec);
}

// 测试TryLock过期锁
TEST_P(DistributedLockBackendTest, TestTryLockExpired) {
    const std::string lock_key = "test_lock";
    const std::string lock_value1 = "value1";
    const std::string lock_value2 = "value2";
    const int64_t short_ttl_ms = 100; // 100毫秒

    // 获取一个短期锁
    ErrorCode ec = backend_->TryLock(lock_key, lock_value1, short_ttl_ms);
    EXPECT_EQ(EC_OK, ec);

    // 立即尝试获取锁（应该失败，锁未过期）
    ec = backend_->TryLock(lock_key, lock_value2, short_ttl_ms);
    EXPECT_EQ(EC_EXIST, ec);

    // 等待锁过期
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // 再次尝试获取锁（应该成功，锁已过期）
    ec = backend_->TryLock(lock_key, lock_value2, short_ttl_ms);
    EXPECT_EQ(EC_OK, ec);
}

// 测试RenewLock基本功能
TEST_P(DistributedLockBackendTest, TestRenewLockBasic) {
    const std::string lock_key = "test_lock";
    const std::string lock_value = "value1";
    const int64_t ttl_ms = 1000;

    // 先获取锁
    ErrorCode ec = backend_->TryLock(lock_key, lock_value, ttl_ms);
    ASSERT_EQ(EC_OK, ec);

    // 续约锁（应该成功）
    ec = backend_->RenewLock(lock_key, lock_value, ttl_ms * 2);
    EXPECT_EQ(EC_OK, ec);

    // 用错误的值续约（应该失败）
    ec = backend_->RenewLock(lock_key, "wrong_value", ttl_ms);
    EXPECT_EQ(EC_MISMATCH, ec);

    // 续约不存在的锁（应该失败）
    ec = backend_->RenewLock("nonexistent_lock", lock_value, ttl_ms);
    EXPECT_EQ(EC_NOENT, ec);
}

// 测试RenewLock过期锁
TEST_P(DistributedLockBackendTest, TestRenewLockExpired) {
    const std::string lock_key = "test_lock";
    const std::string lock_value = "value1";
    const int64_t short_ttl_ms = 100; // 100毫秒

    // 获取一个短期锁
    ErrorCode ec = backend_->TryLock(lock_key, lock_value, short_ttl_ms);
    ASSERT_EQ(EC_OK, ec);

    // 等待锁过期
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // 尝试续约已过期的锁（应该失败）
    ec = backend_->RenewLock(lock_key, lock_value, short_ttl_ms);
    EXPECT_EQ(EC_NOENT, ec);
}

// 测试Unlock基本功能
TEST_P(DistributedLockBackendTest, TestUnlockBasic) {
    const std::string lock_key = "test_lock";
    const std::string lock_value = "value1";
    const int64_t ttl_ms = 1000;

    // 先获取锁
    ErrorCode ec = backend_->TryLock(lock_key, lock_value, ttl_ms);
    ASSERT_EQ(EC_OK, ec);

    // 用正确的值释放锁（应该成功）
    ec = backend_->Unlock(lock_key, lock_value);
    EXPECT_EQ(EC_OK, ec);

    // 再次释放同一个锁（应该失败，锁不存在）
    ec = backend_->Unlock(lock_key, lock_value);
    EXPECT_EQ(EC_NOENT, ec);

    // 获取锁并再次测试
    ec = backend_->TryLock(lock_key, lock_value, ttl_ms);
    ASSERT_EQ(EC_OK, ec);

    // 用错误的值释放锁（应该失败）
    ec = backend_->Unlock(lock_key, "wrong_value");
    EXPECT_EQ(EC_MISMATCH, ec);
}

// 测试Unlock不存在的锁
TEST_P(DistributedLockBackendTest, TestUnlockNonexistent) {
    // 释放不存在的锁（应该失败）
    ErrorCode ec = backend_->Unlock("nonexistent_lock", "value");
    EXPECT_EQ(EC_NOENT, ec);
}

// 测试GetLockHolder基本功能
TEST_P(DistributedLockBackendTest, TestGetLockHolderBasic) {
    const std::string lock_key = "test_lock";
    const std::string lock_value = "value1";
    const int64_t ttl_ms = 1000;

    // 获取锁前检查（应该不存在）
    std::string current_value;
    int64_t expire_time_ms;
    ErrorCode ec = backend_->GetLockHolder(lock_key, current_value, expire_time_ms);
    EXPECT_EQ(EC_NOENT, ec);

    // 获取锁
    ec = backend_->TryLock(lock_key, lock_value, ttl_ms);
    ASSERT_EQ(EC_OK, ec);

    // 获取锁持有者（应该成功）
    current_value.clear();
    ec = backend_->GetLockHolder(lock_key, current_value, expire_time_ms);
    EXPECT_EQ(EC_OK, ec);
    EXPECT_EQ(lock_value, current_value);

    // 释放锁
    ec = backend_->Unlock(lock_key, lock_value);
    ASSERT_EQ(EC_OK, ec);

    // 再次检查（应该不存在）
    ec = backend_->GetLockHolder(lock_key, current_value, expire_time_ms);
    EXPECT_EQ(EC_NOENT, ec);
}

// 测试GetLockHolder过期锁
TEST_P(DistributedLockBackendTest, TestGetLockHolderExpired) {
    const std::string lock_key = "test_lock";
    const std::string lock_value = "value1";
    const int64_t short_ttl_ms = 100; // 100毫秒

    // 获取一个短期锁
    ErrorCode ec = backend_->TryLock(lock_key, lock_value, short_ttl_ms);
    ASSERT_EQ(EC_OK, ec);

    // 立即检查（应该存在）
    std::string current_value;
    int64_t expire_time_ms;
    ec = backend_->GetLockHolder(lock_key, current_value, expire_time_ms);
    EXPECT_EQ(EC_OK, ec);
    EXPECT_EQ(lock_value, current_value);

    // 等待锁过期
    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    // 再次检查（应该不存在）
    current_value.clear();
    ec = backend_->GetLockHolder(lock_key, current_value, expire_time_ms);
    EXPECT_EQ(EC_NOENT, ec);
}

// 测试多个不同的锁键
TEST_P(DistributedLockBackendTest, TestMultipleLockKeys) {
    const std::string lock_key1 = "lock1";
    const std::string lock_key2 = "lock2";
    const std::string value1 = "value1";
    const std::string value2 = "value2";
    const int64_t ttl_ms = 1000;

    // 获取第一个锁
    ErrorCode ec = backend_->TryLock(lock_key1, value1, ttl_ms);
    EXPECT_EQ(EC_OK, ec);

    // 获取第二个锁（应该成功，不同的锁键）
    ec = backend_->TryLock(lock_key2, value2, ttl_ms);
    EXPECT_EQ(EC_OK, ec);

    // 验证第一个锁仍然存在
    ec = backend_->TryLock(lock_key1, "other_value", ttl_ms);
    EXPECT_EQ(EC_EXIST, ec);

    // 验证第二个锁仍然存在
    ec = backend_->TryLock(lock_key2, "other_value", ttl_ms);
    EXPECT_EQ(EC_EXIST, ec);

    // 释放第一个锁
    ec = backend_->Unlock(lock_key1, value1);
    EXPECT_EQ(EC_OK, ec);

    // 释放第二个锁
    ec = backend_->Unlock(lock_key2, value2);
    EXPECT_EQ(EC_OK, ec);
}

// 测试特殊字符的锁键
TEST_P(DistributedLockBackendTest, TestSpecialLockKey) {
    // 包含特殊字符的锁键
    const std::string lock_key = "test/lock/key\\with/special@chars";
    const std::string lock_value = "value";
    const int64_t ttl_ms = 1000;

    // 获取锁（应该成功，特殊字符会被处理）
    ErrorCode ec = backend_->TryLock(lock_key, lock_value, ttl_ms);
    EXPECT_EQ(EC_OK, ec);

    // 验证锁存在
    ec = backend_->TryLock(lock_key, "other_value", ttl_ms);
    EXPECT_EQ(EC_EXIST, ec);

    // 释放锁
    ec = backend_->Unlock(lock_key, lock_value);
    EXPECT_EQ(EC_OK, ec);
}

// 测试并发场景（简单版本）
TEST_P(DistributedLockBackendTest, TestConcurrentAccess) {
    const std::string lock_key = "concurrent_lock";
    const std::string lock_value = "value";
    const int64_t ttl_ms = 1000;

    // 线程1：获取锁
    std::thread thread1([&]() {
        ErrorCode ec_local = backend_->TryLock(lock_key, lock_value, ttl_ms);
        EXPECT_EQ(EC_OK, ec_local);
    });

    thread1.join();

    // 线程2：尝试获取锁（应该失败）
    std::thread thread2([&]() {
        ErrorCode ec_local = backend_->TryLock(lock_key, "other_value", ttl_ms);
        EXPECT_EQ(EC_EXIST, ec_local);
    });

    thread2.join();

    // 释放锁
    ErrorCode ec = backend_->Unlock(lock_key, lock_value);
    EXPECT_EQ(EC_OK, ec);
}

// 测试完整的锁生命周期
TEST_P(DistributedLockBackendTest, TestFullLockLifecycle) {
    const std::string lock_key = "lifecycle_lock";
    const std::string lock_value = "process_123";
    const int64_t ttl_ms = 5000;

    // 阶段1：获取锁
    ErrorCode ec = backend_->TryLock(lock_key, lock_value, ttl_ms);
    ASSERT_EQ(EC_OK, ec);

    // 阶段2：验证锁持有者
    std::string current_value;
    int64_t expire_time_ms;
    ec = backend_->GetLockHolder(lock_key, current_value, expire_time_ms);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(lock_value, current_value);

    // 阶段3：续约锁
    ec = backend_->RenewLock(lock_key, lock_value, ttl_ms * 2);
    ASSERT_EQ(EC_OK, ec);

    // 阶段4：再次验证锁持有者
    current_value.clear();
    ec = backend_->GetLockHolder(lock_key, current_value, expire_time_ms);
    ASSERT_EQ(EC_OK, ec);
    ASSERT_EQ(lock_value, current_value);

    // 阶段5：释放锁
    ec = backend_->Unlock(lock_key, lock_value);
    ASSERT_EQ(EC_OK, ec);

    // 阶段6：验证锁已释放
    ec = backend_->GetLockHolder(lock_key, current_value, expire_time_ms);
    EXPECT_EQ(EC_NOENT, ec);
}

// 测试无效参数
TEST_P(DistributedLockBackendTest, TestInvalidArguments) {
    // 测试空键名
    ErrorCode ec = backend_->TryLock("", "value", 1000);
    EXPECT_EQ(EC_BADARGS, ec);

    // 测试空值
    ec = backend_->TryLock("key", "", 1000);
    EXPECT_EQ(EC_BADARGS, ec);

    // 测试无效TTL
    ec = backend_->TryLock("key", "value", 0);
    EXPECT_EQ(EC_BADARGS, ec);

    ec = backend_->TryLock("key", "value", -100);
    EXPECT_EQ(EC_BADARGS, ec);
}

} // namespace kv_cache_manager