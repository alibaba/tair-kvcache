#pragma once

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/config/distributed_lock_backend.h"

namespace kv_cache_manager {

/**
 * @brief Memory-based distributed lock backend (dummy implementation)
 *
 * This backend is designed for single-process scenarios where no actual
 * cross-process locking is required. It stores all lock information in
 * memory using thread-safe data structures.
 *
 * URI format: memory:// (any path or parameters are ignored)
 */
class DistributedLockMemoryBackend : public DistributedLockBackend {
public:
    DistributedLockMemoryBackend();
    ~DistributedLockMemoryBackend() override;

    // 禁止拷贝和赋值
    DistributedLockMemoryBackend(const DistributedLockMemoryBackend &) = delete;
    DistributedLockMemoryBackend &operator=(const DistributedLockMemoryBackend &) = delete;

    // 允许移动
    DistributedLockMemoryBackend(DistributedLockMemoryBackend &&) = default;
    DistributedLockMemoryBackend &operator=(DistributedLockMemoryBackend &&) = default;

public:
    ErrorCode Init(const StandardUri &standard_uri) noexcept override;
    ErrorCode TryLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) override;
    ErrorCode RenewLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) override;
    ErrorCode Unlock(const std::string &lock_key, const std::string &lock_value) override;
    ErrorCode
    GetLockHolder(const std::string &lock_key, std::string &out_current_value, int64_t &out_expire_time_ms) override;

private:
    // 锁信息结构
    struct LockInfo {
        std::string value;                                 // 锁持有者标识
        std::chrono::steady_clock::time_point expire_time; // 过期时间点

        // 检查锁是否过期
        bool IsExpired() const { return std::chrono::steady_clock::now() >= expire_time; }

        // 获取剩余过期时间（毫秒）
        int64_t GetRemainingTTLMs() const {
            auto now = std::chrono::steady_clock::now();
            if (now >= expire_time) {
                return 0;
            }
            return std::chrono::duration_cast<std::chrono::milliseconds>(expire_time - now).count();
        }

        // 获取过期时间戳（毫秒）
        int64_t GetExpireTimeMs() const {
            return std::chrono::duration_cast<std::chrono::milliseconds>(expire_time.time_since_epoch()).count();
        }
    };

private:
    // 清理过期的锁条目（调用前必须持有锁）
    void CleanupExpiredLocks();

    // 获取当前时间点
    static std::chrono::steady_clock::time_point Now() { return std::chrono::steady_clock::now(); }

    // 从毫秒时间戳转换为时间点
    static std::chrono::steady_clock::time_point FromMsTimestamp(int64_t ms_timestamp) {
        return std::chrono::steady_clock::time_point(std::chrono::milliseconds(ms_timestamp));
    }

private:
    std::mutex mutex_;
    std::unordered_map<std::string, LockInfo> locks_;
    bool initialized_{false};
};

} // namespace kv_cache_manager