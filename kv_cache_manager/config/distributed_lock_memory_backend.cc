#include "kv_cache_manager/config/distributed_lock_memory_backend.h"

#include <chrono>
#include <mutex>
#include <string>

namespace kv_cache_manager {

DistributedLockMemoryBackend::DistributedLockMemoryBackend() = default;

DistributedLockMemoryBackend::~DistributedLockMemoryBackend() = default;

ErrorCode DistributedLockMemoryBackend::Init(const StandardUri &standard_uri) noexcept {
    if (initialized_) {
        KVCM_LOG_WARN("Memory lock backend already initialized");
        return EC_OK;
    }

    if (standard_uri.GetProtocol() != "memory") {
        KVCM_LOG_ERROR("Invalid protocol for memory lock backend: %s", standard_uri.GetProtocol().c_str());
        return EC_BADARGS;
    }

    initialized_ = true;
    KVCM_LOG_INFO("DistributedLockMemoryBackend initialized successfully");
    return EC_OK;
}

void DistributedLockMemoryBackend::CleanupExpiredLocks() {
    auto now = Now();
    for (auto it = locks_.begin(); it != locks_.end();) {
        if (now >= it->second.expire_time) {
            it = locks_.erase(it);
        } else {
            ++it;
        }
    }
}

ErrorCode
DistributedLockMemoryBackend::TryLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) {
    if (!initialized_) {
        KVCM_LOG_ERROR("Memory lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty() || lock_value.empty() || ttl_ms <= 0) {
        KVCM_LOG_ERROR("Invalid arguments for TryLock: key=%s, ttl_ms=%ld", lock_key.c_str(), ttl_ms);
        return EC_BADARGS;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 清理过期的锁
    CleanupExpiredLocks();

    auto it = locks_.find(lock_key);
    if (it != locks_.end()) {
        // 锁已存在，检查是否过期
        if (it->second.IsExpired()) {
            // 锁已过期，可以重新获取
            it->second.value = lock_value;
            it->second.expire_time = Now() + std::chrono::milliseconds(ttl_ms);
            return EC_OK;
        } else {
            // 锁未过期，检查是否同一个持有者（可重入）
            if (it->second.value == lock_value) {
                // 同一个持有者，更新过期时间
                it->second.expire_time = Now() + std::chrono::milliseconds(ttl_ms);
                return EC_OK;
            } else {
                // 锁被其他客户端持有
                return EC_EXIST;
            }
        }
    } else {
        // 锁不存在，创建新锁
        LockInfo lock_info;
        lock_info.value = lock_value;
        lock_info.expire_time = Now() + std::chrono::milliseconds(ttl_ms);
        locks_[lock_key] = lock_info;
        return EC_OK;
    }
}

ErrorCode
DistributedLockMemoryBackend::RenewLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) {
    if (!initialized_) {
        KVCM_LOG_ERROR("Memory lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty() || lock_value.empty() || ttl_ms <= 0) {
        KVCM_LOG_ERROR("Invalid arguments for RenewLock: key=%s, ttl_ms=%ld", lock_key.c_str(), ttl_ms);
        return EC_BADARGS;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 清理过期的锁
    CleanupExpiredLocks();

    auto it = locks_.find(lock_key);
    if (it == locks_.end()) {
        // 锁不存在
        return EC_NOENT;
    }

    // 检查锁是否过期
    if (it->second.IsExpired()) {
        locks_.erase(it);
        return EC_NOENT;
    }

    // 检查锁值是否匹配
    if (it->second.value != lock_value) {
        return EC_MISMATCH;
    }

    // 更新过期时间
    it->second.expire_time = Now() + std::chrono::milliseconds(ttl_ms);
    return EC_OK;
}

ErrorCode DistributedLockMemoryBackend::Unlock(const std::string &lock_key, const std::string &lock_value) {
    if (!initialized_) {
        KVCM_LOG_ERROR("Memory lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty() || lock_value.empty()) {
        KVCM_LOG_ERROR("Invalid arguments for Unlock: key=%s", lock_key.c_str());
        return EC_BADARGS;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 清理过期的锁
    CleanupExpiredLocks();

    auto it = locks_.find(lock_key);
    if (it == locks_.end()) {
        // 锁不存在
        return EC_NOENT;
    }

    // 检查锁是否过期
    if (it->second.IsExpired()) {
        locks_.erase(it);
        return EC_NOENT;
    }

    // 检查锁值是否匹配
    if (it->second.value != lock_value) {
        return EC_MISMATCH;
    }

    // 释放锁
    locks_.erase(it);
    return EC_OK;
}

ErrorCode DistributedLockMemoryBackend::GetLockHolder(const std::string &lock_key,
                                                      std::string &out_current_value,
                                                      int64_t &out_expire_time_ms) {
    if (!initialized_) {
        KVCM_LOG_ERROR("Memory lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty()) {
        KVCM_LOG_ERROR("Invalid arguments for GetLockHolder: key is empty");
        return EC_BADARGS;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 清理过期的锁
    CleanupExpiredLocks();

    auto it = locks_.find(lock_key);
    if (it == locks_.end()) {
        // 锁不存在
        out_current_value.clear();
        out_expire_time_ms = 0;
        return EC_NOENT;
    }

    // 检查锁是否过期
    if (it->second.IsExpired()) {
        locks_.erase(it);
        out_current_value.clear();
        out_expire_time_ms = 0;
        return EC_NOENT;
    }

    // 返回锁信息
    out_current_value = it->second.value;
    out_expire_time_ms = it->second.GetExpireTimeMs();
    return EC_OK;
}

} // namespace kv_cache_manager