#include "kv_cache_manager/config/distributed_lock_redis_backend.h"

#include <memory>
#include <string>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/redis_client_ext.h"
#include "kv_cache_manager/common/standard_uri.h"

namespace kv_cache_manager {

DistributedLockRedisBackend::DistributedLockRedisBackend() = default;

DistributedLockRedisBackend::~DistributedLockRedisBackend() = default;

ErrorCode DistributedLockRedisBackend::Init(const StandardUri &standard_uri) noexcept {
    // 验证URI协议
    if (standard_uri.GetProtocol() != "redis") {
        KVCM_LOG_ERROR("Invalid protocol for Redis lock backend: %s", standard_uri.GetProtocol().c_str());
        return EC_BADARGS;
    }

    if (initialized_) {
        KVCM_LOG_WARN("Redis lock backend already initialized");
        return EC_OK;
    }

    // 创建Redis客户端
    try {
        redis_client_ = std::make_unique<RedisClientExt>(standard_uri);
    } catch (const std::exception &e) {
        KVCM_LOG_ERROR("Failed to create Redis client: %s", e.what());
        return EC_ERROR;
    }

    // 打开Redis连接
    if (!redis_client_->Open()) {
        KVCM_LOG_ERROR("Failed to open Redis connection");
        redis_client_.reset();
        return EC_IO_ERROR;
    }

    // 设置键前缀
    key_prefix_ = "kvcm_lock:";

    initialized_ = true;

    // 初始化时加载所有Lua脚本
    ErrorCode ec = redis_client_->LoadScript(LUA_TRY_LOCK, try_lock_sha1_);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to load TryLock script: ec=%d", ec);
        return ec;
    }

    ec = redis_client_->LoadScript(LUA_RENEW_LOCK, renew_lock_sha1_);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to load RenewLock script: ec=%d", ec);
        return ec;
    }

    ec = redis_client_->LoadScript(LUA_UNLOCK, unlock_sha1_);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to load Unlock script: ec=%d", ec);
        return ec;
    }

    KVCM_LOG_INFO("Redis lock backend initialized successfully with script caching");
    return EC_OK;
}

std::string DistributedLockRedisBackend::GetRedisKey(const std::string &lock_key) const {
    return key_prefix_ + lock_key;
}

ErrorCode
DistributedLockRedisBackend::TryLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) {
    if (!initialized_) {
        KVCM_LOG_ERROR("Redis lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty() || lock_value.empty() || ttl_ms <= 0) {
        KVCM_LOG_ERROR("Invalid arguments for TryLock: key=%s, ttl_ms=%ld", lock_key.c_str(), ttl_ms);
        return EC_BADARGS;
    }

    std::string redis_key = GetRedisKey(lock_key);

    // 使用Lua脚本原子性地获取锁
    std::string result;
    std::vector<std::string> keys = {redis_key};
    std::vector<std::string> args = {lock_value, std::to_string(ttl_ms)};

    ErrorCode ec = redis_client_->ExecuteScriptWithFallback(LUA_TRY_LOCK, keys, args, try_lock_sha1_, result);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to execute TryLock Lua script: ec=%d", ec);
        return ec;
    }

    // 解析Lua脚本结果
    if (result == "1") {
        // 成功获取锁
        return EC_OK;
    } else if (result == "0") {
        // 锁已被其他客户端持有
        return EC_EXIST;
    } else {
        // 其他错误
        KVCM_LOG_ERROR("Unexpected result from TryLock Lua script: %s", result.c_str());
        return EC_ERROR;
    }
}

ErrorCode
DistributedLockRedisBackend::RenewLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) {
    if (!initialized_) {
        KVCM_LOG_ERROR("Redis lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty() || lock_value.empty() || ttl_ms <= 0) {
        KVCM_LOG_ERROR("Invalid arguments for RenewLock: key=%s, ttl_ms=%ld", lock_key.c_str(), ttl_ms);
        return EC_BADARGS;
    }

    std::string redis_key = GetRedisKey(lock_key);

    // 使用Lua脚本原子性地续约锁
    std::string result;
    std::vector<std::string> keys = {redis_key};
    std::vector<std::string> args = {lock_value, std::to_string(ttl_ms)};

    ErrorCode ec = redis_client_->ExecuteScriptWithFallback(LUA_RENEW_LOCK, keys, args, renew_lock_sha1_, result);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to execute RenewLock Lua script: ec=%d", ec);
        return ec;
    }

    // 解析Lua脚本结果
    if (result == "1") {
        // 成功续约锁
        return EC_OK;
    } else if (result == "0") {
        // 锁不存在或已过期
        return EC_NOENT;
    } else if (result == "-1") {
        // 值不匹配
        return EC_MISMATCH;
    } else {
        // 其他错误
        KVCM_LOG_ERROR("Unexpected result from RenewLock Lua script: %s", result.c_str());
        return EC_ERROR;
    }
}

ErrorCode DistributedLockRedisBackend::Unlock(const std::string &lock_key, const std::string &lock_value) {
    if (!initialized_) {
        KVCM_LOG_ERROR("Redis lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty() || lock_value.empty()) {
        KVCM_LOG_ERROR("Invalid arguments for Unlock: key=%s", lock_key.c_str());
        return EC_BADARGS;
    }

    std::string redis_key = GetRedisKey(lock_key);

    // 使用Lua脚本原子性地释放锁
    std::string result;
    std::vector<std::string> keys = {redis_key};
    std::vector<std::string> args = {lock_value};

    ErrorCode ec = redis_client_->ExecuteScriptWithFallback(LUA_UNLOCK, keys, args, unlock_sha1_, result);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to execute Unlock Lua script: ec=%d", ec);
        return ec;
    }

    // 解析Lua脚本结果
    if (result == "1") {
        // 成功释放锁
        return EC_OK;
    } else if (result == "0") {
        // 锁不存在
        return EC_NOENT;
    } else if (result == "-1") {
        // 值不匹配
        return EC_MISMATCH;
    } else {
        // 其他错误
        KVCM_LOG_ERROR("Unexpected result from Unlock Lua script: %s", result.c_str());
        return EC_ERROR;
    }
}

ErrorCode DistributedLockRedisBackend::GetLockHolder(const std::string &lock_key,
                                                     std::string &out_current_value,
                                                     int64_t &out_expire_time_ms) {
    if (!initialized_) {
        KVCM_LOG_ERROR("Redis lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty()) {
        KVCM_LOG_ERROR("Invalid arguments for GetLockHolder: key is empty");
        return EC_BADARGS;
    }

    std::string redis_key = GetRedisKey(lock_key);

    // 获取当前锁的值
    ErrorCode ec = redis_client_->Get(redis_key, out_current_value);
    if (ec == EC_NOENT) {
        // 锁不存在
        out_current_value.clear();
        out_expire_time_ms = 0;
        return EC_NOENT;
    } else if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to get lock value: ec=%d", ec);
        return ec;
    }

    // 获取锁的剩余过期时间
    ec = redis_client_->Pttl(redis_key, out_expire_time_ms);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to get lock TTL: ec=%d", ec);
        return ec;
    }

    // 如果锁已过期，返回不存在
    if (out_expire_time_ms <= 0) {
        out_current_value.clear();
        out_expire_time_ms = 0;
        return EC_NOENT;
    }

    return EC_OK;
}

} // namespace kv_cache_manager