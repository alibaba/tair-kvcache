#include "kv_cache_manager/config/coordination_redis_backend.h"

#include <chrono>
#include <memory>
#include <string>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/redis_client_ext.h"
#include "kv_cache_manager/common/standard_uri.h"

namespace kv_cache_manager {

CoordinationRedisBackend::CoordinationRedisBackend() = default;

CoordinationRedisBackend::~CoordinationRedisBackend() = default;

std::shared_ptr<RedisClientExt> CoordinationRedisBackend::CreateRedisClient(const StandardUri &storage_uri) const {
    return std::make_shared<RedisClientExt>(storage_uri);
}

ErrorCode CoordinationRedisBackend::Init(const StandardUri &standard_uri) noexcept {
    constexpr int32_t DEFAULT_CLIENT_MAX_POOL_SIZE = 4;
    constexpr int32_t DEFAULT_CLIENT_MIN_POOL_SIZE = 1;

    // 验证URI协议
    if (standard_uri.GetProtocol() != "redis") {
        KVCM_LOG_ERROR("Invalid protocol for Redis lock backend: %s", standard_uri.GetProtocol().c_str());
        return EC_BADARGS;
    }

    if (initialized_.load(std::memory_order_acquire)) {
        KVCM_LOG_WARN("Redis lock backend already initialized");
        return EC_OK;
    }

    storage_uri_ = standard_uri;

    timeout_ms_ = 1000;
    int64_t tmp_timeout_ms = 0;
    storage_uri_.GetParamAs("timeout_ms", tmp_timeout_ms);
    if (tmp_timeout_ms > 0) {
        timeout_ms_ = tmp_timeout_ms;
    }

    int32_t client_max_pool_size = DEFAULT_CLIENT_MAX_POOL_SIZE;
    int32_t client_min_pool_size = DEFAULT_CLIENT_MIN_POOL_SIZE;
    int64_t tmp_client_max_pool_size = 0;
    storage_uri_.GetParamAs("client_max_pool_size", tmp_client_max_pool_size);
    if (tmp_client_max_pool_size > 0) {
        client_max_pool_size = static_cast<int32_t>(tmp_client_max_pool_size);
    }
    int64_t tmp_client_min_pool_size = 0;
    storage_uri_.GetParamAs("client_min_pool_size", tmp_client_min_pool_size);
    if (tmp_client_min_pool_size > 0) {
        client_min_pool_size = static_cast<int32_t>(tmp_client_min_pool_size);
    }
    if (client_max_pool_size <= 0) {
        client_max_pool_size = DEFAULT_CLIENT_MAX_POOL_SIZE;
    }
    if (client_min_pool_size < 0) {
        client_min_pool_size = DEFAULT_CLIENT_MIN_POOL_SIZE;
    }
    if (client_min_pool_size > client_max_pool_size) {
        KVCM_LOG_WARN("coordination redis client_min_pool_size[%d] > client_max_pool_size[%d], clamp min to max",
                      client_min_pool_size,
                      client_max_pool_size);
        client_min_pool_size = client_max_pool_size;
    }

    auto client_pool = std::make_shared<RedisClientPool>(
        [this]() -> std::shared_ptr<RedisClientExt> {
            auto client = CreateRedisClient(storage_uri_);
            if (!client || !client->Open()) {
                return nullptr;
            }
            return client;
        },
        client_min_pool_size,
        client_max_pool_size);
    if (!client_pool->Initialize()) {
        KVCM_LOG_ERROR("Failed to initialize coordination redis client pool");
        return EC_IO_ERROR;
    }

    // 从URI参数中获取cluster_name，用于多KVCM实例共享同一Redis时的key隔离
    std::string cluster_name = standard_uri.GetParam("cluster_name");
    std::string base_prefix = "kvcm_";
    if (!cluster_name.empty()) {
        base_prefix += cluster_name + "_";
    } else {
        KVCM_LOG_WARN("cluster_name not set for coordination backend, keys will use global namespace. "
                      "Consider setting cluster_name to avoid conflicts in shared-Redis deployments.");
    }
    lock_key_prefix_ = base_prefix + "lock:";
    kv_key_prefix_ = base_prefix + "kv:";

    auto handle = client_pool->AcquireClient(timeout_ms_);
    if (!handle) {
        KVCM_LOG_ERROR("Failed to acquire coordination redis client for script loading");
        return EC_TIMEOUT;
    }

    std::string try_lock_sha1;
    std::string renew_lock_sha1;
    std::string unlock_sha1;

    // 初始化时加载所有Lua脚本
    ErrorCode ec = handle->LoadScript(LUA_TRY_LOCK, try_lock_sha1);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to load TryLock script: ec=%d", ec);
        return ec;
    }

    ec = handle->LoadScript(LUA_RENEW_LOCK, renew_lock_sha1);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to load RenewLock script: ec=%d", ec);
        return ec;
    }

    ec = handle->LoadScript(LUA_UNLOCK, unlock_sha1);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to load Unlock script: ec=%d", ec);
        return ec;
    }

    try_lock_sha1_ = try_lock_sha1;
    renew_lock_sha1_ = renew_lock_sha1;
    unlock_sha1_ = unlock_sha1;
    client_pool_ = client_pool;
    initialized_.store(true, std::memory_order_release);

    KVCM_LOG_INFO("Redis coordination backend initialized, cluster_name[%s], lock_prefix[%s], kv_prefix[%s], "
                  "client pool min[%d], max[%d]",
                  cluster_name.c_str(), lock_key_prefix_.c_str(), kv_key_prefix_.c_str(),
                  client_min_pool_size, client_max_pool_size);
    return EC_OK;
}

std::string CoordinationRedisBackend::GetRedisLockKey(const std::string &lock_key) const {
    return lock_key_prefix_ + lock_key;
}

CoordinationRedisBackend::RedisClientHandle CoordinationRedisBackend::AcquireClient() {
    if (!client_pool_) {
        return RedisClientHandle(nullptr, nullptr);
    }
    return client_pool_->AcquireClient(timeout_ms_);
}

std::string CoordinationRedisBackend::GetCachedScriptSha1(const std::string &cached_sha1) const {
    std::lock_guard<std::mutex> lock(script_sha_mutex_);
    return cached_sha1;
}

void CoordinationRedisBackend::UpdateCachedScriptSha1(std::string &cached_sha1, const std::string &new_sha1) {
    std::lock_guard<std::mutex> lock(script_sha_mutex_);
    cached_sha1 = new_sha1;
}

ErrorCode
CoordinationRedisBackend::TryLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) {
    if (!initialized_.load(std::memory_order_acquire)) {
        KVCM_LOG_ERROR("Redis lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty() || lock_value.empty() || ttl_ms <= 0) {
        KVCM_LOG_ERROR("Invalid arguments for TryLock: key=%s, ttl_ms=%ld", lock_key.c_str(), ttl_ms);
        return EC_BADARGS;
    }

    std::string redis_key = GetRedisLockKey(lock_key);

    // 使用Lua脚本原子性地获取锁
    std::string result;
    std::vector<std::string> keys = {redis_key};
    std::vector<std::string> args = {lock_value, std::to_string(ttl_ms)};

    auto handle = AcquireClient();
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "try lock fail, fail to acquire coordination redis client");
        return EC_TIMEOUT;
    }

    std::string script_sha1 = GetCachedScriptSha1(try_lock_sha1_);
    ErrorCode ec = handle->ExecuteScriptWithFallback(LUA_TRY_LOCK, keys, args, script_sha1, result);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to execute TryLock Lua script: ec=%d", ec);
        return ec;
    }
    UpdateCachedScriptSha1(try_lock_sha1_, script_sha1);

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
CoordinationRedisBackend::RenewLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) {
    if (!initialized_.load(std::memory_order_acquire)) {
        KVCM_LOG_ERROR("Redis lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty() || lock_value.empty() || ttl_ms <= 0) {
        KVCM_LOG_ERROR("Invalid arguments for RenewLock: key=%s, ttl_ms=%ld", lock_key.c_str(), ttl_ms);
        return EC_BADARGS;
    }

    std::string redis_key = GetRedisLockKey(lock_key);

    // 使用Lua脚本原子性地续约锁
    std::string result;
    std::vector<std::string> keys = {redis_key};
    std::vector<std::string> args = {lock_value, std::to_string(ttl_ms)};

    auto handle = AcquireClient();
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "renew lock fail, fail to acquire coordination redis client");
        return EC_TIMEOUT;
    }

    std::string script_sha1 = GetCachedScriptSha1(renew_lock_sha1_);
    ErrorCode ec = handle->ExecuteScriptWithFallback(LUA_RENEW_LOCK, keys, args, script_sha1, result);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to execute RenewLock Lua script: ec=%d", ec);
        return ec;
    }
    UpdateCachedScriptSha1(renew_lock_sha1_, script_sha1);

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

ErrorCode CoordinationRedisBackend::Unlock(const std::string &lock_key, const std::string &lock_value) {
    if (!initialized_.load(std::memory_order_acquire)) {
        KVCM_LOG_ERROR("Redis lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty() || lock_value.empty()) {
        KVCM_LOG_ERROR("Invalid arguments for Unlock: key=%s", lock_key.c_str());
        return EC_BADARGS;
    }

    std::string redis_key = GetRedisLockKey(lock_key);

    // 使用Lua脚本原子性地释放锁
    std::string result;
    std::vector<std::string> keys = {redis_key};
    std::vector<std::string> args = {lock_value};

    auto handle = AcquireClient();
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "unlock fail, fail to acquire coordination redis client");
        return EC_TIMEOUT;
    }

    std::string script_sha1 = GetCachedScriptSha1(unlock_sha1_);
    ErrorCode ec = handle->ExecuteScriptWithFallback(LUA_UNLOCK, keys, args, script_sha1, result);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to execute Unlock Lua script: ec=%d", ec);
        return ec;
    }
    UpdateCachedScriptSha1(unlock_sha1_, script_sha1);

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

ErrorCode CoordinationRedisBackend::GetLockHolder(const std::string &lock_key,
                                                  std::string &out_current_value,
                                                  int64_t &out_expire_time_ms) {
    if (!initialized_.load(std::memory_order_acquire)) {
        KVCM_LOG_ERROR("Redis lock backend not initialized");
        return EC_ERROR;
    }

    if (lock_key.empty()) {
        KVCM_LOG_ERROR("Invalid arguments for GetLockHolder: key is empty");
        return EC_BADARGS;
    }

    std::string redis_key = GetRedisLockKey(lock_key);

    auto handle = AcquireClient();
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "get lock holder fail, fail to acquire coordination redis client");
        return EC_TIMEOUT;
    }

    // 获取当前锁的值
    ErrorCode ec = handle->Get(redis_key, out_current_value);
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
    int64_t remaining_ttl_ms = 0;
    ec = handle->Pttl(redis_key, remaining_ttl_ms);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to get lock TTL: ec=%d", ec);
        return ec;
    }

    // 如果锁已过期，返回不存在
    if (remaining_ttl_ms <= 0) {
        out_current_value.clear();
        out_expire_time_ms = 0;
        return EC_NOENT;
    }

    // 将剩余 TTL 转换为 system_clock 绝对时间戳
    out_expire_time_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count() +
        remaining_ttl_ms;

    return EC_OK;
}

std::string CoordinationRedisBackend::GetRedisKVKey(const std::string &key) const { return kv_key_prefix_ + key; }

ErrorCode CoordinationRedisBackend::SetValue(const std::string &key, const std::string &value) {
    if (!initialized_.load(std::memory_order_acquire)) {
        KVCM_LOG_ERROR("Redis coordination backend not initialized");
        return EC_ERROR;
    }
    if (key.empty()) {
        KVCM_LOG_ERROR("Invalid arguments for SetValue: key is empty");
        return EC_BADARGS;
    }

    std::string redis_key = GetRedisKVKey(key);
    auto handle = AcquireClient();
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "set value fail, fail to acquire coordination redis client");
        return EC_TIMEOUT;
    }

    ErrorCode ec = handle->Set(redis_key, value);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to set value for key %s: ec=%d", key.c_str(), ec);
        return ec;
    }
    return EC_OK;
}

ErrorCode CoordinationRedisBackend::GetValue(const std::string &key, std::string &out_value) {
    if (!initialized_.load(std::memory_order_acquire)) {
        KVCM_LOG_ERROR("Redis coordination backend not initialized");
        return EC_ERROR;
    }
    if (key.empty()) {
        KVCM_LOG_ERROR("Invalid arguments for GetValue: key is empty");
        return EC_BADARGS;
    }

    std::string redis_key = GetRedisKVKey(key);
    auto handle = AcquireClient();
    if (!handle) {
        KVCM_INTERVAL_LOG_WARN(10, "get value fail, fail to acquire coordination redis client");
        return EC_TIMEOUT;
    }

    ErrorCode ec = handle->Get(redis_key, out_value);
    if (ec == EC_NOENT) {
        return EC_NOENT;
    }
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to get value for key %s: ec=%d", key.c_str(), ec);
        return ec;
    }
    return EC_OK;
}

} // namespace kv_cache_manager
