#pragma once

#include <memory>
#include <string>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/redis_client_ext.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/config/distributed_lock_backend.h"

namespace kv_cache_manager {

class DistributedLockRedisBackend : public DistributedLockBackend {
public:
    DistributedLockRedisBackend();
    ~DistributedLockRedisBackend() override;

    // 禁止拷贝和赋值
    DistributedLockRedisBackend(const DistributedLockRedisBackend &) = delete;
    DistributedLockRedisBackend &operator=(const DistributedLockRedisBackend &) = delete;

    // 允许移动
    DistributedLockRedisBackend(DistributedLockRedisBackend &&) = default;
    DistributedLockRedisBackend &operator=(DistributedLockRedisBackend &&) = default;

public:
    ErrorCode Init(const StandardUri &standard_uri) noexcept override;
    ErrorCode TryLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) override;
    ErrorCode RenewLock(const std::string &lock_key, const std::string &lock_value, int64_t ttl_ms) override;
    ErrorCode Unlock(const std::string &lock_key, const std::string &lock_value) override;
    ErrorCode
    GetLockHolder(const std::string &lock_key, std::string &out_current_value, int64_t &out_expire_time_ms) override;

private:
    // 生成Redis键名
    std::string GetRedisKey(const std::string &lock_key) const;

    // Lua脚本：尝试获取锁（原子操作）
    static constexpr const char *LUA_TRY_LOCK = R"(
        local value = ARGV[1]
        local ttl_ms = tonumber(ARGV[2])
        
        -- 使用SET NX PX命令原子性地获取锁
        local result = redis.call('SET', KEYS[1], value, 'NX', 'PX', ttl_ms)
        if result then
            -- 成功获取锁
            return 1
        else
            -- 锁已存在，检查值是否匹配（可重入锁）
            local current_value = redis.call('GET', KEYS[1])
            if current_value == value then
                -- 同一个客户端，允许重入
                redis.call('PEXPIRE', KEYS[1], ttl_ms)
                return 1
            else
                -- 锁被其他客户端持有
                return 0
            end
        end
    )";

    // Lua脚本：续约锁（原子操作）
    static constexpr const char *LUA_RENEW_LOCK = R"(
        local value = ARGV[1]
        local ttl_ms = tonumber(ARGV[2])
        
        -- 检查锁是否存在且值匹配
        local current_value = redis.call('GET', KEYS[1])
        if not current_value then
            -- 锁不存在
            return 0
        end
        
        if current_value ~= value then
            -- 值不匹配
            return -1
        end
        
        -- 检查锁是否过期
        local current_ttl = redis.call('PTTL', KEYS[1])
        if current_ttl <= 0 then
            -- 锁已过期
            return 0
        end
        
        -- 续约锁
        redis.call('PEXPIRE', KEYS[1], ttl_ms)
        return 1
    )";

    // Lua脚本：释放锁（原子操作）
    static constexpr const char *LUA_UNLOCK = R"(
        local value = ARGV[1]
        
        -- 检查锁是否存在且值匹配
        local current_value = redis.call('GET', KEYS[1])
        if not current_value then
            -- 锁不存在
            return 0
        end
        
        if current_value ~= value then
            -- 值不匹配
            return -1
        end
        
        -- 删除锁
        redis.call('DEL', KEYS[1])
        return 1
    )";

private:
    std::unique_ptr<RedisClientExt> redis_client_;
    std::string key_prefix_;
    bool initialized_{false};

    // Lua脚本的SHA1缓存
    std::string try_lock_sha1_;
    std::string renew_lock_sha1_;
    std::string unlock_sha1_;
};

} // namespace kv_cache_manager