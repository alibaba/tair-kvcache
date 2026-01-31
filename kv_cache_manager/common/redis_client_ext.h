#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/redis_client.h"

namespace kv_cache_manager {

// RedisClient的扩展类，添加Lua脚本执行等功能.
class RedisClientExt : public RedisClient {
public:
    using RedisClient::RedisClient;

    // 执行Lua脚本
    // script: Lua脚本内容
    // keys: KEYS数组
    // args: ARGV数组
    // out_result: 脚本执行结果
    ErrorCode Eval(const std::string &script,
                   const std::vector<std::string> &keys,
                   const std::vector<std::string> &args,
                   std::string &out_result);

    // 执行缓存的Lua脚本（使用SHA1）
    // sha1: 脚本的SHA1哈希值
    // keys: KEYS数组
    // args: ARGV数组
    // out_result: 脚本执行结果
    ErrorCode EvalSha(const std::string &sha1,
                      const std::vector<std::string> &keys,
                      const std::vector<std::string> &args,
                      std::string &out_result);

    // 加载Lua脚本到Redis并返回SHA1哈希值
    // script: Lua脚本内容
    // out_sha1: 输出的SHA1哈希值
    ErrorCode ScriptLoad(const std::string &script, std::string &out_sha1);

    // 检查脚本是否已加载到Redis
    // sha1: 脚本的SHA1哈希值
    // out_exists: 输出是否存在
    ErrorCode ScriptExists(const std::string &sha1, bool &out_exists);

    // 加载脚本到Redis
    // script: 脚本
    // out_sha1: 脚本的SHA1哈希值
    ErrorCode LoadScript(const std::string &script, std::string &out_sha1);

    // 执行脚本。会尽可能使用EVALSHA
    // script: 脚本
    // keys: KEYS数组
    // args: ARGV数组
    // in_out_cached_sha1: 脚本的SHA1哈希值。需要传入。如果miss会重新load，并把得到的SHA1写回该值。
    // out_result: 脚本执行结果
    ErrorCode ExecuteScriptWithFallback(const std::string &script,
                                        const std::vector<std::string> &keys,
                                        const std::vector<std::string> &args,
                                        std::string &in_out_cached_sha1,
                                        std::string &out_result);

    // 执行简单的GET命令
    ErrorCode Get(const std::string &key, std::string &out_value);

    // 执行简单的SET命令
    ErrorCode Set(const std::string &key, const std::string &value, int64_t ttl_ms = 0);

    // 执行PTTL命令获取剩余毫秒数
    ErrorCode Pttl(const std::string &key, int64_t &out_ttl_ms);

    // 执行DEL命令
    ErrorCode Del(const std::string &key);

    // 执行PEXPIRE命令
    ErrorCode Pexpire(const std::string &key, int64_t ttl_ms);

    // 执行FLUSHALL命令
    ErrorCode FlushAll();
};

} // namespace kv_cache_manager