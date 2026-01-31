#include "kv_cache_manager/common/redis_client_ext.h"

#include <hiredis.h>
#include <string>
#include <vector>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

ErrorCode RedisClientExt::Eval(const std::string &script,
                               const std::vector<std::string> &keys,
                               const std::vector<std::string> &args,
                               std::string &out_result) {
    out_result.clear();

    if (!IsContextOk()) {
        KVCM_LOG_ERROR("Redis context not ok for EVAL");
        return EC_IO_ERROR;
    }

    // 构建EVAL命令
    std::vector<std::string> cmd_args;
    cmd_args.reserve(3 + keys.size() + args.size());
    cmd_args.emplace_back("EVAL");
    cmd_args.emplace_back(script);
    cmd_args.emplace_back(std::to_string(keys.size()));

    // 添加KEYS
    for (const auto &key : keys) {
        cmd_args.emplace_back(key);
    }

    // 添加ARGV
    for (const auto &arg : args) {
        cmd_args.emplace_back(arg);
    }

    // 执行命令
    std::vector<CmdArgs> cmds = {cmd_args};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_LOG_ERROR("EVAL command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_LOG_ERROR("EVAL command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    // 处理不同类型的回复
    switch (reply->type) {
    case REDIS_REPLY_STRING:
        out_result = std::string(reply->str, reply->len);
        return EC_OK;
    case REDIS_REPLY_INTEGER:
        out_result = std::to_string(reply->integer);
        return EC_OK;
    case REDIS_REPLY_NIL:
        // Lua脚本返回nil
        return EC_OK;
    case REDIS_REPLY_STATUS:
        out_result = std::string(reply->str, reply->len);
        return EC_OK;
    case REDIS_REPLY_ERROR:
        KVCM_LOG_ERROR("EVAL command error: %s", reply->str);
        return EC_ERROR;
    default:
        KVCM_LOG_ERROR("EVAL command unexpected reply type: %d", reply->type);
        return EC_ERROR;
    }
}

ErrorCode RedisClientExt::EvalSha(const std::string &sha1,
                                  const std::vector<std::string> &keys,
                                  const std::vector<std::string> &args,
                                  std::string &out_result) {
    out_result.clear();

    if (!IsContextOk()) {
        KVCM_LOG_ERROR("Redis context not ok for EVALSHA");
        return EC_IO_ERROR;
    }

    // 构建EVALSHA命令
    std::vector<std::string> cmd_args;
    cmd_args.reserve(3 + keys.size() + args.size());
    cmd_args.emplace_back("EVALSHA");
    cmd_args.emplace_back(sha1);
    cmd_args.emplace_back(std::to_string(keys.size()));

    // 添加KEYS
    for (const auto &key : keys) {
        cmd_args.emplace_back(key);
    }

    // 添加ARGV
    for (const auto &arg : args) {
        cmd_args.emplace_back(arg);
    }

    // 执行命令
    std::vector<CmdArgs> cmds = {cmd_args};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_LOG_ERROR("EVALSHA command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_LOG_ERROR("EVALSHA command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    // 处理不同类型的回复
    switch (reply->type) {
    case REDIS_REPLY_STRING:
        out_result = std::string(reply->str, reply->len);
        return EC_OK;
    case REDIS_REPLY_INTEGER:
        out_result = std::to_string(reply->integer);
        return EC_OK;
    case REDIS_REPLY_NIL:
        // Lua脚本返回nil
        return EC_OK;
    case REDIS_REPLY_STATUS:
        out_result = std::string(reply->str, reply->len);
        return EC_OK;
    case REDIS_REPLY_ERROR:
        // 检查是否是NOSCRIPT错误
        if (reply->str && std::string(reply->str).find("NOSCRIPT") != std::string::npos) {
            KVCM_LOG_WARN("EVALSHA NOSCRIPT error for sha1: %s", sha1.c_str());
            return EC_NOSCRIPT;
        }
        KVCM_LOG_ERROR("EVALSHA command error: %s", reply->str);
        return EC_ERROR;
    default:
        KVCM_LOG_ERROR("EVALSHA command unexpected reply type: %d", reply->type);
        return EC_ERROR;
    }
}

ErrorCode RedisClientExt::ScriptLoad(const std::string &script, std::string &out_sha1) {
    out_sha1.clear();

    if (!IsContextOk()) {
        KVCM_LOG_ERROR("Redis context not ok for SCRIPT LOAD");
        return EC_IO_ERROR;
    }

    std::vector<CmdArgs> cmds = {{"SCRIPT", "LOAD", script}};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_LOG_ERROR("SCRIPT LOAD command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_LOG_ERROR("SCRIPT LOAD command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    if (reply->type == REDIS_REPLY_STRING) {
        out_sha1 = std::string(reply->str, reply->len);
        return EC_OK;
    }

    KVCM_LOG_ERROR("SCRIPT LOAD command unexpected reply type: %d", reply->type);
    return EC_ERROR;
}

ErrorCode RedisClientExt::ScriptExists(const std::string &sha1, bool &out_exists) {
    out_exists = false;

    if (!IsContextOk()) {
        KVCM_LOG_ERROR("Redis context not ok for SCRIPT EXISTS");
        return EC_IO_ERROR;
    }

    std::vector<CmdArgs> cmds = {{"SCRIPT", "EXISTS", sha1}};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_LOG_ERROR("SCRIPT EXISTS command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_LOG_ERROR("SCRIPT EXISTS command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    if (reply->type == REDIS_REPLY_ARRAY && reply->elements == 1) {
        const redisReply *element = reply->element[0];
        if (element->type == REDIS_REPLY_INTEGER) {
            out_exists = (element->integer == 1);
            return EC_OK;
        }
    }

    KVCM_LOG_ERROR("SCRIPT EXISTS command unexpected reply type: %d", reply->type);
    return EC_ERROR;
}

ErrorCode RedisClientExt::Get(const std::string &key, std::string &out_value) {
    out_value.clear();

    if (!IsContextOk()) {
        KVCM_LOG_ERROR("Redis context not ok for GET");
        return EC_IO_ERROR;
    }

    std::vector<CmdArgs> cmds = {{"GET", key}};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_LOG_ERROR("GET command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_LOG_ERROR("GET command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    if (reply->type == REDIS_REPLY_NIL) {
        // 键不存在
        return EC_NOENT;
    }

    if (reply->type == REDIS_REPLY_STRING) {
        out_value = std::string(reply->str, reply->len);
        return EC_OK;
    }

    KVCM_LOG_ERROR("GET command unexpected reply type: %d", reply->type);
    return EC_ERROR;
}

ErrorCode RedisClientExt::Set(const std::string &key, const std::string &value, int64_t ttl_ms) {
    if (!IsContextOk()) {
        KVCM_LOG_ERROR("Redis context not ok for SET");
        return EC_IO_ERROR;
    }

    std::vector<CmdArgs> cmds;
    if (ttl_ms > 0) {
        cmds = {{"SET", key, value, "PX", std::to_string(ttl_ms)}};
    } else {
        cmds = {{"SET", key, value}};
    }

    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_LOG_ERROR("SET command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_LOG_ERROR("SET command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    // SET命令成功返回"OK"
    if (reply->type == REDIS_REPLY_STATUS && std::string(reply->str, reply->len) == "OK") {
        return EC_OK;
    }

    KVCM_LOG_ERROR("SET command unexpected reply");
    return EC_ERROR;
}

ErrorCode RedisClientExt::Pttl(const std::string &key, int64_t &out_ttl_ms) {
    out_ttl_ms = -2; // Redis中-2表示键不存在

    if (!IsContextOk()) {
        KVCM_LOG_ERROR("Redis context not ok for PTTL");
        return EC_IO_ERROR;
    }

    std::vector<CmdArgs> cmds = {{"PTTL", key}};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_LOG_ERROR("PTTL command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_LOG_ERROR("PTTL command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    if (reply->type == REDIS_REPLY_INTEGER) {
        out_ttl_ms = reply->integer;
        return EC_OK;
    }

    KVCM_LOG_ERROR("PTTL command unexpected reply type: %d", reply->type);
    return EC_ERROR;
}

ErrorCode RedisClientExt::Del(const std::string &key) {
    if (!IsContextOk()) {
        KVCM_LOG_ERROR("Redis context not ok for DEL");
        return EC_IO_ERROR;
    }

    std::vector<CmdArgs> cmds = {{"DEL", key}};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_LOG_ERROR("DEL command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_LOG_ERROR("DEL command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    if (reply->type == REDIS_REPLY_INTEGER) {
        // 返回删除的键数量
        return EC_OK;
    }

    KVCM_LOG_ERROR("DEL command unexpected reply type: %d", reply->type);
    return EC_ERROR;
}

ErrorCode RedisClientExt::Pexpire(const std::string &key, int64_t ttl_ms) {
    if (!IsContextOk()) {
        KVCM_LOG_ERROR("Redis context not ok for PEXPIRE");
        return EC_IO_ERROR;
    }

    if (ttl_ms <= 0) {
        KVCM_LOG_ERROR("Invalid TTL for PEXPIRE: %ld", ttl_ms);
        return EC_BADARGS;
    }

    std::vector<CmdArgs> cmds = {{"PEXPIRE", key, std::to_string(ttl_ms)}};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_LOG_ERROR("PEXPIRE command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_LOG_ERROR("PEXPIRE command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    if (reply->type == REDIS_REPLY_INTEGER) {
        // 1表示成功设置过期时间，0表示键不存在或设置失败
        if (reply->integer == 1) {
            return EC_OK;
        } else {
            return EC_NOENT;
        }
    }

    KVCM_LOG_ERROR("PEXPIRE command unexpected reply type: %d", reply->type);
    return EC_ERROR;
}
ErrorCode RedisClientExt::FlushAll() {
    CmdArgs flushall_cmd{"FLUSHALL"};
    std::vector<ReplyUPtr> flushall_replies = CommandPipeline({flushall_cmd});
    if (1 != flushall_replies.size()) {
        KVCM_LOG_ERROR("redis flushall fail, pipeline [1] != flushall_replies.size[%zu]", flushall_replies.size());
        return EC_ERROR;
    }

    const ReplyUPtr &flushall_reply = flushall_replies[0];
    if (!IsReplyOk(flushall_reply.get())) {
        KVCM_LOG_ERROR("redis flushall fail");
        return EC_ERROR;
    }

    // FLUSHALL 命令返回 "OK" 字符串
    if (flushall_reply->type != REDIS_REPLY_STATUS) {
        KVCM_LOG_ERROR("redis flushall fail, unexpected reply type[%d]", flushall_reply->type);
        return EC_ERROR;
    }

    static const std::string ok_str = "OK";
    if (!flushall_reply->str || std::string(flushall_reply->str) != ok_str) {
        KVCM_LOG_ERROR("redis flushall fail, reply str[%s] is not OK",
                       flushall_reply->str ? flushall_reply->str : "nullptr");
        return EC_ERROR;
    }

    return EC_OK;
}

ErrorCode RedisClientExt::LoadScript(const std::string &script, std::string &out_sha1) {
    ErrorCode ec = ScriptLoad(script, out_sha1);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to load Lua script: ec=%d", ec);
        return ec;
    }

    KVCM_LOG_DEBUG("Loaded Lua script with SHA1: %s", out_sha1.c_str());
    return EC_OK;
}

ErrorCode RedisClientExt::ExecuteScriptWithFallback(const std::string &script,
                                                    const std::vector<std::string> &keys,
                                                    const std::vector<std::string> &args,
                                                    std::string &in_out_cached_sha1,
                                                    std::string &out_result) {
    // 首先尝试使用evalsha
    ErrorCode ec = EvalSha(in_out_cached_sha1, keys, args, out_result);
    if (ec == EC_OK) {
        // evalsha成功
        return EC_OK;
    } else if (ec == EC_NOSCRIPT) {
        // 脚本未加载，重新加载脚本
        KVCM_LOG_WARN("Script not loaded in Redis, reloading: %s", in_out_cached_sha1.c_str());

        std::string new_sha1;
        ec = ScriptLoad(script, new_sha1);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("Failed to reload Lua script: ec=%d", ec);
            return ec;
        }

        // 重新尝试evalsha
        ec = EvalSha(new_sha1, keys, args, out_result);
        if (ec == EC_OK) {
            // 更新缓存
            in_out_cached_sha1 = new_sha1;
            return EC_OK;
        }
    }

    // 如果evalsha失败且不是NOSCRIPT错误，或者重新加载后仍然失败，回退到eval
    KVCM_LOG_WARN("Fallback to EVAL for script: %s", in_out_cached_sha1.c_str());
    return Eval(script, keys, args, out_result);
}

} // namespace kv_cache_manager