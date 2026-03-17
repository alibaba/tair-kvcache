#include "kv_cache_manager/common/redis_client.h"

#include <cassert>
#include <unordered_set>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"
#include "unistd.h"

namespace kv_cache_manager {

#define KVCM_REDIS_LOG_INFO(format, ...)                                                                               \
    KVCM_LOG_INFO("redis client addr[%s:%ld] user[%s] " format, host_.c_str(), port_, user_info_.c_str(), ##__VA_ARGS__)
#define KVCM_REDIS_LOG_WARN(format, ...)                                                                               \
    KVCM_LOG_WARN("redis client addr[%s:%ld] user[%s] " format, host_.c_str(), port_, user_info_.c_str(), ##__VA_ARGS__)
#define KVCM_REDIS_LOG_ERROR(format, ...)                                                                              \
    KVCM_LOG_ERROR(                                                                                                    \
        "redis client addr[%s:%ld] user[%s] " format, host_.c_str(), port_, user_info_.c_str(), ##__VA_ARGS__)

RedisClient::RedisClient(const StandardUri &storage_uri)
    : user_info_(storage_uri.GetUserInfo()), host_(storage_uri.GetHostName()), port_(storage_uri.GetPort()) {
    int64_t tmp_timeout_ms = 0;
    storage_uri.GetParamAs("timeout_ms", tmp_timeout_ms);
    if (tmp_timeout_ms > 0) {
        timeout_ms_ = tmp_timeout_ms;
    }
    int64_t tmp_retry_count = 0;
    storage_uri.GetParamAs("retry_count", tmp_retry_count);
    if (tmp_retry_count > 0) {
        retry_count_ = tmp_retry_count;
    }
    int64_t tmp_randomkey_batch_num = 0;
    storage_uri.GetParamAs("randomkey_batch_num", tmp_randomkey_batch_num);
    if (tmp_randomkey_batch_num > 0) {
        randomkey_batch_num_ = tmp_randomkey_batch_num;
    }
    int64_t tmp_randomkey_key_per_eval = 0;
    storage_uri.GetParamAs("randomkey_key_per_eval", tmp_randomkey_key_per_eval);
    if (tmp_randomkey_key_per_eval > 0) {
        randomkey_key_per_eval_ = tmp_randomkey_key_per_eval;
    }
}

RedisClient::~RedisClient() { Close(); }

bool RedisClient::IsContextOk() const {
    if (!context_ || context_->err) {
        std::string msg = (context_ ? context_->errstr : "Cannot allocate redis context");
        KVCM_REDIS_LOG_WARN("redis context error[%s]", msg.c_str());
        return false;
    }
    return true;
}

bool RedisClient::IsReplyOk(const redisReply *reply) const {
    if (!reply || reply->type == REDIS_REPLY_ERROR) {
        std::string msg = (reply ? reply->str : "invalid nullptr redis reply");
        KVCM_REDIS_LOG_WARN("redis reply error[%s]", msg.c_str());
        return false;
    }
    return true;
}

bool RedisClient::CheckReplyInteger(const redisReply *reply) const {
    if (!reply) {
        KVCM_REDIS_LOG_ERROR("redis check reply integer, reply is nullptr");
        return false;
    }
    if (reply->type != REDIS_REPLY_INTEGER || reply->integer < 0) {
        KVCM_REDIS_LOG_ERROR(
            "redis reply check integer type fail, type[%d] integer[%lld] ", reply->type, reply->integer);
        return false;
    }
    return true;
}

bool RedisClient::CheckReplyArray(const redisReply *reply) const {
    if (!reply) {
        KVCM_REDIS_LOG_ERROR("redis check reply array fail, reply is nullptr");
        return false;
    }
    if (reply->type != REDIS_REPLY_ARRAY) {
        KVCM_REDIS_LOG_ERROR("redis reply check array type fail, type[%d]", reply->type);
        return false;
    }
    return true;
}

// out_str is empty if it is nil
bool RedisClient::GetReplyStrOrNil(const redisReply *reply, std::string &out_str) const {
    out_str.clear();
    if (!reply) {
        KVCM_REDIS_LOG_ERROR("redis get reply str or nil fail, reply is nullptr");
        return false;
    }
    if (reply->type != REDIS_REPLY_STRING && reply->type != REDIS_REPLY_NIL) {
        KVCM_REDIS_LOG_ERROR("redis get reply str or nil fail, unexpected reply type[%d]", reply->type);
        return false;
    }
    out_str = (reply->type == REDIS_REPLY_STRING ? std::string(reply->str) : std::string());
    return true;
}

bool RedisClient::Connect() {
    Disconnect();

    struct timeval timeout;
    timeout.tv_sec = timeout_ms_ / 1000;
    timeout.tv_usec = (timeout_ms_ % 1000) * 1000;
    context_ = redisConnectWithTimeout(host_.c_str(), port_, timeout);
    if (!IsContextOk()) {
        KVCM_REDIS_LOG_WARN("fail to connect to redis");
        return false;
    }

    ReplyUPtr ping_reply_1((redisReply *)redisCommand(context_, "PING"), freeReplyObject);
    if (!ping_reply_1) {
        KVCM_REDIS_LOG_WARN("want to ping first to check if auth needed, but ping fail, get nullptr reply");
        return false;
    }
    if (IsReplyOk(ping_reply_1.get())) {
        // no need to auth, do nothing
    } else {
        assert(ping_reply_1->type == REDIS_REPLY_ERROR);
        std::string error_str(ping_reply_1->str);
        static const std::string auth_str = "NOAUTH";
        if (error_str.size() >= auth_str.size() && error_str.compare(0, auth_str.size(), auth_str.c_str()) == 0) {
            // do auth
            ReplyUPtr auth_reply((redisReply *)redisCommand(context_, "AUTH %s", user_info_.c_str()), freeReplyObject);
            if (!IsReplyOk(auth_reply.get())) {
                KVCM_REDIS_LOG_WARN("auth fail");
                return false;
            }
            ReplyUPtr ping_reply_2((redisReply *)redisCommand(context_, "PING"), freeReplyObject);
            if (!IsReplyOk(ping_reply_2.get())) {
                KVCM_REDIS_LOG_WARN("ping fail after auth");
                return false;
            }
            KVCM_REDIS_LOG_INFO("connect to redis, auth successfully");
        } else {
            KVCM_REDIS_LOG_WARN("unexpected ping error str[%s]", error_str.c_str());
            return false;
        }
    }

    KVCM_REDIS_LOG_INFO("connect to redis timeout_ms[%ld] retry_count[%ld]", timeout_ms_, retry_count_);
    return true;
}

void RedisClient::Disconnect() {
    if (context_) {
        redisFree(context_);
        context_ = nullptr;
    }
}

bool RedisClient::Reconnect() {
    for (int32_t count = 0; count < retry_count_; ++count) {
        if (Connect()) {
            return true;
        }
        usleep(50 * 1000);
    }
    KVCM_REDIS_LOG_ERROR("fail to reconnect redis after [%ld] times", retry_count_);
    return false;
}

// return empty vector if failed
std::vector<RedisClient::ReplyUPtr> RedisClient::TryExecPipeline(const std::vector<CmdArgs> &cmds) {
    std::vector<ReplyUPtr> replies;
    for (const CmdArgs &cmd : cmds) {
        std::vector<const char *> argv;
        std::vector<size_t> argvlen;
        argv.reserve(cmd.size());
        argvlen.reserve(cmd.size());
        for (const auto &cmd_arg : cmd) {
            argv.push_back(cmd_arg.data());
            argvlen.push_back(cmd_arg.size());
        }
        if (redisAppendCommandArgv(context_, (int)argv.size(), argv.data(), argvlen.data()) != REDIS_OK) {
            KVCM_REDIS_LOG_WARN("redisAppendCommandArgv failed: %s", context_->errstr);
            return replies;
        }
    }

    replies.reserve(cmds.size());
    for (size_t i = 0; i < cmds.size(); ++i) {
        redisReply *r = nullptr;
        if (redisGetReply(context_, (void **)&r) != REDIS_OK) {
            KVCM_REDIS_LOG_WARN("redisGetReply failed: %s", context_->errstr);
            freeReplyObject(r);
            replies.clear();
            return replies;
        }
        replies.emplace_back(r, freeReplyObject);
    }
    return replies;
}

// return empty vector if failed
std::vector<RedisClient::ReplyUPtr> RedisClient::CommandPipeline(const std::vector<CmdArgs> &cmds) {
    std::vector<ReplyUPtr> replies;
    if (cmds.empty()) {
        return replies;
    }
    for (int32_t count = 0; count < retry_count_; ++count) {
        if (!IsContextOk()) {
            if (!Reconnect()) {
                KVCM_REDIS_LOG_ERROR("fail to reconnect before pipeline, try count[%d]", count);
                replies.clear();
                return replies;
            }
        }
        if (IsContextOk()) {
            replies = TryExecPipeline(cmds);
            if (!replies.empty()) {
                return replies;
            } else if (IsContextOk()) {
                KVCM_REDIS_LOG_ERROR("pipeline fail but connection is ok, try count[%d]", count);
                return replies;
            } else {
                KVCM_REDIS_LOG_WARN("pipeline fail, connection not ok, try count[%d]", count);
            }
        }
        usleep(50 * 1000);
    }

    KVCM_REDIS_LOG_ERROR("pipeline all fail, try count[%ld]", retry_count_);
    replies.clear();
    return replies;
}

bool RedisClient::Open() {
    if (!Reconnect()) {
        KVCM_REDIS_LOG_ERROR("fail to connect in open");
        return false;
    }
    return true;
}

void RedisClient::Close() { Disconnect(); }

// cover old key-fields
std::vector<ErrorCode> RedisClient::Set(const std::vector<std::string> &keys,
                                        const std::vector<std::map<std::string, std::string>> &field_maps) {
    if (keys.size() != field_maps.size()) {
        KVCM_REDIS_LOG_ERROR("redis set fail, keys.size[%lu] != field_maps.size[%lu]", keys.size(), field_maps.size());
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }

    std::vector<CmdArgs> cmds;
    cmds.reserve(keys.size() * 2);
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string &key = keys[i];
        const std::map<std::string, std::string> &field_map = field_maps[i];
        CmdArgs del_cmd{"DEL", key};
        CmdArgs hset_cmd;
        hset_cmd.reserve((field_map.size() + 1) * 2);
        hset_cmd.emplace_back("HSET");
        hset_cmd.emplace_back(key);
        for (const auto &[field_name, field_value] : field_map) {
            hset_cmd.emplace_back(field_name);
            hset_cmd.emplace_back(field_value);
        }
        cmds.emplace_back(std::move(del_cmd));
        cmds.emplace_back(std::move(hset_cmd));
    }

    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);
    if (cmds.size() != replies.size()) {
        KVCM_REDIS_LOG_ERROR(
            "redis set fail, pipeline cmds.size[%lu] != replies.size[%lu]", cmds.size(), replies.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    std::vector<ErrorCode> ec_per_key;
    ec_per_key.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const ReplyUPtr &del_reply = replies[i * 2];
        const ReplyUPtr &hset_reply = replies[i * 2 + 1];
        if (!IsReplyOk(del_reply.get()) || !CheckReplyInteger(del_reply.get())) {
            KVCM_REDIS_LOG_ERROR("redis set fail, key[%s] DEL fail", keys[i].c_str());
            ec_per_key.emplace_back(EC_ERROR);
        } else if (!IsReplyOk(hset_reply.get()) || !CheckReplyInteger(hset_reply.get())) {
            KVCM_REDIS_LOG_ERROR("redis set fail, key[%s] HSET fail", keys[i].c_str());
            ec_per_key.emplace_back(EC_ERROR);
        } else {
            ec_per_key.emplace_back(EC_OK);
        }
    }
    return ec_per_key;
}

// return EC_NOENT if key not exist
std::vector<ErrorCode> RedisClient::Update(const std::vector<std::string> &keys,
                                           const std::vector<std::map<std::string, std::string>> &field_maps) {
    if (keys.size() != field_maps.size()) {
        KVCM_REDIS_LOG_ERROR(
            "redis update fail, keys.size[%lu] != field_maps.size[%lu]", keys.size(), field_maps.size());
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }

    std::vector<bool> is_exist_vec;
    std::vector<ErrorCode> ec_per_key = Exists(keys, is_exist_vec);
    std::vector<CmdArgs> hset_cmds;
    hset_cmds.reserve(keys.size());
    size_t exist_count = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (ec_per_key[i] != EC_OK) {
            KVCM_REDIS_LOG_ERROR("redis update fail, key[%s] fail in exists", keys[i].c_str());
        } else if (!is_exist_vec[i]) {
            ec_per_key[i] = EC_NOENT;
        } else {
            const std::map<std::string, std::string> &field_map = field_maps[i];
            CmdArgs hset_cmd;
            hset_cmd.reserve((field_map.size() + 1) * 2);
            hset_cmd.emplace_back("HSET");
            hset_cmd.emplace_back(keys[i]);
            for (const auto &[field_name, field_value] : field_map) {
                hset_cmd.emplace_back(field_name);
                hset_cmd.emplace_back(field_value);
            }
            hset_cmds.emplace_back(std::move(hset_cmd));
            ++exist_count;
        }
    }
    if (exist_count <= 0) {
        return ec_per_key;
    }

    std::vector<ReplyUPtr> hset_replies = CommandPipeline(hset_cmds);
    if (hset_cmds.size() != hset_replies.size()) {
        KVCM_REDIS_LOG_ERROR("redis update fail, pipeline hset_cmds.size[%lu] != hset_replies.size[%lu]",
                             hset_cmds.size(),
                             hset_replies.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }

    size_t hset_reply_index = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (ec_per_key[i] != EC_OK) {
            continue;
        }
        const ReplyUPtr &hset_reply = hset_replies[hset_reply_index++];
        if (!IsReplyOk(hset_reply.get()) || !CheckReplyInteger(hset_reply.get())) {
            KVCM_REDIS_LOG_ERROR("redis update fail, key[%s] HSET fail", keys[i].c_str());
            ec_per_key[i] = EC_ERROR;
        }
    }
    return ec_per_key;
}

std::vector<ErrorCode> RedisClient::Upsert(const std::vector<std::string> &keys,
                                           const std::vector<std::map<std::string, std::string>> &field_maps) {
    if (keys.size() != field_maps.size()) {
        KVCM_REDIS_LOG_ERROR(
            "redis upsert fail, keys.size[%lu] != field_maps.size[%lu]", keys.size(), field_maps.size());
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }

    std::vector<CmdArgs> hset_cmds;
    hset_cmds.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::map<std::string, std::string> &field_map = field_maps[i];
        CmdArgs hset_cmd;
        hset_cmd.reserve((field_map.size() + 1) * 2);
        hset_cmd.emplace_back("HSET");
        hset_cmd.emplace_back(keys[i]);
        for (const auto &[field_name, field_value] : field_map) {
            hset_cmd.emplace_back(field_name);
            hset_cmd.emplace_back(field_value);
        }
        hset_cmds.emplace_back(std::move(hset_cmd));
    }

    std::vector<ReplyUPtr> hset_replies = CommandPipeline(hset_cmds);
    if (hset_cmds.size() != hset_replies.size()) {
        KVCM_REDIS_LOG_ERROR("redis upsert fail, pipeline hset_cmds.size[%lu] != hset_replies.size[%lu]",
                             hset_cmds.size(),
                             hset_replies.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    std::vector<ErrorCode> ec_per_key;
    ec_per_key.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const ReplyUPtr &hset_reply = hset_replies[i];
        if (!IsReplyOk(hset_reply.get()) || !CheckReplyInteger(hset_reply.get())) {
            KVCM_REDIS_LOG_ERROR("redis upsert fail, key[%s] HSET fail", keys[i].c_str());
            ec_per_key.emplace_back(EC_ERROR);
        } else {
            ec_per_key.emplace_back(EC_OK);
        }
    }
    return ec_per_key;
}

std::vector<ErrorCode> RedisClient::Delete(const std::vector<std::string> &keys) {
    std::vector<CmdArgs> del_cmds;
    del_cmds.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        CmdArgs del_cmd{"DEL", keys[i]};
        del_cmds.emplace_back(std::move(del_cmd));
    }

    std::vector<ReplyUPtr> del_replies = CommandPipeline(del_cmds);
    if (del_cmds.size() != del_replies.size()) {
        KVCM_REDIS_LOG_ERROR("redis delete fail, pipeline del_cmds.size[%lu] != del_replies.size[%lu]",
                             del_cmds.size(),
                             del_replies.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    std::vector<ErrorCode> ec_per_key;
    ec_per_key.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const ReplyUPtr &del_reply = del_replies[i];
        if (!IsReplyOk(del_reply.get()) || !CheckReplyInteger(del_reply.get())) {
            KVCM_REDIS_LOG_ERROR("redis delete fail, key[%s] DEL fail", keys[i].c_str());
            ec_per_key.emplace_back(EC_ERROR);
        } else if (del_reply->integer == 0) {
            ec_per_key.emplace_back(EC_NOENT);
        } else {
            ec_per_key.emplace_back(EC_OK);
        }
    }
    return ec_per_key;
}

std::vector<ErrorCode> RedisClient::Get(const std::vector<std::string> &keys,
                                        const std::vector<std::string> &field_names,
                                        std::vector<std::map<std::string, std::string>> &out_field_maps) {
    out_field_maps = std::vector<std::map<std::string, std::string>>(keys.size());

    std::vector<CmdArgs> hmget_cmds;
    hmget_cmds.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        CmdArgs hmget_cmd;
        hmget_cmd.reserve(field_names.size() + 2);
        hmget_cmd.emplace_back("HMGET");
        hmget_cmd.emplace_back(keys[i]);
        for (const auto &field_name : field_names) {
            hmget_cmd.emplace_back(field_name);
        }
        hmget_cmds.emplace_back(std::move(hmget_cmd));
    }
    std::vector<ReplyUPtr> hmget_replies = CommandPipeline(hmget_cmds);
    if (keys.size() != hmget_replies.size()) {
        KVCM_REDIS_LOG_ERROR(
            "redis get fail, pipeline keys.size[%lu] != hmget_replies.size[%lu]", keys.size(), hmget_replies.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }

    std::vector<ErrorCode> ec_per_key;
    ec_per_key.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const ReplyUPtr &hmget_reply = hmget_replies[i];
        if (!IsReplyOk(hmget_reply.get()) || !CheckReplyArray(hmget_reply.get())) {
            KVCM_REDIS_LOG_ERROR("redis get fail, key[%s] HMGET fail", keys[i].c_str());
            ec_per_key.emplace_back(EC_ERROR);
            continue;
        }
        if (hmget_reply->elements != field_names.size()) {
            KVCM_REDIS_LOG_ERROR("redis get fail, key[%s] HMGET reply elements[%lu] != field_names[%lu]",
                                 keys[i].c_str(),
                                 hmget_reply->elements,
                                 field_names.size());
            ec_per_key.emplace_back(EC_ERROR);
            continue;
        }

        bool hasError = false;
        std::map<std::string, std::string> &out_field_map = out_field_maps[i];
        for (size_t j = 0; j < hmget_reply->elements; ++j) {
            const std::string &field_name = field_names[j];
            const redisReply *field_value_reply = hmget_reply->element[j];
            std::string field_value;
            if (!GetReplyStrOrNil(field_value_reply, field_value)) {
                KVCM_REDIS_LOG_ERROR(
                    "redis get fail, key[%s] field_name[%s] get reply str fail", keys[i].c_str(), field_name.c_str());
                out_field_map.clear();
                hasError = true;
                break;
            }
            out_field_map.emplace(field_name, std::move(field_value));
        }
        ec_per_key.emplace_back(hasError ? EC_ERROR : EC_OK);
    }
    return ec_per_key;
}

std::vector<ErrorCode> RedisClient::GetAllFields(const std::vector<std::string> &keys,
                                                 std::vector<std::map<std::string, std::string>> &out_field_maps) {
    out_field_maps = std::vector<std::map<std::string, std::string>>(keys.size());

    std::vector<CmdArgs> hgetall_cmds;
    hgetall_cmds.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        CmdArgs hgetall_cmd{"HGETALL", keys[i]};
        hgetall_cmds.emplace_back(std::move(hgetall_cmd));
    }
    std::vector<ReplyUPtr> hgetall_replies = CommandPipeline(hgetall_cmds);
    if (hgetall_cmds.size() != hgetall_replies.size()) {
        KVCM_REDIS_LOG_ERROR("redis get all fields fail, pipeline hgetall_cmds.size[%lu] != hgetall_replies.size[%lu]",
                             hgetall_cmds.size(),
                             hgetall_replies.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }

    std::vector<ErrorCode> ec_per_key;
    ec_per_key.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const ReplyUPtr &hgetall_reply = hgetall_replies[i];
        if (!IsReplyOk(hgetall_reply.get()) || !CheckReplyArray(hgetall_reply.get())) {
            KVCM_REDIS_LOG_ERROR("redis get all fields fail, key[%s] HGETALL fail", keys[i].c_str());
            ec_per_key.emplace_back(EC_ERROR);
            continue;
        }
        if (hgetall_reply->elements == 0) {
            ec_per_key.emplace_back(EC_NOENT);
            continue;
        }
        if ((hgetall_reply->elements) & 1) { // elements should be even number
            KVCM_REDIS_LOG_ERROR("redis get all fields fail, key[%s] HGETALL reply elements[%lu] is not even",
                                 keys[i].c_str(),
                                 hgetall_reply->elements);
            ec_per_key.emplace_back(EC_ERROR);
            continue;
        }

        bool hasError = false;
        std::map<std::string, std::string> &out_field_map = out_field_maps[i];
        for (size_t j = 0; j < hgetall_reply->elements; j += 2) {
            const redisReply *field_name_reply = hgetall_reply->element[j];
            std::string field_name;
            if (!GetReplyStrOrNil(field_name_reply, field_name)) {
                KVCM_REDIS_LOG_ERROR(
                    "redis get all fields fail, key[%s] field name idx[%lu] get reply str fail", keys[i].c_str(), j);
                ec_per_key[i] = EC_ERROR;
                out_field_map.clear();
                break;
            }
            const redisReply *field_value_reply = hgetall_reply->element[j + 1];
            std::string field_value;
            if (!GetReplyStrOrNil(field_value_reply, field_value)) {
                KVCM_REDIS_LOG_ERROR(
                    "redis get all fields fail, key[%s] field value idx[%lu] get reply str fail", keys[i].c_str(), j);
                hasError = true;
                out_field_map.clear();
                break;
            }
            out_field_map.emplace(std::move(field_name), std::move(field_value));
        }
        ec_per_key.emplace_back(hasError ? EC_ERROR : EC_OK);
    }
    return ec_per_key;
}

std::vector<ErrorCode> RedisClient::Exists(const std::vector<std::string> &keys, std::vector<bool> &out_is_exist_vec) {
    out_is_exist_vec.resize(keys.size(), false);

    std::vector<CmdArgs> exists_cmds;
    exists_cmds.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        CmdArgs exist_cmd{"EXISTS", keys[i]};
        exists_cmds.emplace_back(std::move(exist_cmd));
    }

    std::vector<ReplyUPtr> exists_replies = CommandPipeline(exists_cmds);
    if (exists_cmds.size() != exists_replies.size()) {
        KVCM_REDIS_LOG_ERROR("redis exists fail, pipeline exists_cmds.size[%lu] != exists_replies.size[%lu]",
                             exists_cmds.size(),
                             exists_replies.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }

    std::vector<ErrorCode> ec_per_key;
    ec_per_key.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const ReplyUPtr &exists_reply = exists_replies[i];
        if (!IsReplyOk(exists_reply.get()) || !CheckReplyInteger(exists_reply.get())) {
            KVCM_REDIS_LOG_ERROR("redis exists fail, key[%s] EXISTS fail", keys[i].c_str());
            ec_per_key.emplace_back(EC_ERROR);
            out_is_exist_vec[i] = false;
        } else {
            ec_per_key.emplace_back(EC_OK);
            out_is_exist_vec[i] = (exists_reply->integer > 0);
        }
    }
    return ec_per_key;
}

ErrorCode RedisClient::Scan(const std::string &matching_prefix,
                            const std::string &cursor,
                            const int64_t limit,
                            std::string &out_next_cursor,
                            std::vector<std::string> &out_keys) {
    out_next_cursor.clear();
    out_keys.clear();

    std::string pattern = matching_prefix + "*";
    CmdArgs scan_cmd{"SCAN", cursor, "MATCH", pattern, "COUNT", std::to_string(limit)};
    std::vector<ReplyUPtr> scan_replies = CommandPipeline({scan_cmd});
    if (/*scan_cmds.size()*/ 1 != scan_replies.size()) {
        KVCM_REDIS_LOG_ERROR("redis scan fail, pipeline [1] != scan_replies.size[%lu]", scan_replies.size());
        return EC_ERROR;
    }

    const ReplyUPtr &scan_reply = scan_replies[0];
    if (!IsReplyOk(scan_reply.get()) || !CheckReplyArray(scan_reply.get())) {
        KVCM_REDIS_LOG_ERROR("redis scan fail");
        return EC_ERROR;
    }
    if ((scan_reply->elements) != 2) { // [next_cursor, [keys...]]
        KVCM_REDIS_LOG_ERROR("redis scan fail, scan reply elements[%lu] is not 2", scan_reply->elements);
        return EC_ERROR;
    }

    const redisReply *next_cursor_reply = scan_reply->element[0];
    const redisReply *keys_reply = scan_reply->element[1];
    if (!IsReplyOk(next_cursor_reply) || !GetReplyStrOrNil(next_cursor_reply, out_next_cursor)) {
        KVCM_REDIS_LOG_ERROR("redis scan fail, get next cursor fail");
        out_next_cursor.clear();
        return EC_ERROR;
    }
    if (!IsReplyOk(keys_reply) || !CheckReplyArray(keys_reply)) {
        KVCM_REDIS_LOG_ERROR("redis scan fail, check keys reply fail");
        out_next_cursor.clear();
        return EC_ERROR;
    }
    for (size_t i = 0; i < keys_reply->elements; ++i) {
        const redisReply *key_reply = keys_reply->element[i];
        std::string key;
        if (!IsReplyOk(key_reply) || !GetReplyStrOrNil(key_reply, key) || key.empty()) {
            KVCM_REDIS_LOG_ERROR("redis scan fail, get key from reply fail");
            out_next_cursor.clear();
            out_keys.clear();
            return EC_ERROR;
        }
        out_keys.emplace_back(std::move(key));
    }
    return EC_OK;
}

ErrorCode RedisClient::Eval(const std::string &script,
                            const std::vector<std::string> &keys,
                            const std::vector<std::string> &args,
                            std::string &out_result) {
    out_result.clear();

    if (!IsContextOk()) {
        KVCM_REDIS_LOG_ERROR("redis context not ok for EVAL");
        return EC_IO_ERROR;
    }

    std::vector<std::string> cmd_args;
    cmd_args.reserve(3 + keys.size() + args.size());
    cmd_args.emplace_back("EVAL");
    cmd_args.emplace_back(script);
    cmd_args.emplace_back(std::to_string(keys.size()));
    for (const auto &key : keys) {
        cmd_args.emplace_back(key);
    }
    for (const auto &arg : args) {
        cmd_args.emplace_back(arg);
    }

    std::vector<CmdArgs> cmds = {cmd_args};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_REDIS_LOG_ERROR("EVAL command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_REDIS_LOG_ERROR("EVAL command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    switch (reply->type) {
    case REDIS_REPLY_STRING:
        out_result = std::string(reply->str, reply->len);
        return EC_OK;
    case REDIS_REPLY_INTEGER:
        out_result = std::to_string(reply->integer);
        return EC_OK;
    case REDIS_REPLY_NIL:
        return EC_OK;
    case REDIS_REPLY_STATUS:
        out_result = std::string(reply->str, reply->len);
        return EC_OK;
    case REDIS_REPLY_ERROR:
        KVCM_REDIS_LOG_ERROR("EVAL command error: %s", reply->str);
        return EC_ERROR;
    default:
        KVCM_REDIS_LOG_ERROR("EVAL command unexpected reply type: %d", reply->type);
        return EC_ERROR;
    }
}

ErrorCode RedisClient::EvalSha(const std::string &sha1,
                               const std::vector<std::string> &keys,
                               const std::vector<std::string> &args,
                               std::string &out_result) {
    out_result.clear();

    if (!IsContextOk()) {
        KVCM_REDIS_LOG_ERROR("redis context not ok for EVALSHA");
        return EC_IO_ERROR;
    }

    std::vector<std::string> cmd_args;
    cmd_args.reserve(3 + keys.size() + args.size());
    cmd_args.emplace_back("EVALSHA");
    cmd_args.emplace_back(sha1);
    cmd_args.emplace_back(std::to_string(keys.size()));
    for (const auto &key : keys) {
        cmd_args.emplace_back(key);
    }
    for (const auto &arg : args) {
        cmd_args.emplace_back(arg);
    }

    std::vector<CmdArgs> cmds = {cmd_args};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_REDIS_LOG_ERROR("EVALSHA command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_REDIS_LOG_ERROR("EVALSHA command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    switch (reply->type) {
    case REDIS_REPLY_STRING:
        out_result = std::string(reply->str, reply->len);
        return EC_OK;
    case REDIS_REPLY_INTEGER:
        out_result = std::to_string(reply->integer);
        return EC_OK;
    case REDIS_REPLY_NIL:
        return EC_OK;
    case REDIS_REPLY_STATUS:
        out_result = std::string(reply->str, reply->len);
        return EC_OK;
    case REDIS_REPLY_ERROR:
        if (reply->str && std::string(reply->str).find("NOSCRIPT") != std::string::npos) {
            KVCM_REDIS_LOG_WARN("EVALSHA NOSCRIPT error for sha1: %s", sha1.c_str());
            return EC_NOSCRIPT;
        }
        KVCM_REDIS_LOG_ERROR("EVALSHA command error: %s", reply->str);
        return EC_ERROR;
    default:
        KVCM_REDIS_LOG_ERROR("EVALSHA command unexpected reply type: %d", reply->type);
        return EC_ERROR;
    }
}

ErrorCode RedisClient::ScriptLoad(const std::string &script, std::string &out_sha1) {
    out_sha1.clear();

    if (!IsContextOk()) {
        KVCM_REDIS_LOG_ERROR("redis context not ok for SCRIPT LOAD");
        return EC_IO_ERROR;
    }

    std::vector<CmdArgs> cmds = {{"SCRIPT", "LOAD", script}};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_REDIS_LOG_ERROR("SCRIPT LOAD command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_REDIS_LOG_ERROR("SCRIPT LOAD command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    if (reply->type == REDIS_REPLY_STRING) {
        out_sha1 = std::string(reply->str, reply->len);
        return EC_OK;
    }

    KVCM_REDIS_LOG_ERROR("SCRIPT LOAD command unexpected reply type: %d", reply->type);
    return EC_ERROR;
}

ErrorCode RedisClient::ScriptExists(const std::string &sha1, bool &out_exists) {
    out_exists = false;

    if (!IsContextOk()) {
        KVCM_REDIS_LOG_ERROR("redis context not ok for SCRIPT EXISTS");
        return EC_IO_ERROR;
    }

    std::vector<CmdArgs> cmds = {{"SCRIPT", "EXISTS", sha1}};
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);

    if (replies.empty()) {
        KVCM_REDIS_LOG_ERROR("SCRIPT EXISTS command failed, no reply");
        return EC_ERROR;
    }

    const redisReply *reply = replies[0].get();
    if (!IsReplyOk(reply)) {
        KVCM_REDIS_LOG_ERROR("SCRIPT EXISTS command failed: %s", reply ? reply->str : "null reply");
        return EC_ERROR;
    }

    if (reply->type == REDIS_REPLY_ARRAY && reply->elements == 1) {
        const redisReply *element = reply->element[0];
        if (element->type == REDIS_REPLY_INTEGER) {
            out_exists = (element->integer == 1);
            return EC_OK;
        }
    }

    KVCM_REDIS_LOG_ERROR("SCRIPT EXISTS command unexpected reply type: %d", reply->type);
    return EC_ERROR;
}

ErrorCode
RedisClient::Rand(const std::string &matching_prefix, const int64_t count, std::vector<std::string> &out_keys) {
    if (randomkey_key_per_eval_ > 0) {
        return RandByLuaBatch(matching_prefix, count, out_keys);
    } else {
        return RandByBatch(matching_prefix, count, out_keys);
    }
}

ErrorCode
RedisClient::RandByBatch(const std::string &matching_prefix, const int64_t count, std::vector<std::string> &out_keys) {
    out_keys.clear();

    std::vector<CmdArgs> randomkey_cmds(randomkey_batch_num_, CmdArgs{"RANDOMKEY"});
    std::unordered_set<std::string> seen;
    size_t consecutive_misses = 0;
    while (out_keys.size() < count && consecutive_misses < 3) {
        std::vector<ReplyUPtr> randomkey_replies = CommandPipeline(randomkey_cmds);
        if (randomkey_cmds.size() != randomkey_replies.size()) {
            KVCM_REDIS_LOG_ERROR("redis rand fail, pipeline randomkey_cmds.size[%lu] != randomkey_replies.size[%lu]",
                                 randomkey_cmds.size(),
                                 randomkey_replies.size());
            out_keys.clear();
            return EC_ERROR;
        }
        bool found = false;
        for (size_t i = 0; i < randomkey_replies.size(); ++i) {
            const ReplyUPtr &randomkey_reply = randomkey_replies[i];
            std::string key;
            if (!IsReplyOk(randomkey_reply.get()) || !GetReplyStrOrNil(randomkey_reply.get(), key)) {
                KVCM_REDIS_LOG_ERROR("redis rand fail, get rand key from reply fail");
                out_keys.clear();
                return EC_ERROR;
            }
            if (key.size() >= matching_prefix.size() && key.compare(0, matching_prefix.size(), matching_prefix) == 0) {
                if (seen.insert(key).second) {
                    out_keys.emplace_back(std::move(key));
                    found = true;
                    if (out_keys.size() >= count) {
                        break;
                    }
                }
            }
        }
        if (found) {
            consecutive_misses = 0;
        } else {
            ++consecutive_misses;
        }
    }
    return EC_OK;
}

namespace {
// Lua script: batch RANDOMKEY, filtering nil/false to avoid array truncation.
// KEYS[1]: routing key (for proxy slot routing, not actually used by the script)
// ARGV[1]: number of RANDOMKEY calls (M)
constexpr const char *kRandomKeyLuaScript = "local n = tonumber(ARGV[1]) or 1\n"
                                            "local results = {}\n"
                                            "local count = 0\n"
                                            "for i = 1, n do\n"
                                            "    local key = redis.call('RANDOMKEY')\n"
                                            "    if key then\n"
                                            "        count = count + 1\n"
                                            "        results[count] = key\n"
                                            "    end\n"
                                            "end\n"
                                            "return results\n";
} // anonymous namespace

ErrorCode RedisClient::RandByLuaBatch(const std::string &matching_prefix,
                                      const int64_t count,
                                      std::vector<std::string> &out_keys) {
    out_keys.clear();

    // Load Lua script if SHA is not cached
    if (randomkey_script_sha_.empty()) {
        ErrorCode ec = ScriptLoad(kRandomKeyLuaScript, randomkey_script_sha_);
        if (ec != EC_OK) {
            KVCM_REDIS_LOG_ERROR("redis rand fail, SCRIPT LOAD failed");
            return EC_ERROR;
        }
    }

    std::string randomkey_per_eval = std::to_string(randomkey_key_per_eval_);
    std::unordered_set<std::string> seen;
    size_t consecutive_misses = 0;

    while (out_keys.size() < static_cast<size_t>(count) && consecutive_misses < 3) {
        // Build pipeline_size EVALSHA commands, each with a different routing key
        std::vector<CmdArgs> eval_cmds;
        size_t pipeline_size = (std::min(randomkey_batch_num_, count - static_cast<int64_t>(out_keys.size())) +
                                randomkey_key_per_eval_ - 1) /
                               randomkey_key_per_eval_;
        eval_cmds.reserve(pipeline_size);
        for (int64_t i = 0; i < pipeline_size; ++i) {
            eval_cmds.push_back(
                {"EVALSHA", randomkey_script_sha_, "1", StringUtil::GenerateRandomString(8), randomkey_per_eval});
        }

        std::vector<ReplyUPtr> eval_replies = CommandPipeline(eval_cmds);
        if (static_cast<int64_t>(eval_replies.size()) != pipeline_size) {
            KVCM_REDIS_LOG_ERROR("redis rand fail, pipeline eval_cmds.size[%ld] != eval_replies.size[%lu]",
                                 pipeline_size,
                                 eval_replies.size());
            out_keys.clear();
            return EC_ERROR;
        }

        // Check for NOSCRIPT error — if so, reload script and retry this iteration
        bool noscript = false;
        for (const auto &reply : eval_replies) {
            if (reply && reply->type == REDIS_REPLY_ERROR && reply->str &&
                std::string(reply->str).find("NOSCRIPT") != std::string::npos) {
                noscript = true;
                break;
            }
        }
        if (noscript) {
            KVCM_REDIS_LOG_WARN("redis rand NOSCRIPT, reloading script");
            randomkey_script_sha_.clear();
            ErrorCode ec = ScriptLoad(kRandomKeyLuaScript, randomkey_script_sha_);
            if (ec != EC_OK) {
                KVCM_REDIS_LOG_ERROR("redis rand fail, SCRIPT LOAD retry failed");
                out_keys.clear();
                return EC_ERROR;
            }
            continue; // Retry this iteration with the new SHA
        }

        // Parse results
        bool found = false;
        for (size_t i = 0; i < eval_replies.size(); ++i) {
            const ReplyUPtr &reply = eval_replies[i];
            if (!reply || !IsReplyOk(reply.get())) {
                KVCM_REDIS_LOG_ERROR("redis rand fail, EVALSHA reply error at index[%lu]", i);
                out_keys.clear();
                return EC_ERROR;
            }
            if (reply->type == REDIS_REPLY_ARRAY) {
                for (size_t j = 0; j < reply->elements; ++j) {
                    const redisReply *elem = reply->element[j];
                    if (!elem || elem->type != REDIS_REPLY_STRING || !elem->str) {
                        continue; // skip nil elements
                    }
                    std::string key(elem->str, elem->len);
                    if (key.size() >= matching_prefix.size() &&
                        key.compare(0, matching_prefix.size(), matching_prefix) == 0) {
                        if (seen.insert(key).second) {
                            out_keys.emplace_back(std::move(key));
                            found = true;
                            if (out_keys.size() >= static_cast<size_t>(count)) {
                                return EC_OK;
                            }
                        }
                    }
                }
            }
            // REDIS_REPLY_NIL means empty DB for that EVAL, just skip
        }

        if (found) {
            consecutive_misses = 0;
        } else {
            ++consecutive_misses;
        }
    }
    return EC_OK;
}

#undef KVCM_REDIS_LOG_INFO
#undef KVCM_REDIS_LOG_WARN
#undef KVCM_REDIS_LOG_ERROR

} // namespace kv_cache_manager
