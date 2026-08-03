#include "kv_cache_manager/common/redis/redis_client.h"

#include <cassert>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

#define KVCM_REDIS_LOG_WARN(format, ...) KVCM_LOG_WARN("redis client[%s] " format, description_.c_str(), ##__VA_ARGS__)
#define KVCM_REDIS_LOG_ERROR(format, ...)                                                                              \
    KVCM_LOG_ERROR("redis client[%s] " format, description_.c_str(), ##__VA_ARGS__)

RedisClient::RedisClient(const StandardUri &storage_uri, std::shared_ptr<IRedisExecutor> executor)
    : executor_(std::move(executor))
    , description_(storage_uri.GetProtocol() + "://" + storage_uri.GetHostName() + ":" +
                   std::to_string(storage_uri.GetPort())) {
    int64_t tmp_randomkey_batch_num = 0;
    storage_uri.GetParamAs("randomkey_batch_num", tmp_randomkey_batch_num);
    if (tmp_randomkey_batch_num > 0) {
        randomkey_batch_num_ = tmp_randomkey_batch_num;
    }
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

// return empty vector if failed
std::vector<RedisClient::ReplyUPtr> RedisClient::CommandPipeline(const std::vector<CmdArgs> &cmds) {
    if (cmds.empty()) {
        return {};
    }
    if (!executor_) {
        KVCM_REDIS_LOG_ERROR("executor is closed");
        return {};
    }
    return executor_->ExecuteBatch(cmds);
}

std::vector<ErrorCode> RedisClient::BatchWrite(const std::vector<CmdArgs> &cmds, bool &out_all_ok) {
    out_all_ok = false;
    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);
    if (replies.size() != cmds.size()) {
        return std::vector<ErrorCode>(cmds.size(), EC_ERROR);
    }
    std::vector<ErrorCode> error_codes;
    error_codes.reserve(replies.size());
    bool has_error = false;
    for (ReplyUPtr &reply : replies) {
        if (!IsReplyOk(reply.get()) || !CheckReplyInteger(reply.get())) {
            error_codes.push_back(EC_ERROR);
            has_error = true;
        } else {
            error_codes.push_back(EC_OK);
        }
    }
    out_all_ok = !has_error;
    return error_codes;
}

bool RedisClient::Open() {
    if (!executor_ || !executor_->Open()) {
        KVCM_REDIS_LOG_ERROR("fail to connect in open");
        return false;
    }
    return true;
}

void RedisClient::Close() { executor_.reset(); }

bool RedisClient::IsReady() const noexcept { return executor_ && executor_->IsReady(); }

// --- Static command builders ---

void RedisClient::BuildSetCmds(const std::vector<std::string> &keys,
                               const std::vector<std::map<std::string, std::string>> &field_maps,
                               std::vector<CmdArgs> &out_cmds) {
    assert(keys.size() == field_maps.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        out_cmds.push_back({"DEL", keys[i]});
        if (!field_maps[i].empty()) {
            CmdArgs hset_cmd;
            hset_cmd.reserve(field_maps[i].size() * 2 + 2);
            hset_cmd.emplace_back("HSET");
            hset_cmd.emplace_back(keys[i]);
            for (const std::pair<const std::string, std::string> &field : field_maps[i]) {
                hset_cmd.emplace_back(field.first);
                hset_cmd.emplace_back(field.second);
            }
            out_cmds.emplace_back(std::move(hset_cmd));
        }
    }
}

void RedisClient::BuildHashSetCmds(const std::vector<std::string> &keys,
                                   const std::vector<std::map<std::string, std::string>> &field_maps,
                                   std::vector<CmdArgs> &out_cmds) {
    assert(keys.size() == field_maps.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (field_maps[i].empty())
            continue;
        CmdArgs hset_cmd;
        hset_cmd.reserve(field_maps[i].size() * 2 + 2);
        hset_cmd.emplace_back("HSET");
        hset_cmd.emplace_back(keys[i]);
        for (const std::pair<const std::string, std::string> &field : field_maps[i]) {
            hset_cmd.emplace_back(field.first);
            hset_cmd.emplace_back(field.second);
        }
        out_cmds.emplace_back(std::move(hset_cmd));
    }
}

void RedisClient::BuildDeleteCmds(const std::vector<std::string> &keys, std::vector<CmdArgs> &out_cmds) {
    for (const std::string &key : keys) {
        out_cmds.push_back({"DEL", key});
    }
}

void RedisClient::BuildHashDeleteCmds(const std::vector<std::string> &keys,
                                      const std::vector<std::vector<std::string>> &field_names_vec,
                                      std::vector<CmdArgs> &out_cmds) {
    assert(keys.size() == field_names_vec.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (field_names_vec[i].empty())
            continue;
        CmdArgs hdel_cmd;
        hdel_cmd.reserve(field_names_vec[i].size() + 2);
        hdel_cmd.emplace_back("HDEL");
        hdel_cmd.emplace_back(keys[i]);
        for (const std::string &field_name : field_names_vec[i]) {
            hdel_cmd.emplace_back(field_name);
        }
        out_cmds.emplace_back(std::move(hdel_cmd));
    }
}

// cover old key-fields
std::vector<ErrorCode> RedisClient::Set(const std::vector<std::string> &keys,
                                        const std::vector<std::map<std::string, std::string>> &field_maps) {
    if (keys.size() != field_maps.size()) {
        KVCM_REDIS_LOG_ERROR("redis set fail, keys.size[%lu] != field_maps.size[%lu]", keys.size(), field_maps.size());
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }

    std::vector<CmdArgs> cmds;
    cmds.reserve(keys.size() * 2);
    BuildSetCmds(keys, field_maps, cmds);

    std::vector<ReplyUPtr> replies = CommandPipeline(cmds);
    if (cmds.size() != replies.size()) {
        KVCM_REDIS_LOG_ERROR(
            "redis set fail, pipeline cmds.size[%lu] != replies.size[%lu]", cmds.size(), replies.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    std::vector<ErrorCode> ec_per_key;
    ec_per_key.reserve(keys.size());
    size_t reply_idx = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        const ReplyUPtr &del_reply = replies[reply_idx++];
        if (!IsReplyOk(del_reply.get()) || !CheckReplyInteger(del_reply.get())) {
            KVCM_REDIS_LOG_ERROR("redis set fail, key[%s] DEL fail", keys[i].c_str());
            ec_per_key.emplace_back(EC_ERROR);
            if (!field_maps[i].empty()) {
                ++reply_idx;
            }
        } else if (!field_maps[i].empty()) {
            const ReplyUPtr &hset_reply = replies[reply_idx++];
            if (!IsReplyOk(hset_reply.get()) || !CheckReplyInteger(hset_reply.get())) {
                KVCM_REDIS_LOG_ERROR("redis set fail, key[%s] HSET fail", keys[i].c_str());
                ec_per_key.emplace_back(EC_ERROR);
            } else {
                ec_per_key.emplace_back(EC_OK);
            }
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
        } else if (field_maps[i].empty()) {
            // nothing to update
        } else {
            const std::map<std::string, std::string> &field_map = field_maps[i];
            CmdArgs hset_cmd;
            hset_cmd.reserve((field_map.size() + 1) * 2);
            hset_cmd.emplace_back("HSET");
            hset_cmd.emplace_back(keys[i]);
            for (const std::pair<const std::string, std::string> &field : field_map) {
                hset_cmd.emplace_back(field.first);
                hset_cmd.emplace_back(field.second);
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
        if (ec_per_key[i] != EC_OK || field_maps[i].empty()) {
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
    BuildHashSetCmds(keys, field_maps, hset_cmds);

    std::vector<ReplyUPtr> hset_replies = CommandPipeline(hset_cmds);
    if (hset_cmds.size() != hset_replies.size()) {
        KVCM_REDIS_LOG_ERROR("redis upsert fail, pipeline hset_cmds.size[%lu] != hset_replies.size[%lu]",
                             hset_cmds.size(),
                             hset_replies.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    std::vector<ErrorCode> ec_per_key;
    ec_per_key.reserve(keys.size());
    size_t reply_idx = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (field_maps[i].empty()) {
            ec_per_key.emplace_back(EC_OK);
            continue;
        }
        const ReplyUPtr &hset_reply = hset_replies[reply_idx++];
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
    BuildDeleteCmds(keys, del_cmds);

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

std::vector<ErrorCode> RedisClient::DeleteFields(const std::vector<std::string> &keys,
                                                 const std::vector<std::vector<std::string>> &field_names_vec) {
    if (keys.size() != field_names_vec.size()) {
        KVCM_REDIS_LOG_ERROR("redis delete fields fail, keys.size[%lu] != field_names_vec.size[%lu]",
                             keys.size(),
                             field_names_vec.size());
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }

    std::vector<CmdArgs> hdel_cmds;
    std::vector<size_t> original_indexes;
    std::vector<ErrorCode> ec_per_key(keys.size(), EC_OK);
    hdel_cmds.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (field_names_vec[i].empty()) {
            continue;
        }
        original_indexes.emplace_back(i);
    }
    BuildHashDeleteCmds(keys, field_names_vec, hdel_cmds);
    if (hdel_cmds.empty()) {
        return ec_per_key;
    }

    std::vector<ReplyUPtr> hdel_replies = CommandPipeline(hdel_cmds);
    if (hdel_cmds.size() != hdel_replies.size()) {
        KVCM_REDIS_LOG_ERROR("redis delete fields fail, pipeline hdel_cmds.size[%lu] != hdel_replies.size[%lu]",
                             hdel_cmds.size(),
                             hdel_replies.size());
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    assert(original_indexes.size() == hdel_replies.size());
    for (size_t i = 0; i < original_indexes.size(); ++i) {
        const size_t original_idx = original_indexes[i];
        const ReplyUPtr &hdel_reply = hdel_replies[i];
        if (!IsReplyOk(hdel_reply.get()) || !CheckReplyInteger(hdel_reply.get())) {
            KVCM_REDIS_LOG_ERROR("redis delete fields fail, key[%s] HDEL fail", keys[original_idx].c_str());
            ec_per_key[original_idx] = EC_ERROR;
        }
    }
    return ec_per_key;
}

std::vector<ErrorCode> RedisClient::Get(const std::vector<std::string> &keys,
                                        const std::vector<std::string> &field_names,
                                        std::vector<std::map<std::string, std::string>> &out_field_maps) {
    if (field_names.empty()) {
        KVCM_REDIS_LOG_ERROR("invalid empty field names");
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }
    out_field_maps = std::vector<std::map<std::string, std::string>>(keys.size());

    std::vector<CmdArgs> hmget_cmds;
    hmget_cmds.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        CmdArgs hmget_cmd;
        hmget_cmd.reserve(field_names.size() + 2);
        hmget_cmd.emplace_back("HMGET");
        hmget_cmd.emplace_back(keys[i]);
        for (const std::string &field_name : field_names) {
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
            // Skip nil values: treat them as non-existent fields
            if (!field_value.empty()) {
                out_field_map.emplace(field_name, std::move(field_value));
            }
        }
        if (hasError) {
            ec_per_key.emplace_back(EC_ERROR);
        } else if (out_field_map.empty()) {
            // All fields are nil: key does not exist or has no matching fields
            ec_per_key.emplace_back(EC_NOENT);
        } else {
            ec_per_key.emplace_back(EC_OK);
        }
    }
    return ec_per_key;
}

std::vector<ErrorCode> RedisClient::Get(const std::vector<std::string> &keys,
                                        const std::vector<std::vector<std::string>> &field_names_vec,
                                        std::vector<std::map<std::string, std::string>> &out_field_maps) {
    if (keys.size() != field_names_vec.size()) {
        KVCM_REDIS_LOG_ERROR(
            "redis get fail, keys.size[%lu] != field_names_vec.size[%lu]", keys.size(), field_names_vec.size());
        return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
    }

    out_field_maps = std::vector<std::map<std::string, std::string>>(keys.size());

    std::vector<CmdArgs> hmget_cmds;
    hmget_cmds.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::vector<std::string> &field_names = field_names_vec[i];
        if (field_names.empty()) {
            KVCM_REDIS_LOG_ERROR("invalid empty field names for key[%s]", keys[i].c_str());
            return std::vector<ErrorCode>(keys.size(), EC_BADARGS);
        }
        CmdArgs hmget_cmd;
        hmget_cmd.reserve(field_names.size() + 2);
        hmget_cmd.emplace_back("HMGET");
        hmget_cmd.emplace_back(keys[i]);
        for (const std::string &field_name : field_names) {
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
        const std::vector<std::string> &field_names = field_names_vec[i];
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
            if (!field_value.empty()) {
                out_field_map.emplace(field_name, std::move(field_value));
            }
        }
        if (hasError) {
            ec_per_key.emplace_back(EC_ERROR);
        } else if (out_field_map.empty()) {
            ec_per_key.emplace_back(EC_NOENT);
        } else {
            ec_per_key.emplace_back(EC_OK);
        }
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
    out_is_exist_vec.assign(keys.size(), false);

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

std::vector<ErrorCode> RedisClient::ExistsFieldWithPrefix(const std::vector<std::string> &keys,
                                                          const std::string &field_prefix,
                                                          std::vector<bool> &out_exists_vec) {
    out_exists_vec.assign(keys.size(), false);

    // First check which keys exist. Non-existent keys get EC_NOENT immediately.
    std::vector<bool> key_exists_vec;
    std::vector<ErrorCode> ec_per_key = Exists(keys, key_exists_vec);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (ec_per_key[i] == EC_OK && !key_exists_vec[i]) {
            ec_per_key[i] = EC_NOENT;
        }
    }

    const std::string pattern = field_prefix + "*";
    const std::string count_hint = "1000";
    struct PendingKey {
        size_t original_index;
        std::string cursor;
    };
    std::vector<PendingKey> pending;
    pending.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (ec_per_key[i] == EC_OK) {
            pending.push_back({i, "0"});
        }
    }
    while (!pending.empty()) {
        std::vector<CmdArgs> hscan_cmds;
        hscan_cmds.reserve(pending.size());
        for (const PendingKey &entry : pending) {
            hscan_cmds.push_back(
                {"HSCAN", keys[entry.original_index], entry.cursor, "MATCH", pattern, "COUNT", count_hint});
        }

        std::vector<ReplyUPtr> replies = CommandPipeline(hscan_cmds);
        if (replies.size() != hscan_cmds.size()) {
            KVCM_REDIS_LOG_ERROR(
                "redis exists field with prefix fail, pipeline hscan_cmds.size[%lu] != replies.size[%lu]",
                hscan_cmds.size(),
                replies.size());
            for (PendingKey &entry : pending) {
                ec_per_key[entry.original_index] = EC_ERROR;
            }
            break;
        }

        std::vector<PendingKey> next_pending;
        for (size_t i = 0; i < pending.size(); ++i) {
            size_t original_idx = pending[i].original_index;
            const ReplyUPtr &hscan_reply = replies[i];
            if (!IsReplyOk(hscan_reply.get()) || !CheckReplyArray(hscan_reply.get()) || hscan_reply->elements != 2) {
                KVCM_REDIS_LOG_ERROR("redis exists field with prefix fail, key[%s] HSCAN fail",
                                     keys[original_idx].c_str());
                ec_per_key[original_idx] = EC_ERROR;
                continue;
            }

            const redisReply *next_cursor_reply = hscan_reply->element[0];
            const redisReply *fields_reply = hscan_reply->element[1];
            std::string next_cursor;
            if (!GetReplyStrOrNil(next_cursor_reply, next_cursor)) {
                KVCM_REDIS_LOG_ERROR("redis exists field with prefix fail, key[%s] get next cursor fail",
                                     keys[original_idx].c_str());
                ec_per_key[original_idx] = EC_ERROR;
                continue;
            }
            if (!CheckReplyArray(fields_reply)) {
                KVCM_REDIS_LOG_ERROR("redis exists field with prefix fail, key[%s] check fields reply fail",
                                     keys[original_idx].c_str());
                ec_per_key[original_idx] = EC_ERROR;
                continue;
            }
            // HSCAN returns [field1, value1, field2, value2, ...].
            // Skip tombstones (empty value) — they are not valid locations.
            bool found_non_empty = false;
            for (size_t j = 0; j + 1 < fields_reply->elements; j += 2) {
                std::string value;
                if (GetReplyStrOrNil(fields_reply->element[j + 1], value) && !value.empty()) {
                    found_non_empty = true;
                    break;
                }
            }
            if (found_non_empty) {
                out_exists_vec[original_idx] = true;
            } else if (next_cursor != "0") {
                next_pending.push_back({original_idx, std::move(next_cursor)});
            }
        }
        pending = std::move(next_pending);
    }
    return ec_per_key;
}

std::vector<ErrorCode>
RedisClient::GetFieldNamesWithPrefix(const std::vector<std::string> &keys,
                                     const std::string &field_prefix,
                                     std::vector<std::vector<std::string>> &out_field_names_vec) {
    out_field_names_vec.resize(keys.size());

    // First check which keys exist. Non-existent keys get EC_NOENT immediately.
    std::vector<bool> key_exists_vec;
    std::vector<ErrorCode> ec_per_key = Exists(keys, key_exists_vec);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (ec_per_key[i] == EC_OK && !key_exists_vec[i]) {
            ec_per_key[i] = EC_NOENT;
        }
    }

    const std::string pattern = field_prefix + "*";
    const std::string count_hint = "1000";
    struct PendingKey {
        size_t original_index;
        std::string cursor;
    };
    std::vector<PendingKey> pending;
    pending.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (ec_per_key[i] == EC_OK) {
            pending.push_back({i, "0"});
        }
    }
    while (!pending.empty()) {
        std::vector<CmdArgs> hscan_cmds;
        hscan_cmds.reserve(pending.size());
        for (const PendingKey &entry : pending) {
            hscan_cmds.push_back(
                {"HSCAN", keys[entry.original_index], entry.cursor, "MATCH", pattern, "COUNT", count_hint});
        }

        std::vector<ReplyUPtr> replies = CommandPipeline(hscan_cmds);
        if (replies.size() != hscan_cmds.size()) {
            KVCM_REDIS_LOG_ERROR(
                "redis list field names with prefix fail, pipeline hscan_cmds.size[%lu] != replies.size[%lu]",
                hscan_cmds.size(),
                replies.size());
            for (PendingKey &entry : pending) {
                ec_per_key[entry.original_index] = EC_ERROR;
            }
            break;
        }

        std::vector<PendingKey> next_pending;
        for (size_t i = 0; i < pending.size(); ++i) {
            size_t original_idx = pending[i].original_index;
            const ReplyUPtr &hscan_reply = replies[i];
            if (!IsReplyOk(hscan_reply.get()) || !CheckReplyArray(hscan_reply.get()) || hscan_reply->elements != 2) {
                KVCM_REDIS_LOG_ERROR("redis list field names with prefix fail, key[%s] HSCAN fail",
                                     keys[original_idx].c_str());
                ec_per_key[original_idx] = EC_ERROR;
                continue;
            }

            const redisReply *next_cursor_reply = hscan_reply->element[0];
            const redisReply *fields_reply = hscan_reply->element[1];
            std::string next_cursor;
            if (!GetReplyStrOrNil(next_cursor_reply, next_cursor)) {
                KVCM_REDIS_LOG_ERROR("redis list field names with prefix fail, key[%s] get next cursor fail",
                                     keys[original_idx].c_str());
                ec_per_key[original_idx] = EC_ERROR;
                continue;
            }
            if (!CheckReplyArray(fields_reply)) {
                KVCM_REDIS_LOG_ERROR("redis list field names with prefix fail, key[%s] check fields reply fail",
                                     keys[original_idx].c_str());
                ec_per_key[original_idx] = EC_ERROR;
                continue;
            }
            // HSCAN returns [field1, value1, field2, value2, ...]; collect only
            // field names (even indices) whose value is non-empty (skip tombstones).
            for (size_t j = 0; j + 1 < fields_reply->elements; j += 2) {
                std::string field_name;
                std::string field_value;
                if (GetReplyStrOrNil(fields_reply->element[j], field_name) && !field_name.empty() &&
                    GetReplyStrOrNil(fields_reply->element[j + 1], field_value) && !field_value.empty()) {
                    out_field_names_vec[original_idx].emplace_back(std::move(field_name));
                }
            }
            if (next_cursor != "0") {
                next_pending.push_back({original_idx, std::move(next_cursor)});
            }
        }
        pending = std::move(next_pending);
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

ErrorCode
RedisClient::Rand(const std::string &matching_prefix, const int64_t count, std::vector<std::string> &out_keys) {
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

ErrorCode RedisClient::Eval(const std::string &script,
                            const std::vector<std::string> &keys,
                            const std::vector<std::string> &args,
                            std::string &out_result) {
    out_result.clear();

    if (!IsReady()) {
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
    for (const std::string &key : keys) {
        cmd_args.emplace_back(key);
    }

    // 添加ARGV
    for (const std::string &arg : args) {
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

ErrorCode RedisClient::EvalSha(const std::string &sha1,
                               const std::vector<std::string> &keys,
                               const std::vector<std::string> &args,
                               std::string &out_result) {
    out_result.clear();

    if (!IsReady()) {
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
    for (const std::string &key : keys) {
        cmd_args.emplace_back(key);
    }

    // 添加ARGV
    for (const std::string &arg : args) {
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
    if (reply && reply->type == REDIS_REPLY_ERROR && reply->str &&
        std::string(reply->str, reply->len).find("NOSCRIPT") != std::string::npos) {
        KVCM_LOG_WARN("EVALSHA NOSCRIPT error for sha1: %s", sha1.c_str());
        return EC_NOSCRIPT;
    }
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
    default:
        KVCM_LOG_ERROR("EVALSHA command unexpected reply type: %d", reply->type);
        return EC_ERROR;
    }
}

ErrorCode RedisClient::ScriptLoad(const std::string &script, std::string &out_sha1) {
    out_sha1.clear();

    if (!IsReady()) {
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

ErrorCode RedisClient::ScriptExists(const std::string &sha1, bool &out_exists) {
    out_exists = false;

    if (!IsReady()) {
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

ErrorCode RedisClient::Get(const std::string &key, std::string &out_value) {
    out_value.clear();

    if (!IsReady()) {
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

ErrorCode RedisClient::Set(const std::string &key, const std::string &value, int64_t ttl_ms) {
    if (!IsReady()) {
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

ErrorCode RedisClient::Pttl(const std::string &key, int64_t &out_ttl_ms) {
    out_ttl_ms = -2; // Redis中-2表示键不存在

    if (!IsReady()) {
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

ErrorCode RedisClient::Del(const std::string &key) {
    if (!IsReady()) {
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

ErrorCode RedisClient::Pexpire(const std::string &key, int64_t ttl_ms) {
    if (!IsReady()) {
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
ErrorCode RedisClient::FlushAll() {
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

ErrorCode RedisClient::LoadScript(const std::string &script, std::string &out_sha1) {
    ErrorCode ec = ScriptLoad(script, out_sha1);
    if (ec != EC_OK) {
        KVCM_LOG_ERROR("Failed to load Lua script: ec=%d", ec);
        return ec;
    }

    KVCM_LOG_DEBUG("Loaded Lua script with SHA1: %s", out_sha1.c_str());
    return EC_OK;
}

ErrorCode RedisClient::ExecuteScriptWithFallback(const std::string &script,
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

#undef KVCM_REDIS_LOG_WARN
#undef KVCM_REDIS_LOG_ERROR

} // namespace kv_cache_manager
