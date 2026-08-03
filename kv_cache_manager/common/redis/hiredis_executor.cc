#include "kv_cache_manager/common/redis/hiredis_executor.h"

#include <unistd.h>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

#define KVCM_HIREDIS_LOG_INFO(format, ...)                                                                             \
    KVCM_LOG_INFO("redis executor addr[%s:%ld] " format, host_.c_str(), port_, ##__VA_ARGS__)
#define KVCM_HIREDIS_LOG_WARN(format, ...)                                                                             \
    KVCM_LOG_WARN("redis executor addr[%s:%ld] " format, host_.c_str(), port_, ##__VA_ARGS__)
#define KVCM_HIREDIS_LOG_ERROR(format, ...)                                                                            \
    KVCM_LOG_ERROR("redis executor addr[%s:%ld] " format, host_.c_str(), port_, ##__VA_ARGS__)

HiredisExecutor::HiredisExecutor(const StandardUri &storage_uri)
    : user_info_(storage_uri.GetUserInfo()), host_(storage_uri.GetHostName()), port_(storage_uri.GetPort()) {
    int64_t db = 0;
    storage_uri.GetParamAs("db", db);
    if (db >= 0) {
        db_ = db;
    }
    int64_t timeout_ms = 0;
    storage_uri.GetParamAs("timeout_ms", timeout_ms);
    if (timeout_ms > 0) {
        timeout_ms_ = timeout_ms;
    }
    int64_t retry_count = 0;
    storage_uri.GetParamAs("retry_count", retry_count);
    if (retry_count > 0) {
        retry_count_ = retry_count;
    }
}

HiredisExecutor::~HiredisExecutor() { Disconnect(); }

bool HiredisExecutor::IsReady() const noexcept { return context_ && context_->err == REDIS_OK; }

bool HiredisExecutor::Connect() {
    Disconnect();

    struct timeval timeout;
    timeout.tv_sec = timeout_ms_ / 1000;
    timeout.tv_usec = (timeout_ms_ % 1000) * 1000;
    context_ = redisConnectWithTimeout(host_.c_str(), port_, timeout);
    if (!IsReady()) {
        const std::string message = context_ ? context_->errstr : "Cannot allocate redis context";
        KVCM_HIREDIS_LOG_WARN("fail to connect: %s", message.c_str());
        return false;
    }

    RedisReplyUPtr first_ping(static_cast<redisReply *>(redisCommand(context_, "PING")));
    if (!first_ping) {
        KVCM_HIREDIS_LOG_WARN("first PING returned an empty reply");
        return false;
    }
    if (first_ping->type == REDIS_REPLY_ERROR) {
        const std::string error(first_ping->str, first_ping->len);
        static const std::string no_auth = "NOAUTH";
        if (error.compare(0, no_auth.size(), no_auth) != 0) {
            KVCM_HIREDIS_LOG_WARN("unexpected PING error[%s]", error.c_str());
            return false;
        }

        RedisReplyUPtr auth_reply(static_cast<redisReply *>(redisCommand(context_, "AUTH %s", user_info_.c_str())));
        if (!auth_reply || auth_reply->type == REDIS_REPLY_ERROR) {
            KVCM_HIREDIS_LOG_WARN("AUTH failed");
            return false;
        }
        RedisReplyUPtr second_ping(static_cast<redisReply *>(redisCommand(context_, "PING")));
        if (!second_ping || second_ping->type == REDIS_REPLY_ERROR) {
            KVCM_HIREDIS_LOG_WARN("PING failed after AUTH");
            return false;
        }
    }

    if (db_ > 0) {
        RedisReplyUPtr select_reply(static_cast<redisReply *>(redisCommand(context_, "SELECT %ld", db_)));
        if (!select_reply || select_reply->type == REDIS_REPLY_ERROR) {
            KVCM_HIREDIS_LOG_WARN("fail to select db[%ld]", db_);
            return false;
        }
    }

    KVCM_HIREDIS_LOG_INFO("connected, db[%ld] timeout_ms[%ld] retry_count[%ld]", db_, timeout_ms_, retry_count_);
    return true;
}

bool HiredisExecutor::Reconnect() {
    for (int32_t count = 0; count < retry_count_; ++count) {
        if (Connect()) {
            return true;
        }
        usleep(50 * 1000);
    }
    KVCM_HIREDIS_LOG_ERROR("fail to reconnect after [%ld] attempts", retry_count_);
    return false;
}

void HiredisExecutor::Disconnect() {
    if (context_) {
        redisFree(context_);
        context_ = nullptr;
    }
}

bool HiredisExecutor::Open() { return Reconnect(); }

std::vector<RedisReplyUPtr> HiredisExecutor::TryExecuteBatch(const std::vector<RedisCmdArgs> &cmds) {
    std::vector<RedisReplyUPtr> replies;
    for (const RedisCmdArgs &cmd : cmds) {
        std::vector<const char *> argv;
        std::vector<size_t> argv_lengths;
        argv.reserve(cmd.size());
        argv_lengths.reserve(cmd.size());
        for (const std::string &arg : cmd) {
            argv.push_back(arg.data());
            argv_lengths.push_back(arg.size());
        }
        const int append_result =
            redisAppendCommandArgv(context_, static_cast<int>(argv.size()), argv.data(), argv_lengths.data());
        if (append_result != REDIS_OK) {
            KVCM_HIREDIS_LOG_WARN("redisAppendCommandArgv failed: %s", context_->errstr);
            Disconnect();
            return replies;
        }
    }

    replies.reserve(cmds.size());
    for (size_t i = 0; i < cmds.size(); ++i) {
        redisReply *reply = nullptr;
        const int get_result = redisGetReply(context_, reinterpret_cast<void **>(&reply));
        if (get_result != REDIS_OK) {
            KVCM_HIREDIS_LOG_WARN("redisGetReply failed: %s", context_->errstr);
            freeReplyObject(reply);
            replies.clear();
            Disconnect();
            return replies;
        }
        replies.emplace_back(reply);
    }
    return replies;
}

std::vector<RedisReplyUPtr> HiredisExecutor::ExecuteBatch(const std::vector<RedisCmdArgs> &cmds) {
    std::vector<RedisReplyUPtr> replies;
    if (cmds.empty()) {
        return replies;
    }
    for (int32_t count = 0; count < retry_count_; ++count) {
        if (!IsReady() && !Reconnect()) {
            KVCM_HIREDIS_LOG_ERROR("fail to reconnect before pipeline, attempt[%d]", count);
            return replies;
        }
        replies = TryExecuteBatch(cmds);
        if (!replies.empty()) {
            return replies;
        }
        if (IsReady()) {
            KVCM_HIREDIS_LOG_ERROR("pipeline failed while connection is ready, attempt[%d]", count);
            return replies;
        }
        usleep(50 * 1000);
    }
    KVCM_HIREDIS_LOG_ERROR("pipeline failed after [%ld] attempts", retry_count_);
    return replies;
}

#undef KVCM_HIREDIS_LOG_INFO
#undef KVCM_HIREDIS_LOG_WARN
#undef KVCM_HIREDIS_LOG_ERROR

} // namespace kv_cache_manager
