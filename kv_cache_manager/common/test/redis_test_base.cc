#include "kv_cache_manager/common/test/redis_test_base.h"

#include <cstring>

namespace kv_cache_manager {
RedisTestBase::ReplyUPtr RedisTestBase::MakeFakeReply(int type, const std::string &str) {
    redisReply *r = (redisReply *)malloc(sizeof(redisReply));
    memset(r, 0, sizeof(redisReply));
    r->type = type;
    if (!str.empty()) {
        r->str = strdup(str.c_str());
        r->len = str.size();
    }
    return ReplyUPtr(r, freeReplyObject);
}

RedisTestBase::ReplyUPtr RedisTestBase::MakeFakeReplyInteger(const int64_t &integer) {
    redisReply *r = (redisReply *)malloc(sizeof(redisReply));
    memset(r, 0, sizeof(redisReply));
    r->type = REDIS_REPLY_INTEGER;
    r->integer = integer;
    return ReplyUPtr(r, freeReplyObject);
}

RedisTestBase::ReplyUPtr RedisTestBase::MakeFakeReplyArrayString(const std::vector<std::optional<std::string>> &strs) {
    redisReply *r = (redisReply *)malloc(sizeof(redisReply));
    memset(r, 0, sizeof(redisReply));
    r->type = REDIS_REPLY_ARRAY;
    r->elements = strs.size();
    r->element = (redisReply **)malloc(sizeof(redisReply *) * strs.size());
    for (size_t i = 0; i < strs.size(); ++i) {
        if (strs[i].has_value()) {
            ReplyUPtr sub = MakeFakeReply(REDIS_REPLY_STRING, *(strs[i]));
            r->element[i] = sub.release();
        } else {
            ReplyUPtr sub = MakeFakeReply(REDIS_REPLY_NIL, "");
            r->element[i] = sub.release();
        }
    }
    return ReplyUPtr(r, freeReplyObject);
}

RedisTestBase::ReplyUPtr RedisTestBase::MakeFakeReplyScan(const std::string &next_cursor,
                                                          const std::vector<std::optional<std::string>> &keys) {
    redisReply *r = (redisReply *)malloc(sizeof(redisReply));
    memset(r, 0, sizeof(redisReply));
    r->type = REDIS_REPLY_ARRAY;
    r->elements = 2;
    r->element = (redisReply **)malloc(sizeof(redisReply *) * 2);
    // next_cursor
    r->element[0] = MakeFakeReply(REDIS_REPLY_STRING, next_cursor).release();
    // keys
    r->element[1] = MakeFakeReplyArrayString(keys).release();
    return ReplyUPtr(r, freeReplyObject);
}
} // namespace kv_cache_manager
