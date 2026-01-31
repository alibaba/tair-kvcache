#pragma once

#include "kv_cache_manager/common/redis_client.h"

namespace kv_cache_manager {
class RedisTestBase {
protected:
    using ReplyUPtr = RedisClient::ReplyUPtr;
    using CmdArgs = std::vector<std::string>;

    static ReplyUPtr MakeFakeReply(int type, const std::string &str);
    static ReplyUPtr MakeFakeReplyInteger(const int64_t &integer);
    static ReplyUPtr MakeFakeReplyArrayString(const std::vector<std::optional<std::string>> &strs);
    static ReplyUPtr MakeFakeReplyScan(const std::string &next_cursor,
                                       const std::vector<std::optional<std::string>> &keys);
};
} // namespace kv_cache_manager
