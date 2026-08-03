#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/redis/redis_executor.h"
#include "kv_cache_manager/common/standard_uri.h"

namespace kv_cache_manager {

class RedisClient {
public:
    using ReplyUPtr = RedisReplyUPtr;
    using CmdArgs = RedisCmdArgs;

    RedisClient(const StandardUri &storage_uri, std::shared_ptr<IRedisExecutor> executor);
    ~RedisClient() = default;

    RedisClient(const RedisClient &) = delete;
    RedisClient &operator=(const RedisClient &) = delete;
    RedisClient(RedisClient &&) = default;
    RedisClient &operator=(RedisClient &&) = default;

    bool Open();
    void Close();

    std::vector<ErrorCode> Set(const std::vector<std::string> &keys,
                               const std::vector<std::map<std::string, std::string>> &field_maps);
    std::vector<ErrorCode> Update(const std::vector<std::string> &keys,
                                  const std::vector<std::map<std::string, std::string>> &field_maps);
    std::vector<ErrorCode> Upsert(const std::vector<std::string> &keys,
                                  const std::vector<std::map<std::string, std::string>> &field_maps);
    std::vector<ErrorCode> Delete(const std::vector<std::string> &keys);
    std::vector<ErrorCode> DeleteFields(const std::vector<std::string> &keys,
                                        const std::vector<std::vector<std::string>> &field_names_vec);
    std::vector<ErrorCode> Get(const std::vector<std::string> &keys,
                               const std::vector<std::string> &field_names,
                               std::vector<std::map<std::string, std::string>> &out_field_maps);
    std::vector<ErrorCode> Get(const std::vector<std::string> &keys,
                               const std::vector<std::vector<std::string>> &field_names_vec,
                               std::vector<std::map<std::string, std::string>> &out_field_maps);
    std::vector<ErrorCode> GetAllFields(const std::vector<std::string> &keys,
                                        std::vector<std::map<std::string, std::string>> &out_field_maps);
    std::vector<ErrorCode> Exists(const std::vector<std::string> &keys, std::vector<bool> &out_is_exist_vec);
    std::vector<ErrorCode> ExistsFieldWithPrefix(const std::vector<std::string> &keys,
                                                 const std::string &field_prefix,
                                                 std::vector<bool> &out_exists_vec);
    std::vector<ErrorCode> GetFieldNamesWithPrefix(const std::vector<std::string> &keys,
                                                   const std::string &field_prefix,
                                                   std::vector<std::vector<std::string>> &out_field_names_vec);
    ErrorCode Scan(const std::string &matching_prefix,
                   const std::string &cursor,
                   int64_t limit,
                   std::string &out_next_cursor,
                   std::vector<std::string> &out_keys);
    ErrorCode Rand(const std::string &matching_prefix, int64_t count, std::vector<std::string> &out_keys);

    ErrorCode Eval(const std::string &script,
                   const std::vector<std::string> &keys,
                   const std::vector<std::string> &args,
                   std::string &out_result);
    ErrorCode EvalSha(const std::string &sha1,
                      const std::vector<std::string> &keys,
                      const std::vector<std::string> &args,
                      std::string &out_result);
    ErrorCode ScriptLoad(const std::string &script, std::string &out_sha1);
    ErrorCode ScriptExists(const std::string &sha1, bool &out_exists);
    ErrorCode LoadScript(const std::string &script, std::string &out_sha1);
    ErrorCode ExecuteScriptWithFallback(const std::string &script,
                                        const std::vector<std::string> &keys,
                                        const std::vector<std::string> &args,
                                        std::string &in_out_cached_sha1,
                                        std::string &out_result);
    ErrorCode Get(const std::string &key, std::string &out_value);
    ErrorCode Set(const std::string &key, const std::string &value, int64_t ttl_ms);
    ErrorCode Pttl(const std::string &key, int64_t &out_ttl_ms);
    ErrorCode Del(const std::string &key);
    ErrorCode Pexpire(const std::string &key, int64_t ttl_ms);
    ErrorCode FlushAll();

    std::vector<ErrorCode> BatchWrite(const std::vector<CmdArgs> &cmds, bool &out_all_ok);

    static void BuildSetCmds(const std::vector<std::string> &keys,
                             const std::vector<std::map<std::string, std::string>> &field_maps,
                             std::vector<CmdArgs> &out_cmds);
    static void BuildHashSetCmds(const std::vector<std::string> &keys,
                                 const std::vector<std::map<std::string, std::string>> &field_maps,
                                 std::vector<CmdArgs> &out_cmds);
    static void BuildDeleteCmds(const std::vector<std::string> &keys, std::vector<CmdArgs> &out_cmds);
    static void BuildHashDeleteCmds(const std::vector<std::string> &keys,
                                    const std::vector<std::vector<std::string>> &field_names_vec,
                                    std::vector<CmdArgs> &out_cmds);

private:
    bool IsReady() const noexcept;
    bool IsReplyOk(const redisReply *reply) const;
    bool CheckReplyInteger(const redisReply *reply) const;
    bool CheckReplyArray(const redisReply *reply) const;
    bool GetReplyStrOrNil(const redisReply *reply, std::string &out_str) const;
    std::vector<ReplyUPtr> CommandPipeline(const std::vector<CmdArgs> &cmds);

    std::shared_ptr<IRedisExecutor> executor_;
    std::string description_;
    int64_t randomkey_batch_num_ = 20;
};

} // namespace kv_cache_manager
