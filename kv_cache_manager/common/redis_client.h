#pragma once

#include <hiredis.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/common/standard_uri.h"

namespace kv_cache_manager {

class RedisClient {
public:
    RedisClient(const StandardUri &storage_uri);
    virtual ~RedisClient(); // virtual for test

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
                   const int64_t limit,
                   std::string &out_next_cursor,
                   std::vector<std::string> &out_keys);
    ErrorCode Rand(const std::string &matching_prefix, const int64_t count, std::vector<std::string> &out_keys);

    using ReplyUPtr = std::unique_ptr<redisReply, void (*)(void *)>;
    using CmdArgs = std::vector<std::string>;

    // Batch-execute write commands (DEL/HSET/HDEL) via pipeline.
    // All commands are expected to return REDIS_REPLY_INTEGER.
    // Returns per-command ErrorCode; empty vector on total connection failure.
    // Sets all_ok to true if every command succeeded.
    std::vector<ErrorCode> BatchWrite(const std::vector<CmdArgs> &cmds, bool &out_all_ok);

    // --- Static command builders ---
    // Pure utility functions that construct Redis CmdArgs without any connection or state.
    // Callers use these to build command sequences, then execute via BatchExecute/CommandPipeline.

    // DEL + HSET per key (overwrite semantics). Skips keys with empty field_maps.
    static void BuildSetCmds(const std::vector<std::string> &keys,
                             const std::vector<std::map<std::string, std::string>> &field_maps,
                             std::vector<CmdArgs> &out_cmds);

    // HSET per key (merge/upsert semantics). Skips keys with empty field_maps.
    static void BuildHashSetCmds(const std::vector<std::string> &keys,
                                 const std::vector<std::map<std::string, std::string>> &field_maps,
                                 std::vector<CmdArgs> &out_cmds);

    // DEL per key.
    static void BuildDeleteCmds(const std::vector<std::string> &keys, std::vector<CmdArgs> &out_cmds);

    // HDEL per key. Skips keys with empty field_names.
    static void BuildHashDeleteCmds(const std::vector<std::string> &keys,
                                    const std::vector<std::vector<std::string>> &field_names_vec,
                                    std::vector<CmdArgs> &out_cmds);

protected:

    bool IsReplyOk(const redisReply *reply) const;
    bool CheckReplyInteger(const redisReply *reply) const;
    bool CheckReplyArray(const redisReply *reply) const;
    bool GetReplyStrOrNil(const redisReply *reply, std::string &out_str) const;
    bool Connect();
    void Disconnect();
    std::vector<ReplyUPtr> CommandPipeline(const std::vector<CmdArgs> &cmds);

    // virtual for test
    virtual bool IsContextOk() const;
    virtual bool Reconnect();
    virtual std::vector<ReplyUPtr> TryExecPipeline(const std::vector<CmdArgs> &cmds);

private:
    redisContext *context_ = nullptr;
    std::string user_info_;
    std::string host_;
    int64_t port_ = 0;
    int64_t db_ = 0;
    int64_t timeout_ms_ = 2000;
    int64_t retry_count_ = 2;
    int64_t randomkey_batch_num_ = 20;
};
} // namespace kv_cache_manager
