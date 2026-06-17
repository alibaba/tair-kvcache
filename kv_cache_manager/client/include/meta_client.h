#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace kv_cache_manager {

class MetaClient {
public:
    virtual ~MetaClient() = default;
    static std::unique_ptr<MetaClient> Create(const std::string &config, const InitParams &init_params);

    virtual std::pair<ClientErrorCode, Locations>
    MatchLocation(const std::string &trace_id,
                  QueryType query_type,
                  const std::vector<int64_t> &keys,
                  const std::vector<int64_t> &tokens,
                  const BlockMask &block_mask,
                  int32_t sw_size,
                  const std::vector<std::string> &location_spec_names) = 0;

    virtual std::pair<ClientErrorCode, WriteLocation>
    StartWrite(const std::string &trace_id,
               const std::vector<int64_t> &keys,
               const std::vector<int64_t> &tokens,
               const std::vector<std::string> &location_spec_group_names,
               int64_t write_timeout_seconds) = 0;
    // 任务 82620492：block_hashes 与 StartWrite 时上报的 keys 一一对应；空 vector
    // 视为未上报。MetaClientImpl 会把每个 hash 填到 FinishWriteCacheRequest.locations
    // 对应位置的 block_hash 字段透传给 server。
    virtual ClientErrorCode FinishWrite(const std::string &trace_id,
                                        const std::string &write_session_id,
                                        const BlockMask &success_block,
                                        const Locations &locations,
                                        const std::vector<int64_t> &block_hashes = {}) = 0;

    virtual std::pair<ClientErrorCode, Metas> MatchMeta(const std::string &trace_id,
                                                        const std::vector<int64_t> &keys,
                                                        const std::vector<int64_t> &tokens,
                                                        const BlockMask &block_mask,
                                                        int32_t detail_level) = 0;

    virtual std::pair<ClientErrorCode, int64_t> MatchLocationLen(const std::string &trace_id,
                                                                 QueryType query_type,
                                                                 const std::vector<int64_t> &keys,
                                                                 const std::vector<int64_t> &tokens,
                                                                 int32_t sw_size) = 0;

    virtual ClientErrorCode RemoveCache(const std::string &trace_id,
                                        const std::vector<int64_t> &keys,
                                        const std::vector<int64_t> &tokens,
                                        const BlockMask &block_mask) = 0;

    virtual const std::string &GetStorageConfig() const = 0;

protected:
    MetaClient() = default;
    virtual ClientErrorCode Init(const std::string &config, const InitParams &init_params) = 0;
    virtual void Shutdown() = 0;
};
} // namespace kv_cache_manager