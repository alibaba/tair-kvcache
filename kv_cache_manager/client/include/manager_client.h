#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace kv_cache_manager {

class ManagerClient {
public:
    virtual ~ManagerClient() = default;
    static std::unique_ptr<ManagerClient> Create(const std::string &config, InitParams &init_params);

    // for meta client
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
    // 任务 82620492：block_hashes 与 StartWrite 时上报的 keys 一一对应 (长度等于
    // keys.size，而不是 success_block 中"成功"的子集；失败位置填 0)。空 vector
    // 视为 client 未启用 meta_checksum，server 不会修改 CacheLocation 已有 hash。
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

    virtual ClientErrorCode RemoveCache(const std::string &trace_id,
                                        const std::vector<int64_t> &keys,
                                        const std::vector<int64_t> &tokens,
                                        const BlockMask &block_mask) = 0;

    // for transfer client
    // 任务 82620492：out_block_hashes 透传给 TransferClient::SaveKvCaches。
    // 非 nullptr 时 SDK 算 hash 写入；上层之后调 FinishWrite 时把同一个 vector 传回去。
    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                         const BlockBuffers &block_buffers,
                                         const std::vector<int64_t> *expected_hashes = nullptr) = 0;
    virtual std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                               const BlockBuffers &block_buffers,
                                                               std::vector<int64_t> *out_block_hashes = nullptr) = 0;

protected:
    ManagerClient() = default;
    virtual ClientErrorCode Init(const std::string &config, InitParams &init_params) = 0;
    virtual void Shutdown() = 0;
};

} // namespace kv_cache_manager