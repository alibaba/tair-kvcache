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
    virtual ClientErrorCode FinishWrite(const std::string &trace_id,
                                        const std::string &write_session_id,
                                        const BlockMask &success_block,
                                        const Locations &locations) = 0;

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
    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) = 0;
    virtual std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                               const BlockBuffers &block_buffers) = 0;

protected:
    ManagerClient() = default;
    virtual ClientErrorCode Init(const std::string &config, InitParams &init_params) = 0;
    virtual void Shutdown() = 0;
};

} // namespace kv_cache_manager