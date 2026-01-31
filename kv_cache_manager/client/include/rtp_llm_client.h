#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace kv_cache_manager {

class RTPLLMClient {
public:
    virtual ~RTPLLMClient() = default;
    static std::unique_ptr<RTPLLMClient> Create(const std::string &config, InitParams &init_params);
    // for meta client
    virtual std::pair<ClientErrorCode, Locations> Match(const std::string &trace_id,
                                                        QueryType query_type,
                                                        const std::vector<int64_t> &keys,
                                                        const BlockMask &block_mask,
                                                        const ForwardContext &forward_context = ForwardContext()) = 0;

    virtual std::pair<ClientErrorCode, WriteLocation>
    GetWriteLocation(const std::string &trace_id,
                     const std::vector<int64_t> &keys,
                     const std::vector<std::string> &location_spec_group_names,
                     int64_t write_timeout_seconds,
                     const ForwardContext &forward_context = ForwardContext()) = 0;
    virtual ClientErrorCode FinishWrite(const std::string &trace_id,
                                        const std::string &write_session_id,
                                        const BlockMask &success_block,
                                        const Locations &locations) = 0;

    // for transfer client
    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) = 0;
    virtual std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                               const BlockBuffers &block_buffers) = 0;

protected:
    RTPLLMClient() = default;
    virtual void Shutdown() = 0;
};

} // namespace kv_cache_manager