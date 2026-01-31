#pragma once

#include "kv_cache_manager/client/include/rtp_llm_client.h"

namespace kv_cache_manager {
class ManagerClient;

class RTPLLMClientImpl : public RTPLLMClient {
public:
    RTPLLMClientImpl();
    ~RTPLLMClientImpl() override;

    // for meta client
    std::pair<ClientErrorCode, Locations> Match(const std::string &trace_id,
                                                QueryType query_type,
                                                const std::vector<int64_t> &keys,
                                                const BlockMask &block_mask,
                                                const ForwardContext &forward_context) override;

    std::pair<ClientErrorCode, WriteLocation>
    GetWriteLocation(const std::string &trace_id,
                     const std::vector<int64_t> &keys,
                     const std::vector<std::string> &location_spec_group_names,
                     int64_t write_timeout_seconds,
                     const ForwardContext &forward_context) override;
    ClientErrorCode FinishWrite(const std::string &trace_id,
                                const std::string &write_session_id,
                                const BlockMask &success_block,
                                const Locations &locations) override;

    // for transfer client
    ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) override;
    std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                       const BlockBuffers &block_buffers) override;

protected:
    ClientErrorCode Init(const std::string &client_config, InitParams &init_params);
    void Shutdown() override;

private:
    friend class RTPLLMClient;
    std::unique_ptr<ManagerClient> manager_client_;
};
} // namespace kv_cache_manager