#include "kv_cache_manager/client/src/rtp_llm_client_impl.h"

#include "kv_cache_manager/client/src/manager_client_impl.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

RTPLLMClientImpl::RTPLLMClientImpl() {}

RTPLLMClientImpl::~RTPLLMClientImpl() { Shutdown(); }

ClientErrorCode RTPLLMClientImpl::Init(const std::string &client_config, InitParams &init_params) {
    manager_client_ = ManagerClient::Create(client_config, init_params);
    if (manager_client_ == nullptr) {
        KVCM_LOG_ERROR("create manager client failed");
        return ER_MANAGERCLIENT_INIT_ERROR;
    }
    return ER_OK;
}
void RTPLLMClientImpl::Shutdown() {}

std::pair<ClientErrorCode, Locations> RTPLLMClientImpl::Match(const std::string &trace_id,
                                                              QueryType query_type,
                                                              const std::vector<int64_t> &keys,
                                                              const BlockMask &block_mask,
                                                              const ForwardContext &forward_context) {
    return manager_client_->MatchLocation(trace_id, query_type, keys, {}, block_mask, forward_context.sw_size, {});
}

std::pair<ClientErrorCode, WriteLocation>
RTPLLMClientImpl::GetWriteLocation(const std::string &trace_id,
                                   const std::vector<int64_t> &keys,
                                   const std::vector<std::string> &location_spec_group_names,
                                   int64_t write_timeout_seconds,
                                   const ForwardContext &forward_context) {
    return manager_client_->StartWrite(trace_id, keys, {}, location_spec_group_names, write_timeout_seconds);
}
ClientErrorCode RTPLLMClientImpl::FinishWrite(const std::string &trace_id,
                                              const std::string &write_session_id,
                                              const BlockMask &success_block,
                                              const Locations &locations) {
    return manager_client_->FinishWrite(trace_id, write_session_id, success_block, locations);
}

ClientErrorCode RTPLLMClientImpl::LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) {
    return manager_client_->LoadKvCaches(uri_str_vec, block_buffers);
}
std::pair<ClientErrorCode, UriStrVec> RTPLLMClientImpl::SaveKvCaches(const UriStrVec &uri_str_vec,
                                                                     const BlockBuffers &block_buffers) {
    return manager_client_->SaveKvCaches(uri_str_vec, block_buffers);
}

std::unique_ptr<RTPLLMClient> RTPLLMClient::Create(const std::string &client_config, InitParams &init_params) {
    auto client = std::make_unique<RTPLLMClientImpl>();
    auto ec = client->Init(client_config, init_params);
    if (ec == ER_OK) {
        return client;
    }
    KVCM_LOG_ERROR("create rtp llm client failed, errorcode: %d", ec);
    return nullptr;
}
} // namespace kv_cache_manager