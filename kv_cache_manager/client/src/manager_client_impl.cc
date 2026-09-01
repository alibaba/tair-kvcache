#include "kv_cache_manager/client/src/manager_client_impl.h"

#include "kv_cache_manager/client/src/meta_client_impl.h"
#include "kv_cache_manager/client/src/transfer_client_impl.h"
#include "kv_cache_manager/common/logger.h"

#define DEFER(...) __VA_ARGS__
#define CHECK_CLIENT_BASE(client, return_value)                                                                        \
    if (client == nullptr) {                                                                                           \
        KVCM_LOG_ERROR("client is nullptr");                                                                           \
        return return_value;                                                                                           \
    }

#define CHECK_CLIENT(client) CHECK_CLIENT_BASE(client, ER_CLIENT_NOT_EXISTS)
#define CHECK_CLIENT_WITH_TYPE(client) CHECK_CLIENT_BASE(client, DEFER({ER_CLIENT_NOT_EXISTS, {}}))

namespace kv_cache_manager {

ManagerClientImpl::ManagerClientImpl() {}

ManagerClientImpl::~ManagerClientImpl() { Shutdown(); }

ClientErrorCode ManagerClientImpl::Init(const std::string &client_config, InitParams &init_params) {
    if (init_params.role_type == RoleType::UNKNOWN) {
        KVCM_LOG_ERROR("init manager client failed, invalid role type [%s]",
                       RoleTypeToString(init_params.role_type).c_str());
        return ER_INVALID_ROLETYPE;
    }
    if (init_params.role_type & RoleType::SCHEDULER) {
        meta_client_ = MetaClientImpl::Create(client_config, init_params);
        if (meta_client_ == nullptr) {
            KVCM_LOG_ERROR("init meta client failed");
            return ER_METACLIENT_INIT_ERROR;
        }
    }
    if (init_params.role_type & RoleType::WORKER) {
        if (meta_client_) {
            init_params.storage_configs = meta_client_->GetStorageConfig();
        }
        if (init_params.storage_configs.empty()) {
            KVCM_LOG_ERROR("storage config is empty");
            return ER_INVALID_STORAGE_CONFIG;
        }
        transfer_client_ = TransferClientImpl::Create(client_config, init_params);
        if (transfer_client_ == nullptr) {
            KVCM_LOG_ERROR("init transfer client failed");
            return ER_TRANSFERCLIENT_INIT_ERROR;
        }
    }
    KVCM_LOG_INFO("manager client init success");
    return ER_OK;
}

void ManagerClientImpl::Shutdown() {}

std::pair<ClientErrorCode, Locations>
ManagerClientImpl::MatchLocation(const std::string &trace_id,
                                 QueryType query_type,
                                 const std::vector<int64_t> &keys,
                                 const std::vector<int64_t> &tokens,
                                 const BlockMask &block_mask,
                                 int32_t sw_size,
                                 const std::vector<std::string> &location_spec_names) {
    CHECK_CLIENT_WITH_TYPE(meta_client_);
    return meta_client_->MatchLocation(trace_id, query_type, keys, tokens, block_mask, sw_size, location_spec_names);
}

std::pair<ClientErrorCode, WriteLocation>
ManagerClientImpl::StartWrite(const std::string &trace_id,
                              const std::vector<int64_t> &keys,
                              const std::vector<int64_t> &tokens,
                              const std::vector<std::string> &location_spec_group_names,
                              int64_t write_timeout_seconds) {
    CHECK_CLIENT_WITH_TYPE(meta_client_);
    return meta_client_->StartWrite(trace_id, keys, tokens, location_spec_group_names, write_timeout_seconds);
}

ClientErrorCode ManagerClientImpl::FinishWrite(const std::string &trace_id,
                                               const std::string &write_session_id,
                                               const BlockMask &success_block,
                                               const Locations &locations) {
    CHECK_CLIENT(meta_client_);
    return meta_client_->FinishWrite(trace_id, write_session_id, success_block, locations);
}

std::pair<ClientErrorCode, Metas> ManagerClientImpl::MatchMeta(const std::string &trace_id,
                                                               const std::vector<int64_t> &keys,
                                                               const std::vector<int64_t> &tokens,
                                                               const BlockMask &block_mask,
                                                               int32_t detail_level) {
    CHECK_CLIENT_WITH_TYPE(meta_client_);
    return meta_client_->MatchMeta(trace_id, keys, tokens, block_mask, detail_level);
}

ClientErrorCode ManagerClientImpl::RemoveCache(const std::string &trace_id,
                                               const std::vector<int64_t> &keys,
                                               const std::vector<int64_t> &tokens,
                                               const BlockMask &block_mask) {
    CHECK_CLIENT(meta_client_);
    return meta_client_->RemoveCache(trace_id, keys, tokens, block_mask);
}

ClientErrorCode ManagerClientImpl::LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) {
    CHECK_CLIENT(transfer_client_);
    return transfer_client_->LoadKvCaches(uri_str_vec, block_buffers);
}

std::pair<ClientErrorCode, UriStrVec> ManagerClientImpl::SaveKvCaches(const UriStrVec &uri_str_vec,
                                                                      const BlockBuffers &block_buffers) {
    CHECK_CLIENT_WITH_TYPE(transfer_client_);
    return transfer_client_->SaveKvCaches(uri_str_vec, block_buffers);
}

std::unique_ptr<ManagerClient> ManagerClient::Create(const std::string &client_config, InitParams &init_params) {
    LoggerBroker::InitLoggerForClientOnce();
    auto client = std::make_unique<ManagerClientImpl>();
    auto ec = client->Init(client_config, init_params);
    if (ec == ER_OK) {
        return client;
    }
    KVCM_LOG_ERROR("create manager client failed, error code: %d", ec);
    return nullptr;
}

} // namespace kv_cache_manager