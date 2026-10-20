#include "kv_cache_manager/client/src/manager_client_impl.h"

#include "kv_cache_manager/client/include/meta_client.h"
#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/replication_executor.h"
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
        meta_client_ = MetaClient::Create(client_config, init_params);
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
    if (meta_client_ && transfer_client_) {
        ClientConfig config;
        int num_workers = 2;
        if (config.FromJsonString(client_config)) {
            num_workers = std::max(1, static_cast<int>(config.replication_workers()));
            auto_replicate_ = config.auto_replicate();
        }
        ReplicationOptions options;
        options.instance_id = config.instance_id();
        options.metrics_callback = init_params.replication_metrics_callback;
        options.max_buffer_bytes = config.replication_max_buffer_bytes();
        options.max_pending_bytes = config.replication_max_pending_bytes();
        options.node_bytes_per_second = config.replication_node_bytes_per_second();
        options.max_age_ms = config.replication_max_age_ms();
        replication_executor_ =
            std::make_unique<ReplicationExecutor>(meta_client_.get(), transfer_client_.get(), num_workers, 1024, std::move(options));
        KVCM_LOG_INFO("replication executor created: workers=%d auto_replicate=%s",
                      num_workers,
                      auto_replicate_ ? "true" : "false");
    }
    KVCM_LOG_INFO("manager client init success");
    return ER_OK;
}

bool ManagerClientImpl::ReplicateWithBuffers(const ClientReplicationHint &hint,
                                             std::vector<ClientReplicationBuffer> buffers) {
    return ReplicateWithBuffersAsync(hint, std::move(buffers), {});
}

bool ManagerClientImpl::ReplicateWithBuffersAsync(const ClientReplicationHint &hint,
                                                  std::vector<ClientReplicationBuffer> buffers,
                                                  ReplicationResultCallback result_fn) {
    if (replication_executor_) {
        return replication_executor_->SubmitWithBuffers(hint, std::move(buffers), std::move(result_fn));
    }
    if (result_fn) {
        result_fn({hint.block_key,
                   hint.target_node_id,
                   ReplicationOutcome::REJECTED_STOPPED,
                   ER_CLIENT_NOT_EXISTS});
    }
    return false;
}

ReplicationStats ManagerClientImpl::GetReplicationStats() const {
    return replication_executor_ ? replication_executor_->GetStats() : ReplicationStats{};
}

void ManagerClientImpl::Shutdown() {
    if (replication_executor_) {
        replication_executor_->Shutdown();
        replication_executor_.reset();
    }
}

std::pair<ClientErrorCode, Locations>
ManagerClientImpl::MatchLocation(const std::string &trace_id,
                                 QueryType query_type,
                                 const std::vector<int64_t> &keys,
                                 const std::vector<int64_t> &tokens,
                                 const BlockMask &block_mask,
                                 int32_t sw_size,
                                 const std::vector<std::string> &location_spec_names,
                                 std::vector<ClientReplicationHint> &out_hints) {
    CHECK_CLIENT_WITH_TYPE(meta_client_);
    auto result = meta_client_->MatchLocation(
        trace_id, query_type, keys, tokens, block_mask, sw_size, location_spec_names, out_hints);
    if (result.first == ER_OK && !out_hints.empty() && replication_executor_ && auto_replicate_) {
        replication_executor_->Submit(out_hints);
    }
    return result;
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

void ManagerClientImpl::ReplicateWithData(const ClientReplicationHint &hint,
                                          const void *data,
                                          size_t size,
                                          std::function<void()> release_fn) {
    static_cast<void>(ReplicateWithDataAsync(hint, data, size, std::move(release_fn), {}));
}

bool ManagerClientImpl::ReplicateWithDataAsync(const ClientReplicationHint &hint,
                                               const void *data,
                                               size_t size,
                                               std::function<void()> release_fn,
                                               ReplicationResultCallback result_fn) {
    if (replication_executor_) {
        return replication_executor_->SubmitWithData(
            hint, data, size, std::move(release_fn), std::move(result_fn));
    }
    KVCM_LOG_ERROR("[replication] ReplicateWithDataAsync: replication_executor_ is null, "
                   "block_key [%ld] target [%s] cannot be replicated",
                   hint.block_key,
                   hint.target_node_id.c_str());
    if (release_fn) release_fn();
    if (result_fn) {
        result_fn({hint.block_key,
                   hint.target_node_id,
                   ReplicationOutcome::REJECTED_STOPPED,
                   ER_CLIENT_NOT_EXISTS});
    }
    return false;
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

std::string ManagerClientImpl::GetCallerNode() const {
    CHECK_CLIENT_WITH_TYPE(meta_client_);
    return meta_client_->GetCallerNode();
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
