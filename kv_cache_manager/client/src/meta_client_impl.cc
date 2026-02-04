#include "kv_cache_manager/client/src/meta_client_impl.h"

#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/stub/grpc_stub.h"
#include "kv_cache_manager/client/src/internal/util/debug_string_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/string_util.h"

#define DEFER(...) __VA_ARGS__
#define CHECK_INSTANCE_STUB_BASE(return_value)                                                                         \
    GetInstanceId();                                                                                                   \
    if (instance_id.empty() || stub_ == nullptr) {                                                                     \
        KVCM_LOG_ERROR("trace_id [%s] | empty instance id or null stub [%p]", trace_id.c_str(), stub_.get());          \
        return return_value;                                                                                           \
    }

#define CHECK_INSTANCE_STUB() CHECK_INSTANCE_STUB_BASE(ER_INVALID_STUB)
#define CHECK_INSTANCE_STUB_WITH_TYPE() CHECK_INSTANCE_STUB_BASE(DEFER({ER_INVALID_STUB, {}}))

namespace {

std::string GenRandomTraceId(const std::string &prefix_log) {
    std::string random_trace_id = kv_cache_manager::StringUtil::GenerateRandomString(64);
    KVCM_LOG_INFO("[%s] generate random trace id [%s]", prefix_log.c_str(), random_trace_id.c_str());
    return random_trace_id;
}

} // namespace

namespace kv_cache_manager {

MetaClientImpl::MetaClientImpl() {}

MetaClientImpl::~MetaClientImpl() { Shutdown(); }

ClientErrorCode MetaClientImpl::Init(const std::string &client_config, const InitParams &init_params) {
    {
        std::scoped_lock write_guard(config_mutex_);
        // double checkout
        if (client_config_ != nullptr) {
            KVCM_LOG_INFO("client has been inited by others");
            return ER_OK;
        }
        if (!(init_params.role_type & RoleType::SCHEDULER)) {
            KVCM_LOG_INFO("not support role type [%s] on meta client, skip init",
                          RoleTypeToString(init_params.role_type).c_str());
            return ER_SKIPINIT;
        }
        client_config_ = std::make_unique<ClientConfig>();
        if (!client_config_->FromJsonString(client_config)) {
            KVCM_LOG_ERROR("config error! [%s]", client_config.c_str());
            client_config_.reset();
            return ER_INVALID_CLIENT_CONFIG;
        }
        auto ec = IsValid(client_config_);
        if (ec != ER_OK) {
            KVCM_LOG_ERROR("check client config [%s] on scheduler failed", client_config.c_str());
            client_config_.reset();
            return ec;
        }
        const auto &channel_config = client_config_->meta_channel_config();
        if (!stub_) {
            stub_ = std::make_unique<GrpcStub>(channel_config.retry_time(), channel_config.call_timeout());
        }
    }
    // 主从模式
    if (client_config_->addresses().size() > 1) {
        KVCM_LOG_INFO("has multi server address, need find leader");
    }
    ClientErrorCode ec = ER_OK;
    {
        std::scoped_lock write_guard(config_mutex_);
        for (const auto &address : client_config_->addresses()) {
            ec = Connect(address);
            if (ec == ER_OK) {
                KVCM_LOG_INFO("client init with address [%s] success", address.c_str());
                break;
            }
            if (ec == ER_SERVICE_NOT_LEADER) {
                // 当前节点不是Leader，断开当前连接，寻找下一个可用节点
                KVCM_LOG_INFO("address [%s] not leader, try next", address.c_str());
            } else {
                KVCM_LOG_ERROR("client init with address [%s] failed", address.c_str());
                client_config_.reset();
                stub_.reset();
                return ec;
            }
        }
    }
    if (ec != ER_OK) {
        KVCM_LOG_ERROR("meta client init fail, last errorcode [%d]", ec);
    } else {
        KVCM_LOG_INFO("meta client init success");
    }
    return ec;
}

void MetaClientImpl::Shutdown() {}

std::pair<ClientErrorCode, Locations>
MetaClientImpl::MatchLocation(const std::string &trace_id,
                              QueryType query_type,
                              const std::vector<int64_t> &keys,
                              const std::vector<int64_t> &tokens,
                              const BlockMask &block_mask,
                              int32_t sw_size,
                              const std::vector<std::string> &location_spec_names) {
    KVCM_LOG_DEBUG("match location with trace_id [%s], query_type [%d], keys %s, tokens %s, block_mask %s, sw_size "
                   "[%d], location_spec_names %s",
                   trace_id.c_str(),
                   static_cast<int>(query_type),
                   DebugStringUtil::ToString(keys).c_str(),
                   DebugStringUtil::ToString(tokens).c_str(),
                   DebugStringUtil::ToString(block_mask).c_str(),
                   sw_size,
                   DebugStringUtil::ToString(location_spec_names).c_str());
    const std::string &instance_id = CHECK_INSTANCE_STUB_WITH_TYPE();
    return stub_->GetCacheLocation(
        trace_id, instance_id, query_type, keys, tokens, block_mask, sw_size, location_spec_names);
}

std::pair<ClientErrorCode, Metas> MetaClientImpl::MatchMeta(const std::string &trace_id,
                                                            const std::vector<int64_t> &keys,
                                                            const std::vector<int64_t> &tokens,
                                                            const BlockMask &block_mask,
                                                            int32_t detail_level) {
    KVCM_LOG_DEBUG("match meta with trace_id [%s], keys %s, tokens %s, block_mask %s, detail_level [%d]",
                   trace_id.c_str(),
                   DebugStringUtil::ToString(keys).c_str(),
                   DebugStringUtil::ToString(tokens).c_str(),
                   DebugStringUtil::ToString(block_mask).c_str(),
                   detail_level);
    const std::string &instance_id = CHECK_INSTANCE_STUB_WITH_TYPE();
    return stub_->GetCacheMeta(trace_id, instance_id, keys, tokens, block_mask, detail_level);
}

std::pair<ClientErrorCode, WriteLocation>
MetaClientImpl::StartWrite(const std::string &trace_id,
                           const std::vector<int64_t> &keys,
                           const std::vector<int64_t> &tokens,
                           const std::vector<std::string> &location_spec_group_names,
                           int64_t write_timeout_seconds) {
    KVCM_LOG_DEBUG("start write with trace_id [%s], keys %s, tokens %s, location_spec_group_names [%s], "
                   "write_timeout_seconds [%ld]",
                   trace_id.c_str(),
                   DebugStringUtil::ToString(keys).c_str(),
                   DebugStringUtil::ToString(tokens).c_str(),
                   DebugStringUtil::ToString(location_spec_group_names).c_str(),
                   write_timeout_seconds);
    const std::string &instance_id = CHECK_INSTANCE_STUB_WITH_TYPE();
    return stub_->StartWriteCache(
        trace_id, instance_id, keys, tokens, location_spec_group_names, write_timeout_seconds);
}
ClientErrorCode MetaClientImpl::FinishWrite(const std::string &trace_id,
                                            const std::string &write_session_id,
                                            const BlockMask &success_block,
                                            const Locations &locations) {
    KVCM_LOG_DEBUG("finish write with trace_id [%s], write_session_id [%s], block_mask %s, locations %s",
                   trace_id.c_str(),
                   write_session_id.c_str(),
                   DebugStringUtil::ToString(success_block).c_str(),
                   DebugStringUtil::ToString(locations).c_str());
    const std::string &instance_id = CHECK_INSTANCE_STUB();
    return stub_->FinishWriteCache(trace_id, instance_id, write_session_id, success_block, locations);
}

ClientErrorCode MetaClientImpl::RemoveCache(const std::string &trace_id,
                                            const std::vector<int64_t> &keys,
                                            const std::vector<int64_t> &tokens,
                                            const BlockMask &block_mask) {
    KVCM_LOG_DEBUG("remove cache with trace_id [%s], keys %s, tokens %s, block_mask %s",
                   trace_id.c_str(),
                   DebugStringUtil::ToString(keys).c_str(),
                   DebugStringUtil::ToString(tokens).c_str(),
                   DebugStringUtil::ToString(block_mask).c_str());
    const std::string &instance_id = CHECK_INSTANCE_STUB();
    return stub_->RemoveCache(trace_id, instance_id, keys, tokens, block_mask);
}

const std::string &MetaClientImpl::GetStorageConfig() const {
    KVCM_LOG_DEBUG("get storage config");
    return storage_config_;
}

ClientErrorCode MetaClientImpl::IsValid(const std::unique_ptr<ClientConfig> &client_config) const {
    if (client_config == nullptr) {
        KVCM_LOG_ERROR("client config is null");
        return ER_INVALID_CLIENT_CONFIG;
    }
    if (client_config->addresses().empty()) {
        KVCM_LOG_ERROR("addresses is empty on scheduler");
        return ER_INVALID_CLIENT_CONFIG;
    }
    if (client_config->model_deployment().model_name().empty() || client_config->model_deployment().tp_size() < 1) {
        KVCM_LOG_ERROR("model_deployment info error, model_name [%s], tp_size [%d]",
                       client_config->model_deployment().model_name().c_str(),
                       client_config->model_deployment().tp_size());
        return ER_INVALID_CLIENT_CONFIG;
    }
    return ER_OK;
}

ClientErrorCode MetaClientImpl::Connect(const std::string &address) {
    // TODO, rigister here
    auto client_config = GetClientConfigUnsafe();
    if (client_config == nullptr) {
        KVCM_LOG_ERROR("client not init");
        return ER_INVALID_CLIENT_CONFIG;
    }
    auto ec = stub_->AddConnection(address, client_config->meta_channel_config().connection_timeout());
    if (ec != ER_OK) {
        KVCM_LOG_ERROR("meta client connect to %s failed", address.c_str());
        return ec;
    }
    KVCM_LOG_INFO("meta client connect to %s success", address.c_str());
    auto [reg_ec, storage_config] = stub_->RegisterInstance(GenRandomTraceId("AddStubConnection"),
                                                            client_config->instance_group(),
                                                            client_config->instance_id(),
                                                            client_config->block_size(),
                                                            client_config->location_spec_infos(),
                                                            client_config->model_deployment(),
                                                            client_config->location_spec_groups());
    if (reg_ec == ER_OK) {
        storage_config_ = storage_config;
    }
    if (reg_ec == ER_SERVICE_NOT_LEADER) {
        KVCM_LOG_INFO("address %s is not leader, remove all connections", address.c_str());
        stub_->RemoveAllConnections();
    }
    KVCM_LOG_INFO("register instance_group [%s] instance [%s] result is [%s]",
                  client_config->instance_group().c_str(),
                  client_config->instance_id().c_str(),
                  reg_ec == ER_OK ? "success" : "failed");
    return reg_ec;
}

const std::string &MetaClientImpl::GetInstanceId() const {
    auto client_config = GetClientConfig();
    if (client_config == nullptr) {
        static std::string empty_instance;
        return empty_instance;
    }
    return client_config->instance_id();
}

const ClientConfig *MetaClientImpl::GetClientConfig() const {
    std::shared_lock read_guard(config_mutex_);
    return GetClientConfigUnsafe();
}

const ClientConfig *MetaClientImpl::GetClientConfigUnsafe() const { return client_config_.get(); }

std::unique_ptr<MetaClient> MetaClient::Create(const std::string &client_config, const InitParams &init_params) {
    LoggerBroker::InitLoggerForClientOnce();
    auto client = std::make_unique<MetaClientImpl>();
    auto ec = client->Init(client_config, init_params);
    if (ec == ER_OK) {
        KVCM_LOG_INFO("create meta client success, @client=%p", client.get());
        return client;
    }
    KVCM_LOG_ERROR("create meta client failed, errorcode: %d", ec);
    return nullptr;
}
} // namespace kv_cache_manager