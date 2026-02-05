#include "kv_cache_manager/client/src/transfer_client_impl.h"

#include <sstream>
#ifdef USING_CUDA
#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"
#include "kv_cache_manager/common/env_util.h"
#endif
#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/client/src/internal/util/debug_string_util.h"
#include "kv_cache_manager/common/logger.h"

#define DEFER(...) __VA_ARGS__
#define CHECK_SDK_BASE(return_value)                                                                                   \
    if (sdk_wrapper_ == nullptr) {                                                                                     \
        KVCM_LOG_ERROR("sdk wrapper is null");                                                                         \
        return return_value;                                                                                           \
    }

#define CHECK_SDK() CHECK_SDK_BASE(ER_INVALID_SDKWRAPPER_CONFIG)
#define CHECK_SDK_WITH_TYPE() CHECK_SDK_BASE(DEFER({ER_INVALID_SDKWRAPPER_CONFIG, {}}))

namespace kv_cache_manager {

TransferClientImpl::TransferClientImpl() {}

TransferClientImpl::~TransferClientImpl() {}

ClientErrorCode TransferClientImpl::Init(const std::string &client_config, const InitParams &init_params) {
    {
        std::shared_lock read_guard(config_mutex_);
        if (client_config_ != nullptr) {
            KVCM_LOG_INFO("transfer client has been inited by others");
            return ER_OK;
        }
    }
    {
        std::scoped_lock write_guard(config_mutex_);
        // double checkout
        if (client_config_ != nullptr) {
            KVCM_LOG_INFO("transfer client has been inited by others");
            return ER_OK;
        }
        if (!(init_params.role_type & RoleType::WORKER)) {
            KVCM_LOG_INFO("not support role type [%s] on transfer client, skip init",
                          RoleTypeToString(init_params.role_type).c_str());
            return ER_SKIPINIT;
        }
        if (init_params.self_location_spec_name.empty()) {
            KVCM_LOG_ERROR("init transfer client failed, self location spec name is empty");
            return ER_INVALID_PARAMS;
        }
        init_params_ = init_params;
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
        sdk_wrapper_ = std::make_unique<SdkWrapper>();
        KVCM_LOG_INFO("transfer client init params: role_type[%d], regist_span[%p], self_location_spec_name[%s], "
                      "storage_configs[%s]",
                      static_cast<int>(init_params_.role_type),
                      init_params_.regist_span,
                      init_params_.self_location_spec_name.c_str(),
                      init_params_.storage_configs.c_str());
        ec = sdk_wrapper_->Init(client_config_, init_params_);
        if (ec != ER_OK) {
            KVCM_LOG_ERROR("init sdk wrapper failed");
            client_config_.reset();
            sdk_wrapper_.reset();
            return ec;
        }
#ifdef USING_CUDA
        is_check_buffer_ = EnvUtil::GetEnv("KVCM_SDK_CHECK", false);
        if (is_check_buffer_) {
            size_t sdk_check_cell_num = EnvUtil::GetEnv("KVCM_SDK_CHECK_CELL_NUM", 4);
            max_check_iov_num_ = EnvUtil::GetEnv("KVCM_SDK_MAX_CHECK_IOV_NUM", 500 * 1000);
            sdk_buffer_check_pool_ = std::make_shared<SdkBufferCheckPool>(sdk_check_cell_num);
            if (!sdk_buffer_check_pool_->Init(max_check_iov_num_)) {
                KVCM_LOG_ERROR("sdk_buffer_check_pool init faild, sdk_check_cell_num[%lu], max_check_iov_num[%lu]",
                               sdk_check_cell_num,
                               max_check_iov_num_);
                return ER_INIT_CHECK_BUFFER_ERROR;
            }
        }
#endif
        KVCM_LOG_INFO("transfer client init success");
        return ER_OK;
    }
}

void TransferClientImpl::PrintBlockHashAndUri(const std::string &prefix,
                                              const UriStrVec &uri_str_vec,
                                              const std::vector<int64_t> &block_hashs) const {
    std::stringstream ss;
    ss << prefix << "; self_location_spec_name : " << init_params_.self_location_spec_name
       << "; uri size : " << uri_str_vec.size() << "; real size : " << block_hashs.size() << '\n';
    ss << "{\n";
    for (size_t i = 0; i < block_hashs.size(); ++i) {
        ss << "  \"" << prefix << uri_str_vec[i] << "\":" << block_hashs[i];
        if (i != (block_hashs.size() - 1)) {
            ss << ',';
        }
        ss << '\n';
    }
    ss << "}";
    KVCM_LOG_INFO("%s", ss.str().c_str());
}

ClientErrorCode TransferClientImpl::LoadKvCaches(const UriStrVec &uri_str_vec, const BlockBuffers &block_buffers) {
    KVCM_LOG_DEBUG("load kv caches with uri_str_vec %s, block_buffers %s",
                   DebugStringUtil::ToString(uri_str_vec).c_str(),
                   DebugStringUtil::ToString(block_buffers).c_str());
    CHECK_SDK();
    auto remote_uris = ParseLocations(uri_str_vec);
    auto ec = sdk_wrapper_->Get(remote_uris, block_buffers);
    if (ec != ER_OK) {
        return ec;
    }
#ifdef USING_CUDA
    if (is_check_buffer_) {
        auto handle = sdk_buffer_check_pool_->GetCell();
        auto block_hashs = SdkBufferCheckUtil::GetBlocksHash(
            block_buffers, handle->d_iovs, handle->d_crcs, handle->h_iovs, max_check_iov_num_, handle->cuda_stream);
        PrintBlockHashAndUri("get_", uri_str_vec, block_hashs);
    }
#endif
    return ec;
}

std::pair<ClientErrorCode, UriStrVec> TransferClientImpl::SaveKvCaches(const UriStrVec &uri_str_vec,
                                                                       const BlockBuffers &block_buffers) {
    KVCM_LOG_DEBUG("save kv caches with uri_str_vec %s, block_buffers %s",
                   DebugStringUtil::ToString(uri_str_vec).c_str(),
                   DebugStringUtil::ToString(block_buffers).c_str());
    CHECK_SDK_WITH_TYPE();
#ifdef USING_CUDA
    if (is_check_buffer_) {
        auto handle = sdk_buffer_check_pool_->GetCell();
        auto block_hashs = SdkBufferCheckUtil::GetBlocksHash(
            block_buffers, handle->d_iovs, handle->d_crcs, handle->h_iovs, max_check_iov_num_, handle->cuda_stream);
        PrintBlockHashAndUri("put_", uri_str_vec, block_hashs);
    }
#endif
    auto remote_uris = ParseLocations(uri_str_vec);
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    auto ec = sdk_wrapper_->Put(remote_uris, block_buffers, actual_remote_uris);
    if (ec != ER_OK) {
        KVCM_LOG_ERROR("save kv cache failed");
        return {ec, {}};
    }
    return {ER_OK, ConstructLocations(*actual_remote_uris)};
}

ClientErrorCode TransferClientImpl::IsValid(const std::unique_ptr<ClientConfig> &client_config) const {
    if (client_config == nullptr) {
        KVCM_LOG_ERROR("client config is null");
        return ER_INVALID_CLIENT_CONFIG;
    }
    if (client_config->sdk_wrapper_config() == nullptr) {
        KVCM_LOG_ERROR("sdk config is null");
        return ER_INVALID_SDKWRAPPER_CONFIG;
    }
    if (!client_config->sdk_wrapper_config()->Validate()) {
        KVCM_LOG_ERROR("sdk config is invalid");
        return ER_INVALID_SDKWRAPPER_CONFIG;
    }
    return ER_OK;
}

std::vector<DataStorageUri> TransferClientImpl::ParseLocations(const UriStrVec &uri_str_vec) {
    std::vector<DataStorageUri> remote_uris;
    for (const auto &uri_str : uri_str_vec) {
        remote_uris.push_back(DataStorageUri(uri_str));
    }
    return remote_uris;
}

UriStrVec TransferClientImpl::ConstructLocations(const std::vector<DataStorageUri> &uris) {
    UriStrVec uri_str_vec;
    for (const auto &uri : uris) {
        uri_str_vec.push_back(uri.ToUriString());
    }
    return uri_str_vec;
}

std::unique_ptr<TransferClient> TransferClient::Create(const std::string &client_config,
                                                       const InitParams &init_params) {
    LoggerBroker::InitLoggerForClientOnce();
    auto client = std::make_unique<TransferClientImpl>();
    auto ec = client->Init(client_config, init_params);
    if (ec == ER_OK) {
        return client;
    }
    KVCM_LOG_ERROR("create transfer client failed with errocode: %d", ec);
    return nullptr;
}

} // namespace kv_cache_manager

#undef DEFER
#undef CHECK_SDK_BASE
#undef CHECK_SDK
#undef CHECK_SDK_WITH_TYPE