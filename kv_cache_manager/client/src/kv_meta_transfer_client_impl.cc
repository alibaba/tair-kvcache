#include "kv_cache_manager/client/src/kv_meta_transfer_client_impl.h"

#include <utility>

#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"

namespace kv_cache_manager {

ClientErrorCode KvMetaTransferClientImpl::Init(
    const std::string &client_config,
    const InitParams &init_params,
    std::uint64_t max_object_bytes,
    const SharedMemoryRegistration *shared_memory_registration) {
    if (!(init_params.role_type & RoleType::WORKER) || init_params.self_location_spec_name.empty() ||
        init_params.storage_configs.empty() || max_object_bytes == 0) {
        return ER_INVALID_PARAMS;
    }
    client_config_ = std::make_unique<ClientConfig>();
    if (!client_config_->FromJsonString(client_config)) {
        client_config_.reset();
        return ER_INVALID_CLIENT_CONFIG;
    }
    sdk_wrapper_ = std::make_unique<SdkWrapper>();
    const auto ec = sdk_wrapper_->InitForKvMeta(
        client_config_, init_params, max_object_bytes, shared_memory_registration);
    if (ec != ER_OK) {
        sdk_wrapper_.reset();
        client_config_.reset();
    }
    return ec;
}

ClientErrorCode KvMetaTransferClientImpl::LoadObjects(
    const UriStrVec &uri_str_vec,
    const std::vector<std::uint64_t> &value_sizes,
    const BlockBuffers &object_buffers) {
    if (!sdk_wrapper_) {
        return ER_INVALID_SDKWRAPPER_CONFIG;
    }
    return sdk_wrapper_->GetKvMetaObjects(ParseLocations(uri_str_vec), value_sizes, object_buffers);
}

std::pair<ClientErrorCode, UriStrVec> KvMetaTransferClientImpl::SaveObjects(
    const UriStrVec &uri_str_vec,
    const std::vector<std::uint64_t> &value_sizes,
    const BlockBuffers &object_buffers) {
    if (!sdk_wrapper_) {
        return {ER_INVALID_SDKWRAPPER_CONFIG, {}};
    }
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    const auto ec = sdk_wrapper_->PutKvMetaObjects(
        ParseLocations(uri_str_vec), value_sizes, object_buffers, actual_remote_uris);
    if (ec != ER_OK) {
        return {ec, {}};
    }
    return {ER_OK, ConstructLocations(*actual_remote_uris)};
}

std::vector<DataStorageUri> KvMetaTransferClientImpl::ParseLocations(const UriStrVec &uri_str_vec) {
    std::vector<DataStorageUri> result;
    result.reserve(uri_str_vec.size());
    for (const auto &uri : uri_str_vec) {
        result.emplace_back(uri);
    }
    return result;
}

UriStrVec KvMetaTransferClientImpl::ConstructLocations(const std::vector<DataStorageUri> &uris) {
    UriStrVec result;
    result.reserve(uris.size());
    for (const auto &uri : uris) {
        result.push_back(uri.ToUriString());
    }
    return result;
}

std::unique_ptr<KvMetaTransferClient> KvMetaTransferClient::Create(
    const std::string &client_config,
    const InitParams &init_params,
    std::uint64_t max_object_bytes) {
    LoggerBroker::InitLoggerForClientOnce();
    auto client = std::make_unique<KvMetaTransferClientImpl>();
    if (client->Init(client_config, init_params, max_object_bytes, nullptr) != ER_OK) {
        return nullptr;
    }
    return client;
}

std::unique_ptr<KvMetaTransferClient> KvMetaTransferClient::Create(
    const std::string &client_config,
    const InitParams &init_params,
    std::uint64_t max_object_bytes,
    const SharedMemoryRegistration &shared_memory_registration) {
    LoggerBroker::InitLoggerForClientOnce();
    auto client = std::make_unique<KvMetaTransferClientImpl>();
    if (client->Init(client_config, init_params, max_object_bytes, &shared_memory_registration) != ER_OK) {
        return nullptr;
    }
    return client;
}

} // namespace kv_cache_manager
