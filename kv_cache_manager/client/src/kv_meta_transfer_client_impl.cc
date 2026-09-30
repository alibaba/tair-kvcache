#include "kv_cache_manager/client/src/kv_meta_transfer_client_impl.h"

#include <algorithm>
#include <utility>

#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"

namespace kv_cache_manager {

namespace {

constexpr const char *kKvMetaValueSpecName = "value";
constexpr std::uint64_t kMaxKvMetaObjectBytes = 1ULL * 1024 * 1024 * 1024;

bool HasOnlyUnambiguousUriText(const UriStrVec &uris) {
    return std::all_of(uris.begin(), uris.end(), [](const std::string &uri) {
        return uri.size() <= kMaxKvMetaLocationUriBytes && HasUnambiguousKvMetaUriText(uri);
    });
}

} // namespace

ClientErrorCode ValidateKvMetaTransferClientConfig(const std::string &client_config,
                                                   const InitParams &init_params,
                                                   const std::string *expected_instance_group,
                                                   const std::string *expected_instance_id) {
    if (client_config.empty() || init_params.self_location_spec_name != kKvMetaValueSpecName) {
        return ER_INVALID_PARAMS;
    }

    ClientConfig parsed_config;
    if (!parsed_config.FromJsonString(client_config)) {
        return ER_INVALID_CLIENT_CONFIG;
    }
    const auto &location_specs = parsed_config.location_spec_infos();
    const auto value_spec = location_specs.find(kKvMetaValueSpecName);
    if (parsed_config.block_size() != 1 || location_specs.size() != 1 || value_spec == location_specs.end() ||
        value_spec->second != 1 || !parsed_config.location_spec_groups().empty()) {
        KVCM_LOG_WARN("KVMeta transfer config must use only the fixed schema marker value=1");
        return ER_INVALID_CLIENT_CONFIG;
    }
    if ((expected_instance_group != nullptr && parsed_config.instance_group() != *expected_instance_group) ||
        (expected_instance_id != nullptr && parsed_config.instance_id() != *expected_instance_id)) {
        KVCM_LOG_WARN("KVMeta metadata and transfer client identities do not match");
        return ER_INVALID_CLIENT_CONFIG;
    }
    const auto wrapper_config = parsed_config.sdk_wrapper_config();
    if (!wrapper_config || wrapper_config->thread_num() == 0 || wrapper_config->queue_size() == 0 ||
        !wrapper_config->Validate()) {
        KVCM_LOG_WARN("KVMeta sdk wrapper config is invalid");
        return ER_INVALID_SDKWRAPPER_CONFIG;
    }
    return ER_OK;
}

ClientErrorCode KvMetaTransferClientImpl::Init(const std::string &client_config,
                                               const InitParams &init_params,
                                               std::uint64_t max_object_bytes,
                                               const SharedMemoryRegistration *shared_memory_registration) {
    if (!(init_params.role_type & RoleType::WORKER) || init_params.self_location_spec_name.empty() ||
        init_params.storage_configs.empty() || max_object_bytes == 0 || max_object_bytes > kMaxKvMetaObjectBytes) {
        return ER_INVALID_PARAMS;
    }
    const auto validation_ec = ValidateKvMetaTransferClientConfig(client_config, init_params);
    if (validation_ec != ER_OK) {
        return validation_ec;
    }
    client_config_ = std::make_unique<ClientConfig>();
    if (!client_config_->FromJsonString(client_config)) {
        client_config_.reset();
        return ER_INVALID_CLIENT_CONFIG;
    }
    sdk_wrapper_ = std::make_unique<SdkWrapper>();
    const auto ec =
        sdk_wrapper_->InitForKvMeta(client_config_, init_params, max_object_bytes, shared_memory_registration);
    if (ec != ER_OK) {
        sdk_wrapper_.reset();
        client_config_.reset();
    }
    return ec;
}

ClientErrorCode KvMetaTransferClientImpl::LoadObjects(const UriStrVec &uri_str_vec,
                                                      const std::vector<std::uint64_t> &value_sizes,
                                                      const BlockBuffers &object_buffers) {
    if (!sdk_wrapper_) {
        return ER_INVALID_SDKWRAPPER_CONFIG;
    }
    if (!HasOnlyUnambiguousUriText(uri_str_vec)) {
        return ER_INVALID_PARAMS;
    }
    return sdk_wrapper_->GetKvMetaObjects(ParseLocations(uri_str_vec), value_sizes, object_buffers);
}

std::pair<ClientErrorCode, UriStrVec> KvMetaTransferClientImpl::SaveObjects(
    const UriStrVec &uri_str_vec, const std::vector<std::uint64_t> &value_sizes, const BlockBuffers &object_buffers) {
    if (!sdk_wrapper_) {
        return {ER_INVALID_SDKWRAPPER_CONFIG, {}};
    }
    if (!HasOnlyUnambiguousUriText(uri_str_vec)) {
        return {ER_INVALID_PARAMS, {}};
    }
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    const auto ec =
        sdk_wrapper_->PutKvMetaObjects(ParseLocations(uri_str_vec), value_sizes, object_buffers, actual_remote_uris);
    if (ec != ER_OK) {
        return {ec, {}};
    }
    UriStrVec actual_uris = ConstructLocations(*actual_remote_uris);
    if (actual_uris.size() != uri_str_vec.size()) {
        return {ER_SDKWRITE_ERROR, {}};
    }
    for (std::size_t i = 0; i < uri_str_vec.size(); ++i) {
        if (!HasSameCanonicalKvMetaUri(uri_str_vec[i], actual_uris[i])) {
            KVCM_LOG_WARN("KVMeta SDK rewrote an exact-object allocation URI");
            return {ER_SDKWRITE_ERROR, {}};
        }
    }
    return {ER_OK, std::move(actual_uris)};
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

std::unique_ptr<KvMetaTransferClient> KvMetaTransferClient::Create(const std::string &client_config,
                                                                   const InitParams &init_params,
                                                                   std::uint64_t max_object_bytes) {
    LoggerBroker::InitLoggerForClientOnce();
    auto client = std::make_unique<KvMetaTransferClientImpl>();
    if (client->Init(client_config, init_params, max_object_bytes, nullptr) != ER_OK) {
        return nullptr;
    }
    return client;
}

std::unique_ptr<KvMetaTransferClient>
KvMetaTransferClient::Create(const std::string &client_config,
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
