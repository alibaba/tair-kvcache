#include "kv_cache_manager/client/src/kv_meta_transfer_client_impl.h"

#include <limits>
#include <map>
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
constexpr std::size_t kMaxKvMetaBatchItems = 64;
constexpr std::uint64_t kMaxKvMetaObjectBytes = 1ULL * 1024 * 1024 * 1024;
constexpr std::uint64_t kMaxKvMetaBatchBytes = 4ULL * 1024 * 1024 * 1024;

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
    const auto ec = sdk_wrapper_->Init(client_config_, init_params, shared_memory_registration);
    if (ec != ER_OK) {
        sdk_wrapper_.reset();
        client_config_.reset();
        return ec;
    }
    max_object_bytes_ = max_object_bytes;
    return ER_OK;
}

ClientErrorCode KvMetaTransferClientImpl::LoadObjects(const UriStrVec &uri_str_vec,
                                                      const std::vector<std::uint64_t> &value_sizes,
                                                      const BlockBuffers &object_buffers) {
    if (!sdk_wrapper_) {
        return ER_INVALID_SDKWRAPPER_CONFIG;
    }
    if (const auto ec = ValidateObjects(uri_str_vec, value_sizes, object_buffers); ec != ER_OK) {
        return ec;
    }
    std::lock_guard<std::mutex> lock(io_mutex_);
    SetAllowedObjectSizes(value_sizes);
    return sdk_wrapper_->Get(ParseLocations(uri_str_vec), object_buffers);
}

std::pair<ClientErrorCode, UriStrVec> KvMetaTransferClientImpl::SaveObjects(
    const UriStrVec &uri_str_vec, const std::vector<std::uint64_t> &value_sizes, const BlockBuffers &object_buffers) {
    if (!sdk_wrapper_) {
        return {ER_INVALID_SDKWRAPPER_CONFIG, {}};
    }
    if (const auto ec = ValidateObjects(uri_str_vec, value_sizes, object_buffers); ec != ER_OK) {
        return {ec, {}};
    }
    std::lock_guard<std::mutex> lock(io_mutex_);
    SetAllowedObjectSizes(value_sizes);
    auto actual_locations = std::make_shared<std::vector<DataStorageUri>>();
    const auto ec = sdk_wrapper_->Put(ParseLocations(uri_str_vec), object_buffers, actual_locations);
    if (ec != ER_OK) {
        return {ec, {}};
    }
    auto actual_uris = ConstructLocations(*actual_locations);
    if (actual_uris.size() != uri_str_vec.size()) {
        return {ER_SDKWRITE_ERROR, {}};
    }
    for (std::size_t i = 0; i < uri_str_vec.size(); ++i) {
        if (uri_str_vec[i] != actual_uris[i]) {
            KVCM_LOG_WARN("KVMeta SDK rewrote an exact-object allocation URI");
            return {ER_SDKWRITE_ERROR, {}};
        }
    }
    return {ER_OK, std::move(actual_uris)};
}

ClientErrorCode KvMetaTransferClientImpl::ValidateObjects(const UriStrVec &uri_str_vec,
                                                          const std::vector<std::uint64_t> &value_sizes,
                                                          const BlockBuffers &object_buffers) const {
    if (max_object_bytes_ == 0 || uri_str_vec.empty() || uri_str_vec.size() > kMaxKvMetaBatchItems ||
        uri_str_vec.size() != value_sizes.size() || uri_str_vec.size() != object_buffers.size()) {
        return ER_INVALID_PARAMS;
    }
    std::uint64_t batch_bytes = 0;
    for (std::size_t i = 0; i < uri_str_vec.size(); ++i) {
        const std::uint64_t expected_size = value_sizes[i];
        std::uint64_t uri_size = 0;
        if (expected_size == 0 || expected_size > max_object_bytes_ || expected_size > kMaxKvMetaBatchBytes ||
            batch_bytes > kMaxKvMetaBatchBytes - expected_size ||
            !IsValidKvMetaLocation(uri_str_vec[i], DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, uri_size) ||
            uri_size != expected_size || object_buffers[i].iovs.empty()) {
            return ER_INVALID_PARAMS;
        }
        batch_bytes += expected_size;

        std::uint64_t buffer_size = 0;
        for (const auto &iov : object_buffers[i].iovs) {
            const auto base = reinterpret_cast<std::uintptr_t>(iov.base);
            if (iov.ignore || iov.base == nullptr || iov.size == 0 ||
                (iov.type != MemoryType::CPU && iov.type != MemoryType::GPU) || buffer_size > expected_size ||
                iov.size > expected_size - buffer_size ||
                iov.size > std::numeric_limits<std::uintptr_t>::max() - base) {
                return ER_INVALID_LOCAL_BUFFERS;
            }
            buffer_size += iov.size;
        }
        if (buffer_size != expected_size) {
            return ER_INVALID_LOCAL_BUFFERS;
        }
    }
    return ER_OK;
}

void KvMetaTransferClientImpl::SetAllowedObjectSizes(const std::vector<std::uint64_t> &value_sizes) {
    std::map<std::string, std::int64_t> sizes;
    for (const std::uint64_t size : value_sizes) {
        sizes.emplace(std::to_string(size), static_cast<std::int64_t>(size));
    }
    const auto wrapper_config = client_config_->sdk_wrapper_config();
    for (const auto type :
         {DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL, DataStorageType::DATA_STORAGE_TYPE_TAIR_MEMPOOL_SSD}) {
        if (const auto config = wrapper_config->GetSdkBackendConfig(type)) {
            config->set_spec_byte_sizes_per_block(sizes);
        }
    }
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
