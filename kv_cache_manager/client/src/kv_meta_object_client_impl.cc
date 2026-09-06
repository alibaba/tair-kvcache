#include "kv_cache_manager/client/src/kv_meta_object_client_impl.h"

#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
namespace {

constexpr const char *kKvMetaValueSpecName = "value";

std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
CreateObjectClient(const std::string &trace_id,
                   const KvMetaObjectClientConfig &config,
                   const SharedMemoryRegistration *shared_memory_registration) {
    if (config.instance_group.empty() || config.transfer_client_config.empty() || config.max_object_bytes == 0 ||
        config.write_timeout_seconds <= 0 || !(config.transfer_init_params.role_type & RoleType::WORKER) ||
        config.transfer_init_params.self_location_spec_name != kKvMetaValueSpecName) {
        return {ER_INVALID_PARAMS, nullptr};
    }
    auto metadata_client = KvMetaClient::Create(config.metadata);
    if (!metadata_client) {
        return {ER_METACLIENT_INIT_ERROR, nullptr};
    }
    auto [register_ec, storage_configs] =
        metadata_client->RegisterInstance(trace_id, config.instance_group, config.user_data);
    if (register_ec != ER_OK) {
        return {register_ec, nullptr};
    }
    InitParams transfer_init_params = config.transfer_init_params;
    transfer_init_params.storage_configs = std::move(storage_configs);
    std::unique_ptr<KvMetaTransferClient> transfer_client;
    if (shared_memory_registration == nullptr) {
        transfer_client = KvMetaTransferClient::Create(
            config.transfer_client_config, transfer_init_params, config.max_object_bytes);
    } else {
        transfer_client = KvMetaTransferClient::Create(config.transfer_client_config,
                                                       transfer_init_params,
                                                       config.max_object_bytes,
                                                       *shared_memory_registration);
    }
    if (!transfer_client) {
        return {ER_TRANSFERCLIENT_INIT_ERROR, nullptr};
    }
    std::unique_ptr<KvMetaObjectClient> object_client =
        std::make_unique<KvMetaObjectClientImpl>(std::move(metadata_client),
                                                 std::move(transfer_client),
                                                 config.max_object_bytes,
                                                 config.write_timeout_seconds);
    return {ER_OK, std::move(object_client)};
}

} // namespace

KvMetaObjectClientImpl::KvMetaObjectClientImpl(std::unique_ptr<KvMetaClient> metadata_client,
                                               std::unique_ptr<KvMetaTransferClient> transfer_client,
                                               std::uint64_t max_object_bytes,
                                               std::int32_t write_timeout_seconds)
    : metadata_client_(std::move(metadata_client)),
      transfer_client_(std::move(transfer_client)),
      max_object_bytes_(max_object_bytes),
      write_timeout_seconds_(write_timeout_seconds) {}

ClientErrorCode KvMetaObjectClientImpl::ValidateRequest(const std::vector<std::string> &keys,
                                                        const std::vector<std::uint64_t> &value_sizes,
                                                        const BlockBuffers &object_buffers,
                                                        std::uint64_t max_object_bytes) {
    if (max_object_bytes == 0 || keys.empty() || keys.size() != value_sizes.size() ||
        keys.size() != object_buffers.size()) {
        return ER_INVALID_PARAMS;
    }
    std::unordered_set<std::string> unique_keys;
    unique_keys.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (keys[i].empty() || !unique_keys.insert(keys[i]).second || value_sizes[i] == 0 ||
            value_sizes[i] > max_object_bytes || object_buffers[i].iovs.empty()) {
            return ER_INVALID_PARAMS;
        }
        std::uint64_t buffer_size = 0;
        for (const auto &iov : object_buffers[i].iovs) {
            if (iov.ignore || iov.size == 0 || iov.base == nullptr ||
                (iov.type != MemoryType::CPU && iov.type != MemoryType::GPU) || buffer_size > value_sizes[i] ||
                iov.size > value_sizes[i] - buffer_size) {
                return ER_INVALID_LOCAL_BUFFERS;
            }
            buffer_size += iov.size;
        }
        if (buffer_size != value_sizes[i]) {
            return ER_INVALID_LOCAL_BUFFERS;
        }
    }
    return ER_OK;
}

ClientErrorCode KvMetaObjectClientImpl::ExtractUris(const std::vector<KvMetaValueLocation> &locations,
                                                    const std::vector<std::uint64_t> &value_sizes,
                                                    UriStrVec &uris) {
    if (locations.empty() || locations.size() != value_sizes.size()) {
        return ER_SERVICE_INTERNAL_ERROR;
    }
    uris.clear();
    uris.reserve(locations.size());
    for (std::size_t i = 0; i < locations.size(); ++i) {
        const auto &location = locations[i];
        if (location.type == KvMetaStorageType::UNSPECIFIED || location.value_size != value_sizes[i] ||
            location.location_specs.size() != 1 || location.location_specs[0].spec_name != kKvMetaValueSpecName ||
            location.location_specs[0].uri.empty()) {
            uris.clear();
            return ER_SERVICE_SIZE_MISMATCH;
        }
        uris.push_back(location.location_specs[0].uri);
    }
    return ER_OK;
}

ClientErrorCode KvMetaObjectClientImpl::AbortWrite(const std::string &trace_id,
                                                   const std::string &write_session_id,
                                                   std::size_t location_count,
                                                   ClientErrorCode original_error) {
    if (write_session_id.empty() || location_count == 0) {
        return original_error;
    }
    const auto abort_ec =
        metadata_client_->FinishWrite(trace_id, write_session_id, std::vector<bool>(location_count, false));
    if (abort_ec != ER_OK) {
        KVCM_LOG_WARN("KVMeta object write rollback failed, original error [%d], rollback error [%d]",
                      static_cast<int>(original_error),
                      static_cast<int>(abort_ec));
        return abort_ec;
    }
    return original_error;
}

ClientErrorCode KvMetaObjectClientImpl::SaveObjects(const std::string &trace_id,
                                                    const std::vector<std::string> &keys,
                                                    const std::vector<std::uint64_t> &value_sizes,
                                                    const BlockBuffers &object_buffers) {
    const auto validation_ec = ValidateRequest(keys, value_sizes, object_buffers, max_object_bytes_);
    if (validation_ec != ER_OK) {
        return validation_ec;
    }

    auto [start_ec, start_result] =
        metadata_client_->StartWrite(trace_id, keys, value_sizes, write_timeout_seconds_);
    if (start_ec != ER_OK) {
        return start_ec;
    }
    if (start_result.key_mask.size() != keys.size()) {
        return AbortWrite(trace_id,
                          start_result.write_session_id,
                          start_result.locations.size(),
                          ER_SERVICE_INTERNAL_ERROR);
    }

    std::vector<std::uint64_t> missing_sizes;
    BlockBuffers missing_buffers;
    missing_sizes.reserve(start_result.locations.size());
    missing_buffers.reserve(start_result.locations.size());
    for (std::size_t i = 0; i < start_result.key_mask.size(); ++i) {
        if (!start_result.key_mask[i]) {
            missing_sizes.push_back(value_sizes[i]);
            missing_buffers.push_back(object_buffers[i]);
        }
    }
    if (missing_sizes.empty()) {
        return start_result.locations.empty() ? ER_OK : ER_SERVICE_INTERNAL_ERROR;
    }
    if (missing_sizes.size() != start_result.locations.size() || start_result.write_session_id.empty()) {
        return AbortWrite(trace_id,
                          start_result.write_session_id,
                          start_result.locations.size(),
                          ER_SERVICE_INTERNAL_ERROR);
    }

    UriStrVec requested_uris;
    const auto location_ec = ExtractUris(start_result.locations, missing_sizes, requested_uris);
    if (location_ec != ER_OK) {
        return AbortWrite(trace_id, start_result.write_session_id, start_result.locations.size(), location_ec);
    }
    auto [save_ec, actual_uris] = transfer_client_->SaveObjects(requested_uris, missing_sizes, missing_buffers);
    if (save_ec != ER_OK) {
        return AbortWrite(trace_id, start_result.write_session_id, start_result.locations.size(), save_ec);
    }
    if (actual_uris != requested_uris) {
        return AbortWrite(trace_id,
                          start_result.write_session_id,
                          start_result.locations.size(),
                          ER_SDKWRITE_ERROR);
    }
    return metadata_client_->FinishWrite(
        trace_id, start_result.write_session_id, std::vector<bool>(start_result.locations.size(), true));
}

ClientErrorCode KvMetaObjectClientImpl::LoadObjects(const std::string &trace_id,
                                                    const std::vector<std::string> &keys,
                                                    const std::vector<std::uint64_t> &expected_value_sizes,
                                                    const BlockBuffers &object_buffers) {
    const auto validation_ec = ValidateRequest(keys, expected_value_sizes, object_buffers, max_object_bytes_);
    if (validation_ec != ER_OK) {
        return validation_ec;
    }
    auto [get_ec, get_result] = metadata_client_->Get(trace_id, keys);
    if (get_ec != ER_OK) {
        return get_ec;
    }
    if (get_result.hit_mask.size() != keys.size() || get_result.locations.size() != keys.size()) {
        return ER_SERVICE_INTERNAL_ERROR;
    }
    for (bool hit : get_result.hit_mask) {
        if (!hit) {
            return ER_SERVICE_NOT_FOUND;
        }
    }
    UriStrVec uris;
    const auto location_ec = ExtractUris(get_result.locations, expected_value_sizes, uris);
    if (location_ec != ER_OK) {
        return location_ec;
    }
    return transfer_client_->LoadObjects(uris, expected_value_sizes, object_buffers);
}

ClientErrorCode KvMetaObjectClientImpl::Remove(const std::string &trace_id,
                                               const std::vector<std::string> &keys) {
    if (keys.empty()) {
        return ER_INVALID_PARAMS;
    }
    return metadata_client_->Remove(trace_id, keys);
}

std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
KvMetaObjectClient::Create(const std::string &trace_id, const KvMetaObjectClientConfig &config) {
    LoggerBroker::InitLoggerForClientOnce();
    return CreateObjectClient(trace_id, config, nullptr);
}

std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
KvMetaObjectClient::Create(const std::string &trace_id,
                           const KvMetaObjectClientConfig &config,
                           const SharedMemoryRegistration &shared_memory_registration) {
    LoggerBroker::InitLoggerForClientOnce();
    return CreateObjectClient(trace_id, config, &shared_memory_registration);
}

} // namespace kv_cache_manager
