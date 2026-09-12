#include "kv_cache_manager/client/src/kv_meta_object_client_impl.h"

#include <algorithm>
#include <limits>
#include <sys/stat.h>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/client/src/kv_meta_transfer_client_impl.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
namespace {

constexpr const char *kKvMetaValueSpecName = "value";
constexpr std::size_t kMaxBatchItems = 64;
constexpr std::size_t kMaxKeyBytes = 512;
constexpr std::uint64_t kMaxServiceObjectBytes = 1ULL * 1024 * 1024 * 1024;
constexpr std::uint64_t kMaxServiceBatchBytes = 4ULL * 1024 * 1024 * 1024;
constexpr std::int32_t kMaxWriteTimeoutSeconds = 1800;

bool AddressRangeIsRepresentable(const void *base, std::size_t size) {
    if (base == nullptr) {
        return size == 0;
    }
    const auto address = reinterpret_cast<std::uintptr_t>(base);
    return size <= std::numeric_limits<std::uintptr_t>::max() - address;
}

bool IsKnownStorageType(KvMetaStorageType type) {
    switch (type) {
    case KvMetaStorageType::HF3FS:
    case KvMetaStorageType::MOONCAKE:
    case KvMetaStorageType::TAIR_MEMPOOL:
    case KvMetaStorageType::NFS:
    case KvMetaStorageType::VCNS_HF3FS:
    case KvMetaStorageType::DUMMY:
    case KvMetaStorageType::EVENT_REPORT_L1P5:
    case KvMetaStorageType::EVENT_REPORT_L2:
    case KvMetaStorageType::TAIR_MEMPOOL_SSD:
        return true;
    case KvMetaStorageType::UNSPECIFIED:
    default:
        return false;
    }
}

ClientErrorCode ValidateLocalRegistration(const InitParams &init_params,
                                          const SharedMemoryRegistration *shared_memory_registration) {
    if (init_params.regist_span != nullptr) {
        const auto &span = *init_params.regist_span;
        if ((span.base == nullptr) != (span.size == 0) || !AddressRangeIsRepresentable(span.base, span.size)) {
            KVCM_LOG_WARN("KVMeta registered span is incomplete or its address range overflows");
            return ER_INVALID_PARAMS;
        }
    }
    if (shared_memory_registration == nullptr) {
        return ER_OK;
    }

    const auto &registration = *shared_memory_registration;
    const bool disabled = registration.fd == -1 && registration.base == nullptr && registration.size == 0;
    if (disabled) {
        return ER_OK;
    }
    if (registration.fd < 0 || registration.base == nullptr || registration.size == 0 ||
        !AddressRangeIsRepresentable(registration.base, registration.size)) {
        KVCM_LOG_WARN("KVMeta shared-memory registration is incomplete or its address range overflows");
        return ER_INVALID_PARAMS;
    }
    struct stat file_stat {};
    if (fstat(registration.fd, &file_stat) != 0 || file_stat.st_size < 0 ||
        static_cast<std::uintmax_t>(file_stat.st_size) < registration.size) {
        KVCM_LOG_WARN("KVMeta shared-memory fd is invalid or smaller than the registered range");
        return ER_INVALID_PARAMS;
    }
    return ER_OK;
}

std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
CreateObjectClient(const std::string &trace_id,
                   const KvMetaObjectClientConfig &config,
                   const SharedMemoryRegistration *shared_memory_registration) {
    if (config.instance_group.empty() || config.transfer_client_config.empty() || config.max_object_bytes == 0 ||
        config.max_object_bytes > kMaxServiceObjectBytes || config.write_timeout_seconds <= 0 ||
        config.write_timeout_seconds > kMaxWriteTimeoutSeconds ||
        !(config.transfer_init_params.role_type & RoleType::WORKER) ||
        config.transfer_init_params.self_location_spec_name != kKvMetaValueSpecName) {
        return {ER_INVALID_PARAMS, nullptr};
    }
    const auto registration_ec =
        ValidateLocalRegistration(config.transfer_init_params, shared_memory_registration);
    if (registration_ec != ER_OK) {
        return {registration_ec, nullptr};
    }
    const auto transfer_config_ec = ValidateKvMetaTransferClientConfig(config.transfer_client_config,
                                                                       config.transfer_init_params,
                                                                       &config.instance_group,
                                                                       &config.metadata.instance_id,
                                                                       config.write_timeout_seconds,
                                                                       config.metadata.call_timeout_ms);
    if (transfer_config_ec != ER_OK) {
        return {transfer_config_ec, nullptr};
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

std::uint32_t GetKvMetaObjectClientApiVersion() noexcept {
    return kKvMetaObjectClientApiVersion;
}

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
    if (max_object_bytes == 0 || max_object_bytes > kMaxServiceObjectBytes || keys.empty() ||
        keys.size() > kMaxBatchItems || keys.size() != value_sizes.size() ||
        keys.size() != object_buffers.size()) {
        return ER_INVALID_PARAMS;
    }
    std::unordered_set<std::string> unique_keys;
    unique_keys.reserve(keys.size());
    std::uint64_t batch_bytes = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (keys[i].empty() || keys[i].size() > kMaxKeyBytes || !unique_keys.insert(keys[i]).second ||
            value_sizes[i] == 0 || value_sizes[i] > max_object_bytes || value_sizes[i] > kMaxServiceBatchBytes ||
            batch_bytes > kMaxServiceBatchBytes - value_sizes[i] || object_buffers[i].iovs.empty()) {
            return ER_INVALID_PARAMS;
        }
        batch_bytes += value_sizes[i];
        std::uint64_t buffer_size = 0;
        for (const auto &iov : object_buffers[i].iovs) {
            const auto base = reinterpret_cast<std::uintptr_t>(iov.base);
            if (iov.ignore || iov.size == 0 || iov.base == nullptr ||
                (iov.type != MemoryType::CPU && iov.type != MemoryType::GPU) || buffer_size > value_sizes[i] ||
                iov.size > value_sizes[i] - buffer_size ||
                iov.size > std::numeric_limits<std::uintptr_t>::max() - base) {
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
        if (!IsKnownStorageType(location.type) || location.location_specs.size() != 1 ||
            location.location_specs[0].spec_name != kKvMetaValueSpecName || location.location_specs[0].uri.empty()) {
            uris.clear();
            return ER_SERVICE_INTERNAL_ERROR;
        }
        if (location.value_size != value_sizes[i]) {
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
    ClientErrorCode abort_ec = ER_INVALID_GRPCSTATUS;
    try {
        abort_ec =
            metadata_client_->FinishWrite(trace_id, write_session_id, std::vector<bool>(location_count, false));
    } catch (...) {
        KVCM_LOG_WARN("KVMeta object write rollback threw, original error [%d]; rollback outcome is unknown",
                      static_cast<int>(original_error));
        return ER_INVALID_GRPCSTATUS;
    }
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
    if (!metadata_client_ || !transfer_client_) {
        return ER_CLIENT_NOT_EXISTS;
    }

    ClientErrorCode start_ec = ER_INVALID_GRPCSTATUS;
    KvMetaStartWriteResult start_result;
    try {
        auto result = metadata_client_->StartWrite(trace_id, keys, value_sizes, write_timeout_seconds_);
        start_ec = result.first;
        start_result = std::move(result.second);
    } catch (...) {
        KVCM_LOG_WARN("KVMeta object StartWrite threw; write-session outcome is unknown");
        return ER_INVALID_GRPCSTATUS;
    }
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
    std::vector<std::string> hit_keys;
    std::vector<std::uint64_t> hit_sizes;
    missing_sizes.reserve(start_result.locations.size());
    missing_buffers.reserve(start_result.locations.size());
    hit_keys.reserve(keys.size());
    hit_sizes.reserve(keys.size());
    for (std::size_t i = 0; i < start_result.key_mask.size(); ++i) {
        if (!start_result.key_mask[i]) {
            missing_sizes.push_back(value_sizes[i]);
            missing_buffers.push_back(object_buffers[i]);
        } else {
            hit_keys.push_back(keys[i]);
            hit_sizes.push_back(value_sizes[i]);
        }
    }
    if (missing_sizes.empty()) {
        if (!start_result.locations.empty() || !start_result.write_session_id.empty()) {
            return AbortWrite(trace_id,
                              start_result.write_session_id,
                              start_result.locations.size(),
                              ER_SERVICE_INTERNAL_ERROR);
        }
    } else if (missing_sizes.size() != start_result.locations.size() || start_result.write_session_id.empty()) {
        return AbortWrite(trace_id,
                          start_result.write_session_id,
                          missing_sizes.size(),
                          ER_SERVICE_INTERNAL_ERROR);
    }

    if (!hit_keys.empty()) {
        // A rolling-upgrade peer may still use the original protocol behavior
        // that masked active reservations as hits. Verify that every masked
        // key is actually committed/readable before reporting SaveObjects
        // success or writing this call's misses.
        ClientErrorCode get_ec = ER_SERVICE_INTERNAL_ERROR;
        KvMetaGetResult hit_result;
        try {
            auto result = metadata_client_->Get(trace_id, hit_keys);
            get_ec = result.first;
            hit_result = std::move(result.second);
        } catch (...) {
            KVCM_LOG_WARN("KVMeta compatibility Get threw while an object write session was active");
            return AbortWrite(trace_id,
                              start_result.write_session_id,
                              missing_sizes.size(),
                              ER_SERVICE_INTERNAL_ERROR);
        }
        if (get_ec != ER_OK) {
            return AbortWrite(trace_id,
                              start_result.write_session_id,
                              missing_sizes.size(),
                              get_ec);
        }
        if (hit_result.hit_mask.size() != hit_keys.size() ||
            hit_result.locations.size() != hit_keys.size()) {
            return AbortWrite(trace_id,
                              start_result.write_session_id,
                              missing_sizes.size(),
                              ER_SERVICE_INTERNAL_ERROR);
        }
        if (std::any_of(hit_result.hit_mask.begin(), hit_result.hit_mask.end(), [](bool hit) { return !hit; })) {
            return AbortWrite(trace_id,
                              start_result.write_session_id,
                              missing_sizes.size(),
                              ER_SERVICE_WRITE_IN_PROGRESS);
        }
        UriStrVec hit_uris;
        const auto hit_location_ec = ExtractUris(hit_result.locations, hit_sizes, hit_uris);
        if (hit_location_ec != ER_OK) {
            return AbortWrite(trace_id,
                              start_result.write_session_id,
                              missing_sizes.size(),
                              hit_location_ec);
        }
    }
    if (missing_sizes.empty()) {
        return ER_OK;
    }

    UriStrVec requested_uris;
    const auto location_ec = ExtractUris(start_result.locations, missing_sizes, requested_uris);
    if (location_ec != ER_OK) {
        return AbortWrite(trace_id, start_result.write_session_id, start_result.locations.size(), location_ec);
    }
    ClientErrorCode save_ec = ER_SDKWRITE_ERROR;
    UriStrVec actual_uris;
    try {
        auto result = transfer_client_->SaveObjects(requested_uris, missing_sizes, missing_buffers);
        save_ec = result.first;
        actual_uris = std::move(result.second);
    } catch (...) {
        KVCM_LOG_WARN("KVMeta object data-plane save threw; aborting the write session");
        return AbortWrite(trace_id,
                          start_result.write_session_id,
                          start_result.locations.size(),
                          ER_SDKWRITE_ERROR);
    }
    if (save_ec != ER_OK) {
        return AbortWrite(trace_id, start_result.write_session_id, start_result.locations.size(), save_ec);
    }
    if (actual_uris != requested_uris) {
        return AbortWrite(trace_id,
                          start_result.write_session_id,
                          start_result.locations.size(),
                          ER_SDKWRITE_ERROR);
    }
    try {
        return metadata_client_->FinishWrite(
            trace_id, start_result.write_session_id, std::vector<bool>(start_result.locations.size(), true));
    } catch (...) {
        KVCM_LOG_WARN("KVMeta object FinishWrite threw; commit outcome is unknown");
        return ER_INVALID_GRPCSTATUS;
    }
}

ClientErrorCode KvMetaObjectClientImpl::LoadObjects(const std::string &trace_id,
                                                    const std::vector<std::string> &keys,
                                                    const std::vector<std::uint64_t> &expected_value_sizes,
                                                    const BlockBuffers &object_buffers) {
    const auto validation_ec = ValidateRequest(keys, expected_value_sizes, object_buffers, max_object_bytes_);
    if (validation_ec != ER_OK) {
        return validation_ec;
    }
    if (!metadata_client_ || !transfer_client_) {
        return ER_CLIENT_NOT_EXISTS;
    }
    ClientErrorCode get_ec = ER_SERVICE_INTERNAL_ERROR;
    KvMetaGetResult get_result;
    try {
        auto result = metadata_client_->Get(trace_id, keys);
        get_ec = result.first;
        get_result = std::move(result.second);
    } catch (...) {
        KVCM_LOG_WARN("KVMeta object Get threw before data-plane load");
        return ER_SERVICE_INTERNAL_ERROR;
    }
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
    try {
        return transfer_client_->LoadObjects(uris, expected_value_sizes, object_buffers);
    } catch (...) {
        KVCM_LOG_WARN("KVMeta object data-plane load threw");
        return ER_SDKREAD_ERROR;
    }
}

ClientErrorCode KvMetaObjectClientImpl::Remove(const std::string &trace_id,
                                               const std::vector<std::string> &keys) {
    if (keys.empty() || keys.size() > kMaxBatchItems) {
        return ER_INVALID_PARAMS;
    }
    std::unordered_set<std::string> unique_keys;
    unique_keys.reserve(keys.size());
    for (const auto &key : keys) {
        if (key.empty() || key.size() > kMaxKeyBytes || !unique_keys.insert(key).second) {
            return ER_INVALID_PARAMS;
        }
    }
    if (!metadata_client_) {
        return ER_CLIENT_NOT_EXISTS;
    }
    try {
        return metadata_client_->Remove(trace_id, keys);
    } catch (...) {
        KVCM_LOG_WARN("KVMeta object Remove threw; mutation outcome is unknown");
        return ER_INVALID_GRPCSTATUS;
    }
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
