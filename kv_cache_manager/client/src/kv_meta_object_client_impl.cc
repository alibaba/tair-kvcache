#include "kv_cache_manager/client/src/kv_meta_object_client_impl.h"

#include <chrono>
#include <limits>
#include <sys/stat.h>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/client/src/kv_meta_transfer_client_impl.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/kv_meta_uri.h"

namespace kv_cache_manager {
namespace {

bool IsKnownStorageType(KvMetaStorageType type) {
    return type == KvMetaStorageType::TAIR_MEMPOOL || type == KvMetaStorageType::TAIR_MEMPOOL_SSD;
}

bool ValidateStorageUri(KvMetaStorageType type, const std::string &uri_text, std::uint64_t expected_size) {
    if (!IsKnownStorageType(type)) {
        return false;
    }
    std::uint64_t size = 0;
    return IsValidKvMetaLocation(uri_text, size) && size == expected_size;
}

bool SameStorageUris(const UriStrVec &expected, const UriStrVec &actual) {
    if (expected.size() != actual.size()) {
        return false;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            return false;
        }
    }
    return true;
}

ClientErrorCode ValidateBuffer(const BlockBuffer &buffer, std::uint64_t value_size) {
    if (buffer.iovs.empty()) {
        return ER_INVALID_PARAMS;
    }
    std::uint64_t buffer_size = 0;
    for (const auto &iov : buffer.iovs) {
        const auto base = reinterpret_cast<std::uintptr_t>(iov.base);
        if (iov.ignore || iov.size == 0 || iov.base == nullptr ||
            (iov.type != MemoryType::CPU && iov.type != MemoryType::GPU) || buffer_size > value_size ||
            iov.size > value_size - buffer_size || iov.size > std::numeric_limits<std::uintptr_t>::max() - base) {
            return ER_INVALID_LOCAL_BUFFERS;
        }
        buffer_size += iov.size;
    }
    if (buffer_size != value_size) {
        return ER_INVALID_LOCAL_BUFFERS;
    }
    return ER_OK;
}

ClientErrorCode ValidateLocation(const KvMetaValueLocation &location, std::uint64_t size) {
    if (!IsKnownStorageType(location.type) || location.location_specs.size() != 1 ||
        location.location_specs[0].spec_name != kKvMetaValueSpecName || location.location_specs[0].uri.empty()) {
        return ER_SERVICE_INTERNAL_ERROR;
    }
    if (location.value_size != size) {
        return ER_SERVICE_SIZE_MISMATCH;
    }
    return ValidateStorageUri(location.type, location.location_specs[0].uri, size) ? ER_OK : ER_SERVICE_INTERNAL_ERROR;
}

std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
CreateObjectClient(const std::string &trace_id,
                   const KvMetaObjectClientConfig &config,
                   const ClientMemoryRegistrations *memory_registrations) {
    if (config.instance_group.empty() || config.transfer_client_config.empty() || config.max_object_bytes == 0 ||
        config.max_object_bytes > kKvMetaMaxValueBytes || config.write_timeout_seconds <= 0 ||
        config.write_timeout_seconds > kKvMetaMaxWriteTimeoutSeconds ||
        !(config.transfer_init_params.role_type & RoleType::WORKER) ||
        config.transfer_init_params.self_location_spec_name != kKvMetaValueSpecName) {
        return {ER_INVALID_PARAMS, nullptr};
    }
    const auto registration_ec = ValidateKvMetaMemoryRegistrations(config.transfer_init_params, memory_registrations);
    if (registration_ec != ER_OK) {
        return {registration_ec, nullptr};
    }
    const auto transfer_config_ec = ValidateKvMetaTransferClientConfig(config.transfer_client_config,
                                                                       config.transfer_init_params,
                                                                       &config.instance_group,
                                                                       &config.metadata.instance_id);
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
    if (memory_registrations == nullptr) {
        transfer_client =
            KvMetaTransferClient::Create(config.transfer_client_config, transfer_init_params, config.max_object_bytes);
    } else {
        transfer_client = KvMetaTransferClient::CreateWithMemoryRegistrations(
            config.transfer_client_config, transfer_init_params, config.max_object_bytes, *memory_registrations);
    }
    if (!transfer_client) {
        return {ER_TRANSFERCLIENT_INIT_ERROR, nullptr};
    }
    std::unique_ptr<KvMetaObjectClient> object_client = std::make_unique<KvMetaObjectClientImpl>(
        std::move(metadata_client), std::move(transfer_client), config.max_object_bytes, config.write_timeout_seconds);
    return {ER_OK, std::move(object_client)};
}

} // namespace

std::uint32_t GetKvMetaObjectClientApiVersion() noexcept { return kKvMetaObjectClientApiVersion; }

std::uint32_t GetKvMetaObjectTransferStateVersion() noexcept { return 1; }

std::uint32_t GetKvMetaObjectAddressUpdateVersion() noexcept { return 1; }

std::uint32_t GetKvMetaObjectMetadataVersion() noexcept { return 1; }

KvMetaObjectClientImpl::KvMetaObjectClientImpl(std::unique_ptr<KvMetaClient> metadata_client,
                                               std::unique_ptr<KvMetaTransferClient> transfer_client,
                                               std::uint64_t max_object_bytes,
                                               std::int32_t write_timeout_seconds)
    : metadata_client_(std::move(metadata_client))
    , transfer_client_(std::move(transfer_client))
    , max_object_bytes_(max_object_bytes)
    , write_timeout_seconds_(write_timeout_seconds) {}

KvMetaObjectClientImpl::OperationGuard::OperationGuard(KvMetaObjectClientImpl *owner, bool require_transfer)
    : owner_(owner), admitted_(owner_ != nullptr && owner_->TryBeginOperation(require_transfer)) {}

KvMetaObjectClientImpl::OperationGuard::~OperationGuard() noexcept {
    if (admitted_) {
        owner_->EndOperation();
    }
}

bool KvMetaObjectClientImpl::TryBeginOperation(bool require_transfer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closing_ || closed_ || !metadata_client_ || (require_transfer && !transfer_client_)) {
        return false;
    }
    ++active_operations_;
    return true;
}

void KvMetaObjectClientImpl::EndOperation() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (--active_operations_ == 0) {
        lifecycle_condition_.notify_all();
    }
}

ClientErrorCode KvMetaObjectClientImpl::ValidateRequest(const std::vector<std::string> &keys,
                                                        const std::vector<std::uint64_t> &value_sizes,
                                                        const BlockBuffers &object_buffers,
                                                        std::uint64_t max_object_bytes) {
    if (max_object_bytes == 0 || max_object_bytes > kKvMetaMaxValueBytes || keys.empty() ||
        keys.size() > kKvMetaMaxBatchItems || keys.size() != value_sizes.size() ||
        keys.size() != object_buffers.size()) {
        return ER_INVALID_PARAMS;
    }
    std::unordered_set<std::string> unique_keys;
    unique_keys.reserve(keys.size());
    std::uint64_t batch_bytes = 0;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (keys[i].empty() || keys[i].size() > kKvMetaMaxKeyBytes || !unique_keys.insert(keys[i]).second ||
            value_sizes[i] == 0 || value_sizes[i] > max_object_bytes || value_sizes[i] > kKvMetaMaxBatchBytes ||
            batch_bytes > kKvMetaMaxBatchBytes - value_sizes[i] || object_buffers[i].iovs.empty()) {
            return ER_INVALID_PARAMS;
        }
        batch_bytes += value_sizes[i];
        const auto ec = ValidateBuffer(object_buffers[i], value_sizes[i]);
        if (ec != ER_OK) {
            return ec;
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
        const auto ec = ValidateLocation(location, value_sizes[i]);
        if (ec != ER_OK) {
            uris.clear();
            return ec;
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
        abort_ec = metadata_client_->FinishWrite(trace_id, write_session_id, std::vector<bool>(location_count, false));
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
    return SaveObjectsWithTransferState(trace_id, keys, value_sizes, object_buffers).ec;
}

KvMetaObjectResult KvMetaObjectClientImpl::SaveObjectsWithTransferState(const std::string &trace_id,
                                                                        const std::vector<std::string> &keys,
                                                                        const std::vector<std::uint64_t> &value_sizes,
                                                                        const BlockBuffers &object_buffers) {
    return SaveObjectsWithUserData(trace_id, keys, value_sizes, object_buffers, {});
}

KvMetaObjectResult KvMetaObjectClientImpl::SaveObjectsWithUserData(
    const std::string &trace_id,
    const std::vector<std::string> &keys,
    const std::vector<std::uint64_t> &value_sizes,
    const BlockBuffers &object_buffers,
    const std::vector<KvMetaObjectUserData> &user_data) {
    bool may_be_inflight = false;
    const auto ec = SaveObjectsImpl(trace_id, keys, value_sizes, object_buffers, may_be_inflight, user_data);
    return {ec, may_be_inflight};
}

ClientErrorCode KvMetaObjectClientImpl::SaveObjectsImpl(const std::string &trace_id,
                                                        const std::vector<std::string> &keys,
                                                        const std::vector<std::uint64_t> &value_sizes,
                                                        const BlockBuffers &object_buffers,
                                                        bool &data_transfer_may_be_inflight,
                                                        const std::vector<KvMetaObjectUserData> &user_data) {
    const auto validation_ec = ValidateRequest(keys, value_sizes, object_buffers, max_object_bytes_);
    if (validation_ec != ER_OK) {
        return validation_ec;
    }
    OperationGuard operation(this, true);
    if (!operation.admitted()) {
        return ER_CLIENT_NOT_EXISTS;
    }

    ClientErrorCode start_ec = ER_INVALID_GRPCSTATUS;
    KvMetaStartWriteResult start_result;
    try {
        auto result = metadata_client_->StartWriteWithUserData(
            trace_id, keys, value_sizes, write_timeout_seconds_, user_data);
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
        return AbortWrite(
            trace_id, start_result.write_session_id, start_result.locations.size(), ER_SERVICE_INTERNAL_ERROR);
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
        if (!start_result.locations.empty() || !start_result.write_session_id.empty()) {
            return AbortWrite(
                trace_id, start_result.write_session_id, start_result.locations.size(), ER_SERVICE_INTERNAL_ERROR);
        }
    } else if (missing_sizes.size() != start_result.locations.size() || start_result.write_session_id.empty()) {
        return AbortWrite(trace_id, start_result.write_session_id, missing_sizes.size(), ER_SERVICE_INTERNAL_ERROR);
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
    data_transfer_may_be_inflight = true;
    try {
        auto result = transfer_client_->SaveObjects(requested_uris, missing_sizes, missing_buffers);
        save_ec = result.first;
        actual_uris = std::move(result.second);
    } catch (...) {
        KVCM_LOG_WARN("KVMeta object data-plane save threw; retaining allocations until metadata-only expiry");
        return ER_SDKWRITE_ERROR;
    }
    if (save_ec != ER_OK) {
        // BUFFER_SAFE permits caller-buffer reuse, but does not establish
        // that a remote allocation can be recycled. Keep those two decisions
        // separate, so a safe local failure need not retire the client.
        if (save_ec == ER_SDK_ERROR_BUFFER_SAFE || save_ec == ER_SDK_TIMEOUT_BUFFER_SAFE) {
            data_transfer_may_be_inflight = false;
            return save_ec;
        }
        // These are pre-I/O rejections from the transfer client. Other errors
        // do not prove quiescence: FinishWrite(false) would free addresses an
        // outstanding write can still reach. Session expiry only hides them.
        if (save_ec == ER_INVALID_PARAMS || save_ec == ER_INVALID_LOCAL_BUFFERS ||
            save_ec == ER_INVALID_SDKWRAPPER_CONFIG) {
            data_transfer_may_be_inflight = false;
            return AbortWrite(trace_id, start_result.write_session_id, start_result.locations.size(), save_ec);
        }
        return save_ec;
    }
    data_transfer_may_be_inflight = false;
    // Keep the complete transaction admitted until its metadata commit ends;
    // Close drains it, while unrelated requests can continue transferring.
    if (!SameStorageUris(requested_uris, actual_uris)) {
        return AbortWrite(trace_id, start_result.write_session_id, start_result.locations.size(), ER_SDKWRITE_ERROR);
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
    return LoadObjectsWithTransferState(trace_id, keys, expected_value_sizes, object_buffers).ec;
}

KvMetaObjectResult
KvMetaObjectClientImpl::LoadObjectsWithTransferState(const std::string &trace_id,
                                                     const std::vector<std::string> &keys,
                                                     const std::vector<std::uint64_t> &expected_value_sizes,
                                                     const BlockBuffers &object_buffers) {
    bool may_be_inflight = false;
    const auto ec = LoadObjectsImpl(trace_id, keys, expected_value_sizes, object_buffers, may_be_inflight);
    return {ec, may_be_inflight};
}

ClientErrorCode KvMetaObjectClientImpl::LoadObjectsImpl(const std::string &trace_id,
                                                        const std::vector<std::string> &keys,
                                                        const std::vector<std::uint64_t> &expected_value_sizes,
                                                        const BlockBuffers &object_buffers,
                                                        bool &data_transfer_may_be_inflight) {
    const auto validation_ec = ValidateRequest(keys, expected_value_sizes, object_buffers, max_object_bytes_);
    if (validation_ec != ER_OK) {
        return validation_ec;
    }
    OperationGuard operation(this, true);
    if (!operation.admitted()) {
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
    const auto result = LoadPayload(trace_id, uris, expected_value_sizes, object_buffers);
    data_transfer_may_be_inflight = result.data_transfer_may_be_inflight;
    return result.ec;
}

KvMetaObjectResult KvMetaObjectClientImpl::LoadPayload(const std::string &trace_id,
                                                       const UriStrVec &uris,
                                                       const std::vector<std::uint64_t> &sizes,
                                                       const BlockBuffers &buffers) {
    ClientErrorCode ec = ER_SDKREAD_ERROR;
    try {
        ec = transfer_client_->LoadObjects(uris, sizes, buffers);
    } catch (...) {
        KVCM_LOG_WARN("KVMeta object data-plane load threw, trace_id[%s]", trace_id.c_str());
        return {ec, true};
    }
    if (ec != ER_OK) {
        KVCM_LOG_WARN(
            "KVMeta object data-plane load failed, trace_id[%s] code[%d]", trace_id.c_str(), static_cast<int>(ec));
    }
    // Preserve the transfer SDK's buffer-safety contract on failure.
    return {ec, ec != ER_OK && ec != ER_SDK_ERROR_BUFFER_SAFE && ec != ER_SDK_TIMEOUT_BUFFER_SAFE};
}

ClientErrorCode KvMetaObjectClientImpl::Remove(const std::string &trace_id, const std::vector<std::string> &keys) {
    if (keys.empty() || keys.size() > kKvMetaMaxBatchItems) {
        return ER_INVALID_PARAMS;
    }
    std::unordered_set<std::string> unique_keys;
    unique_keys.reserve(keys.size());
    for (const auto &key : keys) {
        if (key.empty() || key.size() > kKvMetaMaxKeyBytes || !unique_keys.insert(key).second) {
            return ER_INVALID_PARAMS;
        }
    }
    OperationGuard operation(this, false);
    if (!operation.admitted()) {
        return ER_CLIENT_NOT_EXISTS;
    }
    try {
        return metadata_client_->Remove(trace_id, keys);
    } catch (...) {
        KVCM_LOG_WARN("KVMeta object Remove threw; mutation outcome is unknown");
        return ER_INVALID_GRPCSTATUS;
    }
}

std::pair<ClientErrorCode, std::optional<ObjectMeta>>
KvMetaObjectClientImpl::GetObjectMeta(const std::string &key, std::uint32_t timeout_ms, const std::string &trace_id) {
    if (timeout_ms == 0 || timeout_ms > 600000) {
        return {ER_INVALID_PARAMS, std::nullopt};
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    OperationGuard guard(this, /*require_transfer=*/false);
    if (!guard.admitted()) {
        return {ER_CLIENT_NOT_EXISTS, std::nullopt};
    }
    auto [ec, result] = metadata_client_->GetWithDeadline(trace_id, {key}, deadline);
    if (ec != ER_OK) {
        return {ec, std::nullopt};
    }
    if (result.hit_mask.size() != 1 || result.locations.size() != 1) {
        return {ER_SERVICE_INTERNAL_ERROR, std::nullopt};
    }
    if (!result.hit_mask.front()) {
        return {ER_OK, std::nullopt};
    }
    const auto size = result.locations.front().value_size;
    if (size == 0 || size > max_object_bytes_) {
        return {ER_SERVICE_SIZE_MISMATCH, std::nullopt};
    }
    return {ER_OK, ObjectMeta{metadata_owner_, std::move(result.locations.front())}};
}

KvMetaObjectResult
KvMetaObjectClientImpl::LoadFromMeta(const ObjectMeta &meta, const BlockBuffer &buffer, const std::string &trace_id) {
    if (meta.owner_.lock() != metadata_owner_) {
        return {ER_INVALID_PARAMS, false};
    }
    const auto size = meta.SizeBytes();
    const auto buffer_ec = ValidateBuffer(buffer, size);
    if (buffer_ec != ER_OK) {
        return {buffer_ec, false};
    }
    const auto location_ec = ValidateLocation(meta.location_, size);
    if (location_ec != ER_OK) {
        return {location_ec, false};
    }
    OperationGuard operation(this, /*require_transfer=*/true);
    if (!operation.admitted()) {
        return {ER_CLIENT_NOT_EXISTS, false};
    }
    return LoadPayload(trace_id, {meta.location_.location_specs.front().uri}, {size}, {buffer});
}

ClientErrorCode KvMetaObjectClientImpl::UpdateMetadataAddresses(const std::vector<std::string> &addresses) {
    OperationGuard operation(this, false);
    if (!operation.admitted()) {
        return ER_CLIENT_NOT_EXISTS;
    }
    try {
        return metadata_client_->UpdateAddresses(addresses);
    } catch (...) {
        KVCM_LOG_WARN("KVMeta metadata address update failed; retaining previous endpoints");
        return ER_CONNECT_FAIL;
    }
}

void KvMetaObjectClientImpl::Close() noexcept {
    std::unique_ptr<KvMetaTransferClient> transfer_client;
    std::unique_ptr<KvMetaClient> metadata_client;
    std::unique_lock<std::mutex> lock(mutex_);
    if (closed_) {
        return;
    }
    if (closing_) {
        lifecycle_condition_.wait(lock, [&]() { return closed_; });
        return;
    }
    closing_ = true;
    lifecycle_condition_.wait(lock, [&]() { return active_operations_ == 0; });
    // Destroy the data plane first: it owns worker pools and may still refer
    // to storage configuration returned by the metadata registration.
    transfer_client = std::move(transfer_client_);
    metadata_client = std::move(metadata_client_);
    lock.unlock();
    transfer_client.reset();
    metadata_client.reset();
    lock.lock();
    closed_ = true;
    lifecycle_condition_.notify_all();
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
    ClientMemoryRegistrations registrations;
    registrations.host = shared_memory_registration;
    return CreateObjectClient(trace_id, config, &registrations);
}

std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
KvMetaObjectClient::CreateWithMemoryRegistrations(const std::string &trace_id,
                                                  const KvMetaObjectClientConfig &config,
                                                  const ClientMemoryRegistrations &registrations) {
    LoggerBroker::InitLoggerForClientOnce();
    return CreateObjectClient(trace_id, config, &registrations);
}

} // namespace kv_cache_manager
