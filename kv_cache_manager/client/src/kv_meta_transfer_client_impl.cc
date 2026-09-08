#include "kv_cache_manager/client/src/kv_meta_transfer_client_impl.h"

#include <utility>

#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/config/sdk_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/data_storage_uri.h"

namespace kv_cache_manager {

namespace {

constexpr const char *kKvMetaValueSpecName = "value";
constexpr std::size_t kMaxKvMetaBatchItems = 64;
constexpr std::uint64_t kMaxKvMetaObjectBytes = 1ULL * 1024 * 1024 * 1024;

} // namespace

ClientErrorCode ValidateKvMetaTransferClientConfig(const std::string &client_config,
                                                   const InitParams &init_params,
                                                   const std::string *expected_instance_group,
                                                   const std::string *expected_instance_id,
                                                   std::int32_t write_timeout_seconds,
                                                   std::uint32_t metadata_call_timeout_ms) {
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
    // One exact-object request may contain the full KVMeta service batch.
    // RunWithTimeoutParallel deliberately uses non-blocking submission for
    // caller-owned buffers, so accepting a smaller static queue would make an
    // otherwise valid 64-object request fail partway through admission. This
    // check is exclusive to the KVMeta transfer path; the regular fixed-block
    // TransferClient keeps its existing queue-size behavior.
    if (wrapper_config->queue_size() < kMaxKvMetaBatchItems) {
        KVCM_LOG_WARN("KVMeta sdk wrapper queue size must be at least %zu, got %zu",
                      kMaxKvMetaBatchItems,
                      wrapper_config->queue_size());
        return ER_INVALID_SDKWRAPPER_CONFIG;
    }
    // KvMetaObjectClient starts the server-side write lease before invoking
    // the data plane. Conservatively reserve one metadata timeout for the
    // PutStart response/request hand-off, one for the compatibility Get that
    // verifies masked hits, and one for PutFinish, in addition to the
    // data-plane timeout. Reject a nominally impossible lease before
    // RegisterInstance can mutate remote state. A zero pair keeps the
    // standalone KvMetaTransferClient API free of metadata-transaction policy.
    if (write_timeout_seconds != 0 || metadata_call_timeout_ms != 0) {
        if (write_timeout_seconds <= 0 || metadata_call_timeout_ms == 0) {
            KVCM_LOG_WARN("KVMeta write lease and metadata timeout must both be positive");
            return ER_INVALID_CLIENT_CONFIG;
        }
        const std::uint64_t write_lease_ms = static_cast<std::uint64_t>(write_timeout_seconds) * 1000;
        const std::uint64_t minimum_completion_ms =
            static_cast<std::uint64_t>(wrapper_config->timeout_config().put_timeout_ms()) +
            static_cast<std::uint64_t>(metadata_call_timeout_ms) * 3;
        if (write_lease_ms <= minimum_completion_ms) {
            KVCM_LOG_WARN(
                "KVMeta write_timeout_seconds must exceed put_timeout_ms plus three metadata call_timeout_ms windows");
            return ER_INVALID_CLIENT_CONFIG;
        }
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
    return sdk_wrapper_->GetKvMetaObjects(ParseLocations(uri_str_vec), value_sizes, object_buffers);
}

std::pair<ClientErrorCode, UriStrVec> KvMetaTransferClientImpl::SaveObjects(
    const UriStrVec &uri_str_vec, const std::vector<std::uint64_t> &value_sizes, const BlockBuffers &object_buffers) {
    if (!sdk_wrapper_) {
        return {ER_INVALID_SDKWRAPPER_CONFIG, {}};
    }
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    const auto ec =
        sdk_wrapper_->PutKvMetaObjects(ParseLocations(uri_str_vec), value_sizes, object_buffers, actual_remote_uris);
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
