#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"

#include "kv_cache_manager/client/src/internal/sdk/lock_free_thread_pool.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_factory.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

SdkWrapper::SdkWrapper() : sdk_factory_(SdkFactory::GetInstance()) {}

SdkWrapper::~SdkWrapper() {}

ClientErrorCode SdkWrapper::Init(const std::unique_ptr<ClientConfig> &client_config, const InitParams &init_params) {
    if (!client_config) {
        KVCM_LOG_WARN("client config is null");
        return ER_INVALID_CLIENT_CONFIG;
    }
    wrapper_config_ = client_config->sdk_wrapper_config();
    if (!wrapper_config_) {
        KVCM_LOG_WARN("sdk wrapper config is null");
        return ER_INVALID_SDKWRAPPER_CONFIG;
    }
    const std::string &storage_configs = init_params.storage_configs;
    if (!Jsonizable::FromJsonString(storage_configs, storage_configs_)) {
        KVCM_LOG_WARN("parse storage config failed, storage config: %s", storage_configs.c_str());
        return ER_INVALID_STORAGE_CONFIG;
    }
    if (storage_configs_.empty()) {
        KVCM_LOG_WARN("storage config is empty");
        return ER_INVALID_STORAGE_CONFIG;
    }

    wait_task_thread_pool_ = std::make_unique<LockFreeThreadPool>(
        wrapper_config_->thread_num(), wrapper_config_->queue_size(), "SdkWaitTaskPool");
    if (!wait_task_thread_pool_->start()) {
        KVCM_LOG_WARN("start wait task thread pool failed, thread num: %zu, queue size: %zu",
                      wrapper_config_->thread_num(),
                      wrapper_config_->queue_size());
        return ER_THREADPOOL_ERROR;
    }

    auto regist_span = init_params.regist_span;
    // For now, use the size of first location_spec_info and assuming all sizes in location_spec_infos are
    // consistent.
    const auto &location_spec_infos = client_config->location_spec_infos();
    if (location_spec_infos.empty()) {
        KVCM_LOG_WARN("location_spec_infos is empty");
        return ER_INVALID_CLIENT_CONFIG;
    }

    auto iter = location_spec_infos.find(init_params.self_location_spec_name);
    if (iter == location_spec_infos.end()) {
        KVCM_LOG_WARN("location_spec_infos does not contain self_location_spec_name [%s]",
                      init_params.self_location_spec_name.c_str());
        return ER_INVALID_CLIENT_CONFIG;
    }
    int64_t byte_size_per_block = iter->second;
    for (const auto &storage_config : storage_configs_) {
        DataStorageType type = storage_config->type();
        const auto &sdk_backend_config = wrapper_config_->GetSdkBackendConfig(type);
        if (!sdk_backend_config) {
            KVCM_LOG_WARN("sdk backend config is null, storage config: %s", storage_config->ToString().c_str());
            return ER_INVALID_SDKBACKEND_CONFIG;
        }
        auto ec = UpdateMooncakeSdkConfig(sdk_backend_config, regist_span, init_params.self_location_spec_name);
        if (ec != ER_OK) {
            KVCM_LOG_WARN("fill span failed, storage config: %s", storage_config->ToString().c_str());
            return ec;
        }
        sdk_backend_config->set_byte_size_per_block(byte_size_per_block);
        auto sdk = sdk_factory_->CreateSdk(type, sdk_backend_config, storage_config);
        if (!sdk) {
            KVCM_LOG_WARN("create sdk failed, storage config: %s", storage_config->ToString().c_str());
            return ER_CREATESDK_ERROR;
        }
        sdk_map_.insert({storage_config->global_unique_name(), sdk});
    }
    return ER_OK;
}

ClientErrorCode SdkWrapper::Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) {
    auto ec = Valid(remote_uris, local_buffers);
    if (ec != ER_OK) {
        return ec;
    }
    auto sdk = GetSdk(remote_uris[0]);
    if (!sdk) {
        return ER_GETSDK_ERROR;
    }
    auto task = [sdk, remote_uris, &local_buffers]() { return sdk->Get(remote_uris, local_buffers); };
    return RunWithTimeout(OpType::GET, task, wrapper_config_->timeout_config().get_timeout_ms());
}

ClientErrorCode SdkWrapper::Put(const std::vector<DataStorageUri> &remote_uris,
                                const BlockBuffers &local_buffers,
                                std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) {
    auto ec = Valid(remote_uris, local_buffers);
    if (ec != ER_OK) {
        KVCM_LOG_WARN("put failed, remote_uris or local_buffers invalid.");
        return ec;
    }
    auto sdk = GetSdk(remote_uris[0]);
    if (!sdk) {
        KVCM_LOG_WARN("put failed. can not get sdk by uri: %s", remote_uris[0].ToUriString().c_str());
        return ER_GETSDK_ERROR;
    }
    auto task = [sdk, remote_uris, local_buffers, actual_remote_uris]() {
        return sdk->Put(remote_uris, local_buffers, actual_remote_uris);
    };
    return RunWithTimeout(OpType::PUT, task, wrapper_config_->timeout_config().put_timeout_ms());
}

ClientErrorCode SdkWrapper::Valid(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers local_buffers) {
    if (remote_uris.empty() || local_buffers.empty() || remote_uris.size() != local_buffers.size()) {
        KVCM_LOG_WARN(
            "Check failed, remote_uris or local_buffers invalid, remote_uris size: %zu, local_buffers size: %zu",
            remote_uris.size(),
            local_buffers.size());
        return ER_INVALID_PARAMS;
    }

    const auto it = std::find_if(remote_uris.begin(), remote_uris.end(), [](const auto &uri) { return !uri.Valid(); });
    if (it != remote_uris.end()) {
        KVCM_LOG_WARN("Check failed, remote_uri %s invalid", it->ToUriString().c_str());
        return ER_INVALID_PARAMS;
    }
    return ER_OK;
}

std::shared_ptr<SdkInterface> SdkWrapper::GetSdk(const DataStorageUri &remote_uri) {
    std::string host_name = remote_uri.GetHostName();
    if (host_name.empty()) {
        KVCM_LOG_WARN("get sdk for remote_uri %s failed, remote_uri's host name is empty",
                      remote_uri.ToUriString().c_str());
        return nullptr;
    }
    auto it = sdk_map_.find(host_name);
    if (it != sdk_map_.end()) {
        return it->second;
    }
    return nullptr;
}

std::string SdkWrapper::getOpTypeString(OpType op_type) const {
    switch (op_type) {
    case OpType::GET: {
        return "get";
    }
    case OpType::PUT: {
        return "put";
    }
    }
    return "unknown";
}

ClientErrorCode
SdkWrapper::RunWithTimeout(OpType op_type, const std::function<ClientErrorCode()> &func, int timeout_ms) const {
    if (wait_task_thread_pool_->isFull()) {
        KVCM_LOG_WARN("run %s failed, wait task thread pool is full, something maybe wrong",
                      getOpTypeString(op_type).c_str());
        return ER_THREADPOOL_ERROR;
    }
    // wrap func with stop flag inside
    auto stop = std::make_shared<std::atomic<bool>>(false);
    auto wrapped = [stop, func]() -> ClientErrorCode {
        if (stop->load()) {
            return ER_SDK_TIMEOUT;
        }
        return func();
    };

    auto future = wait_task_thread_pool_->async(wrapped);
    if (future.wait_for(std::chrono::milliseconds(timeout_ms)) == std::future_status::ready) {
        return future.get();
    }

    KVCM_LOG_WARN("run %s but timeout: %d ms", getOpTypeString(op_type).c_str(), timeout_ms);
    stop->store(true);
    return ER_SDK_TIMEOUT;
}

ClientErrorCode SdkWrapper::UpdateMooncakeSdkConfig(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                                    RegistSpan *span,
                                                    const std::string &self_location_spec_name) {
    if (DataStorageType::DATA_STORAGE_TYPE_MOONCAKE != sdk_backend_config->type()) {
        return ER_OK;
    }
    auto config = std::dynamic_pointer_cast<MooncakeSdkConfig>(sdk_backend_config);
    if (!config) {
        KVCM_LOG_WARN("convert to mooncake config failed");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    if (span == nullptr) {
        KVCM_LOG_WARN("regist span is null but mooncake config is not null");
        return ER_INVALID_PARAMS;
    }
    if (config->local_mem_ptr() != nullptr || config->local_buffer_size() != 0) {
        KVCM_LOG_WARN("local mem ptr already set, not support register multi mooncake sdk");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    config->set_local_mem_ptr(span->base);
    config->set_local_buffer_size(span->size);
    config->set_self_location_spec_name(self_location_spec_name);
    return ER_OK;
}

} // namespace kv_cache_manager