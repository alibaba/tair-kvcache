#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"

#include <charconv>
#include <fcntl.h>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#include "kv_cache_manager/client/src/internal/sdk/lock_free_thread_pool.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_factory.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/storage_config.h"

namespace kv_cache_manager {

SdkWrapper::SdkWrapper() : sdk_factory_(SdkFactory::GetInstance()) {}

SdkWrapper::~SdkWrapper() {
    if (wait_task_thread_pool_) {
        wait_task_thread_pool_->stop();
        wait_task_thread_pool_.reset();
    }
    sdk_map_.clear();
    if (owned_shm_fd_ >= 0) {
        close(owned_shm_fd_);
        owned_shm_fd_ = -1;
    }
}

ClientErrorCode SdkWrapper::Init(const std::unique_ptr<ClientConfig> &client_config,
                                 const InitParams &init_params,
                                 const SharedMemoryRegistration *shared_memory_registration) {
    return InitInternal(client_config, init_params, false, 0, shared_memory_registration);
}

ClientErrorCode SdkWrapper::InitForKvMeta(const std::unique_ptr<ClientConfig> &client_config,
                                          const InitParams &init_params,
                                          std::uint64_t max_object_bytes,
                                          const SharedMemoryRegistration *shared_memory_registration) {
    if (max_object_bytes == 0 || max_object_bytes > std::numeric_limits<std::size_t>::max()) {
        KVCM_LOG_WARN("KVMeta max object bytes is invalid: %llu",
                      static_cast<unsigned long long>(max_object_bytes));
        return ER_INVALID_PARAMS;
    }
    return InitInternal(client_config, init_params, true, max_object_bytes, shared_memory_registration);
}

ClientErrorCode SdkWrapper::InitInternal(const std::unique_ptr<ClientConfig> &client_config,
                                         const InitParams &init_params,
                                         bool variable_object_size_enabled,
                                         std::uint64_t max_object_bytes,
                                         const SharedMemoryRegistration *shared_memory_registration) {
    if (!client_config) {
        KVCM_LOG_WARN("client config is null");
        return ER_INVALID_CLIENT_CONFIG;
    }
    variable_object_size_enabled_ = variable_object_size_enabled;
    max_variable_object_bytes_ = max_object_bytes;
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

    SharedMemoryRegistration prepared_registration;
    const SharedMemoryRegistration *active_registration = nullptr;
    if (shared_memory_registration != nullptr) {
        auto ec = PrepareSharedMemoryRegistration(*shared_memory_registration, prepared_registration);
        if (ec != ER_OK) {
            return ec;
        }
        active_registration = &prepared_registration;
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
    const auto &location_spec_infos = client_config->location_spec_infos();
    if (location_spec_infos.empty()) {
        KVCM_LOG_WARN("location_spec_infos is empty");
        return ER_INVALID_CLIENT_CONFIG;
    }

    // 验证 self_location_spec_name 存在于 location_spec_infos 中
    if (location_spec_infos.find(init_params.self_location_spec_name) == location_spec_infos.end()) {
        KVCM_LOG_WARN("location_spec_infos does not contain self_location_spec_name [%s]",
                      init_params.self_location_spec_name.c_str());
        return ER_INVALID_CLIENT_CONFIG;
    }

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
        ec = UpdateTairMempoolSdkConfig(sdk_backend_config, active_registration);
        if (ec != ER_OK) {
            KVCM_LOG_WARN("fill tair mempool span failed, storage config: %s", storage_config->ToString().c_str());
            return ec;
        }

        // 将完整的 spec → byte_size_per_block 映射传给 SDK
        sdk_backend_config->set_spec_byte_sizes_per_block(location_spec_infos);
        sdk_backend_config->set_variable_object_size_policy(variable_object_size_enabled, max_object_bytes);
        // 注入静态超时预算：后端用它从自身任务起点起算 deadline 并自律（内部取消）。
        // 不读取该字段的后端（tair_mempool 等）行为不受影响。
        sdk_backend_config->set_timeout_config(wrapper_config_->timeout_config());

        auto sdk = sdk_factory_->CreateSdk(type, sdk_backend_config, storage_config);
        if (!sdk) {
            KVCM_LOG_WARN("create sdk failed, storage config: %s", storage_config->ToString().c_str());
            return ER_CREATESDK_ERROR;
        }
        sdk_map_.insert({storage_config->global_unique_name(), sdk});
    }
    return ER_OK;
}

ClientErrorCode SdkWrapper::GroupBySdk(const std::vector<DataStorageUri> &remote_uris,
                                       const BlockBuffers &local_buffers,
                                       std::vector<SdkGroup> &groups) {
    std::unordered_map<std::string, size_t> hostname_to_group;
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        std::string hostname = remote_uris[i].GetHostName();
        if (hostname.empty()) {
            KVCM_LOG_WARN("GroupBySdk: remote_uri %s has empty hostname", remote_uris[i].ToUriString().c_str());
            return ER_GETSDK_ERROR;
        }
        auto it = hostname_to_group.find(hostname);
        if (it == hostname_to_group.end()) {
            auto sdk = GetSdk(remote_uris[i]);
            if (!sdk) {
                KVCM_LOG_WARN("GroupBySdk: no sdk found for hostname: %s", hostname.c_str());
                return ER_GETSDK_ERROR;
            }
            hostname_to_group[hostname] = groups.size();
            groups.push_back({sdk, {}, {}, {}});
        }
        auto &group = groups[hostname_to_group[hostname]];
        group.indices.push_back(i);
        group.uris.push_back(remote_uris[i]);
        group.buffers.push_back(local_buffers[i]);
    }
    return ER_OK;
}

ClientErrorCode SdkWrapper::Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) {
    auto ec = Valid(remote_uris, local_buffers);
    if (ec != ER_OK) {
        return ec;
    }
    std::vector<SdkGroup> groups;
    ec = GroupBySdk(remote_uris, local_buffers, groups);
    if (ec != ER_OK) {
        return ec;
    }

    // Build task vector for parallel dispatch
    std::vector<std::function<ClientErrorCode()>> tasks;
    tasks.reserve(groups.size());
    for (const auto &group : groups) {
        // Capture group by value to prevent use-after-free on timeout
        tasks.push_back([group]() { return group.sdk->Get(group.uris, group.buffers); });
    }

    // 静态预算：同一份已在 Init 时注入各后端（SdkBackendConfig::timeout_config），
    // 后端从自身任务起点起算 deadline 并自律。
    int timeout_ms = wrapper_config_->timeout_config().get_timeout_ms();
    return RunWithTimeoutParallel(OpType::GET, std::move(tasks), timeout_ms);
}

ClientErrorCode SdkWrapper::Put(const std::vector<DataStorageUri> &remote_uris,
                                const BlockBuffers &local_buffers,
                                std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) {
    auto ec = Valid(remote_uris, local_buffers);
    if (ec != ER_OK) {
        KVCM_LOG_WARN("put failed, remote_uris or local_buffers invalid.");
        return ec;
    }
    std::vector<SdkGroup> groups;
    ec = GroupBySdk(remote_uris, local_buffers, groups);
    if (ec != ER_OK) {
        return ec;
    }
    actual_remote_uris->resize(remote_uris.size());

    // Build task vector and result containers for parallel dispatch
    std::vector<std::function<ClientErrorCode()>> tasks;
    std::vector<std::shared_ptr<std::vector<DataStorageUri>>> group_results;
    tasks.reserve(groups.size());
    group_results.reserve(groups.size());

    for (const auto &group : groups) {
        auto group_actual_uris = std::make_shared<std::vector<DataStorageUri>>();
        group_results.push_back(group_actual_uris);
        // Capture group by value to prevent use-after-free on timeout
        tasks.push_back([group, group_actual_uris]() {
            return group.sdk->Put(group.uris, group.buffers, group_actual_uris);
        });
    }

    // 与 Get 同理：静态预算已在 Init 时注入后端。
    int timeout_ms = wrapper_config_->timeout_config().put_timeout_ms();
    ec = RunWithTimeoutParallel(OpType::PUT, std::move(tasks), timeout_ms);
    if (ec != ER_OK) {
        KVCM_LOG_WARN("put failed, sdk error: %d", static_cast<int>(ec));
        return ec;
    }

    // Result aggregation and validation
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto &group = groups[i];
        const auto &group_actual_uris = group_results[i];

        if (group_actual_uris->size() != group.indices.size()) {
            KVCM_LOG_WARN("sdk returned mismatched actual_uris size: %zu vs %zu",
                          group_actual_uris->size(),
                          group.indices.size());
            return ER_SDKWRITE_ERROR;
        }
        for (size_t j = 0; j < group.indices.size(); ++j) {
            (*actual_remote_uris)[group.indices[j]] = (*group_actual_uris)[j];
        }
    }
    return ER_OK;
}

ClientErrorCode SdkWrapper::GetKvMetaObjects(const std::vector<DataStorageUri> &remote_uris,
                                             const std::vector<std::uint64_t> &value_sizes,
                                             const BlockBuffers &local_buffers) {
    auto ec = ValidateKvMetaObjects(remote_uris, value_sizes, local_buffers);
    if (ec != ER_OK) {
        return ec;
    }

    // Fixed-size backends may interpret every element in one SDK call as a
    // block in the same allocation (for example path + blkid * block_size).
    // KVMeta objects are allocated independently and can have different
    // sizes, so never rebatch them through the regular fixed-size data path.
    // Resolve every SDK before submitting work to avoid partial I/O when one
    // URI is invalid or points at an unavailable backend.
    std::vector<std::function<ClientErrorCode()>> tasks;
    tasks.reserve(remote_uris.size());
    for (std::size_t i = 0; i < remote_uris.size(); ++i) {
        auto sdk = GetSdk(remote_uris[i]);
        if (!sdk) {
            KVCM_LOG_WARN("get KVMeta object failed, no sdk found for hostname: %s",
                          remote_uris[i].GetHostName().c_str());
            return ER_GETSDK_ERROR;
        }
        std::vector<DataStorageUri> object_uri{remote_uris[i]};
        BlockBuffers object_buffer{local_buffers[i]};
        tasks.push_back([sdk, object_uri = std::move(object_uri), object_buffer = std::move(object_buffer)]() {
            return sdk->Get(object_uri, object_buffer);
        });
    }

    // Callers own the exact-size buffers. On timeout/error, wait for an
    // already-running backend operation to stop before returning so it can no
    // longer access those buffers. This stronger drain is KVMeta-only.
    return RunWithTimeoutParallel(
        OpType::GET, std::move(tasks), wrapper_config_->timeout_config().get_timeout_ms(), true);
}

ClientErrorCode SdkWrapper::PutKvMetaObjects(
    const std::vector<DataStorageUri> &remote_uris,
    const std::vector<std::uint64_t> &value_sizes,
    const BlockBuffers &local_buffers,
    std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) {
    if (!actual_remote_uris) {
        return ER_INVALID_PARAMS;
    }
    actual_remote_uris->clear();
    auto ec = ValidateKvMetaObjects(remote_uris, value_sizes, local_buffers);
    if (ec != ER_OK) {
        return ec;
    }

    std::vector<std::function<ClientErrorCode()>> tasks;
    std::vector<std::shared_ptr<std::vector<DataStorageUri>>> object_results;
    tasks.reserve(remote_uris.size());
    object_results.reserve(remote_uris.size());
    for (std::size_t i = 0; i < remote_uris.size(); ++i) {
        auto sdk = GetSdk(remote_uris[i]);
        if (!sdk) {
            KVCM_LOG_WARN("put KVMeta object failed, no sdk found for hostname: %s",
                          remote_uris[i].GetHostName().c_str());
            return ER_GETSDK_ERROR;
        }
        std::vector<DataStorageUri> object_uri{remote_uris[i]};
        BlockBuffers object_buffer{local_buffers[i]};
        auto object_result = std::make_shared<std::vector<DataStorageUri>>();
        object_results.push_back(object_result);
        tasks.push_back([sdk,
                         object_uri = std::move(object_uri),
                         object_buffer = std::move(object_buffer),
                         object_result]() {
            return sdk->Put(object_uri, object_buffer, object_result);
        });
    }

    ec = RunWithTimeoutParallel(
        OpType::PUT, std::move(tasks), wrapper_config_->timeout_config().put_timeout_ms(), true);
    if (ec != ER_OK) {
        return ec;
    }

    std::vector<DataStorageUri> aggregated_uris;
    aggregated_uris.reserve(object_results.size());
    for (const auto &object_result : object_results) {
        if (object_result->size() != 1) {
            KVCM_LOG_WARN("KVMeta sdk returned mismatched actual_uris size: %zu vs 1", object_result->size());
            return ER_SDKWRITE_ERROR;
        }
        aggregated_uris.push_back((*object_result)[0]);
    }
    *actual_remote_uris = std::move(aggregated_uris);
    return ER_OK;
}

ClientErrorCode SdkWrapper::ValidateKvMetaObjects(const std::vector<DataStorageUri> &remote_uris,
                                                  const std::vector<std::uint64_t> &value_sizes,
                                                  const BlockBuffers &local_buffers) const {
    if (!variable_object_size_enabled_ || max_variable_object_bytes_ == 0 || remote_uris.empty() ||
        remote_uris.size() != value_sizes.size() || remote_uris.size() != local_buffers.size()) {
        return ER_INVALID_PARAMS;
    }
    for (std::size_t i = 0; i < remote_uris.size(); ++i) {
        const auto expected_size = value_sizes[i];
        const auto &uri = remote_uris[i];
        const auto &buffer = local_buffers[i];
        if (expected_size == 0 || expected_size > max_variable_object_bytes_ || !uri.Valid() ||
            uri.GetHostName().empty() || !uri.HasParam("size") || buffer.iovs.empty()) {
            return ER_INVALID_PARAMS;
        }

        const std::string uri_size_text = uri.GetParam("size");
        std::uint64_t uri_size = 0;
        const auto parse_result =
            std::from_chars(uri_size_text.data(), uri_size_text.data() + uri_size_text.size(), uri_size);
        if (uri_size_text.empty() || parse_result.ec != std::errc{} ||
            parse_result.ptr != uri_size_text.data() + uri_size_text.size() || uri_size != expected_size) {
            return ER_INVALID_PARAMS;
        }

        std::uint64_t buffer_size = 0;
        for (const auto &iov : buffer.iovs) {
            if (iov.ignore || iov.size == 0 || buffer_size > expected_size ||
                iov.size > expected_size - buffer_size || iov.base == nullptr ||
                (iov.type != MemoryType::CPU && iov.type != MemoryType::GPU)) {
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

ClientErrorCode SdkWrapper::RunWithTimeoutParallel(OpType op_type,
                                                   std::vector<std::function<ClientErrorCode()>> &&tasks,
                                                   int timeout_ms,
                                                   bool wait_for_inflight) const {
    if (tasks.empty()) {
        return ER_OK;
    }

    // Check capacity before submitting any tasks
    if (wait_task_thread_pool_->isFull()) {
        KVCM_LOG_WARN("run %s parallel failed, wait task thread pool is full",
                      getOpTypeString(op_type).c_str());
        return ER_THREADPOOL_ERROR;
    }

    // Submit all tasks with shared stop flag
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);

    // stop：错误/超时路径置位后，排队中尚未被拾起的任务直接短路，不再发起新的 I/O。
    // 时间准入（now >= deadline）只覆盖"已到 deadline"的情形；普通错误往往发生在
    // deadline 之前，若不显式拦截，排在后面的 group 会在 caller 拿到错误返回后
    // 依旧发起 I/O、写 caller buffer（多后端混布时放大暴露面）。
    auto stop = std::make_shared<std::atomic<bool>>(false);
    std::vector<std::future<ClientErrorCode>> futures;
    futures.reserve(tasks.size());

    for (auto &task : tasks) {
        auto wrapped = [stop, deadline, task]() -> ClientErrorCode {
            if (stop->load()) {
                return ER_SDK_TIMEOUT;
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                auto overdue_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - deadline)
                        .count();
                KVCM_LOG_WARN("deadline passed (overdue_ms=%lld), skip I/O", static_cast<long long>(overdue_ms));
                return ER_SDK_TIMEOUT;
            }
            return task();
        };
        futures.push_back(wait_task_thread_pool_->async(wrapped));
    }

    // Regular calls keep the existing bounded drain: stop queued work and wait
    // for peers only until the deadline. KVMeta uses a stronger drain because
    // its caller-owned exact-size buffers can be released immediately after
    // return; in-flight backend access must therefore finish first.
    auto drain = [&](size_t from) {
        stop->store(true);
        for (size_t j = from; j < futures.size(); ++j) {
            if (wait_for_inflight) {
                futures[j].wait();
            } else {
                futures[j].wait_until(deadline);
            }
        }
    };

    for (size_t i = 0; i < futures.size(); ++i) {
        if (futures[i].wait_until(deadline) != std::future_status::ready) {
            auto overdue_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - deadline)
                    .count();
            KVCM_LOG_WARN("run %s parallel but timeout: %d ms (group %zu/%zu), overdue_ms: %lld, "
                          "return immediately (in-flight I/O is not cancelled and may still write caller buffer)",
                          getOpTypeString(op_type).c_str(),
                          timeout_ms,
                          i + 1,
                          futures.size(),
                          static_cast<long long>(overdue_ms));
            // The regular fixed-size path preserves its existing immediate
            // timeout behavior. KVMeta drains the timed-out task as well,
            // because its caller-owned variable-size buffer may be released
            // as soon as this method returns.
            drain(wait_for_inflight ? i : i + 1);
            return ER_SDK_TIMEOUT;
        }

        auto ec = futures[i].get();
        if (ec != ER_OK) {
            KVCM_LOG_WARN("run %s parallel failed, error: %d, drain in-flight peers until deadline",
                          getOpTypeString(op_type).c_str(),
                          static_cast<int>(ec));
            drain(i + 1);
            return ec;
        }
    }

    return ER_OK;
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

ClientErrorCode SdkWrapper::PrepareSharedMemoryRegistration(const SharedMemoryRegistration &shared_memory_registration,
                                                            SharedMemoryRegistration &prepared_registration) {
    const bool disabled = shared_memory_registration.fd == -1 && shared_memory_registration.base == nullptr &&
                          shared_memory_registration.size == 0;
    if (disabled) {
        prepared_registration = shared_memory_registration;
        return ER_OK;
    }

    if (shared_memory_registration.fd < 0 || shared_memory_registration.base == nullptr ||
        shared_memory_registration.size == 0) {
        KVCM_LOG_WARN("shared memory registration must provide fd, base and size together");
        return ER_INVALID_PARAMS;
    }

    const uintptr_t base = reinterpret_cast<uintptr_t>(shared_memory_registration.base);
    if (shared_memory_registration.size > std::numeric_limits<uintptr_t>::max() - base) {
        KVCM_LOG_WARN("shared memory registration address range overflows");
        return ER_INVALID_PARAMS;
    }

    struct stat file_stat{};
    if (fstat(shared_memory_registration.fd, &file_stat) != 0 || file_stat.st_size < 0 ||
        static_cast<uintmax_t>(file_stat.st_size) < shared_memory_registration.size) {
        KVCM_LOG_WARN("shared memory fd is invalid or smaller than the registered range");
        return ER_INVALID_PARAMS;
    }

    int duplicated_fd = fcntl(shared_memory_registration.fd, F_DUPFD_CLOEXEC, 0);
    if (duplicated_fd < 0) {
        KVCM_LOG_WARN("duplicate shared memory fd failed");
        return ER_INVALID_PARAMS;
    }
    if (owned_shm_fd_ >= 0) {
        close(owned_shm_fd_);
    }
    owned_shm_fd_ = duplicated_fd;
    prepared_registration = shared_memory_registration;
    prepared_registration.fd = owned_shm_fd_;
    return ER_OK;
}

ClientErrorCode SdkWrapper::UpdateTairMempoolSdkConfig(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                                                       const SharedMemoryRegistration *shared_memory_registration) {
    if (!IsTairMempoolStorageType(sdk_backend_config->type())) {
        return ER_OK;
    }
    auto config = std::dynamic_pointer_cast<TairMempoolSdkConfig>(sdk_backend_config);
    if (!config) {
        KVCM_LOG_WARN("convert to tair mempool config failed");
        return ER_INVALID_SDKBACKEND_CONFIG;
    }
    if (shared_memory_registration == nullptr || shared_memory_registration->fd < 0) {
        return ER_OK;
    }
    config->set_shm_fd(shared_memory_registration->fd);
    config->set_shm_size(shared_memory_registration->size);
    config->set_client_base(shared_memory_registration->base);
    return ER_OK;
}

} // namespace kv_cache_manager
