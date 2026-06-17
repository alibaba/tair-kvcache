#include "kv_cache_manager/client/src/transfer_client_impl.h"

#include <sstream>
#if defined(USING_CUDA) || defined(USING_MUSA)
#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"
#include "kv_cache_manager/common/env_util.h"
#endif
#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/client/src/internal/util/debug_string_util.h"
#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/data_storage/storage_config.h"

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

ClientErrorCode TransferClientImpl::ValidateStorageConfigsForIntegrity(const std::string &storage_configs_json,
                                                                       bool &any_meta_checksum_enabled) const {
    any_meta_checksum_enabled = false;
    if (storage_configs_json.empty()) {
        return ER_OK;
    }
    std::vector<std::shared_ptr<StorageConfig>> parsed;
    if (!Jsonizable::FromJsonString(storage_configs_json, parsed)) {
        // 让下游 sdk_wrapper 报具体的 ER_INVALID_STORAGE_CONFIG，本函数保持沉默。
        return ER_OK;
    }
    for (const auto &cfg : parsed) {
        if (!cfg) {
            continue;
        }
        const auto &integrity = cfg->integrity();
        if (integrity.enable_inline_header()) {
            KVCM_LOG_ERROR("storage config [%s] enables inline_header, which is reserved (not implemented in this "
                           "release); reject init",
                           cfg->global_unique_name().c_str());
            return ER_INLINE_HEADER_INVALID;
        }
        if (integrity.inline_header_version() != 0) {
            KVCM_LOG_ERROR(
                "storage config [%s] sets inline_header_version=%u but enable_inline_header is false; reject init",
                cfg->global_unique_name().c_str(),
                integrity.inline_header_version());
            return ER_INLINE_HEADER_INVALID;
        }
        if (integrity.enable_meta_checksum()) {
            any_meta_checksum_enabled = true;
        }
    }
    return ER_OK;
}

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
        // 任务 82620492：解析 storage_configs，拒绝 inline_header (方案 B 预留) +
        // 自动启用 hash pool (方案 A 开关 enable_meta_checksum)。
        bool any_meta_checksum_enabled = false;
        ec = ValidateStorageConfigsForIntegrity(init_params_.storage_configs, any_meta_checksum_enabled);
        if (ec != ER_OK) {
            client_config_.reset();
            sdk_wrapper_.reset();
            return ec;
        }
#if defined(USING_CUDA) || defined(USING_MUSA)
        is_check_buffer_ = EnvUtil::GetEnv("KVCM_SDK_CHECK", false);
        meta_checksum_enabled_ = any_meta_checksum_enabled;
        if (is_check_buffer_ || meta_checksum_enabled_) {
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
#else
        if (any_meta_checksum_enabled) {
            KVCM_LOG_WARN("storage spec enables meta_checksum but build is not CUDA/MUSA; "
                          "Save/Load hash compute will be skipped, verification will degrade to no-op");
        }
#endif
        KVCM_LOG_INFO("transfer client init success");
        return ER_OK;
    }
}

void TransferClientImpl::PrintBlockHashAndUri(const std::string &prefix,
                                              const UriStrVec &uri_str_vec,
                                              const std::vector<int64_t> &block_hashs,
                                              const std::shared_ptr<TransferTraceInfo> &trace_info) const {
    std::stringstream ss;
    ss << prefix << "; self_location_spec_name : " << init_params_.self_location_spec_name
       << "; uri size : " << uri_str_vec.size() << "; real size : " << block_hashs.size();
    ss << "{";
    bool invalid = (trace_info != nullptr) && (trace_info->block_ids.size() >= block_hashs.size());
    for (size_t i = 0; i < block_hashs.size(); ++i) {
        ss << "\"" << prefix;
        if (invalid) {
            ss << "_" << trace_info->block_ids[i] << "_";
        }
        ss << uri_str_vec[i] << "\":" << block_hashs[i];
        if (i != (block_hashs.size() - 1)) {
            ss << ',';
        }
    }
    ss << "}";
    KVCM_LOG_INFO("%s", ss.str().c_str());
}

ClientErrorCode TransferClientImpl::LoadKvCaches(const UriStrVec &uri_str_vec,
                                                 const BlockBuffers &block_buffers,
                                                 std::shared_ptr<TransferTraceInfo> trace_info,
                                                 const std::vector<int64_t> *expected_hashes) {
    KVCM_LOG_DEBUG("load kv caches with uri_str_vec %s, block_buffers %s",
                   DebugStringUtil::ToString(uri_str_vec).c_str(),
                   DebugStringUtil::ToString(block_buffers).c_str());
    CHECK_SDK();
    auto remote_uris = ParseLocations(uri_str_vec);
    auto ec = sdk_wrapper_->Get(remote_uris, block_buffers);
    if (ec != ER_OK) {
        return ec;
    }
#if defined(USING_CUDA) || defined(USING_MUSA)
    if (is_check_buffer_) {
        bool need_print = (trace_info == nullptr) ? true : trace_info->need_print;
        std::vector<int64_t> block_hashs;
        if (need_print) {
            auto handle = sdk_buffer_check_pool_->GetCell();
            block_hashs = SdkBufferCheckUtil::GetBlocksHash(
                block_buffers, handle->d_iovs, handle->d_crcs, handle->h_iovs, max_check_iov_num_, handle->gpu_stream);
        }
        PrintBlockHashAndUri("get_", uri_str_vec, block_hashs, trace_info);
    }
    // 任务 82620492：读端校验路径。expected_hashes 非空 + 至少一项 != 0 时算 actual。
    if (expected_hashes != nullptr && !expected_hashes->empty()) {
        if (expected_hashes->size() != block_buffers.size()) {
            KVCM_LOG_ERROR("expected_hashes size [%zu] mismatches block_buffers size [%zu]",
                           expected_hashes->size(),
                           block_buffers.size());
            return ER_CHECKSUM_MISMATCH;
        }
        if (!sdk_buffer_check_pool_) {
            // 未启用 pool (spec 没开 enable_meta_checksum，但 caller 又传了 expected)。
            // 这是配置不一致；按 fail-open 处理 + 警告，避免推理路径强阻塞。
            KVCM_LOG_WARN("expected_hashes given but sdk_buffer_check_pool is not enabled; skip verification");
        } else {
            auto handle = sdk_buffer_check_pool_->GetCell();
            auto actual = SdkBufferCheckUtil::GetBlocksHash(
                block_buffers, handle->d_iovs, handle->d_crcs, handle->h_iovs, max_check_iov_num_, handle->gpu_stream);
            if (actual.size() != expected_hashes->size()) {
                KVCM_LOG_ERROR("actual hashes size [%zu] != expected [%zu]", actual.size(), expected_hashes->size());
                return ER_CHECKSUM_MISMATCH;
            }
            for (size_t i = 0; i < actual.size(); ++i) {
                const int64_t expected = (*expected_hashes)[i];
                if (expected == 0) {
                    continue; // sentinel: 老数据 / 老 client 未上报 hash
                }
                if (expected != actual[i]) {
                    KVCM_LOG_ERROR("checksum mismatch at block %zu: expected=0x%lx, actual=0x%lx, uri=%s",
                                   i,
                                   static_cast<unsigned long>(expected),
                                   static_cast<unsigned long>(actual[i]),
                                   i < uri_str_vec.size() ? uri_str_vec[i].c_str() : "<oob>");
                    return ER_CHECKSUM_MISMATCH;
                }
            }
        }
    }
#else
    if (expected_hashes != nullptr && !expected_hashes->empty()) {
        KVCM_LOG_WARN("expected_hashes given but build is not CUDA/MUSA; verification skipped (no-op)");
    }
#endif
    return ec;
}

std::pair<ClientErrorCode, UriStrVec> TransferClientImpl::SaveKvCaches(const UriStrVec &uri_str_vec,
                                                                       const BlockBuffers &block_buffers,
                                                                       std::shared_ptr<TransferTraceInfo> trace_info,
                                                                       std::vector<int64_t> *out_block_hashes) {
    KVCM_LOG_DEBUG("save kv caches with uri_str_vec %s, block_buffers %s",
                   DebugStringUtil::ToString(uri_str_vec).c_str(),
                   DebugStringUtil::ToString(block_buffers).c_str());
    CHECK_SDK_WITH_TYPE();
#if defined(USING_CUDA) || defined(USING_MUSA)
    // 任务 82620492：写端算 hash 路径。out_block_hashes 非空时算并写入。
    // 复用 is_check_buffer_ 已有的 print-only 路径输出，不重复算 hash。
    std::vector<int64_t> block_hashs;
    bool block_hashs_computed = false;
    if (out_block_hashes != nullptr || is_check_buffer_) {
        if (!sdk_buffer_check_pool_) {
            if (out_block_hashes != nullptr) {
                KVCM_LOG_WARN("out_block_hashes requested but sdk_buffer_check_pool is not enabled; "
                              "return empty vector (caller should treat as 'hash not available')");
                out_block_hashes->clear();
            }
        } else {
            bool need_print = (trace_info == nullptr) ? true : trace_info->need_print;
            if (out_block_hashes != nullptr || need_print) {
                auto handle = sdk_buffer_check_pool_->GetCell();
                block_hashs = SdkBufferCheckUtil::GetBlocksHash(block_buffers,
                                                                handle->d_iovs,
                                                                handle->d_crcs,
                                                                handle->h_iovs,
                                                                max_check_iov_num_,
                                                                handle->gpu_stream);
                block_hashs_computed = true;
            }
        }
        if (is_check_buffer_) {
            PrintBlockHashAndUri("put_", uri_str_vec, block_hashs, trace_info);
        }
        if (out_block_hashes != nullptr && block_hashs_computed) {
            *out_block_hashes = block_hashs;
        }
    }
#else
    if (out_block_hashes != nullptr) {
        KVCM_LOG_WARN("out_block_hashes requested but build is not CUDA/MUSA; return empty vector");
        out_block_hashes->clear();
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