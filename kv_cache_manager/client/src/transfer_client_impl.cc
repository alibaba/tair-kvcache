#include "kv_cache_manager/client/src/transfer_client_impl.h"

#include <sstream>
#if defined(USING_CUDA) || defined(USING_MUSA)
#include <algorithm>

#include "kv_cache_manager/client/src/internal/sdk/sdk_buffer_check_util.h"
#include "kv_cache_manager/common/env_util.h"
#endif
#include "kv_cache_manager/client/src/internal/config/client_config.h"
#include "kv_cache_manager/client/src/internal/sdk/sdk_wrapper.h"
#include "kv_cache_manager/client/src/internal/util/checksum_verify_util.h"
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
        // Stay silent and let sdk_wrapper report ER_INVALID_STORAGE_CONFIG downstream.
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
        // Reject any other integrity validation failure (e.g. enable_meta_checksum=true
        // with algo=CA_UNSPECIFIED). Without this the client silently falls back to
        // CRC32 even though server-side StorageConfig::ValidateRequiredFields would
        // reject the same config -- inconsistency between server and client init.
        std::string integrity_invalid_fields;
        if (!integrity.ValidateRequiredFields(integrity_invalid_fields)) {
            KVCM_LOG_ERROR("storage config [%s] has invalid integrity config: %s; reject init",
                           cfg->global_unique_name().c_str(),
                           integrity_invalid_fields.c_str());
            return ER_INVALID_STORAGE_CONFIG;
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
        // Parse storage_configs to reject Scheme B (inline_header) early and to
        // auto-initialize the checksum pool when any spec opts into meta_checksum.
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
                          "Save/Load checksum compute will be skipped, verification will degrade to no-op");
        }
#endif
        KVCM_LOG_INFO("transfer client init success");
        return ER_OK;
    }
}

void TransferClientImpl::PrintBlockChecksumAndUri(const std::string &prefix,
                                                  const UriStrVec &uri_str_vec,
                                                  const std::vector<int64_t> &block_checksums,
                                                  const std::shared_ptr<TransferTraceInfo> &trace_info) const {
    std::stringstream ss;
    ss << prefix << "; self_location_spec_name : " << init_params_.self_location_spec_name
       << "; uri size : " << uri_str_vec.size() << "; real size : " << block_checksums.size();
    ss << "{";
    bool invalid = (trace_info != nullptr) && (trace_info->block_ids.size() >= block_checksums.size());
    for (size_t i = 0; i < block_checksums.size(); ++i) {
        ss << "\"" << prefix;
        if (invalid) {
            ss << "_" << trace_info->block_ids[i] << "_";
        }
        ss << uri_str_vec[i] << "\":" << block_checksums[i];
        if (i != (block_checksums.size() - 1)) {
            ss << ',';
        }
    }
    ss << "}";
    KVCM_LOG_INFO("%s", ss.str().c_str());
}

ClientErrorCode TransferClientImpl::LoadKvCaches(const UriStrVec &uri_str_vec,
                                                 const BlockBuffers &block_buffers,
                                                 std::shared_ptr<TransferTraceInfo> trace_info,
                                                 const std::vector<int64_t> *expected_checksums) {
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
        std::vector<int64_t> block_checksums;
        if (need_print) {
            auto handle = sdk_buffer_check_pool_->GetCell();
            block_checksums = SdkBufferCheckUtil::GetBlocksHash(
                block_buffers, handle->d_iovs, handle->d_crcs, handle->h_iovs, max_check_iov_num_, handle->gpu_stream);
        }
        PrintBlockChecksumAndUri("get_", uri_str_vec, block_checksums, trace_info);
    }
    // Read-side verification path: only kicks in when caller supplies expected checksums
    // (typically forwarded from CacheLocation.checksum returned by meta service).
    //
    // Verification is two-stage by default: a fast XOR aggregate over the whole batch
    // catches the common case in O(1) comparisons. On mismatch the slow per-block walk
    // pinpoints the offending blocks for diagnostics. Set KVCM_CHECKSUM_STRICT_MODE=1
    // to skip the fast aggregate entirely and force per-block comparison (useful for
    // triaging cases the XOR could in theory hide via paired cancellation, p ~2^-64).
    //
    // Blocks whose buffer contains any iov with ignore=true are partial reads: the
    // underlying SDK leaves the ignored ranges untouched, so hashing them would
    // include stale memory and always disagree with the stored checksum. We force the
    // expected slot to the 0 sentinel for those blocks so the verifier skips them; an
    // empty / no-iov block is equally unverifiable and gets the same treatment.
    if (expected_checksums != nullptr && !expected_checksums->empty()) {
        if (expected_checksums->size() != block_buffers.size()) {
            KVCM_LOG_ERROR("expected_checksums size [%zu] mismatches block_buffers size [%zu]",
                           expected_checksums->size(),
                           block_buffers.size());
            return ER_CHECKSUM_MISMATCH;
        }
        if (!sdk_buffer_check_pool_) {
            // Caller asked for verification but the pool was not initialized
            // (spec did not enable meta_checksum). Fail open with a warning to avoid
            // blocking the inference path on a config mismatch.
            KVCM_LOG_WARN("expected_checksums given but sdk_buffer_check_pool is not enabled; "
                          "skip verification");
        } else {
            std::vector<int64_t> effective_expected = *expected_checksums;
            size_t skipped_blocks = 0;
            for (size_t i = 0; i < block_buffers.size(); ++i) {
                if (effective_expected[i] == 0) {
                    continue; // already sentinel
                }
                const auto &iovs = block_buffers[i].iovs;
                bool unverifiable = iovs.empty();
                if (!unverifiable) {
                    for (const auto &iov : iovs) {
                        if (iov.ignore) {
                            unverifiable = true;
                            break;
                        }
                    }
                }
                if (unverifiable) {
                    effective_expected[i] = 0;
                    ++skipped_blocks;
                }
            }
            if (skipped_blocks > 0) {
                KVCM_LOG_DEBUG("checksum verification skipped %zu block(s) due to ignored or empty iovs",
                               skipped_blocks);
            }
            // Short-circuit when nothing is left to verify (all-legacy batch or every
            // block downgraded to sentinel by the partial-read / empty-iov handling
            // above). Avoids one wasted GPU hash compute and — more importantly —
            // avoids GetBlocksHash dereferencing block_buffers on shapes it cannot
            // hash (empty iovs, etc.) after we already declared them unverifiable.
            const bool has_any_expected =
                std::any_of(effective_expected.begin(), effective_expected.end(), [](int64_t v) { return v != 0; });
            if (!has_any_expected) {
                KVCM_LOG_DEBUG("checksum verification: no non-sentinel expected entries; skip");
                return ec;
            }
            auto handle = sdk_buffer_check_pool_->GetCell();
            auto actual = SdkBufferCheckUtil::GetBlocksHash(
                block_buffers, handle->d_iovs, handle->d_crcs, handle->h_iovs, max_check_iov_num_, handle->gpu_stream);
            const bool strict_mode = EnvUtil::GetEnv("KVCM_CHECKSUM_STRICT_MODE", false);
            const auto verify_result = VerifyBatchChecksums(effective_expected, actual, strict_mode);
            if (verify_result.mismatch) {
                if (verify_result.faulty_indices.empty()) {
                    // Size mismatch surfaced from the helper (we already pre-checked
                    // expected vs block_buffers above, so this is paranoid coverage).
                    KVCM_LOG_ERROR(
                        "actual checksums size [%zu] != expected [%zu]", actual.size(), effective_expected.size());
                } else {
                    // Structured per-block log; field set mirrors ChecksumMismatchEvent
                    // so a log scraper can build the same observability surface until
                    // the client SDK grows a real EventManager hook (follow-up).
                    for (auto idx : verify_result.faulty_indices) {
                        const char *block_id_str = (trace_info != nullptr && idx < trace_info->block_ids.size())
                                                       ? trace_info->block_ids[idx].c_str()
                                                       : "<unknown>";
                        KVCM_LOG_ERROR("ChecksumMismatchEvent {block_index=%zu, expected_checksum=0x%lx, "
                                       "actual_checksum=0x%lx, storage_uri=\"%s\", block_id=\"%s\"}",
                                       idx,
                                       static_cast<unsigned long>(effective_expected[idx]),
                                       static_cast<unsigned long>(actual[idx]),
                                       idx < uri_str_vec.size() ? uri_str_vec[idx].c_str() : "<oob>",
                                       block_id_str);
                    }
                }
                return ER_CHECKSUM_MISMATCH;
            }
        }
    }
#else
    if (expected_checksums != nullptr && !expected_checksums->empty()) {
        KVCM_LOG_WARN("expected_checksums given but build is not CUDA/MUSA; verification skipped (no-op)");
    }
#endif
    return ec;
}

std::pair<ClientErrorCode, UriStrVec> TransferClientImpl::SaveKvCaches(const UriStrVec &uri_str_vec,
                                                                       const BlockBuffers &block_buffers,
                                                                       std::shared_ptr<TransferTraceInfo> trace_info,
                                                                       std::vector<int64_t> *out_checksums) {
    KVCM_LOG_DEBUG("save kv caches with uri_str_vec %s, block_buffers %s",
                   DebugStringUtil::ToString(uri_str_vec).c_str(),
                   DebugStringUtil::ToString(block_buffers).c_str());
    CHECK_SDK_WITH_TYPE();
#if defined(USING_CUDA) || defined(USING_MUSA)
    // Write-side checksum compute: triggered by either an explicit out_checksums
    // request or the legacy KVCM_SDK_CHECK print-only fallback. The two paths share
    // the same computation so the checksum is computed at most once per call.
    //
    // SdkBufferCheckUtil::GetBlocksHash dereferences block_buffers.front(), so we
    // must guard against empty / malformed input here. The full URI / Iov validity
    // check still happens inside sdk_wrapper_->Put() below; we only short-circuit
    // the obviously invalid shapes that would crash the hashing kernel.
    std::vector<int64_t> block_checksums;
    bool block_checksums_computed = false;
    const bool checksum_input_usable = !block_buffers.empty() && !block_buffers.front().iovs.empty();
    if (out_checksums != nullptr || is_check_buffer_) {
        if (!checksum_input_usable) {
            if (out_checksums != nullptr) {
                KVCM_LOG_WARN("block_buffers empty or first block has no iovs; "
                              "skip checksum compute and return empty out_checksums");
                out_checksums->clear();
            }
        } else if (!sdk_buffer_check_pool_) {
            if (out_checksums != nullptr) {
                KVCM_LOG_WARN("out_checksums requested but sdk_buffer_check_pool is not enabled; "
                              "return empty vector (caller should treat as 'checksum not available')");
                out_checksums->clear();
            }
        } else {
            bool need_print = (trace_info == nullptr) ? true : trace_info->need_print;
            if (out_checksums != nullptr || need_print) {
                auto handle = sdk_buffer_check_pool_->GetCell();
                block_checksums = SdkBufferCheckUtil::GetBlocksHash(block_buffers,
                                                                    handle->d_iovs,
                                                                    handle->d_crcs,
                                                                    handle->h_iovs,
                                                                    max_check_iov_num_,
                                                                    handle->gpu_stream);
                block_checksums_computed = true;
                // GetBlocksHash caps at max_check_iov_num_ and silently returns a
                // shorter vector; if the caller wants checksums we cannot let the
                // write proceed with a length that will fail FinishWrite's per-
                // location check after the data has already been persisted.
                if (out_checksums != nullptr && block_checksums.size() != block_buffers.size()) {
                    KVCM_LOG_ERROR("block_checksums size [%zu] != block_buffers size [%zu]; iov count likely "
                                   "exceeded max_check_iov_num_ [%d]; reject write before it commits",
                                   block_checksums.size(),
                                   block_buffers.size(),
                                   max_check_iov_num_);
                    out_checksums->clear();
                    return {ER_INVALID_PARAMS, {}};
                }
            }
        }
        if (is_check_buffer_ && checksum_input_usable) {
            PrintBlockChecksumAndUri("put_", uri_str_vec, block_checksums, trace_info);
        }
        // Deliberately DO NOT assign to *out_checksums here — see below. A prior
        // version filled the vector before Put, which meant a failed Put returned
        // checksums for data that never landed on disk; a caller then persisting
        // those via FinishWrite would associate a checksum with a nonexistent block.
    }
#else
    if (out_checksums != nullptr) {
        KVCM_LOG_WARN("out_checksums requested but build is not CUDA/MUSA; return empty vector");
        out_checksums->clear();
    }
#endif
    auto remote_uris = ParseLocations(uri_str_vec);
    auto actual_remote_uris = std::make_shared<std::vector<DataStorageUri>>();
    auto ec = sdk_wrapper_->Put(remote_uris, block_buffers, actual_remote_uris);
    if (ec != ER_OK) {
        KVCM_LOG_ERROR("save kv cache failed");
        return {ec, {}};
    }
#if defined(USING_CUDA) || defined(USING_MUSA)
    // Put succeeded; only now hand computed checksums back to the caller.
    if (out_checksums != nullptr && block_checksums_computed) {
        *out_checksums = std::move(block_checksums);
    }
#endif
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