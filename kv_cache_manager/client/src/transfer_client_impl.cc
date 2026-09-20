#include "kv_cache_manager/client/src/transfer_client_impl.h"

#include <algorithm>
#include <sstream>
#if defined(USING_CUDA) || defined(USING_MUSA)
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

extern "C" uint32_t KVCMStagedChecksumRuntimeApiVersion() noexcept { return KVCM_STAGED_CHECKSUM_API_VERSION; }

namespace kv_cache_manager {

bool TransferClientImpl::IsChecksumWorkloadWithinLimit(const BlockBuffers &block_buffers,
                                                       size_t sample_bytes,
                                                       size_t max_sampled_bytes,
                                                       const std::vector<bool> &included_blocks) {
    if (sample_bytes == 0 || max_sampled_bytes == 0 ||
        (!included_blocks.empty() && included_blocks.size() != block_buffers.size())) {
        return false;
    }
    size_t sampled_bytes = 0;
    for (size_t block_index = 0; block_index < block_buffers.size(); ++block_index) {
        if (!included_blocks.empty() && !included_blocks[block_index]) {
            continue;
        }
        for (const auto &iov : block_buffers[block_index].iovs) {
            // The persisted legacy-v0 algorithm reads equal head/tail spans.
            // min(sample, size/2) makes the multiplication overflow-safe and
            // exactly mirrors the CUDA/MUSA kernels, including odd sizes.
            const size_t bytes_for_iov = 2 * std::min(sample_bytes, iov.size / 2);
            if (bytes_for_iov > max_sampled_bytes - sampled_bytes) {
                return false;
            }
            sampled_bytes += bytes_for_iov;
        }
    }
    return true;
}

#if defined(USING_CUDA) || defined(USING_MUSA)
namespace {

constexpr size_t kMaxChecksumMismatchLogs = 16;

std::string SafeStorageEndpointForLog(const std::string &uri_text) {
    const DataStorageUri uri(uri_text);
    if (!uri.Valid()) {
        return "<invalid>";
    }
    // URI userinfo, path and query may contain credentials, allocation
    // capabilities, or tenant object names. Stage/block identity is enough to
    // locate an integrity failure; log only the non-secret routing endpoint.
    return uri.GetProtocol() + "://" + uri.GetHostPort();
}

bool IsChecksumHashableBlock(const BlockBuffer &block_buffer) {
    if (block_buffer.iovs.empty()) {
        return false;
    }
    return std::all_of(block_buffer.iovs.begin(), block_buffer.iovs.end(), [](const Iov &iov) {
        // The current CRC kernel dereferences device addresses. CPU buffers,
        // ignored slices, and incomplete iovs cannot represent every expected
        // component of the block's checksum input.
        return iov.type == MemoryType::GPU && !iov.ignore && iov.base != nullptr && iov.size >= 2;
    });
}

bool HashBlocksInChunks(const BlockBuffers &block_buffers,
                        SdkBufferCheckPool::CellHandle &handle,
                        size_t max_check_iov_num,
                        std::vector<int64_t> &block_checksums) {
    block_checksums.clear();
    block_checksums.resize(block_buffers.size());
    if (block_buffers.empty()) {
        return true;
    }

    BlockBuffers chunk;
    std::vector<size_t> chunk_indices;
    size_t chunk_iov_num = 0;
    size_t chunk_total_iovs = 0;

    auto flush_chunk = [&]() -> bool {
        if (chunk.empty()) {
            return true;
        }
        auto checksums = SdkBufferCheckUtil::GetBlocksHash(
            chunk, handle->d_iovs, handle->d_crcs, handle->h_iovs, max_check_iov_num, handle->gpu_stream);
        if (checksums.size() != chunk.size()) {
            KVCM_LOG_ERROR("checksum hash returned [%zu] entries for [%zu] blocks", checksums.size(), chunk.size());
            return false;
        }
        for (size_t i = 0; i < checksums.size(); ++i) {
            block_checksums[chunk_indices[i]] = checksums[i];
        }
        chunk.clear();
        chunk_indices.clear();
        chunk_iov_num = 0;
        chunk_total_iovs = 0;
        return true;
    };

    for (size_t i = 0; i < block_buffers.size(); ++i) {
        const size_t iov_num = block_buffers[i].iovs.size();
        if (iov_num == 0 || iov_num > max_check_iov_num) {
            KVCM_LOG_ERROR("block [%zu] has invalid iov_num [%zu] for checksum hash", i, iov_num);
            return false;
        }
        // Hash aggregation only requires a uniform iov count. Each iov's CRC
        // kernel derives its sample size independently, so differing byte sizes
        // can share a launch and should not fragment the batch.
        if (!chunk.empty() && (iov_num != chunk_iov_num || chunk_total_iovs + iov_num > max_check_iov_num)) {
            if (!flush_chunk()) {
                return false;
            }
        }
        if (chunk.empty()) {
            chunk_iov_num = iov_num;
        }
        chunk.push_back(block_buffers[i]);
        chunk_indices.push_back(i);
        chunk_total_iovs += iov_num;
    }
    return flush_chunk();
}

void LogChecksumMismatches(ChecksumValidationStage stage,
                           const ChecksumVerifyResult &verify_result,
                           const std::vector<int64_t> &expected,
                           const std::vector<int64_t> &actual,
                           const UriStrVec &uri_str_vec,
                           const std::vector<size_t> &original_indices,
                           const std::string &request_trace_id,
                           const std::shared_ptr<TransferTraceInfo> &trace_info) {
    const char *trace_id = request_trace_id.empty() ? "<unknown>" : request_trace_id.c_str();
    if (verify_result.faulty_indices.empty()) {
        KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, trace_id=\"%s\", error=\"invalid batch shape\", "
                       "expected_count=%zu, actual_count=%zu}",
                       ChecksumValidationStageToString(stage),
                       trace_id,
                       expected.size(),
                       actual.size());
        return;
    }
    const size_t logged = std::min(verify_result.faulty_indices.size(), kMaxChecksumMismatchLogs);
    for (size_t faulty_index = 0; faulty_index < logged; ++faulty_index) {
        const size_t compact_idx = verify_result.faulty_indices[faulty_index];
        const size_t idx = original_indices.empty() ? compact_idx : original_indices[compact_idx];
        const char *block_id = (trace_info != nullptr && idx < trace_info->block_ids.size())
                                   ? trace_info->block_ids[idx].c_str()
                                   : "<unknown>";
        const std::string storage_endpoint =
            idx < uri_str_vec.size() ? SafeStorageEndpointForLog(uri_str_vec[idx]) : "<oob>";
        KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, trace_id=\"%s\", block_index=%zu, "
                       "expected_checksum=0x%lx, actual_checksum=0x%lx, storage_endpoint=\"%s\", "
                       "block_id=\"%s\"}",
                       ChecksumValidationStageToString(stage),
                       trace_id,
                       idx,
                       static_cast<unsigned long>(expected[compact_idx]),
                       static_cast<unsigned long>(actual[compact_idx]),
                       storage_endpoint.c_str(),
                       block_id);
    }
    if (verify_result.faulty_indices.size() > logged) {
        KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, trace_id=\"%s\", error=\"mismatch log truncated\", "
                       "mismatch_count=%zu, logged_count=%zu}",
                       ChecksumValidationStageToString(stage),
                       trace_id,
                       verify_result.faulty_indices.size(),
                       logged);
    }
}

} // namespace
#endif

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
        KVCM_LOG_ERROR("storage configs cannot be parsed while validating data integrity");
        return ER_INVALID_STORAGE_CONFIG;
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
    return InitInternal(client_config, init_params, nullptr);
}

ClientErrorCode TransferClientImpl::InitWithSharedMemory(const std::string &client_config,
                                                         const InitParams &init_params,
                                                         const SharedMemoryRegistration &shared_memory_registration) {
    return InitInternal(client_config, init_params, &shared_memory_registration);
}

ClientErrorCode TransferClientImpl::InitInternal(const std::string &client_config,
                                                 const InitParams &init_params,
                                                 const SharedMemoryRegistration *shared_memory_registration) {
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
        KVCM_LOG_INFO("transfer client init params: role_type[%d], regist_span[%p], self_location_spec_name[%s], "
                      "storage_configs[%s]",
                      static_cast<int>(init_params_.role_type),
                      init_params_.regist_span,
                      init_params_.self_location_spec_name.c_str(),
                      init_params_.storage_configs.c_str());
        // Validate integrity before initializing storage SDKs so an invalid
        // config cannot produce backend side effects and then fail afterward.
        bool any_meta_checksum_enabled = false;
        ec = ValidateStorageConfigsForIntegrity(init_params_.storage_configs, any_meta_checksum_enabled);
        if (ec != ER_OK) {
            client_config_.reset();
            return ec;
        }
        sdk_wrapper_ = std::make_unique<SdkWrapper>();
        ec = sdk_wrapper_->Init(client_config_, init_params_, shared_memory_registration);
        if (ec != ER_OK) {
            KVCM_LOG_ERROR("init sdk wrapper failed");
            client_config_.reset();
            sdk_wrapper_.reset();
            return ec;
        }
#if defined(USING_CUDA) || defined(USING_MUSA)
        is_check_buffer_ = EnvUtil::GetEnv("KVCM_SDK_CHECK", false);
        meta_checksum_enabled_ = any_meta_checksum_enabled;
        if (is_check_buffer_ || meta_checksum_enabled_) {
            const int64_t sdk_check_cell_num_config = EnvUtil::GetEnv<int64_t>("KVCM_SDK_CHECK_CELL_NUM", 4);
            const int64_t max_check_iov_num_config = EnvUtil::GetEnv<int64_t>("KVCM_SDK_MAX_CHECK_IOV_NUM", 500 * 1000);
            const int64_t check_iov_byte_size = EnvUtil::GetEnv<int64_t>("KVCM_CHECK_IOV_BYTE_SIZE", 4);
            const int64_t max_sampled_bytes_config =
                EnvUtil::GetEnv<int64_t>("KVCM_CHECKSUM_MAX_SAMPLED_BYTES_PER_REQUEST",
                                         static_cast<int64_t>(kDefaultMaxChecksumSampledBytesPerRequest));
            // A zero-cell pool reports successful initialization but GetCell()
            // can never make progress. A zero iov limit is likewise unusable for
            // every non-empty block. Reject both configurations up front.
            if (sdk_check_cell_num_config <= 0 || max_check_iov_num_config <= 0 || check_iov_byte_size <= 0 ||
                check_iov_byte_size > static_cast<int64_t>(KVCM_CHECKSUM_MAX_SAMPLE_BYTES) ||
                max_sampled_bytes_config <= 0) {
                KVCM_LOG_ERROR("invalid checksum pool config, sdk_check_cell_num[%ld], max_check_iov_num[%ld], "
                               "check_iov_byte_size[%ld], max_sampled_bytes_per_request[%ld]",
                               sdk_check_cell_num_config,
                               max_check_iov_num_config,
                               check_iov_byte_size,
                               max_sampled_bytes_config);
                client_config_.reset();
                sdk_wrapper_.reset();
                meta_checksum_enabled_ = false;
                is_check_buffer_ = false;
                return ER_INIT_CHECK_BUFFER_ERROR;
            }
            checksum_sample_bytes_ = SdkBufferCheckUtil::ChecksumSampleBytes();
            if (checksum_sample_bytes_ == 0 || checksum_sample_bytes_ > KVCM_CHECKSUM_MAX_SAMPLE_BYTES) {
                KVCM_LOG_ERROR("checksum kernel resolved invalid sample size [%zu], supported range is [1, %zu]",
                               checksum_sample_bytes_,
                               KVCM_CHECKSUM_MAX_SAMPLE_BYTES);
                client_config_.reset();
                sdk_wrapper_.reset();
                meta_checksum_enabled_ = false;
                is_check_buffer_ = false;
                return ER_INIT_CHECK_BUFFER_ERROR;
            }
            if (static_cast<size_t>(check_iov_byte_size) != checksum_sample_bytes_) {
                KVCM_LOG_ERROR("checksum sample size changed after the GPU implementation was loaded: "
                               "configured_now[%ld], kernel_resolved[%zu]; reject init to avoid advertising a "
                               "checksum contract that differs from the executed kernel",
                               check_iov_byte_size,
                               checksum_sample_bytes_);
                client_config_.reset();
                sdk_wrapper_.reset();
                meta_checksum_enabled_ = false;
                is_check_buffer_ = false;
                return ER_INIT_CHECK_BUFFER_ERROR;
            }
            const size_t sdk_check_cell_num = static_cast<size_t>(sdk_check_cell_num_config);
            max_check_iov_num_ = static_cast<size_t>(max_check_iov_num_config);
            max_checksum_sampled_bytes_per_request_ = static_cast<size_t>(max_sampled_bytes_config);
            sdk_buffer_check_pool_ = std::make_shared<SdkBufferCheckPool>(sdk_check_cell_num);
            if (!sdk_buffer_check_pool_->Init(max_check_iov_num_)) {
                KVCM_LOG_ERROR("sdk_buffer_check_pool init faild, sdk_check_cell_num[%lu], max_check_iov_num[%lu]",
                               sdk_check_cell_num,
                               max_check_iov_num_);
                client_config_.reset();
                sdk_wrapper_.reset();
                sdk_buffer_check_pool_.reset();
                meta_checksum_enabled_ = false;
                is_check_buffer_ = false;
                return ER_INIT_CHECK_BUFFER_ERROR;
            }
        }
#else
        if (any_meta_checksum_enabled) {
            KVCM_LOG_WARN("storage spec enables meta_checksum but build is not CUDA/MUSA; "
                          "explicit Save/Load checksum requests will return ER_CHECKSUM_UNAVAILABLE");
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
    const bool has_block_ids = (trace_info != nullptr) && (trace_info->block_ids.size() >= block_checksums.size());
    for (size_t i = 0; i < block_checksums.size(); ++i) {
        ss << "\"" << prefix;
        if (has_block_ids) {
            ss << "_" << trace_info->block_ids[i] << "_";
        }
        ss << (i < uri_str_vec.size() ? uri_str_vec[i] : "<oob>") << "\":" << block_checksums[i];
        if (i != (block_checksums.size() - 1)) {
            ss << ',';
        }
    }
    ss << "}";
    KVCM_LOG_INFO("%s", ss.str().c_str());
}

ClientErrorCode TransferClientImpl::LoadKvCaches(const UriStrVec &uri_str_vec,
                                                 const BlockBuffers &block_buffers,
                                                 std::shared_ptr<TransferTraceInfo> trace_info) {
    return LoadKvCaches(uri_str_vec, block_buffers, LoadKvCachesOptions::WithTraceInfo(std::move(trace_info)));
}

ClientErrorCode TransferClientImpl::LoadKvCaches(const UriStrVec &uri_str_vec,
                                                 const BlockBuffers &block_buffers,
                                                 const LoadKvCachesOptions &options) {
    const auto &expected_checksums = options.expected_checksums;
    const auto &expected_present = options.expected_checksum_present;
    const bool verify_checksums = options.verify_checksums || !expected_checksums.empty() || !expected_present.empty();
    KVCM_LOG_DEBUG("load kv caches with uri_str_vec %s, block_buffers %s",
                   DebugStringUtil::ToString(uri_str_vec).c_str(),
                   DebugStringUtil::ToString(block_buffers).c_str());
    CHECK_SDK();
    if ((!expected_present.empty() && expected_present.size() != expected_checksums.size()) ||
        (verify_checksums && expected_checksums.size() != block_buffers.size())) {
        KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, error=\"invalid batch shape\", "
                       "expected_count=%zu, presence_count=%zu, buffer_count=%zu}",
                       ChecksumValidationStageToString(ChecksumValidationStage::CVS_READ_OUTPUT),
                       expected_checksums.size(),
                       expected_present.size(),
                       block_buffers.size());
        return ER_CHECKSUM_MISMATCH;
    }
    bool has_expected_checksum = false;
    for (size_t i = 0; i < expected_checksums.size(); ++i) {
        if (expected_present.empty() || expected_present[i]) {
            has_expected_checksum = true;
            break;
        }
    }
#if defined(USING_CUDA) || defined(USING_MUSA)
    if (has_expected_checksum) {
        if (!sdk_buffer_check_pool_) {
            KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, error=\"checksum pool is not enabled\"}",
                           ChecksumValidationStageToString(ChecksumValidationStage::CVS_READ_OUTPUT));
            return ER_CHECKSUM_UNAVAILABLE;
        }
        for (size_t i = 0; i < block_buffers.size(); ++i) {
            if ((!expected_present.empty() && !expected_present[i]) || IsChecksumHashableBlock(block_buffers[i])) {
                continue;
            }
            KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, block_index=%zu, "
                           "error=\"block is partial, too small, invalid, or not GPU memory\"}",
                           ChecksumValidationStageToString(ChecksumValidationStage::CVS_READ_OUTPUT),
                           i);
            return ER_CHECKSUM_UNAVAILABLE;
        }
        if (!IsChecksumWorkloadWithinLimit(
                block_buffers, checksum_sample_bytes_, max_checksum_sampled_bytes_per_request_, expected_present)) {
            KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, error=\"checksum workload exceeds request budget\", "
                           "sample_bytes=%zu, max_sampled_bytes=%zu}",
                           ChecksumValidationStageToString(ChecksumValidationStage::CVS_READ_OUTPUT),
                           checksum_sample_bytes_,
                           max_checksum_sampled_bytes_per_request_);
            return ER_CHECKSUM_UNAVAILABLE;
        }
    }
#else
    if (has_expected_checksum) {
        KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, error=\"build has no CUDA/MUSA checksum kernel\"}",
                       ChecksumValidationStageToString(ChecksumValidationStage::CVS_READ_OUTPUT));
        return ER_CHECKSUM_UNAVAILABLE;
    }
#endif
    auto remote_uris = ParseLocations(uri_str_vec);
    auto ec = sdk_wrapper_->Get(remote_uris, block_buffers);
    if (ec != ER_OK) {
        return ec;
    }
#if defined(USING_CUDA) || defined(USING_MUSA)
    const auto &trace_info = options.trace_info;
    if (is_check_buffer_) {
        const bool need_print = (trace_info == nullptr) ? true : trace_info->need_print;
        if (need_print) {
            auto invalid_it = std::find_if(block_buffers.begin(), block_buffers.end(), [](const BlockBuffer &block) {
                return !IsChecksumHashableBlock(block);
            });
            if (invalid_it != block_buffers.end()) {
                const size_t idx = std::distance(block_buffers.begin(), invalid_it);
                KVCM_LOG_WARN("block [%zu] has ignored, empty, null, or zero-size iovs; skip checksum print", idx);
            } else if (!IsChecksumWorkloadWithinLimit(
                           block_buffers, checksum_sample_bytes_, max_checksum_sampled_bytes_per_request_)) {
                KVCM_LOG_WARN("checksum print workload exceeds request budget [%zu]; skip checksum print",
                              max_checksum_sampled_bytes_per_request_);
            } else if (sdk_buffer_check_pool_) {
                auto handle = sdk_buffer_check_pool_->GetCell();
                std::vector<int64_t> block_checksums;
                if (HashBlocksInChunks(block_buffers, handle, max_check_iov_num_, block_checksums)) {
                    PrintBlockChecksumAndUri("get_", uri_str_vec, block_checksums, trace_info);
                } else {
                    KVCM_LOG_WARN("checksum print failed to hash blocks safely; skip checksum print");
                }
            } else {
                KVCM_LOG_WARN("KVCM_SDK_CHECK is enabled but sdk_buffer_check_pool is not initialized; "
                              "skip checksum print");
            }
        }
    }
    // Verify only positions whose explicit presence bit is true. Calling
    // VerifyWith is strict: once at least one value is present, lack of compute
    // capability or an incomplete buffer is an error rather than a silent no-op.
    if (has_expected_checksum) {
        BlockBuffers verifiable_block_buffers;
        std::vector<int64_t> verifiable_expected;
        std::vector<size_t> original_indices;
        verifiable_block_buffers.reserve(block_buffers.size());
        verifiable_expected.reserve(block_buffers.size());
        original_indices.reserve(block_buffers.size());
        for (size_t i = 0; i < block_buffers.size(); ++i) {
            if (!expected_present.empty() && !expected_present[i]) {
                continue;
            }
            verifiable_block_buffers.push_back(block_buffers[i]);
            verifiable_expected.push_back(expected_checksums[i]);
            original_indices.push_back(i);
        }
        auto handle = sdk_buffer_check_pool_->GetCell();
        std::vector<int64_t> actual;
        if (!HashBlocksInChunks(verifiable_block_buffers, handle, max_check_iov_num_, actual)) {
            KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, error=\"checksum compute failed\"}",
                           ChecksumValidationStageToString(ChecksumValidationStage::CVS_READ_OUTPUT));
            return ER_CHECKSUM_UNAVAILABLE;
        }
        const auto verify_result =
            VerifyBatchChecksums(verifiable_expected, actual, ChecksumValidationStage::CVS_READ_OUTPUT);
        if (verify_result.mismatch) {
            LogChecksumMismatches(ChecksumValidationStage::CVS_READ_OUTPUT,
                                  verify_result,
                                  verifiable_expected,
                                  actual,
                                  uri_str_vec,
                                  original_indices,
                                  options.trace_id,
                                  trace_info);
            return ER_CHECKSUM_MISMATCH;
        }
    }
#endif
    return ec;
}

std::pair<ClientErrorCode, UriStrVec> TransferClientImpl::SaveKvCaches(const UriStrVec &uri_str_vec,
                                                                       const BlockBuffers &block_buffers,
                                                                       std::shared_ptr<TransferTraceInfo> trace_info) {
    auto [ec, result] =
        SaveKvCaches(uri_str_vec, block_buffers, SaveKvCachesOptions::WithTraceInfo(std::move(trace_info)));
    return {ec, std::move(result.uri_str_vec)};
}

std::pair<ClientErrorCode, SaveKvCachesResult> TransferClientImpl::SaveKvCaches(const UriStrVec &uri_str_vec,
                                                                                const BlockBuffers &block_buffers,
                                                                                const SaveKvCachesOptions &options) {
    SaveKvCachesResult result;
    result.location_spec_name = init_params_.self_location_spec_name;
    // Keep manually assembled options consistent with LoadKvCachesOptions and
    // the public base-class fallback: providing an expected batch is itself an
    // explicit request to validate it, even if the boolean helper flag was not
    // set.
    const bool verify_caller_checksums = options.verify_caller_checksums || !options.expected_checksums.empty();
    const bool explicit_checksum_request = options.include_checksums || verify_caller_checksums;
    KVCM_LOG_DEBUG("save kv caches with uri_str_vec %s, block_buffers %s",
                   DebugStringUtil::ToString(uri_str_vec).c_str(),
                   DebugStringUtil::ToString(block_buffers).c_str());
    CHECK_SDK_WITH_TYPE();
    if (uri_str_vec.empty() || block_buffers.empty() || uri_str_vec.size() != block_buffers.size()) {
        KVCM_LOG_ERROR("save checksum/input validation failed: uri count [%zu], buffer count [%zu]",
                       uri_str_vec.size(),
                       block_buffers.size());
        return {ER_INVALID_PARAMS, {}};
    }
    if (verify_caller_checksums && options.expected_checksums.size() != block_buffers.size()) {
        KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, error=\"invalid batch shape\", "
                       "expected_count=%zu, buffer_count=%zu}",
                       ChecksumValidationStageToString(ChecksumValidationStage::CVS_WRITE_INPUT),
                       options.expected_checksums.size(),
                       block_buffers.size());
        return {ER_CHECKSUM_MISMATCH, {}};
    }
#if defined(USING_CUDA) || defined(USING_MUSA)
    const auto &trace_info = options.trace_info;
    const bool should_compute_checksums = explicit_checksum_request || is_check_buffer_;
    // Write-side checksum compute is triggered by an explicit collect/verify
    // request or the legacy KVCM_SDK_CHECK print-only fallback. The computation
    // is shared and happens at most once per call.
    //
    // Validate every block first, then hash chunks with a uniform iov count.
    // Iov byte sizes may differ because sampling is calculated per iov.
    std::vector<int64_t> block_checksums;
    bool block_checksums_computed = false;
    if (should_compute_checksums) {
        if (!sdk_buffer_check_pool_) {
            if (explicit_checksum_request) {
                KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, error=\"checksum pool is not enabled\"}",
                               ChecksumValidationStageToString(ChecksumValidationStage::CVS_WRITE_INPUT));
                return {ER_CHECKSUM_UNAVAILABLE, {}};
            }
        } else {
            bool need_print = (trace_info == nullptr) ? true : trace_info->need_print;
            if (explicit_checksum_request || need_print) {
                auto invalid_it =
                    std::find_if(block_buffers.begin(), block_buffers.end(), [](const BlockBuffer &block_buffer) {
                        return !IsChecksumHashableBlock(block_buffer);
                    });
                if (invalid_it != block_buffers.end()) {
                    const size_t idx = std::distance(block_buffers.begin(), invalid_it);
                    if (explicit_checksum_request) {
                        KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, block_index=%zu, "
                                       "error=\"block is partial, too small, invalid, or not GPU memory\"}",
                                       ChecksumValidationStageToString(ChecksumValidationStage::CVS_WRITE_INPUT),
                                       idx);
                        return {ER_CHECKSUM_UNAVAILABLE, {}};
                    }
                    KVCM_LOG_WARN("block [%zu] cannot be hashed; skip checksum print", idx);
                } else if (!IsChecksumWorkloadWithinLimit(
                               block_buffers, checksum_sample_bytes_, max_checksum_sampled_bytes_per_request_)) {
                    if (explicit_checksum_request) {
                        KVCM_LOG_ERROR(
                            "ChecksumValidationLog {stage=%s, error=\"checksum workload exceeds request budget\", "
                            "sample_bytes=%zu, max_sampled_bytes=%zu}",
                            ChecksumValidationStageToString(ChecksumValidationStage::CVS_WRITE_INPUT),
                            checksum_sample_bytes_,
                            max_checksum_sampled_bytes_per_request_);
                        return {ER_CHECKSUM_UNAVAILABLE, {}};
                    }
                    KVCM_LOG_WARN("checksum print workload exceeds request budget [%zu]; skip checksum print",
                                  max_checksum_sampled_bytes_per_request_);
                } else {
                    auto handle = sdk_buffer_check_pool_->GetCell();
                    if (!HashBlocksInChunks(block_buffers, handle, max_check_iov_num_, block_checksums)) {
                        if (explicit_checksum_request) {
                            KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, error=\"checksum compute failed\"}",
                                           ChecksumValidationStageToString(ChecksumValidationStage::CVS_WRITE_INPUT));
                            return {ER_CHECKSUM_UNAVAILABLE, {}};
                        }
                    } else {
                        block_checksums_computed = true;
                    }
                }
                if (explicit_checksum_request && block_checksums_computed &&
                    block_checksums.size() != block_buffers.size()) {
                    KVCM_LOG_ERROR(
                        "block_checksums size [%zu] != block_buffers size [%zu]; reject write before it commits",
                        block_checksums.size(),
                        block_buffers.size());
                    return {ER_CHECKSUM_UNAVAILABLE, {}};
                }
            }
        }
        if (is_check_buffer_ && block_checksums_computed) {
            PrintBlockChecksumAndUri("put_", uri_str_vec, block_checksums, trace_info);
        }
        // Deliberately DO NOT assign to result.checksums here; a failed Put must
        // not return checksums for data that never landed on disk.
    }
    if (verify_caller_checksums) {
        const auto verify_result =
            VerifyBatchChecksums(options.expected_checksums, block_checksums, ChecksumValidationStage::CVS_WRITE_INPUT);
        if (verify_result.mismatch) {
            LogChecksumMismatches(ChecksumValidationStage::CVS_WRITE_INPUT,
                                  verify_result,
                                  options.expected_checksums,
                                  block_checksums,
                                  uri_str_vec,
                                  {},
                                  options.trace_id,
                                  trace_info);
            return {ER_CHECKSUM_MISMATCH, {}};
        }
    }
#else
    if (explicit_checksum_request) {
        KVCM_LOG_ERROR("ChecksumValidationLog {stage=%s, error=\"build has no CUDA/MUSA checksum kernel\"}",
                       ChecksumValidationStageToString(ChecksumValidationStage::CVS_WRITE_INPUT));
        return {ER_CHECKSUM_UNAVAILABLE, {}};
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
    if (options.include_checksums && block_checksums_computed) {
        result.checksums = std::move(block_checksums);
    }
#endif
    result.uri_str_vec = ConstructLocations(*actual_remote_uris);
    return {ER_OK, std::move(result)};
}

std::pair<ClientErrorCode, std::vector<int64_t>>
TransferClientImpl::CalculateChecksums(const BlockBuffers &block_buffers) {
    CHECK_SDK_WITH_TYPE();
    if (block_buffers.empty()) {
        KVCM_LOG_ERROR("calculate checksums requires at least one block");
        return {ER_INVALID_PARAMS, {}};
    }
#if defined(USING_CUDA) || defined(USING_MUSA)
    if (!sdk_buffer_check_pool_) {
        KVCM_LOG_ERROR("calculate checksums failed: checksum pool is not enabled");
        return {ER_CHECKSUM_UNAVAILABLE, {}};
    }
    const auto invalid_it =
        std::find_if(block_buffers.begin(), block_buffers.end(), [](const BlockBuffer &block_buffer) {
            return !IsChecksumHashableBlock(block_buffer);
        });
    if (invalid_it != block_buffers.end()) {
        KVCM_LOG_ERROR("calculate checksums failed: block [%zu] is partial, too small, invalid, or not GPU memory",
                       static_cast<size_t>(std::distance(block_buffers.begin(), invalid_it)));
        return {ER_CHECKSUM_UNAVAILABLE, {}};
    }
    if (!IsChecksumWorkloadWithinLimit(
            block_buffers, checksum_sample_bytes_, max_checksum_sampled_bytes_per_request_)) {
        KVCM_LOG_ERROR("calculate checksums failed: checksum workload exceeds request budget, "
                       "sample_bytes [%zu], max_sampled_bytes [%zu]",
                       checksum_sample_bytes_,
                       max_checksum_sampled_bytes_per_request_);
        return {ER_CHECKSUM_UNAVAILABLE, {}};
    }
    auto handle = sdk_buffer_check_pool_->GetCell();
    std::vector<int64_t> checksums;
    if (!HashBlocksInChunks(block_buffers, handle, max_check_iov_num_, checksums) ||
        checksums.size() != block_buffers.size()) {
        KVCM_LOG_ERROR(
            "calculate checksums failed: expected [%zu] results, got [%zu]", block_buffers.size(), checksums.size());
        return {ER_CHECKSUM_UNAVAILABLE, {}};
    }
    return {ER_OK, std::move(checksums)};
#else
    KVCM_LOG_ERROR("calculate checksums failed: build has no CUDA/MUSA checksum kernel");
    return {ER_CHECKSUM_UNAVAILABLE, {}};
#endif
}

ChecksumCapability TransferClientImpl::GetChecksumCapability() const {
#if defined(USING_CUDA) || defined(USING_MUSA)
    std::shared_lock read_guard(config_mutex_);
    if (client_config_ == nullptr || sdk_buffer_check_pool_ == nullptr || checksum_sample_bytes_ == 0) {
        return {};
    }
    return {true, std::string(KVCM_CHECKSUM_ALGORITHM_CRC32_XOR_INT64_LEGACY_V0), checksum_sample_bytes_};
#else
    return {};
#endif
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

std::unique_ptr<TransferClient> TransferClient::Create(const std::string &client_config,
                                                       const InitParams &init_params,
                                                       const SharedMemoryRegistration &shared_memory_registration) {
    LoggerBroker::InitLoggerForClientOnce();
    auto client = std::make_unique<TransferClientImpl>();
    auto ec = client->InitWithSharedMemory(client_config, init_params, shared_memory_registration);
    if (ec == ER_OK) {
        return client;
    }
    KVCM_LOG_ERROR("create transfer client with shared memory failed with errocode: %d", ec);
    return nullptr;
}

} // namespace kv_cache_manager

#undef DEFER
#undef CHECK_SDK_BASE
#undef CHECK_SDK
#undef CHECK_SDK_WITH_TYPE
