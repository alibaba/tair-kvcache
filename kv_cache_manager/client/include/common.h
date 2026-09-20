#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

// Compile-time capability marker for clients that expose the additive staged
// checksum APIs (options/results and the appended virtual overloads).  Callers
// must still treat runtime support as a server/client deployment capability.
#define KVCM_STAGED_CHECKSUM_API_VERSION 4

// Exported by libkv_cache_manager_client.so. Embedders that may load an older
// client DSO should weak-probe this C symbol before dispatching any virtual
// method added by KVCM_STAGED_CHECKSUM_API_VERSION. A missing symbol means the
// loaded DSO predates the staged checksum ABI and must be treated as version 0.
extern "C" uint32_t KVCMStagedChecksumRuntimeApiVersion() noexcept;

namespace kv_cache_manager {

enum [[nodiscard]] ClientErrorCode : int32_t{
    // client & sdkwrapper
    ER_OK = 0,
    ER_INVALID_STUB = 1,
    ER_INVALID_GRPCSTATUS = 2,
    ER_INVALID_PARAMS = 3,
    ER_INVALID_ROLETYPE = 4,
    ER_INVALID_CLIENT_CONFIG = 5,
    ER_INVALID_STORAGE_CONFIG = 6,
    ER_INVALID_SDKWRAPPER_CONFIG = 7,
    ER_INVALID_SDKBACKEND_CONFIG = 8,

    ER_CONNECT_FAIL = 9,
    ER_THREADPOOL_ERROR = 10,
    ER_SKIPINIT = 11,

    ER_METACLIENT_INIT_ERROR = 12,
    ER_TRANSFERCLIENT_INIT_ERROR = 13,
    ER_MANAGERCLIENT_INIT_ERROR = 14,
    ER_CLIENT_NOT_EXISTS = 15,
    ER_INIT_CHECK_BUFFER_ERROR = 16,

    // service status code
    ER_SERVICE_NO_STATUS = 50,
    ER_SERVICE_INTERNAL_ERROR = 51,
    ER_SERVICE_UNSUPPORTED = 52,
    ER_SERVICE_INVALID_ARGUMENT = 53,
    ER_SERVICE_DUPLICATE_ENTITY = 54,
    ER_SERVICE_INSTANCE_NOT_EXIST = 55,
    ER_SERVICE_NOT_LEADER = 56,

    // sdk
    ER_SDK_TIMEOUT = 100,
    ER_GETSDK_ERROR = 101,
    ER_CREATESDK_ERROR = 102,

    ER_SDKINIT_ERROR = 103,
    ER_SDKREAD_ERROR = 104,
    ER_SDKWRITE_ERROR = 105,
    ER_SDKALLOC_ERROR = 106,

    ER_INVALID_ADDRESS = 107,
    ER_INVALID_LOCAL_BUFFERS = 108,
    ER_UNSUPPORTED_MEMORY_TYPE = 109,
    ER_UNCONSISTENT_MEMORY_TYPE = 110,
    ER_FILE_IO_ERROR = 111,
    ER_CUDAMEMCPY_ERROR = 112,
    ER_EXTRACT_SLICES_ERROR = 113,
    ER_CUDA_STREAM_CREATE_ERROR = 114,
    ER_CUDA_STREAM_SYNCHRONIZE_ERROR = 115,
    ER_CUDA_STREAM_DESTROY_ERROR = 116,
    ER_CUDA_HOST_REGISTER_ERROR = 117,

    // data integrity
    ER_CHECKSUM_MISMATCH = 118,     // a computed checksum differs from its trusted value
    ER_INLINE_HEADER_INVALID = 119, // inline header check failed (Scheme B, reserved, disabled)
    ER_CHECKSUM_UNAVAILABLE = 120,  // explicit checksum compute/verify cannot be performed
};

enum class QueryType : int {
    QT_UNSPECIFIED = 0,
    QT_BATCH_GET = 1,
    QT_PREFIX_MATCH = 2,
    QT_REVERSE_ROLL_SW_MATCH = 3,
    QT_PREFIX_MATCH_WITH_MAMBA = 4,
};

struct LocationSpecUnit {
    bool operator==(const LocationSpecUnit &other) const { return spec_name == other.spec_name && uri == other.uri; }
    std::string spec_name;
    std::string uri;
};
using Location = std::vector<LocationSpecUnit>; // one block key may have multiple location_specs
using Locations = std::vector<Location>;
using UriStrVec = std::vector<std::string>;
struct Metas {
    Locations locations;
    std::vector<std::string> metas;
};

using BlockMaskVector = std::vector<bool>;
using BlockMaskOffset = size_t;
using BlockMask = std::variant<BlockMaskVector, BlockMaskOffset>;
struct WriteLocation {
    std::string write_session_id;
    BlockMask block_mask;
    Locations locations;
};

struct ClusterInfo {
    std::string self_node_id;
    std::string leader_node_id;
    struct NodeEndpoint {
        std::string node_id;
        std::string host;
        int32_t meta_rpc_port{0};
        int32_t meta_http_port{0};
        std::string custom_info;
    };
    NodeEndpoint leader_endpoint;
};

enum class MemoryType : uint8_t {
    CPU = 0,
    GPU = 1,
};

// 一块连续的内存，存放一层的 K 或 V 数据
struct Iov {
    MemoryType type{MemoryType::GPU};
    void *base{nullptr};
    size_t size{0};
    bool ignore{false};

    void set_base_as_uint64(uint64_t base_ptr) { base = reinterpret_cast<void *>(base_ptr); }
    [[nodiscard]] uint64_t base_as_uint64() const { return reinterpret_cast<uint64_t>(base); }
};

/*
 * 一个block buffer内部存放一个block的数据，包含layer_num*(k_len + v_len)个Iov
 * 按层存放，后续要考虑支持按层读取
 */
struct BlockBuffer {
    std::vector<Iov> iovs;
};

using BlockBuffers = std::vector<BlockBuffer>;

enum class RoleType : uint8_t {
    UNKNOWN = 0b00000000,
    WORKER = 0b00000001,
    SCHEDULER = 0b00000010,
    HYBRID = 0b00000011,
};

inline bool operator&(RoleType lhs, RoleType rhs) { return static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs); }

inline RoleType RoleTypeFromString(const std::string &role_type_str) {
    if (role_type_str == "worker") {
        return RoleType::WORKER;
    } else if (role_type_str == "scheduler") {
        return RoleType::SCHEDULER;
    } else if (role_type_str == "hybrid") {
        return RoleType::HYBRID;
    } else {
        return RoleType::UNKNOWN;
    }
}

inline std::string RoleTypeToString(RoleType role_type) {
    switch (role_type) {
    case RoleType::WORKER:
        return "worker";
    case RoleType::SCHEDULER:
        return "scheduler";
    case RoleType::HYBRID:
        return "hybrid";
    default:
        return "unknown";
    }
}

struct RegistSpan {
    void *base{nullptr};
    size_t size{0};
    void set_base_as_uint64(uint64_t base_ptr) { base = reinterpret_cast<void *>(base_ptr); }
    [[nodiscard]] uint64_t base_as_uint64() const { return reinterpret_cast<uint64_t>(base); }
};

// Keep shared-memory metadata separate from RegistSpan so existing callers
// retain the original ABI.
struct SharedMemoryRegistration {
    void *base{nullptr};
    size_t size{0};
    int fd{-1};
};

struct InitParams {
    RoleType role_type{RoleType::UNKNOWN};
    RegistSpan *regist_span{nullptr};    // used by worker
    std::string self_location_spec_name; // used by worker
    std::string storage_configs;         // used by worker
};

struct ForwardContext {
    std::map<std::string, std::string> metas;
    int32_t sw_size{-1};
};

struct TransferTraceInfo {
    bool need_print = false;
    std::vector<std::string> block_ids; // block_ids.size() must be equal to block_buffer.size()
};

// Runtime descriptor for the checksum implementation in the loaded client
// library. A compile-time API marker alone cannot detect a process that loaded
// a different DSO or changed KVCM_CHECK_IOV_BYTE_SIZE after the CUDA / MUSA
// implementation captured it during static initialization.
inline constexpr std::string_view KVCM_CHECKSUM_ALGORITHM_CRC32_XOR_INT64_LEGACY_V0 = "crc32_xor_int64.legacy-v0";
inline constexpr size_t KVCM_CHECKSUM_MAX_SAMPLE_BYTES = 4096;

struct ChecksumCapability {
    bool available{false};
    std::string algorithm_id;
    size_t sample_bytes{0};
};

enum class ChecksumValidationStage : uint8_t {
    CVS_UNSPECIFIED = 0,
    CVS_WRITE_INPUT = 1,
    CVS_META_ROUND_TRIP = 2,
    CVS_READ_OUTPUT = 3,
};

inline const char *ChecksumValidationStageToString(ChecksumValidationStage stage) {
    switch (stage) {
    case ChecksumValidationStage::CVS_WRITE_INPUT:
        return "WRITE_INPUT";
    case ChecksumValidationStage::CVS_META_ROUND_TRIP:
        return "META_ROUND_TRIP";
    case ChecksumValidationStage::CVS_READ_OUTPUT:
        return "READ_OUTPUT";
    case ChecksumValidationStage::CVS_UNSPECIFIED:
    default:
        return "UNSPECIFIED";
    }
}

struct ChecksumVerifyResult {
    ChecksumValidationStage stage{ChecksumValidationStage::CVS_UNSPECIFIED};
    bool mismatch{false};
    // Empty with mismatch=true means one of the batch/presence vectors had an
    // invalid size. Otherwise every missing or unequal position is reported.
    std::vector<std::size_t> faulty_indices;
};

// Compare two position-aligned checksum batches. An empty presence vector means
// every value in that batch is present. A false expected_present entry skips the
// position (legacy/no-checksum data); an expected value with no actual value is a
// mismatch. Zero is an ordinary checksum whenever its presence bit is true.
inline ChecksumVerifyResult
VerifyBatchChecksums(const std::vector<int64_t> &expected,
                     const std::vector<int64_t> &actual,
                     ChecksumValidationStage stage = ChecksumValidationStage::CVS_UNSPECIFIED,
                     const std::vector<bool> &expected_present = {},
                     const std::vector<bool> &actual_present = {}) {
    ChecksumVerifyResult result;
    result.stage = stage;
    if (expected.size() != actual.size() || (!expected_present.empty() && expected_present.size() != expected.size()) ||
        (!actual_present.empty() && actual_present.size() != actual.size())) {
        result.mismatch = true;
        return result;
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const bool has_expected = expected_present.empty() || expected_present[i];
        if (!has_expected) {
            continue;
        }
        const bool has_actual = actual_present.empty() || actual_present[i];
        if (!has_actual || expected[i] != actual[i]) {
            result.faulty_indices.push_back(i);
        }
    }
    result.mismatch = !result.faulty_indices.empty();
    return result;
}

// Optional operation controls keep rarely used knobs out of the hot-path
// parameter list while leaving default calls concise.
struct MatchLocationOptions {
    // Slide-window size for QT_REVERSE_ROLL_SW_MATCH. The default keeps the
    // existing non-window query behavior.
    int32_t sw_size{-1};
    bool include_checksums{false};

    static MatchLocationOptions WithSlideWindowSize(int32_t sw_size) {
        MatchLocationOptions options;
        options.sw_size = sw_size;
        return options;
    }

    static MatchLocationOptions WithChecksums(int32_t sw_size = -1) {
        MatchLocationOptions options;
        options.sw_size = sw_size;
        options.include_checksums = true;
        return options;
    }
};

// A write-side checksum batch names the independently stored LocationSpec whose
// payload was hashed. Values are aligned with the compact StartWrite session.
struct LocationSpecChecksumBatch {
    std::string location_spec_name;
    std::vector<int64_t> checksums;
};

// Query-side values are also grouped by LocationSpec. Each values/presence
// vector is aligned with the returned block locations; false presence means
// that block has no such spec or stores legacy metadata without a checksum.
struct LocationSpecChecksumResult {
    std::string location_spec_name;
    std::vector<int64_t> checksums;
    std::vector<bool> checksum_present;
};

inline const LocationSpecChecksumResult *
FindLocationSpecChecksums(const std::vector<LocationSpecChecksumResult> &checksum_results,
                          std::string_view location_spec_name) {
    const auto it = std::find_if(checksum_results.begin(), checksum_results.end(), [location_spec_name](const auto &v) {
        return v.location_spec_name == location_spec_name;
    });
    return it == checksum_results.end() ? nullptr : &*it;
}

struct MatchLocationResult {
    Locations locations;
    std::vector<LocationSpecChecksumResult> checksum_results;

    const LocationSpecChecksumResult *FindChecksums(std::string_view location_spec_name) const {
        return FindLocationSpecChecksums(checksum_results, location_spec_name);
    }

    ChecksumVerifyResult VerifyChecksums(std::string_view location_spec_name,
                                         const std::vector<int64_t> &expected) const {
        const auto *result = FindChecksums(location_spec_name);
        if (result == nullptr) {
            ChecksumVerifyResult verify_result;
            verify_result.stage = ChecksumValidationStage::CVS_META_ROUND_TRIP;
            verify_result.mismatch = true;
            return verify_result;
        }
        return VerifyBatchChecksums(
            expected, result->checksums, ChecksumValidationStage::CVS_META_ROUND_TRIP, {}, result->checksum_present);
    }
};

struct MatchLocationLenOptions {
    int32_t sw_size{-1};

    static MatchLocationLenOptions WithSlideWindowSize(int32_t sw_size) {
        MatchLocationLenOptions options;
        options.sw_size = sw_size;
        return options;
    }
};

struct MatchMetaOptions {
    int32_t detail_level{0};
    bool include_checksums{false};

    static MatchMetaOptions WithDetailLevel(int32_t detail_level) {
        MatchMetaOptions options;
        options.detail_level = detail_level;
        return options;
    }

    static MatchMetaOptions WithChecksums(int32_t detail_level = 0) {
        MatchMetaOptions options;
        options.detail_level = detail_level;
        options.include_checksums = true;
        return options;
    }
};

struct MatchMetaResult {
    Metas metas;
    std::vector<LocationSpecChecksumResult> checksum_results;

    const LocationSpecChecksumResult *FindChecksums(std::string_view location_spec_name) const {
        return FindLocationSpecChecksums(checksum_results, location_spec_name);
    }

    ChecksumVerifyResult VerifyChecksums(std::string_view location_spec_name,
                                         const std::vector<int64_t> &expected) const {
        const auto *result = FindChecksums(location_spec_name);
        if (result == nullptr) {
            ChecksumVerifyResult verify_result;
            verify_result.stage = ChecksumValidationStage::CVS_META_ROUND_TRIP;
            verify_result.mismatch = true;
            return verify_result;
        }
        return VerifyBatchChecksums(
            expected, result->checksums, ChecksumValidationStage::CVS_META_ROUND_TRIP, {}, result->checksum_present);
    }
};

struct FinishWriteOptions {
    // Every batch is aligned with the compact keys captured by the StartWrite
    // session, including a placeholder at failed positions. Batches for specs
    // not present on a block are ignored at that position.
    std::vector<LocationSpecChecksumBatch> checksum_batches;

    static FinishWriteOptions WithChecksums(std::string location_spec_name, std::vector<int64_t> checksums) {
        FinishWriteOptions options;
        options.checksum_batches.push_back({std::move(location_spec_name), std::move(checksums)});
        return options;
    }

    static FinishWriteOptions WithChecksumBatches(std::vector<LocationSpecChecksumBatch> checksum_batches) {
        FinishWriteOptions options;
        options.checksum_batches = std::move(checksum_batches);
        return options;
    }
};

// Version-4 FinishWrite payload. Keep this separate from FinishWriteOptions:
// that version-3 type crosses a released C++ ABI boundary and its size must
// never change.
struct LocationSpecUriBatch {
    std::string location_spec_name;
    // Parallel to the compact keys captured by StartWrite. A false presence
    // bit keeps the placeholder URI out of metadata for blocks which do not
    // carry this independently stored spec.
    std::vector<std::string> uris;
    std::vector<bool> uri_present;
};

struct FinishWriteIntegrityOptions {
    std::vector<LocationSpecChecksumBatch> checksum_batches;
    std::vector<LocationSpecUriBatch> uri_batches;
};

struct LoadKvCachesOptions {
    std::shared_ptr<TransferTraceInfo> trace_info{nullptr};
    // Kept in the new options object rather than extending TransferTraceInfo,
    // whose size is part of the released C++ client ABI.
    std::string trace_id;
    bool verify_checksums{false};
    std::vector<int64_t> expected_checksums;
    // Empty means all expected_checksums are present. A non-empty vector must
    // be parallel to expected_checksums and allows legacy/missing entries to be
    // skipped without reserving a checksum value as a sentinel.
    std::vector<bool> expected_checksum_present;

    static LoadKvCachesOptions WithTraceInfo(std::shared_ptr<TransferTraceInfo> trace_info, std::string trace_id = {}) {
        LoadKvCachesOptions options;
        options.trace_info = std::move(trace_info);
        options.trace_id = std::move(trace_id);
        return options;
    }

    static LoadKvCachesOptions VerifyWith(std::vector<int64_t> checksums) {
        LoadKvCachesOptions options;
        options.verify_checksums = true;
        options.expected_checksums = std::move(checksums);
        return options;
    }

    static LoadKvCachesOptions VerifyWith(std::vector<int64_t> checksums, std::vector<bool> checksum_present) {
        LoadKvCachesOptions options;
        options.verify_checksums = true;
        options.expected_checksums = std::move(checksums);
        options.expected_checksum_present = std::move(checksum_present);
        return options;
    }

    static LoadKvCachesOptions VerifyWith(std::vector<int64_t> checksums,
                                          std::shared_ptr<TransferTraceInfo> trace_info,
                                          std::string trace_id = {}) {
        LoadKvCachesOptions options;
        options.trace_info = std::move(trace_info);
        options.trace_id = std::move(trace_id);
        options.verify_checksums = true;
        options.expected_checksums = std::move(checksums);
        return options;
    }

    static LoadKvCachesOptions VerifyWith(std::vector<int64_t> checksums,
                                          std::vector<bool> checksum_present,
                                          std::shared_ptr<TransferTraceInfo> trace_info,
                                          std::string trace_id = {}) {
        LoadKvCachesOptions options;
        options.trace_info = std::move(trace_info);
        options.trace_id = std::move(trace_id);
        options.verify_checksums = true;
        options.expected_checksums = std::move(checksums);
        options.expected_checksum_present = std::move(checksum_present);
        return options;
    }

    static LoadKvCachesOptions VerifyWith(const MatchLocationResult &result, std::string_view location_spec_name) {
        const auto *checksums = result.FindChecksums(location_spec_name);
        if (checksums == nullptr) {
            LoadKvCachesOptions options;
            options.verify_checksums = true;
            return options;
        }
        return VerifyWith(checksums->checksums, checksums->checksum_present);
    }

    static LoadKvCachesOptions VerifyWith(const MatchMetaResult &result, std::string_view location_spec_name) {
        const auto *checksums = result.FindChecksums(location_spec_name);
        if (checksums == nullptr) {
            LoadKvCachesOptions options;
            options.verify_checksums = true;
            return options;
        }
        return VerifyWith(checksums->checksums, checksums->checksum_present);
    }
};

struct SaveKvCachesOptions {
    std::shared_ptr<TransferTraceInfo> trace_info{nullptr};
    std::string trace_id;
    bool include_checksums{false};
    bool verify_caller_checksums{false};
    std::vector<int64_t> expected_checksums;

    static SaveKvCachesOptions WithTraceInfo(std::shared_ptr<TransferTraceInfo> trace_info, std::string trace_id = {}) {
        SaveKvCachesOptions options;
        options.trace_info = std::move(trace_info);
        options.trace_id = std::move(trace_id);
        return options;
    }

    static SaveKvCachesOptions WithChecksums() {
        SaveKvCachesOptions options;
        options.include_checksums = true;
        return options;
    }

    static SaveKvCachesOptions WithChecksums(std::shared_ptr<TransferTraceInfo> trace_info, std::string trace_id = {}) {
        SaveKvCachesOptions options;
        options.trace_info = std::move(trace_info);
        options.trace_id = std::move(trace_id);
        options.include_checksums = true;
        return options;
    }

    // The caller asserts that these values use KVCM's built-in algorithm.
    // TransferClient recomputes and compares them before Put; any mismatch or
    // unavailable compute capability fails the call without writing data.
    static SaveKvCachesOptions VerifyCallerChecksums(std::vector<int64_t> checksums) {
        SaveKvCachesOptions options;
        options.include_checksums = true;
        options.verify_caller_checksums = true;
        options.expected_checksums = std::move(checksums);
        return options;
    }

    static SaveKvCachesOptions VerifyCallerChecksums(std::vector<int64_t> checksums,
                                                     std::shared_ptr<TransferTraceInfo> trace_info,
                                                     std::string trace_id = {}) {
        auto options = VerifyCallerChecksums(std::move(checksums));
        options.trace_info = std::move(trace_info);
        options.trace_id = std::move(trace_id);
        return options;
    }
};

struct SaveKvCachesResult {
    UriStrVec uri_str_vec;
    std::string location_spec_name;
    std::vector<int64_t> checksums;

    LocationSpecChecksumBatch ToChecksumBatch() const { return {location_spec_name, checksums}; }
};

} // namespace kv_cache_manager
