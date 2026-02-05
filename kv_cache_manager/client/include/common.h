#pragma once

#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

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
};

enum class QueryType : int {
    QT_UNSPECIFIED = 0,
    QT_BATCH_GET = 1,
    QT_PREFIX_MATCH = 2,
    QT_REVERSE_ROLL_SW_MATCH = 3,
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

} // namespace kv_cache_manager