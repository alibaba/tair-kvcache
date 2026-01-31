#pragma once

namespace kv_cache_manager {

// TODO 完善错误码，可以表示各个组件的各个细节错误，全局一套
enum [[nodiscard]] ErrorCode : int32_t{
    EC_OK = 0,
    EC_ERROR = 1,
    EC_NOENT = 2, // entry not exist
    EC_TIMEOUT = 3,
    EC_EXIST = 4,
    EC_IO_ERROR = 5,
    EC_BADARGS = 6, // invalid arg
    EC_UNIMPLEMENTED = 7,
    EC_CORRUPTION = 8,
    EC_NOSPC = 9, // out of space
    EC_PARTIAL_OK = 10,
    EC_INSTANCE_NOT_EXIST = 11,
    EC_DUPLICATE_ENTITY = 12,
    EC_CONFIG_ERROR = 13,
    EC_OUT_OF_LIMIT = 14,
    EC_OUT_OF_RANGE = 15,
    EC_MISMATCH = 16,
    EC_NOSCRIPT = 17,
    EC_SERVICE_NOT_LEADER = 18,
    EC_UNKNOWN = 127,
    EC_KVCM_MAX,
};

} // namespace kv_cache_manager
