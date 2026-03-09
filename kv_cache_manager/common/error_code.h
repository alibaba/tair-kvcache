#pragma once

#include <unordered_map>

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

// Convert internal ErrorCode to protobuf proto::meta::ErrorCode or proto::admin::ErrorCode.
template <typename PbErrorCode>
inline PbErrorCode ToPbError(ErrorCode internal_error) {
    static const std::unordered_map<ErrorCode, PbErrorCode> error_map{
        {EC_UNIMPLEMENTED, PbErrorCode::UNSUPPORTED},
        {EC_BADARGS, PbErrorCode::INVALID_ARGUMENT},
        {EC_DUPLICATE_ENTITY, PbErrorCode::DUPLICATE_ENTITY},
        {EC_EXIST, PbErrorCode::DUPLICATE_ENTITY},
        {EC_INSTANCE_NOT_EXIST, PbErrorCode::INSTANCE_NOT_EXIST},
        {EC_NOENT, PbErrorCode::INSTANCE_NOT_EXIST},
        {EC_SERVICE_NOT_LEADER, PbErrorCode::SERVER_NOT_LEADER},
        {EC_IO_ERROR, PbErrorCode::IO_ERROR},
        {EC_OUT_OF_LIMIT, PbErrorCode::REACH_MAX_ENTITY_CAPACITY},
        {EC_UNKNOWN, PbErrorCode::UNKNOWN_ERROR},
    };
    if (auto iter = error_map.find(internal_error); iter != error_map.end()) {
        return iter->second;
    }
    return PbErrorCode::INTERNAL_ERROR;
}

} // namespace kv_cache_manager
