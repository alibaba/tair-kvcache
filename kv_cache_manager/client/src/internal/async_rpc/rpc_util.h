#pragma once

#include <google/protobuf/message.h>

#include "kv_cache_manager/client/src/internal/async_rpc/async_rpc_client.h"

namespace kv_cache_manager::async_rpc {

inline void ApplyServiceStatus(const google::protobuf::Message &response, RpcResult *result) {
    result->service_status = ExtractServiceStatus(response);
    if (result->transport_error != TransportError::kNone) {
        result->ok = false;
        return;
    }
    if (result->service_status == 0) {
        result->ok = true;
        return;
    }
    result->ok = result->service_status == kStatusOk;
    if (!result->ok && result->raw_error.empty()) {
        result->raw_error = ExtractServiceMessage(response);
    }
}

} // namespace kv_cache_manager::async_rpc
