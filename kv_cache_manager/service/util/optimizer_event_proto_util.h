#pragma once

#include <cstdint>
#include <vector>

#include "kv_cache_manager/common/error_code.h"
#include "kv_cache_manager/protocol/protobuf/optimizer_service.pb.h"

namespace kv_cache_manager {

class RequestContext;

ErrorCode DecodeOptimizerEventTokens(const proto::optimizer::TraceQueryRequest &request,
                                     RequestContext *request_context,
                                     std::vector<int64_t> *tokens);

} // namespace kv_cache_manager
