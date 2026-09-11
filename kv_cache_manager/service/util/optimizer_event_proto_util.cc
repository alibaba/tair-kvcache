#include "kv_cache_manager/service/util/optimizer_event_proto_util.h"

#include <cstdint>
#include <cstring>

#include "kv_cache_manager/common/request_context.h"

namespace kv_cache_manager {

ErrorCode DecodeOptimizerEventTokens(const proto::optimizer::TraceQueryRequest &request,
                                     RequestContext *request_context,
                                     std::vector<int64_t> *tokens) {
    if (request.token_ids_size() != 0 && !request.token_ids_le64().empty()) {
        request_context->error_tracer()->AddErrorMsg("token_ids and token_ids_le64 are mutually exclusive");
        return EC_BADARGS;
    }
    if (request.token_ids_le64().empty()) {
        tokens->assign(request.token_ids().begin(), request.token_ids().end());
        return EC_OK;
    }
    if (request.token_ids_le64().size() % sizeof(int64_t) != 0) {
        request_context->error_tracer()->AddErrorMsg("token_ids_le64 size must be a multiple of 8");
        return EC_BADARGS;
    }

    const auto token_count = request.token_ids_le64().size() / sizeof(int64_t);
    tokens->resize(token_count);
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    std::memcpy(tokens->data(), request.token_ids_le64().data(), request.token_ids_le64().size());
#else
    const auto *data = reinterpret_cast<const unsigned char *>(request.token_ids_le64().data());
    for (size_t token_index = 0; token_index < token_count; ++token_index) {
        uint64_t value = 0;
        for (size_t byte_index = 0; byte_index < sizeof(int64_t); ++byte_index) {
            value |= static_cast<uint64_t>(data[token_index * sizeof(int64_t) + byte_index]) << (byte_index * 8);
        }
        (*tokens)[token_index] = static_cast<int64_t>(value);
    }
#endif
    return EC_OK;
}

} // namespace kv_cache_manager
