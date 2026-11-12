#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

#define KVCM_HAS_TRANSFER_DEADLINE 1

namespace kv_cache_manager {

class TransferClient {
public:
    virtual ~TransferClient() = default;
    static std::unique_ptr<TransferClient> Create(const std::string &client_config, const InitParams &init_params);
    static std::unique_ptr<TransferClient>
    Create(const std::string &client_config,
           const InitParams &init_params,
           const SharedMemoryRegistration &shared_memory_registration);
    // Callers that use braced initialization must name the registration type
    // explicitly. Bare `{}` or `{base, size, fd}` arguments are ambiguous with
    // the SharedMemoryRegistration overload above.
    static std::unique_ptr<TransferClient>
    Create(const std::string &client_config,
           const InitParams &init_params,
           const ClientMemoryRegistrations &memory_registrations);

    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                         const BlockBuffers &block_buffers,
                                         std::shared_ptr<TransferTraceInfo> trace_info = nullptr) = 0;
    virtual std::pair<ClientErrorCode, UriStrVec>
    SaveKvCaches(const UriStrVec &uri_str_vec,
                 const BlockBuffers &block_buffers,
                 std::shared_ptr<TransferTraceInfo> trace_info = nullptr) = 0;

    // Absolute steady_clock milliseconds, in this process; <=0 uses the configured
    // SDK budget. Wrapping clients override these methods to forward the deadline.
    virtual ClientErrorCode LoadKvCachesWithDeadline(const UriStrVec &uri_str_vec,
                                                     const BlockBuffers &block_buffers,
                                                     int64_t /*deadline_ms*/,
                                                     std::shared_ptr<TransferTraceInfo> trace_info = nullptr) {
        return LoadKvCaches(uri_str_vec, block_buffers, std::move(trace_info));
    }
    virtual std::pair<ClientErrorCode, UriStrVec>
    SaveKvCachesWithDeadline(const UriStrVec &uri_str_vec,
                             const BlockBuffers &block_buffers,
                             int64_t /*deadline_ms*/,
                             std::shared_ptr<TransferTraceInfo> trace_info = nullptr) {
        return SaveKvCaches(uri_str_vec, block_buffers, std::move(trace_info));
    }

protected:
    TransferClient() = default;
    virtual ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) = 0;
};
} // namespace kv_cache_manager
