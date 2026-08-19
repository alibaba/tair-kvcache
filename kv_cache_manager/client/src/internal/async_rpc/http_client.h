// Plaintext HTTP/1.1 transport.
//
// Every logical client owns a context with its own small, lazily grown
// connection pool per endpoint, plus reserved control-lane connections so a
// saturated business lane cannot block heartbeats or health probes. All
// contexts share the same fixed reactor thread pool.
#pragma once

#include <memory>

#include "kv_cache_manager/client/src/internal/async_rpc/async_rpc_client.h"

namespace kv_cache_manager::async_rpc {

class HttpClientRuntime {
public:
    explicit HttpClientRuntime(uint32_t reactor_threads);
    ~HttpClientRuntime();

    uint32_t thread_count() const;
    void Shutdown();

    struct Impl;
    Impl &impl() { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<AsyncRpcClient> MakeHttpClient(std::shared_ptr<HttpClientRuntime> runtime,
                                               const ContinuationScheduler *scheduler,
                                               EndpointSet endpoints,
                                               ClientLimits limits);

} // namespace kv_cache_manager::async_rpc
