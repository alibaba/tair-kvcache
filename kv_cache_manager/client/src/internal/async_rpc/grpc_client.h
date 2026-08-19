// Insecure gRPC transport.
//
// One channel per unique endpoint per logical client, concurrent RPCs
// multiplexed over HTTP/2. A small fixed pool of completion-queue threads
// waits for network events; nothing else blocks on I/O.
#pragma once

#include <memory>

#include "kv_cache_manager/client/src/internal/async_rpc/async_rpc_client.h"

namespace kv_cache_manager::async_rpc {

class GrpcClientRuntime {
public:
    explicit GrpcClientRuntime(uint32_t completion_queues);
    ~GrpcClientRuntime();

    uint32_t thread_count() const;
    void Shutdown();

    struct Impl;
    Impl &impl() { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<AsyncRpcClient> MakeGrpcClient(std::shared_ptr<GrpcClientRuntime> runtime,
                                               const ContinuationScheduler *scheduler,
                                               EndpointSet endpoints,
                                               ClientLimits limits);

} // namespace kv_cache_manager::async_rpc
