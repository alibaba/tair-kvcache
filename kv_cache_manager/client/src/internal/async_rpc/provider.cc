#include "kv_cache_manager/client/src/internal/async_rpc/async_rpc_client.h"

#include <mutex>

#include "kv_cache_manager/client/src/internal/async_rpc/grpc_client.h"
#include "kv_cache_manager/client/src/internal/async_rpc/http_client.h"

namespace kv_cache_manager::async_rpc {

class AsyncRpcClientProvider::Impl {
public:
    Impl(ContinuationScheduler scheduler,
         EndpointSet endpoints,
         ClientLimits limits,
         uint32_t http_reactor_threads,
         uint32_t grpc_completion_queues)
        : scheduler(std::move(scheduler))
        , endpoints(std::move(endpoints))
        , limits(limits)
        , http_reactor_threads(http_reactor_threads == 0 ? 1u : http_reactor_threads)
        , grpc_completion_queues(grpc_completion_queues == 0 ? 1u : grpc_completion_queues) {}

    ContinuationScheduler scheduler;
    EndpointSet endpoints;
    ClientLimits limits;
    uint32_t http_reactor_threads;
    uint32_t grpc_completion_queues;
    mutable std::mutex mutex;
    std::shared_ptr<HttpClientRuntime> http_runtime;
    std::shared_ptr<GrpcClientRuntime> grpc_runtime;
    bool shutdown = false;
};

AsyncRpcClientProvider::AsyncRpcClientProvider(ContinuationScheduler scheduler,
                                               EndpointSet endpoints,
                                               ClientLimits limits,
                                               uint32_t http_reactor_threads,
                                               uint32_t grpc_completion_queues)
    : impl_(std::make_unique<Impl>(std::move(scheduler),
                                   std::move(endpoints),
                                   limits,
                                   http_reactor_threads,
                                   grpc_completion_queues)) {}

AsyncRpcClientProvider::~AsyncRpcClientProvider() { Shutdown(); }

std::unique_ptr<AsyncRpcClient> AsyncRpcClientProvider::Create(TransportKind kind) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->shutdown) {
        return nullptr;
    }
    if (kind == TransportKind::kHttp) {
        if (!impl_->http_runtime) {
            impl_->http_runtime = std::make_shared<HttpClientRuntime>(impl_->http_reactor_threads);
        }
        return MakeHttpClient(impl_->http_runtime, &impl_->scheduler, impl_->endpoints, impl_->limits);
    }
    if (!impl_->grpc_runtime) {
        impl_->grpc_runtime = std::make_shared<GrpcClientRuntime>(impl_->grpc_completion_queues);
    }
    return MakeGrpcClient(impl_->grpc_runtime, &impl_->scheduler, impl_->endpoints, impl_->limits);
}

uint32_t AsyncRpcClientProvider::io_thread_count() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    uint32_t total = 0;
    if (impl_->http_runtime) {
        total += impl_->http_runtime->thread_count();
    }
    if (impl_->grpc_runtime) {
        total += impl_->grpc_runtime->thread_count();
    }
    return total;
}

void AsyncRpcClientProvider::Shutdown() {
    std::shared_ptr<HttpClientRuntime> http;
    std::shared_ptr<GrpcClientRuntime> grpc;
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->shutdown) {
            return;
        }
        impl_->shutdown = true;
        http = std::move(impl_->http_runtime);
        grpc = std::move(impl_->grpc_runtime);
    }
    if (grpc) {
        grpc->Shutdown();
    }
    if (http) {
        http->Shutdown();
    }
}

} // namespace kv_cache_manager::async_rpc
