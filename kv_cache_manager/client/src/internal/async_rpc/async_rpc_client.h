// Internal asynchronous KVCM RPC client.
//
// This target is intentionally not part of kv_cache_manager_client.so yet.
// It exposes the raw protobuf request/response contract needed by internal
// high-concurrency callers without imposing retries or domain interpretation.
#pragma once

#include <async_simple/coro/Lazy.h>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/client/src/internal/async_rpc/api.h"
#include "kv_cache_manager/client/src/internal/async_rpc/cancellation.h"

namespace google {
namespace protobuf {
class Message;
} // namespace protobuf
} // namespace google

namespace kv_cache_manager::async_rpc {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::nanoseconds;

inline TimePoint Now() { return Clock::now(); }

inline double ToMillis(Duration duration) {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(duration).count();
}

template <typename T = void>
using Task = async_simple::coro::Lazy<T>;

enum class TransportKind {
    kHttp,
    kGrpc
};

const char *TransportKindName(TransportKind kind);

enum class RpcLane {
    kBusiness,
    kControl
};

enum class TransportError {
    kNone = 0,
    kConnect,
    kTimeout,
    kDisconnect,
    kEncode,
    kDecode,
    kCancelled,
    kNoPermit,
    kUnsupported,
    kOther,
};

const char *TransportErrorName(TransportError error);

inline bool IsUncertain(TransportError error) {
    return error == TransportError::kTimeout || error == TransportError::kDisconnect;
}

struct RpcResult {
    TransportError transport_error = TransportError::kNone;
    int service_status = 0;
    int raw_status = 0;
    std::string raw_error;
    Duration rpc_latency{};
    bool ok = false;
};

struct EndpointSet {
    std::string meta_http;
    std::string meta_grpc;
    std::string admin_http;
    std::string admin_grpc;
};

struct ClientLimits {
    uint32_t http_connections_per_endpoint = 8;
    uint32_t http_control_connections_per_endpoint = 2;
    Duration connect_timeout = std::chrono::seconds(3);
    Duration default_rpc_timeout = std::chrono::seconds(10);
};

struct EndpointStats {
    std::string endpoint;
    std::string role;
    uint64_t channels = 0;
    uint64_t connections_current = 0;
    uint64_t connections_peak = 0;
    uint64_t connections_created = 0;
    uint64_t connections_reused = 0;
    uint64_t in_flight_current = 0;
    uint64_t in_flight_peak = 0;
    std::vector<double> establish_latency_ms;
};

struct ClientStats {
    TransportKind kind = TransportKind::kHttp;
    std::vector<EndpointStats> endpoints;
};

struct CallOptions {
    RpcLane lane = RpcLane::kBusiness;
    TimePoint deadline{};
    CancellationToken cancel;
};

// Completion-queue and reactor threads use this callback pair to return a
// suspended call to the caller's continuation executor. The RPC client owns no
// general-purpose worker pool.
struct ContinuationScheduler {
    using Callback = std::function<void()>;
    std::function<bool(Callback)> schedule;
    std::function<void(TimePoint, Callback)> schedule_at;
};

class AsyncRpcClient {
public:
    virtual ~AsyncRpcClient() = default;

    virtual Task<RpcResult> Call(Api api,
                                 const google::protobuf::Message &request,
                                 google::protobuf::Message *response,
                                 CallOptions options) = 0;

    virtual void SetMetaEndpoint(const std::string &endpoint) = 0;
    virtual std::string MetaEndpoint() const = 0;
    virtual ClientStats Stats() const = 0;
    virtual void Shutdown() = 0;
};

class AsyncRpcClientProvider {
public:
    AsyncRpcClientProvider(ContinuationScheduler scheduler,
                           EndpointSet endpoints,
                           ClientLimits limits,
                           uint32_t http_reactor_threads,
                           uint32_t grpc_completion_queues);
    ~AsyncRpcClientProvider();

    std::unique_ptr<AsyncRpcClient> Create(TransportKind kind);
    uint32_t io_thread_count() const;
    void Shutdown();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

bool ValidateInsecureEndpoint(const std::string &endpoint, bool expect_http_scheme, std::string *error);

} // namespace kv_cache_manager::async_rpc
