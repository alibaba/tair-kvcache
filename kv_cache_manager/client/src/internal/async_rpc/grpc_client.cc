#include "kv_cache_manager/client/src/internal/async_rpc/grpc_client.h"

#include <algorithm>
#include <atomic>
#include <grpcpp/generic/generic_stub.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "kv_cache_manager/client/src/internal/async_rpc/async_slot.h"
#include "kv_cache_manager/client/src/internal/async_rpc/rpc_util.h"

namespace kv_cache_manager::async_rpc {
namespace {

struct GrpcCallState {
    grpc::ClientContext context;
    grpc::ByteBuffer response_buffer;
    grpc::Status status;
    std::unique_ptr<grpc::GenericClientAsyncResponseReader> reader;
    std::shared_ptr<AsyncSlot<bool>> slot;
    // Keeps the call state alive until the completion queue delivers the tag.
    std::shared_ptr<GrpcCallState> self;
};

} // namespace

struct GrpcClientRuntime::Impl {
    explicit Impl(uint32_t queue_count) {
        const uint32_t count = queue_count == 0 ? 1u : queue_count;
        for (uint32_t i = 0; i < count; ++i) {
            queues.push_back(std::make_unique<grpc::CompletionQueue>());
        }
        for (auto &queue : queues) {
            grpc::CompletionQueue *raw = queue.get();
            threads.emplace_back([raw]() {
                void *tag = nullptr;
                bool ok = false;
                while (raw->Next(&tag, &ok)) {
                    auto *state = static_cast<GrpcCallState *>(tag);
                    // Only signal here; deserialisation happens on the
                    // Executor after the awaiting coroutine resumes.
                    auto keep_alive = state->self;
                    state->self.reset();
                    state->slot->Complete(ok);
                }
            });
        }
    }

    ~Impl() { Stop(); }

    void Stop() {
        if (stopped) {
            return;
        }
        stopped = true;
        for (auto &queue : queues) {
            queue->Shutdown();
        }
        for (auto &thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    grpc::CompletionQueue *Next() {
        const size_t index = next.fetch_add(1, std::memory_order_relaxed) % queues.size();
        return queues[index].get();
    }

    std::vector<std::unique_ptr<grpc::CompletionQueue>> queues;
    std::vector<std::thread> threads;
    std::atomic<size_t> next{0};
    bool stopped = false;
};

GrpcClientRuntime::GrpcClientRuntime(uint32_t completion_queues)
    : impl_(std::make_unique<Impl>(completion_queues)) {}

GrpcClientRuntime::~GrpcClientRuntime() = default;

uint32_t GrpcClientRuntime::thread_count() const { return static_cast<uint32_t>(impl_->threads.size()); }

void GrpcClientRuntime::Shutdown() { impl_->Stop(); }

namespace {

class GrpcClient : public AsyncRpcClient {
public:
    GrpcClient(std::shared_ptr<GrpcClientRuntime> runtime,
               const ContinuationScheduler *scheduler,
               EndpointSet endpoints,
               ClientLimits limits)
        : runtime_(std::move(runtime))
        , scheduler_(scheduler)
        , endpoints_(std::move(endpoints))
        , limits_(limits)
        , meta_endpoint_(endpoints_.meta_grpc) {}

    ~GrpcClient() override { Shutdown(); }

    Task<RpcResult> Call(Api api,
                         const google::protobuf::Message &request,
                         google::protobuf::Message *response,
                         CallOptions options) override;

    void SetMetaEndpoint(const std::string &endpoint) override {
        std::lock_guard<std::mutex> lock(mutex_);
        meta_endpoint_ = endpoint;
    }

    std::string MetaEndpoint() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return meta_endpoint_;
    }

    ClientStats Stats() const override {
        ClientStats stats;
        stats.kind = TransportKind::kGrpc;
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &entry : channels_) {
            EndpointStats endpoint_stats;
            endpoint_stats.endpoint = entry.first;
            endpoint_stats.role = entry.second->role;
            endpoint_stats.channels = 1;
            endpoint_stats.connections_current = entry.second->created ? 1 : 0;
            endpoint_stats.connections_peak = entry.second->created ? 1 : 0;
            endpoint_stats.connections_created = entry.second->created ? 1 : 0;
            endpoint_stats.connections_reused = entry.second->reused;
            endpoint_stats.in_flight_current = entry.second->in_flight;
            endpoint_stats.in_flight_peak = entry.second->in_flight_peak;
            endpoint_stats.establish_latency_ms = entry.second->establish_latency_ms;
            stats.endpoints.push_back(std::move(endpoint_stats));
        }
        return stats;
    }

    void Shutdown() override {
        std::lock_guard<std::mutex> lock(mutex_);
        channels_.clear();
    }

private:
    struct ChannelEntry {
        std::string role;
        std::shared_ptr<grpc::Channel> channel;
        std::unique_ptr<grpc::GenericStub> stub;
        bool created = false;
        uint64_t reused = 0;
        uint64_t in_flight = 0;
        uint64_t in_flight_peak = 0;
        std::vector<double> establish_latency_ms;
    };

    ChannelEntry *ChannelFor(const std::string &target, const char *role);

    std::string ResolveEndpoint(ServiceEndpoint endpoint, const char **role) const {
        if (endpoint == ServiceEndpoint::kAdmin) {
            *role = "admin";
            return endpoints_.admin_grpc;
        }
        *role = "meta";
        std::lock_guard<std::mutex> lock(mutex_);
        return meta_endpoint_;
    }

    std::shared_ptr<GrpcClientRuntime> runtime_;
    const ContinuationScheduler *scheduler_;
    EndpointSet endpoints_;
    ClientLimits limits_;

    mutable std::mutex mutex_;
    std::string meta_endpoint_;
    std::map<std::string, std::unique_ptr<ChannelEntry>> channels_;
};

GrpcClient::ChannelEntry *GrpcClient::ChannelFor(const std::string &target, const char *role) {
    auto it = channels_.find(target);
    if (it != channels_.end()) {
        ++it->second->reused;
        return it->second.get();
    }
    const TimePoint start = Now();
    auto entry = std::make_unique<ChannelEntry>();
    entry->role = role;
    grpc::ChannelArguments args;
    args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);
    args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
    args.SetInt(GRPC_ARG_MAX_CONCURRENT_STREAMS, 10000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 20000);
    args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
    args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
    entry->channel = grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), args);
    entry->stub = std::make_unique<grpc::GenericStub>(entry->channel);
    entry->created = true;
    entry->establish_latency_ms.push_back(ToMillis(Now() - start));
    ChannelEntry *raw = entry.get();
    channels_.emplace(target, std::move(entry));
    return raw;
}

Task<RpcResult> GrpcClient::Call(Api api,
                                 const google::protobuf::Message &request,
                                 google::protobuf::Message *response,
                                 CallOptions options) {
    const ApiInfo &info = GetApiInfo(api);
    TimePoint deadline = options.deadline;
    if (deadline.time_since_epoch().count() == 0) {
        deadline = Now() + limits_.default_rpc_timeout;
    }

    RpcResult result;
    grpc::ByteBuffer request_buffer;
    bool own_buffer = false;
    const grpc::Status serialize_status = grpc::GenericSerialize<grpc::ProtoBufferWriter, google::protobuf::Message>(
        request, &request_buffer, &own_buffer);
    if (!serialize_status.ok()) {
        result.transport_error = TransportError::kEncode;
        result.raw_error = serialize_status.error_message();
        co_return result;
    }

    const char *role = "meta";
    const std::string target = ResolveEndpoint(info.endpoint, &role);
    auto state = std::make_shared<GrpcCallState>();
    state->slot = std::make_shared<AsyncSlot<bool>>(*scheduler_);
    const TimePoint submitted_at = Now();
    Duration remaining = deadline - submitted_at;
    if (remaining <= Duration::zero()) {
        remaining = std::chrono::milliseconds(1);
    }
    state->context.set_deadline(std::chrono::system_clock::now() +
                                std::chrono::duration_cast<std::chrono::milliseconds>(remaining));

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ChannelEntry *entry = ChannelFor(target, role);
        ++entry->in_flight;
        entry->in_flight_peak = std::max(entry->in_flight_peak, entry->in_flight);
        state->reader = entry->stub->PrepareUnaryCall(
            &state->context, std::string(info.grpc_method), request_buffer, runtime_->impl().Next());
    }

    if (!state->reader) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = channels_.find(target);
        if (it != channels_.end() && it->second->in_flight > 0) {
            --it->second->in_flight;
        }
        result.transport_error = TransportError::kConnect;
        result.raw_error = "gRPC call registration failed";
        result.rpc_latency = Now() - submitted_at;
        co_return result;
    }

    state->self = state;
    state->reader->StartCall();
    state->reader->Finish(&state->response_buffer, &state->status, state.get());

    {
        CancellationCallbackGuard guard(options.cancel, [state]() { state->context.TryCancel(); });
        co_await *state->slot;
    }
    result.rpc_latency = Now() - submitted_at;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = channels_.find(target);
        if (it != channels_.end() && it->second->in_flight > 0) {
            --it->second->in_flight;
        }
    }

    result.raw_status = static_cast<int>(state->status.error_code());
    if (!state->status.ok()) {
        result.raw_error = state->status.error_message();
        switch (state->status.error_code()) {
        case grpc::StatusCode::DEADLINE_EXCEEDED:
            result.transport_error = TransportError::kTimeout;
            break;
        case grpc::StatusCode::CANCELLED:
            result.transport_error =
                options.cancel.StopRequested() ? TransportError::kCancelled : TransportError::kDisconnect;
            break;
        case grpc::StatusCode::UNAVAILABLE:
            result.transport_error = TransportError::kConnect;
            break;
        case grpc::StatusCode::UNIMPLEMENTED:
            result.transport_error = TransportError::kUnsupported;
            break;
        default:
            result.transport_error = TransportError::kOther;
            break;
        }
        co_return result;
    }

    const grpc::Status deserialize_status =
        grpc::GenericDeserialize<grpc::ProtoBufferReader, google::protobuf::Message>(&state->response_buffer, response);
    if (!deserialize_status.ok()) {
        result.transport_error = TransportError::kDecode;
        result.raw_error = deserialize_status.error_message();
        co_return result;
    }

    ApplyServiceStatus(*response, &result);
    co_return result;
}

} // namespace

std::unique_ptr<AsyncRpcClient> MakeGrpcClient(std::shared_ptr<GrpcClientRuntime> runtime,
                                               const ContinuationScheduler *scheduler,
                                               EndpointSet endpoints,
                                               ClientLimits limits) {
    return std::make_unique<GrpcClient>(std::move(runtime), scheduler, std::move(endpoints), limits);
}

} // namespace kv_cache_manager::async_rpc
