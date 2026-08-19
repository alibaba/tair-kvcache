#include "tools/kvcm_swarm/transport/transport_provider.h"

#include "tools/kvcm_swarm/transport/call_recorder.h"

namespace kvcm_swarm {

namespace {

namespace rpc = kv_cache_manager::async_rpc;

rpc::RpcLane ToRpcLane(TrafficLane lane) {
    return lane == TrafficLane::kControl ? rpc::RpcLane::kControl : rpc::RpcLane::kBusiness;
}

class RecordedClientContext : public ClientTransportContext {
public:
    RecordedClientContext(ClientIdentity identity,
                          std::unique_ptr<rpc::AsyncRpcClient> client,
                          CallRecorder recorder,
                          Duration default_rpc_timeout)
        : identity_(std::move(identity))
        , client_(std::move(client))
        , recorder_(std::move(recorder))
        , default_rpc_timeout_(default_rpc_timeout) {}

    Task<RpcResult> Call(Api api,
                         const google::protobuf::Message &request,
                         google::protobuf::Message *response,
                         CallOptions options) override {
        const TimePoint planned_at = options.planned_at.time_since_epoch().count() == 0 ? Now() : options.planned_at;
        TimePoint deadline = options.deadline;
        if (deadline.time_since_epoch().count() == 0) {
            deadline = Now() + default_rpc_timeout_;
        }

        Permit permit = co_await recorder_.admission().Acquire(options.lane, deadline, options.stop);
        if (!permit.valid()) {
            RpcResult result;
            result.transport_error = TransportError::kNoPermit;
            result.raw_error = "no admission permit before deadline";
            recorder_.Record(api, options, recorder_.CurrentPhase(), result, planned_at, Duration::zero());
            co_return result;
        }

        const Phase phase = recorder_.CurrentPhase();
        rpc::CallOptions rpc_options;
        rpc_options.lane = ToRpcLane(options.lane);
        rpc_options.deadline = deadline;
        rpc_options.cancel = options.stop;
        RpcResult result = co_await client_->Call(api, request, response, std::move(rpc_options));
        recorder_.Record(api, options, phase, result, planned_at, permit.wait());
        co_return result;
    }

    void SetMetaEndpoint(const std::string &endpoint) override { client_->SetMetaEndpoint(endpoint); }
    std::string MetaEndpoint() const override { return client_->MetaEndpoint(); }

    TransportContextStats Stats() const override {
        const rpc::ClientStats source = client_->Stats();
        TransportContextStats stats;
        stats.identity = identity_;
        stats.kind = source.kind;
        for (const auto &source_endpoint : source.endpoints) {
            EndpointStats endpoint;
            endpoint.endpoint = source_endpoint.endpoint;
            endpoint.role = source_endpoint.role;
            endpoint.channels = source_endpoint.channels;
            endpoint.connections_current = source_endpoint.connections_current;
            endpoint.connections_peak = source_endpoint.connections_peak;
            endpoint.connections_created = source_endpoint.connections_created;
            endpoint.connections_reused = source_endpoint.connections_reused;
            endpoint.in_flight_current = source_endpoint.in_flight_current;
            endpoint.in_flight_peak = source_endpoint.in_flight_peak;
            for (const double latency : source_endpoint.establish_latency_ms) {
                endpoint.establish_latency_ms.Add(latency);
            }
            stats.endpoints.push_back(std::move(endpoint));
        }
        return stats;
    }

    void Shutdown() override { client_->Shutdown(); }

private:
    ClientIdentity identity_;
    std::unique_ptr<rpc::AsyncRpcClient> client_;
    CallRecorder recorder_;
    Duration default_rpc_timeout_;
};

} // namespace

TransportProvider::TransportProvider(SwarmExecutor &executor,
                                     AdmissionController &admission,
                                     EvidenceSink &evidence,
                                     PhaseSource &phase,
                                     EndpointSet endpoints,
                                     TransportLimits limits,
                                     uint32_t reactor_threads,
                                     uint32_t grpc_completion_queues)
    : executor_(executor)
    , admission_(admission)
    , evidence_(evidence)
    , phase_(phase)
    , endpoints_(std::move(endpoints))
    , limits_(limits)
    , reactor_threads_(reactor_threads == 0 ? 1u : reactor_threads)
    , grpc_completion_queues_(grpc_completion_queues == 0 ? 1u : grpc_completion_queues) {
    rpc::ContinuationScheduler scheduler;
    scheduler.schedule = [this](rpc::ContinuationScheduler::Callback callback) {
        return executor_.schedule(std::move(callback));
    };
    scheduler.schedule_at = [this](rpc::TimePoint when, rpc::ContinuationScheduler::Callback callback) {
        executor_.ScheduleAt(when, std::move(callback));
    };
    clients_ = std::make_unique<rpc::AsyncRpcClientProvider>(
        std::move(scheduler), endpoints_, limits_, reactor_threads_, grpc_completion_queues_);
}

TransportProvider::~TransportProvider() { Shutdown(); }

ClientTransportContext *TransportProvider::CreateClientContext(ClientIdentity identity, TransportKind kind) {
    CallRecorder recorder(std::move(identity), kind, executor_, admission_, evidence_, phase_);
    std::unique_ptr<rpc::AsyncRpcClient> client = clients_->Create(kind);
    if (!client) {
        return nullptr;
    }
    ClientIdentity context_identity = recorder.identity();
    std::unique_ptr<ClientTransportContext> context = std::make_unique<RecordedClientContext>(
        std::move(context_identity), std::move(client), std::move(recorder), limits_.default_rpc_timeout);
    ClientTransportContext *raw = context.get();
    std::lock_guard<std::mutex> lock(mutex_);
    contexts_.push_back(std::move(context));
    return raw;
}

uint32_t TransportProvider::io_thread_count() const {
    return clients_->io_thread_count();
}

std::vector<TransportContextStats> TransportProvider::CollectStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<TransportContextStats> stats;
    stats.reserve(contexts_.size());
    for (const auto &context : contexts_) {
        stats.push_back(context->Stats());
    }
    return stats;
}

void TransportProvider::Shutdown() {
    std::vector<std::unique_ptr<ClientTransportContext>> contexts;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_) {
            return;
        }
        shutdown_ = true;
        contexts.swap(contexts_);
    }
    for (auto &context : contexts) {
        context->Shutdown();
    }
    contexts.clear();
    clients_->Shutdown();
}

bool ValidateInsecureEndpoint(const std::string &endpoint, bool expect_http_scheme, std::string *error) {
    return rpc::ValidateInsecureEndpoint(endpoint, expect_http_scheme, error);
}

} // namespace kvcm_swarm
