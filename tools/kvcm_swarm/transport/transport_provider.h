// Creates one transport context per logical client.
//
// Each V6dProcess owns its own context; HealthProbe owns another. Contexts
// never share sockets, but they all share the process-level reactor threads
// and gRPC completion queues: there is no per-context thread.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/transport/transport.h"

namespace kvcm_swarm {

class TransportProvider {
public:
    TransportProvider(SwarmExecutor &executor,
                      AdmissionController &admission,
                      EvidenceSink &evidence,
                      PhaseSource &phase,
                      EndpointSet endpoints,
                      TransportLimits limits,
                      uint32_t reactor_threads,
                      uint32_t grpc_completion_queues);
    ~TransportProvider();

    // The provider owns every context so transport statistics stay valid until
    // the report is written. Callers get a non-owning pointer.
    ClientTransportContext *CreateClientContext(ClientIdentity identity, TransportKind kind);

    // Number of OS threads dedicated to waiting for network events.
    uint32_t io_thread_count() const;
    const EndpointSet &endpoints() const { return endpoints_; }
    const TransportLimits &limits() const { return limits_; }

    std::vector<TransportContextStats> CollectStats() const;
    void Shutdown();

private:
    SwarmExecutor &executor_;
    AdmissionController &admission_;
    EvidenceSink &evidence_;
    PhaseSource &phase_;
    EndpointSet endpoints_;
    TransportLimits limits_;
    uint32_t reactor_threads_;
    uint32_t grpc_completion_queues_;

    std::unique_ptr<kv_cache_manager::async_rpc::AsyncRpcClientProvider> clients_;

    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<ClientTransportContext>> contexts_;
    bool shutdown_ = false;
};

// Rejects any endpoint that is not plaintext HTTP / insecure gRPC. TLS is not
// silently downgraded.
bool ValidateInsecureEndpoint(const std::string &endpoint, bool expect_http_scheme, std::string *error);

} // namespace kvcm_swarm
