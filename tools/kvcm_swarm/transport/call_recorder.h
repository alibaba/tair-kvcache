// Shared plumbing for the HTTP and gRPC contexts: permit admission,
// submit-time phase attribution, latency measurement and evidence recording.
#pragma once

#include <google/protobuf/message.h>
#include <string>

#include "tools/kvcm_swarm/protocol/api.h"
#include "tools/kvcm_swarm/transport/transport.h"

namespace kvcm_swarm {

class CallRecorder {
public:
    CallRecorder(ClientIdentity identity,
                 TransportKind kind,
                 SwarmExecutor &executor,
                 AdmissionController &admission,
                 EvidenceSink &evidence,
                 PhaseSource &phase)
        : identity_(std::move(identity))
        , kind_(kind)
        , executor_(executor)
        , admission_(admission)
        , evidence_(evidence)
        , phase_(phase) {}

    const ClientIdentity &identity() const { return identity_; }
    TransportKind kind() const { return kind_; }
    SwarmExecutor &executor() const { return executor_; }
    AdmissionController &admission() const { return admission_; }

    // Publishes one RPC observation. `submitted_at` fixes the phase.
    void Record(Api api,
                const CallOptions &options,
                Phase phase,
                const RpcResult &result,
                TimePoint planned_at,
                Duration permit_wait) const {
        RpcObservation observation;
        observation.behavior_type = identity_.behavior_type;
        observation.behavior_id = identity_.behavior_id;
        observation.process_id = identity_.process_id;
        observation.api = std::string(ApiName(api));
        observation.phase = phase;
        observation.lane = options.lane;
        observation.result = result;
        observation.permit_wait = permit_wait;
        observation.queue_delay = Duration::zero();
        if (planned_at.time_since_epoch().count() != 0) {
            const Duration delay = Now() - result.rpc_latency - planned_at;
            observation.queue_delay = delay.count() > 0 ? delay : Duration::zero();
        }
        evidence_.RecordRpc(observation);
    }

    Phase CurrentPhase() const { return phase_.Current(); }

private:
    ClientIdentity identity_;
    TransportKind kind_;
    SwarmExecutor &executor_;
    AdmissionController &admission_;
    EvidenceSink &evidence_;
    PhaseSource &phase_;
};

} // namespace kvcm_swarm
