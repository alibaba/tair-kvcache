// SessionManager: the logical owner of workload sessions.
//
// A session owns only its logical token history, class, turn plan and
// schedule. It never holds a process-local cache entry, a cross-turn lease, a
// reporter location owner or a cold allocation owner. Migration only changes
// which process the next turn selects.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tools/kvcm_swarm/clients/v6d/config.h"
#include "tools/kvcm_swarm/clients/v6d/process.h"
#include "tools/kvcm_swarm/clients/v6d/workload.h"
#include "tools/kvcm_swarm/evidence/histogram.h"

namespace kvcm_swarm {

using SessionId = uint64_t;

enum class TurnState {
    kIdle,
    kInFlight,
    kRetiring
};

struct SessionStats {
    uint64_t planned_arrivals = 0;
    uint64_t admitted = 0;
    uint64_t rejected_admission = 0;
    uint64_t active_current = 0;
    uint64_t active_peak = 0;
    uint64_t completed = 0;
    uint64_t aborted = 0;
    uint64_t turns_started = 0;
    uint64_t turns_completed = 0;
    uint64_t skipped_slots = 0;
    uint64_t migrations = 0;
    uint64_t affinity_retained = 0;
    uint64_t shared_prefix_sessions = 0;
    uint64_t no_ready_process = 0;
    Histogram turn_latency_ms;
    Histogram turn_lag_ms;
    Histogram session_lifetime_ms;
    std::vector<uint64_t> admitted_per_class;
};

// The deployment implements this to run the body of one turn.
class TurnRunner {
public:
    virtual ~TurnRunner() = default;
    // Ready processes, in stable index order.
    virtual std::vector<uint32_t> ReadyProcesses() const = 0;
    // Executes one turn. Returns true when the turn completed its planned work
    // (successfully or with observed RPC failures); false only for an
    // unrecoverable local inconsistency, which aborts the session.
    virtual Task<bool>
    RunTurn(SessionId session_id, uint32_t process_index, SessionWorkload &workload, TimePoint deadline) = 0;
};

class SessionManager {
public:
    SessionManager(V6dDeploymentContext &deployment, TurnRunner &runner);

    void Start();
    // steady end: no new session is admitted and no new turn is started.
    void StopAdmission();
    // Waits until every in-flight turn has finished or the deadline passes.
    Task<> DrainTurns(TimePoint deadline);

    SessionStats Stats() const;
    bool quiesced() const {
        return in_flight_turns_.load(std::memory_order_acquire) == 0 &&
               arrival_running_.load(std::memory_order_acquire) == 0;
    }
    uint64_t max_root_tokens() const { return prefix_pool_.max_root_tokens(); }

private:
    struct SessionState {
        SessionId id = 0;
        size_t class_index = 0;
        SessionWorkload workload;
        bool has_last_process = false;
        uint32_t last_process = 0;
        TimePoint created_at{};
        TimePoint next_turn{};
        TurnState state = TurnState::kIdle;
        uint64_t turn_generation = 0;
        uint32_t remaining_turns = 0;
        uint32_t completed_turns = 0;
        // Independent sub-streams: timing, token content, shape and routing.
        Rng timing_rng;
        Rng content_rng;
        Rng shape_rng;
        Rng routing_rng;
    };

    Task<> ArrivalLoop();
    Task<> TurnEntry(SessionId session_id, uint64_t generation, TimePoint planned);
    void ScheduleTurn(SessionId session_id, uint64_t generation, TimePoint when);
    size_t PickClass(Rng &rng) const;
    uint32_t PickProcess(SessionState &session, const std::vector<uint32_t> &ready);
    void RetireLocked(SessionState &session);

    V6dDeploymentContext &deployment_;
    TurnRunner &runner_;
    SharedPrefixPoolState prefix_pool_;
    double total_weight_ = 0.0;

    mutable std::mutex mutex_;
    std::unordered_map<SessionId, std::unique_ptr<SessionState>> sessions_;
    SessionStats stats_;
    SessionId next_session_id_ = 1;
    bool admission_open_ = true;

    std::atomic<uint64_t> in_flight_turns_{0};
    std::atomic<uint32_t> arrival_running_{0};
    StopSource arrival_stop_;
};

} // namespace kvcm_swarm
