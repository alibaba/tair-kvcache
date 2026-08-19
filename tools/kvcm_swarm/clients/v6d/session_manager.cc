#include "tools/kvcm_swarm/clients/v6d/session_manager.h"

#include <algorithm>

namespace kvcm_swarm {

SessionManager::SessionManager(V6dDeploymentContext &deployment, TurnRunner &runner)
    : deployment_(deployment)
    , runner_(runner)
    , prefix_pool_(deployment.config().shared_prefix_pool, deployment.services().seeds, deployment.behavior_id()) {
    for (const auto &session_class : deployment.config().session_classes) {
        total_weight_ += session_class.weight;
    }
    stats_.admitted_per_class.assign(deployment.config().session_classes.size(), 0);
}

void SessionManager::Start() {
    arrival_running_.fetch_add(1, std::memory_order_release);
    ArrivalLoop().via(&deployment_.services().executor).start([](auto &&) {});
}

void SessionManager::StopAdmission() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        admission_open_ = false;
    }
    arrival_stop_.RequestStop();
}

size_t SessionManager::PickClass(Rng &rng) const {
    const auto &classes = deployment_.config().session_classes;
    if (classes.size() == 1) {
        return 0;
    }
    const double roll = rng.NextDouble() * total_weight_;
    double cumulative = 0.0;
    for (size_t i = 0; i < classes.size(); ++i) {
        cumulative += classes[i].weight;
        if (roll < cumulative) {
            return i;
        }
    }
    return classes.size() - 1;
}

Task<> SessionManager::ArrivalLoop() {
    const V6dConfig &config = deployment_.config();
    // The arrival rate is the total rate for the whole deployment, not a
    // per-process rate. The timeline derives from the seed and never reads
    // active session counts, RPC latency or completion order.
    Rng arrival_rng = deployment_.services().seeds.MakeRng("v6d/" + deployment_.behavior_id() + "/arrival");
    Rng class_rng = deployment_.services().seeds.MakeRng("v6d/" + deployment_.behavior_id() + "/session_class");
    TimePoint planned = Now();
    while (!arrival_stop_.StopRequested() && !deployment_.services().stop.StopRequested()) {
        double gap_seconds = 0.0;
        if (config.arrival_mode == ArrivalMode::kEven) {
            gap_seconds = 1.0 / config.session_arrival_rate;
        } else {
            gap_seconds = arrival_rng.NextExponential(config.session_arrival_rate);
        }
        planned += std::chrono::duration_cast<Duration>(std::chrono::duration<double>(gap_seconds));
        const bool slept = co_await SleepUntil(deployment_.services().executor, planned, arrival_stop_.Token());
        if (!slept) {
            break;
        }

        SessionId session_id = 0;
        size_t class_index = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.planned_arrivals;
            if (!admission_open_) {
                break;
            }
            if (sessions_.size() >= config.max_active_sessions) {
                // A planned arrival that hits the resource limit is rejected
                // immediately; it is not deferred and the arrival timeline is
                // unchanged.
                ++stats_.rejected_admission;
                deployment_.services().admission.MarkSaturated("session_admission_rejected");
                continue;
            }
            session_id = next_session_id_++;
            class_index = PickClass(class_rng);
            const auto &session_class = config.session_classes[class_index];
            auto session = std::make_unique<SessionState>();
            session->id = session_id;
            session->class_index = class_index;
            session->created_at = planned;
            session->next_turn = planned;
            session->timing_rng = deployment_.services().seeds.MakeRng(
                "v6d/" + deployment_.behavior_id() + "/session/timing", session_id);
            session->content_rng = deployment_.services().seeds.MakeRng(
                "v6d/" + deployment_.behavior_id() + "/session/content", session_id);
            session->shape_rng =
                deployment_.services().seeds.MakeRng("v6d/" + deployment_.behavior_id() + "/session/shape", session_id);
            session->routing_rng = deployment_.services().seeds.MakeRng(
                "v6d/" + deployment_.behavior_id() + "/session/routing", session_id);
            session->remaining_turns = static_cast<uint32_t>(Sample(session_class.turns, session->shape_rng));
            const bool use_shared_prefix = session->shape_rng.NextDouble() < session_class.shared_prefix_probability;
            session->workload.Init(session_class,
                                   config.groups,
                                   prefix_pool_,
                                   use_shared_prefix,
                                   session->content_rng,
                                   session->shape_rng);
            if (session->workload.used_shared_prefix()) {
                ++stats_.shared_prefix_sessions;
            }
            ++stats_.admitted;
            if (class_index < stats_.admitted_per_class.size()) {
                ++stats_.admitted_per_class[class_index];
            }
            sessions_.emplace(session_id, std::move(session));
            stats_.active_current = sessions_.size();
            stats_.active_peak = std::max<uint64_t>(stats_.active_peak, sessions_.size());
        }
        ScheduleTurn(session_id, 0, planned);
    }
    arrival_running_.fetch_sub(1, std::memory_order_release);
    co_return;
}

void SessionManager::ScheduleTurn(SessionId session_id, uint64_t generation, TimePoint when) {
    deployment_.services().executor.ScheduleAt(when, [this, session_id, generation, when]() {
        TurnEntry(session_id, generation, when).via(&deployment_.services().executor).start([](auto &&) {});
    });
}

uint32_t SessionManager::PickProcess(SessionState &session, const std::vector<uint32_t> &ready) {
    if (!session.has_last_process) {
        // The first turn picks uniformly among ready processes.
        const uint32_t choice = ready[session.routing_rng.NextInRange(0, ready.size() - 1)];
        return choice;
    }
    const double roll = session.routing_rng.NextDouble();
    if (roll < deployment_.config().session_affinity) {
        const auto it = std::find(ready.begin(), ready.end(), session.last_process);
        if (it != ready.end()) {
            ++stats_.affinity_retained;
            return session.last_process;
        }
    }
    // Otherwise choose from the other ready processes.
    std::vector<uint32_t> others;
    others.reserve(ready.size());
    for (const uint32_t index : ready) {
        if (index != session.last_process) {
            others.push_back(index);
        }
    }
    if (others.empty()) {
        ++stats_.affinity_retained;
        return session.last_process;
    }
    ++stats_.migrations;
    return others[session.routing_rng.NextInRange(0, others.size() - 1)];
}

Task<> SessionManager::TurnEntry(SessionId session_id, uint64_t generation, TimePoint planned) {
    const V6dConfig &config = deployment_.config();
    const std::vector<uint32_t> ready = runner_.ReadyProcesses();

    SessionWorkload *workload = nullptr;
    uint32_t process_index = 0;
    TimePoint deadline{};
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            co_return;
        }
        SessionState &session = *it->second;
        if (session.turn_generation != generation) {
            co_return;
        }
        if (!admission_open_) {
            // steady has ended: no new turn starts. Already running turns are
            // finished by DrainTurns.
            if (session.state == TurnState::kIdle) {
                RetireLocked(session);
                sessions_.erase(it);
                stats_.active_current = sessions_.size();
            }
            co_return;
        }
        if (session.state != TurnState::kIdle) {
            // The previous turn is still running: the expired slot is recorded
            // as skipped, does not consume a logical turn and is not burst
            // caught up later.
            ++stats_.skipped_slots;
            TimePoint next = session.next_turn;
            const auto &session_class = config.session_classes[session.class_index];
            do {
                next += Sample(session_class.turn_interval, session.timing_rng);
            } while (next <= Now());
            session.next_turn = next;
            ScheduleTurn(session_id, session.turn_generation, next);
            co_return;
        }
        if (ready.empty()) {
            ++stats_.no_ready_process;
            co_return;
        }
        process_index = PickProcess(session, ready);
        session.has_last_process = true;
        session.last_process = process_index;
        session.state = TurnState::kInFlight;
        ++session.turn_generation;
        ++stats_.turns_started;
        stats_.turn_lag_ms.Add(ToMillis(Now() - planned));
        // A turn advances the logical history exactly once, before its
        // operations run; the proposal is committed at the end of the turn.
        session.workload.ApplyTurn(config.session_classes[session.class_index], session.content_rng, session.shape_rng);
        workload = &session.workload;
        deadline = Now() + config.turn_deadline;
    }

    in_flight_turns_.fetch_add(1, std::memory_order_release);
    const TimePoint start = Now();
    bool ok = false;
    ok = co_await runner_.RunTurn(session_id, process_index, *workload, deadline);
    const Duration latency = Now() - start;
    in_flight_turns_.fetch_sub(1, std::memory_order_release);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = sessions_.find(session_id);
        if (it == sessions_.end()) {
            co_return;
        }
        SessionState &session = *it->second;
        session.state = TurnState::kIdle;
        stats_.turn_latency_ms.Add(ToMillis(latency));
        ++stats_.turns_completed;
        ++session.completed_turns;
        if (session.remaining_turns > 0) {
            --session.remaining_turns;
        }
        if (!ok) {
            ++stats_.aborted;
            RetireLocked(session);
            sessions_.erase(it);
            stats_.active_current = sessions_.size();
            co_return;
        }
        if (session.remaining_turns == 0) {
            // The only normal termination condition is running out of turns.
            ++stats_.completed;
            RetireLocked(session);
            sessions_.erase(it);
            stats_.active_current = sessions_.size();
            co_return;
        }
        if (!admission_open_) {
            RetireLocked(session);
            sessions_.erase(it);
            stats_.active_current = sessions_.size();
            co_return;
        }
        const auto &session_class = config.session_classes[session.class_index];
        TimePoint next = session.next_turn + Sample(session_class.turn_interval, session.timing_rng);
        while (next <= Now()) {
            ++stats_.skipped_slots;
            next += Sample(session_class.turn_interval, session.timing_rng);
        }
        session.next_turn = next;
        ScheduleTurn(session_id, session.turn_generation, next);
    }
    co_return;
}

void SessionManager::RetireLocked(SessionState &session) {
    // Retirement destroys only logical history and schedule state: it never
    // clears a process cache, triggers BLOCK_DELETE or creates a replacement.
    stats_.session_lifetime_ms.Add(ToMillis(Now() - session.created_at));
}

Task<> SessionManager::DrainTurns(TimePoint deadline) {
    while (in_flight_turns_.load(std::memory_order_acquire) > 0 && Now() < deadline) {
        co_await SleepFor(deployment_.services().executor, std::chrono::milliseconds(5), StopToken());
    }
    co_return;
}

SessionStats SessionManager::Stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    SessionStats stats = stats_;
    stats.active_current = sessions_.size();
    return stats;
}

} // namespace kvcm_swarm
