// Admission control: two independent, bounded in-flight lanes.
//
//   business : lookup, BLOCK_ADD, Start/FinishWriteCache, BLOCK_DELETE
//   control  : heartbeat, leader discovery, health probe, drain/cleanup
//
// Waiting for a permit is asynchronous: a due operation suspends without
// occupying an Executor worker, its wait is measured, and there is no
// catch-up burst once permits free up. The control lane has its own reserved
// capacity so a saturated business lane can never starve it.
#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/runtime/clock.h"
#include "tools/kvcm_swarm/runtime/executor.h"
#include "tools/kvcm_swarm/runtime/stop_token.h"

namespace kvcm_swarm {

enum class TrafficLane {
    kBusiness,
    kControl
};

inline const char *TrafficLaneName(TrafficLane lane) { return lane == TrafficLane::kBusiness ? "business" : "control"; }

struct LaneStats {
    uint64_t acquired = 0;
    uint64_t immediate = 0;
    uint64_t waited = 0;
    uint64_t rejected = 0; // permit wait cancelled or timed out
    uint64_t in_flight = 0;
    uint64_t peak_in_flight = 0;
    uint64_t peak_wait_queue = 0;
    uint64_t wait_ns_total = 0;
    uint64_t wait_ns_max = 0;
};

class AdmissionController;

// Holds one lane permit for its lifetime.
class Permit {
public:
    Permit() = default;
    Permit(AdmissionController *controller, TrafficLane lane, Duration wait)
        : controller_(controller), lane_(lane), wait_(wait) {}
    Permit(Permit &&other) noexcept : controller_(other.controller_), lane_(other.lane_), wait_(other.wait_) {
        other.controller_ = nullptr;
    }
    Permit &operator=(Permit &&other) noexcept {
        if (this != &other) {
            Release();
            controller_ = other.controller_;
            lane_ = other.lane_;
            wait_ = other.wait_;
            other.controller_ = nullptr;
        }
        return *this;
    }
    Permit(const Permit &) = delete;
    Permit &operator=(const Permit &) = delete;
    ~Permit() { Release(); }

    bool valid() const { return controller_ != nullptr; }
    Duration wait() const { return wait_; }
    TrafficLane lane() const { return lane_; }

    void Release();

private:
    AdmissionController *controller_ = nullptr;
    TrafficLane lane_ = TrafficLane::kBusiness;
    Duration wait_{};
};

struct AdmissionLimits {
    uint32_t max_in_flight_business_rpcs = 4096;
    uint32_t max_in_flight_control_rpcs = 512;
    // A business permit wait above this threshold marks the generator
    // saturated. Control waits are reported but never used as a threshold.
    Duration business_permit_wait_threshold = std::chrono::seconds(1);
};

class AdmissionController {
public:
    AdmissionController(SwarmExecutor &executor, AdmissionLimits limits);

    // Suspends until a permit is available, the deadline passes, or the stop
    // token fires. An invalid Permit means no permit was granted.
    Task<Permit> Acquire(TrafficLane lane, TimePoint deadline, StopToken stop);

    // Non-blocking attempt, used by tests and by fast paths.
    Permit TryAcquire(TrafficLane lane);

    LaneStats Snapshot(TrafficLane lane) const;
    bool saturated() const;
    uint64_t saturation_events() const;
    const AdmissionLimits &limits() const { return limits_; }

    // Marks generator saturation for reasons outside the permit path (session
    // admission rejection, schedule lag, cache backpressure).
    void MarkSaturated(const std::string &reason);
    std::vector<std::string> saturation_reasons() const;

private:
    friend class Permit;

    struct Waiter {
        std::shared_ptr<AsyncSlot<bool>> slot;
        TimePoint enqueued_at;
    };

    struct Lane {
        uint32_t capacity = 0;
        LaneStats stats;
        std::deque<Waiter> waiters;
    };

    Lane &LaneRef(TrafficLane lane) { return lane == TrafficLane::kBusiness ? business_ : control_; }
    const Lane &LaneRef(TrafficLane lane) const { return lane == TrafficLane::kBusiness ? business_ : control_; }
    void ReleasePermit(TrafficLane lane);

    SwarmExecutor &executor_;
    AdmissionLimits limits_;
    mutable std::mutex mutex_;
    Lane business_;
    Lane control_;
    std::vector<std::string> saturation_reasons_;
    uint64_t saturation_events_ = 0;
};

} // namespace kvcm_swarm
