#include "tools/kvcm_swarm/runtime/admission.h"

#include <algorithm>

namespace kvcm_swarm {

void Permit::Release() {
    if (controller_ != nullptr) {
        controller_->ReleasePermit(lane_);
        controller_ = nullptr;
    }
}

AdmissionController::AdmissionController(SwarmExecutor &executor, AdmissionLimits limits)
    : executor_(executor), limits_(limits) {
    business_.capacity = limits_.max_in_flight_business_rpcs;
    control_.capacity = limits_.max_in_flight_control_rpcs;
}

Permit AdmissionController::TryAcquire(TrafficLane lane) {
    std::lock_guard<std::mutex> lock(mutex_);
    Lane &l = LaneRef(lane);
    if (l.stats.in_flight >= l.capacity) {
        return Permit();
    }
    ++l.stats.in_flight;
    ++l.stats.acquired;
    ++l.stats.immediate;
    l.stats.peak_in_flight = std::max(l.stats.peak_in_flight, l.stats.in_flight);
    return Permit(this, lane, Duration::zero());
}

Task<Permit> AdmissionController::Acquire(TrafficLane lane, TimePoint deadline, StopToken stop) {
    std::shared_ptr<AsyncSlot<bool>> slot;
    const TimePoint enqueued_at = Now();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Lane &l = LaneRef(lane);
        if (l.stats.in_flight < l.capacity && l.waiters.empty()) {
            ++l.stats.in_flight;
            ++l.stats.acquired;
            ++l.stats.immediate;
            l.stats.peak_in_flight = std::max(l.stats.peak_in_flight, l.stats.in_flight);
            co_return Permit(this, lane, Duration::zero());
        }
        slot = std::make_shared<AsyncSlot<bool>>(executor_);
        l.waiters.push_back(Waiter{slot, enqueued_at});
        l.stats.peak_wait_queue = std::max<uint64_t>(l.stats.peak_wait_queue, l.waiters.size());
    }

    // The timer and the stop token race the permit hand-off; whoever completes
    // the slot first wins and the operation is resumed on the Executor.
    executor_.ScheduleAt(deadline, [slot]() { slot->Complete(false); });
    StopCallbackGuard guard(stop, [slot]() { slot->Complete(false); });
    const bool granted = co_await *slot;
    const Duration waited = Now() - enqueued_at;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        Lane &l = LaneRef(lane);
        if (!granted) {
            for (auto it = l.waiters.begin(); it != l.waiters.end(); ++it) {
                if (it->slot == slot) {
                    l.waiters.erase(it);
                    break;
                }
            }
            ++l.stats.rejected;
            if (lane == TrafficLane::kBusiness) {
                ++saturation_events_;
                if (saturation_reasons_.size() < 32) {
                    saturation_reasons_.emplace_back("business_permit_wait_timeout");
                }
            }
            co_return Permit();
        }
        ++l.stats.acquired;
        ++l.stats.waited;
        l.stats.peak_in_flight = std::max(l.stats.peak_in_flight, l.stats.in_flight);
        const uint64_t waited_ns = static_cast<uint64_t>(std::max<int64_t>(0, waited.count()));
        l.stats.wait_ns_total += waited_ns;
        l.stats.wait_ns_max = std::max(l.stats.wait_ns_max, waited_ns);
        if (lane == TrafficLane::kBusiness && waited >= limits_.business_permit_wait_threshold) {
            ++saturation_events_;
            if (saturation_reasons_.size() < 32) {
                saturation_reasons_.emplace_back("business_permit_wait_over_threshold");
            }
        }
    }
    co_return Permit(this, lane, waited);
}

void AdmissionController::ReleasePermit(TrafficLane lane) {
    std::shared_ptr<AsyncSlot<bool>> next;
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            Lane &l = LaneRef(lane);
            if (l.waiters.empty()) {
                if (l.stats.in_flight > 0) {
                    --l.stats.in_flight;
                }
                return;
            }
            // Hand the in-flight slot straight to the oldest waiter (FIFO, no
            // catch-up burst): in_flight stays reserved for the new owner.
            next = l.waiters.front().slot;
            l.waiters.pop_front();
        }
        if (next->Complete(true)) {
            return;
        }
        // The waiter had already been cancelled; retry with the next one.
        next.reset();
    }
}

LaneStats AdmissionController::Snapshot(TrafficLane lane) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return LaneRef(lane).stats;
}

bool AdmissionController::saturated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return saturation_events_ > 0;
}

uint64_t AdmissionController::saturation_events() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return saturation_events_;
}

void AdmissionController::MarkSaturated(const std::string &reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++saturation_events_;
    if (std::find(saturation_reasons_.begin(), saturation_reasons_.end(), reason) == saturation_reasons_.end() &&
        saturation_reasons_.size() < 32) {
        saturation_reasons_.push_back(reason);
    }
}

std::vector<std::string> AdmissionController::saturation_reasons() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> unique;
    for (const auto &reason : saturation_reasons_) {
        if (std::find(unique.begin(), unique.end(), reason) == unique.end()) {
            unique.push_back(reason);
        }
    }
    return unique;
}

} // namespace kvcm_swarm
