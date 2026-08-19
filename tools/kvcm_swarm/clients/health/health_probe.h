// health_probe: an independent liveness prober that only calls CheckHealth.
//
// It owns its own transport context, so it never shares business connections
// with V6D, and it uses the control lane so a saturated business lane cannot
// delay or starve it.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "tools/kvcm_swarm/clients/client_behavior.h"
#include "tools/kvcm_swarm/clients/health/config.h"
#include "tools/kvcm_swarm/evidence/histogram.h"

namespace kvcm_swarm {

class HealthProbe : public ClientBehavior {
public:
    HealthProbe(BehaviorSpec spec, HealthProbeConfig config, RuntimeServices services);
    ~HealthProbe() override;

    Task<bool> Initialize(TimePoint deadline) override;
    void StartTraffic() override;
    Task<> Drain(TimePoint deadline) override;

    std::string_view TypeName() const override { return "health_probe"; }
    const std::string &Id() const override { return spec_.id; }

    void WriteReport(JsonWriter &writer) const override;
    void WriteEffectiveConfig(JsonWriter &writer) const override;
    std::vector<InvariantObservation> Invariants() const override;
    bool Quiesced() const override { return active_streams_.load(std::memory_order_acquire) == 0; }

private:
    struct Stats {
        uint64_t probes = 0;
        uint64_t responded = 0;
        uint64_t failed = 0;
        uint64_t deadline_exceeded = 0;
        uint64_t reported_unhealthy = 0;
        uint64_t leader_observations = 0;
        Histogram latency_ms;
    };

    Task<> ProbeLoop(uint32_t stream_index);

    BehaviorSpec spec_;
    HealthProbeConfig config_;
    RuntimeServices services_;
    StopSource own_stop_;
    ClientTransportContext *transport_ = nullptr;

    mutable std::mutex mutex_;
    Stats stats_;
    std::atomic<uint32_t> active_streams_{0};
    std::atomic<bool> draining_{false};
};

std::unique_ptr<BehaviorFactory> MakeHealthProbeFactory();

} // namespace kvcm_swarm
