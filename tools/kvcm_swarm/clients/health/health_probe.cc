#include "tools/kvcm_swarm/clients/health/health_probe.h"

#include <algorithm>

#include "tools/kvcm_swarm/evidence/json_writer.h"
#include "tools/kvcm_swarm/protocol/proto_alias.h"
#include "tools/kvcm_swarm/scenario/duration.h"

namespace kvcm_swarm {
namespace {
constexpr const char *kCheckName = "C5_health_probe_bounded_response";
} // namespace

HealthProbe::HealthProbe(BehaviorSpec spec, HealthProbeConfig config, RuntimeServices services)
    : spec_(std::move(spec)), config_(config), services_(services) {}

HealthProbe::~HealthProbe() = default;

Task<bool> HealthProbe::Initialize(TimePoint /*deadline*/) {
    ClientIdentity identity;
    identity.behavior_type = std::string(TypeName());
    identity.behavior_id = spec_.id;
    transport_ = services_.transports.CreateClientContext(identity, spec_.transport);
    co_return transport_ != nullptr;
}

void HealthProbe::StartTraffic() {
    for (uint32_t stream = 0; stream < config_.streams; ++stream) {
        active_streams_.fetch_add(1, std::memory_order_release);
        ProbeLoop(stream).via(&services_.executor).start([](auto &&) {});
    }
}

Task<> HealthProbe::ProbeLoop(uint32_t stream_index) {
    // Independent per-stream clock: the schedule advances from the previous
    // planned time so a slow probe cannot make the cadence drift.
    Rng jitter = services_.seeds.MakeRng("health_probe/" + spec_.id + "/stream", stream_index);
    TimePoint planned =
        Now() + Duration(static_cast<int64_t>(jitter.NextDouble() * static_cast<double>(config_.interval.count())));
    while (!own_stop_.StopRequested() && !services_.stop.StopRequested()) {
        const bool slept = co_await SleepUntil(services_.executor, planned, own_stop_.Token());
        if (!slept) {
            break;
        }
        admin::CheckHealthRequest request;
        request.set_trace_id("swarm-health-" + spec_.id + "-" + std::to_string(stream_index) + "-" +
                             std::to_string(planned.time_since_epoch().count()));
        admin::CheckHealthResponse response;
        CallOptions options;
        options.lane = TrafficLane::kControl;
        options.planned_at = planned;
        options.deadline = Now() + config_.probe_deadline;
        options.stop = services_.stop;
        const RpcResult result = co_await transport_->Call(Api::kCheckHealth, request, &response, options);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++stats_.probes;
            stats_.latency_ms.Add(ToMillis(result.rpc_latency));
            if (result.ok) {
                ++stats_.responded;
                if (!response.is_health()) {
                    ++stats_.reported_unhealthy;
                }
                if (response.is_leader()) {
                    ++stats_.leader_observations;
                }
            } else {
                ++stats_.failed;
            }
            if (result.rpc_latency > config_.probe_deadline || !result.ok) {
                ++stats_.deadline_exceeded;
            }
        }
        if (!result.ok || result.rpc_latency > config_.probe_deadline) {
            JsonWriter writer(false);
            writer.BeginObject();
            writer.KeyString("behavior_id", spec_.id);
            writer.KeyUint("stream", stream_index);
            writer.KeyDouble("latency_ms", ToMillis(result.rpc_latency));
            writer.KeyDouble("deadline_ms", ToMillis(config_.probe_deadline));
            writer.KeyString("transport_error", TransportErrorName(result.transport_error));
            writer.KeyInt("service_status", result.service_status);
            writer.KeyBool("is_health", response.is_health());
            writer.KeyString("raw_error", result.raw_error);
            writer.EndObject();
            services_.evidence.violations().Record(kCheckName, writer.Take());
        }

        // Advance from the planned time, never from the completion time.
        planned += config_.interval;
        const TimePoint now = Now();
        while (planned <= now) {
            planned += config_.interval;
        }
    }
    active_streams_.fetch_sub(1, std::memory_order_release);
    co_return;
}

Task<> HealthProbe::Drain(TimePoint deadline) {
    // Drain must be repeatable: the second call is a no-op.
    if (draining_.exchange(true, std::memory_order_acq_rel)) {
        co_return;
    }
    // Probing continues through drain until the global drain deadline, then
    // this behavior stops its own loops.
    co_await SleepUntil(services_.executor, deadline, services_.stop);
    own_stop_.RequestStop();
    co_return;
}

void HealthProbe::WriteReport(JsonWriter &writer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    writer.BeginObject();
    writer.KeyString("type", std::string(TypeName()));
    writer.KeyUint("streams", config_.streams);
    writer.KeyUint("probes", stats_.probes);
    writer.KeyUint("responded", stats_.responded);
    writer.KeyUint("failed", stats_.failed);
    writer.KeyUint("deadline_exceeded", stats_.deadline_exceeded);
    writer.KeyUint("reported_unhealthy", stats_.reported_unhealthy);
    writer.KeyUint("leader_observations", stats_.leader_observations);
    writer.Key("latency_ms");
    writer.BeginObject();
    writer.KeyUint("count", stats_.latency_ms.count());
    writer.KeyDouble("mean", stats_.latency_ms.mean_ms());
    writer.KeyDouble("p50", stats_.latency_ms.Quantile(0.5));
    writer.KeyDouble("p99", stats_.latency_ms.Quantile(0.99));
    writer.KeyDouble("max", stats_.latency_ms.max_ms());
    writer.EndObject();
    writer.EndObject();
}

void HealthProbe::WriteEffectiveConfig(JsonWriter &writer) const { writer.RawValue(config_.ToJsonString()); }

std::vector<InvariantObservation> HealthProbe::Invariants() const {
    std::lock_guard<std::mutex> lock(mutex_);
    InvariantObservation observation;
    observation.behavior_type = std::string(TypeName());
    observation.check_name = kCheckName;
    observation.checked = stats_.probes;
    observation.violations = stats_.deadline_exceeded;
    observation.counters["probes"] = static_cast<int64_t>(stats_.probes);
    observation.counters["responded"] = static_cast<int64_t>(stats_.responded);
    observation.counters["failed"] = static_cast<int64_t>(stats_.failed);
    observation.counters["deadline_exceeded"] = static_cast<int64_t>(stats_.deadline_exceeded);
    observation.counters["deadline_ms"] = static_cast<int64_t>(ToMillis(config_.probe_deadline));
    if (stats_.probes == 0) {
        observation.status = CheckStatus::kNotRun;
        observation.reason = "no CheckHealth probe completed";
    } else if (stats_.deadline_exceeded > 0) {
        observation.status = CheckStatus::kFail;
        observation.reason = "CheckHealth failed or exceeded the configured deadline";
    } else {
        observation.status = CheckStatus::kPass;
        observation.reason = "every CheckHealth answered within the configured deadline";
    }
    observation.detail_preview = services_.evidence.violations().Preview(kCheckName);
    return {observation};
}

namespace {

class HealthProbeFactory : public BehaviorFactory {
public:
    std::string_view TypeName() const override { return "health_probe"; }

    ValidationResult Validate(const BehaviorSpec &spec) const override {
        ValidationResult result;
        HealthProbeConfig config;
        std::vector<std::string> errors;
        ParseHealthProbeConfig(spec, &config, &errors);
        for (auto &error : errors) {
            result.Fail("behaviors[" + spec.id + "]: " + error);
        }
        return result;
    }

    std::unique_ptr<ClientBehavior> Create(const BehaviorSpec &spec, RuntimeServices services) const override {
        HealthProbeConfig config;
        std::vector<std::string> errors;
        if (!ParseHealthProbeConfig(spec, &config, &errors)) {
            return nullptr;
        }
        return std::make_unique<HealthProbe>(spec, config, services);
    }
};

} // namespace

std::unique_ptr<BehaviorFactory> MakeHealthProbeFactory() { return std::make_unique<HealthProbeFactory>(); }

} // namespace kvcm_swarm
