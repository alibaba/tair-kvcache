// Run report: the stable JSON fact source plus the human-readable summary,
// both rendered from the same model.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/client_behavior.h"
#include "tools/kvcm_swarm/evidence/sink.h"
#include "tools/kvcm_swarm/scenario/config.h"
#include "tools/kvcm_swarm/transport/transport_provider.h"

namespace kvcm_swarm {

struct PhaseRecord {
    Phase phase = Phase::kValidate;
    TimePoint start{};
    TimePoint end{};
    bool entered = false;
};

struct PreflightReport {
    bool executed = false;
    bool passed = false;
    std::string failure_stage;
    std::string failure_detail;
    std::vector<std::pair<std::string, bool>> steps;
    uint64_t remove_cache_calls = 0;
    std::string temporary_instance_id;
    std::string temporary_host_ip_port;
    std::vector<std::string> cleanup_notes;
};

struct ResourceUsage {
    uint64_t threads = 0;
    uint64_t rss_bytes = 0;
    uint64_t peak_rss_bytes = 0;
    double user_cpu_seconds = 0.0;
    double system_cpu_seconds = 0.0;
    uint64_t open_sockets = 0;
};

ResourceUsage CollectResourceUsage();

struct RunReportInput {
    const ScenarioConfig *config = nullptr;
    const EvidenceSink *evidence = nullptr;
    const AdmissionController *admission = nullptr;
    const SwarmExecutor *executor = nullptr;
    const TransportProvider *transports = nullptr;
    const std::vector<ClientBehavior *> *behaviors = nullptr;
    const std::vector<PhaseRecord> *phases = nullptr;
    const PreflightReport *preflight = nullptr;
    int64_t started_wall_ms = 0;
    int64_t ended_wall_ms = 0;
    Duration total_duration{};
    std::string exit_reason;
    bool initialize_ok = false;
    bool drain_complete = false;
    bool quiesced = false;
    ResourceUsage resources;
};

std::string BuildRunReportJson(const RunReportInput &input);
std::string RenderRunSummary(const RunReportInput &input);

} // namespace kvcm_swarm
