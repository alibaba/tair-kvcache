// RunCoordinator: the single owner of phase transitions, the stop source and
// the final report.
//
// local validation -> preflight -> initialize -> warmup -> steady -> drain ->
// report. warmup to steady only changes the phase: sessions, caches,
// connections, histograms and RNG streams are never rebuilt.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/registry.h"
#include "tools/kvcm_swarm/evidence/report.h"
#include "tools/kvcm_swarm/scenario/config.h"

namespace kvcm_swarm {

enum class ExitCode : int {
    kOk = 0,
    kConfigInvalid = 2,
    kPreflightFailed = 3,
    kInitializeFailed = 4,
    kReportFailed = 5,
};

class RunCoordinator {
public:
    RunCoordinator(ScenarioConfig config, const BehaviorRegistry &registry);
    ~RunCoordinator();

    ExitCode Run();

private:
    void EnterPhase(Phase phase);
    void LeaveCurrentPhase();
    bool WriteOutputs(const RunReportInput &input);

    ScenarioConfig config_;
    const BehaviorRegistry &registry_;
    std::vector<PhaseRecord> phases_;
    size_t current_phase_ = 0;
    bool has_current_phase_ = false;
};

} // namespace kvcm_swarm
