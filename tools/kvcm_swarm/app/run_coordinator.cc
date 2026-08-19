#include "tools/kvcm_swarm/app/run_coordinator.h"

#include <cstdio>
#include <iostream>
#include <thread>

#include "async_simple/coro/SyncAwait.h"
#include "tools/kvcm_swarm/app/preflight.h"
#include "tools/kvcm_swarm/evidence/json_writer.h"
#include "tools/kvcm_swarm/runtime/executor.h"

namespace kvcm_swarm {
namespace {

bool WriteFile(const std::string &path, const std::string &content) {
    std::FILE *file = std::fopen(path.c_str(), "w");
    if (file == nullptr) {
        return false;
    }
    const size_t written = std::fwrite(content.data(), 1, content.size(), file);
    const bool flushed = std::fflush(file) == 0;
    std::fclose(file);
    return written == content.size() && flushed;
}

} // namespace

RunCoordinator::RunCoordinator(ScenarioConfig config, const BehaviorRegistry &registry)
    : config_(std::move(config)), registry_(registry) {}

RunCoordinator::~RunCoordinator() = default;

void RunCoordinator::EnterPhase(Phase phase) {
    LeaveCurrentPhase();
    PhaseRecord record;
    record.phase = phase;
    record.start = Now();
    record.end = record.start;
    record.entered = true;
    phases_.push_back(record);
    current_phase_ = phases_.size() - 1;
    has_current_phase_ = true;
}

void RunCoordinator::LeaveCurrentPhase() {
    if (has_current_phase_) {
        phases_[current_phase_].end = Now();
    }
}

bool RunCoordinator::WriteOutputs(const RunReportInput &input) {
    const std::string json = BuildRunReportJson(input);
    if (json.empty()) {
        return false;
    }
    if (!WriteFile(config_.evidence.output_json, json)) {
        std::cerr << "kvcm_swarm: failed to write report to " << config_.evidence.output_json << "\n";
        return false;
    }
    const std::string summary = RenderRunSummary(input);
    if (!config_.evidence.markdown_summary.empty() && !WriteFile(config_.evidence.markdown_summary, summary)) {
        std::cerr << "kvcm_swarm: failed to write summary to " << config_.evidence.markdown_summary << "\n";
        return false;
    }
    std::cout << summary << std::flush;
    return true;
}

ExitCode RunCoordinator::Run() {
    const int64_t started_wall_ms = WallClockMs();
    const TimePoint run_start = Now();

    SwarmExecutor executor(config_.runtime.workers);
    AdmissionController admission(executor, config_.runtime.limits);
    EvidenceSink evidence;
    PhaseSource phase_source;
    StopSource stop_source;
    SeedDeriver seeds(config_.seed);

    if (!evidence.violations().Open(config_.evidence.violations_jsonl)) {
        std::cerr << "kvcm_swarm: cannot open violations file " << config_.evidence.violations_jsonl << "\n";
        executor.Shutdown();
        return ExitCode::kReportFailed;
    }

    TransportProvider transports(executor,
                                 admission,
                                 evidence,
                                 phase_source,
                                 config_.target.endpoints,
                                 config_.runtime.transport,
                                 config_.runtime.reactor_threads,
                                 config_.runtime.grpc_completion_queues);

    // ---- preflight ----
    EnterPhase(Phase::kPreflight);
    phase_source.Set(Phase::kPreflight);
    PreflightReport preflight;
    if (config_.preflight_enabled) {
        PreflightRunner runner(config_, transports);
        preflight =
            async_simple::coro::syncAwait(std::move(runner.Run(Now() + std::chrono::seconds(60))).via(&executor));
    } else {
        preflight.executed = false;
        preflight.passed = true;
        preflight.cleanup_notes.push_back("preflight was explicitly disabled by the run configuration");
    }

    // ---- create behaviors ----
    std::vector<std::unique_ptr<ClientBehavior>> owned_behaviors;
    std::vector<ClientBehavior *> behaviors;
    for (const auto &spec : config_.behaviors) {
        const BehaviorFactory *factory = registry_.Find(spec.type);
        if (factory == nullptr) {
            continue;
        }
        RuntimeServices services{executor, admission, transports, evidence, seeds, stop_source.Token(), phase_source};
        auto behavior = factory->Create(spec, services);
        if (behavior == nullptr) {
            std::cerr << "kvcm_swarm: behavior '" << spec.id << "' could not be created\n";
            continue;
        }
        behaviors.push_back(behavior.get());
        owned_behaviors.push_back(std::move(behavior));
    }

    auto finish = [&](ExitCode code, const std::string &reason, bool initialize_ok, bool drain_complete) {
        LeaveCurrentPhase();
        EnterPhase(Phase::kReport);
        phase_source.Set(Phase::kReport);
        // Stop every remaining operation and give it a bounded window to
        // release its state before the report snapshot is taken. This matters on
        // the failure paths, where no drain ran.
        stop_source.RequestStop();
        const TimePoint settle = Now() + std::chrono::seconds(3);
        while (Now() < settle) {
            bool settled = true;
            for (ClientBehavior *behavior : behaviors) {
                if (!behavior->Quiesced()) {
                    settled = false;
                }
            }
            if (settled) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        evidence.violations().Close();
        RunReportInput input;
        input.config = &config_;
        input.evidence = &evidence;
        input.admission = &admission;
        input.executor = &executor;
        input.transports = &transports;
        input.behaviors = &behaviors;
        input.phases = &phases_;
        input.preflight = &preflight;
        input.started_wall_ms = started_wall_ms;
        input.ended_wall_ms = WallClockMs();
        input.total_duration = Now() - run_start;
        input.exit_reason = reason;
        input.initialize_ok = initialize_ok;
        input.drain_complete = drain_complete;
        bool quiesced = true;
        for (ClientBehavior *behavior : behaviors) {
            if (!behavior->Quiesced()) {
                quiesced = false;
            }
        }
        input.quiesced = quiesced;
        input.resources = CollectResourceUsage();
        LeaveCurrentPhase();
        const bool written = WriteOutputs(input);
        // Tear down in dependency order: stop running work first, then the
        // transports it used, then the behaviors that own the domain state.
        executor.Shutdown();
        transports.Shutdown();
        owned_behaviors.clear();
        if (!written) {
            return ExitCode::kReportFailed;
        }
        return code;
    };

    if (config_.preflight_enabled && !preflight.passed) {
        std::cerr << "kvcm_swarm: preflight failed at stage '" << preflight.failure_stage
                  << "': " << preflight.failure_detail << "\n";
        return finish(ExitCode::kPreflightFailed, "preflight_failed", false, false);
    }
    if (behaviors.empty()) {
        return finish(ExitCode::kConfigInvalid, "no behavior could be created", false, false);
    }

    // ---- initialize ----
    LeaveCurrentPhase();
    EnterPhase(Phase::kInitialize);
    phase_source.Set(Phase::kInitialize);
    bool initialize_ok = true;
    for (ClientBehavior *behavior : behaviors) {
        const bool ok = async_simple::coro::syncAwait(
            std::move(behavior->Initialize(Now() + std::chrono::seconds(60))).via(&executor));
        if (!ok) {
            std::cerr << "kvcm_swarm: behavior '" << behavior->Id() << "' failed to initialize\n";
            initialize_ok = false;
        }
    }
    if (!initialize_ok) {
        return finish(ExitCode::kInitializeFailed, "initialize_failed", false, false);
    }

    // ---- warmup ----
    LeaveCurrentPhase();
    EnterPhase(Phase::kWarmup);
    phase_source.Set(Phase::kWarmup);
    for (ClientBehavior *behavior : behaviors) {
        behavior->StartTraffic();
    }
    async_simple::coro::syncAwait(
        std::move(SleepFor(executor, config_.runtime.warmup, stop_source.Token())).via(&executor));

    // ---- steady: only the phase changes ----
    LeaveCurrentPhase();
    EnterPhase(Phase::kSteady);
    phase_source.Set(Phase::kSteady);
    async_simple::coro::syncAwait(
        std::move(SleepFor(executor, config_.runtime.steady, stop_source.Token())).via(&executor));

    // ---- drain ----
    LeaveCurrentPhase();
    EnterPhase(Phase::kDrain);
    phase_source.Set(Phase::kDrain);
    const TimePoint drain_deadline = Now() + config_.runtime.drain_timeout;
    for (ClientBehavior *behavior : behaviors) {
        async_simple::coro::syncAwait(std::move(behavior->Drain(drain_deadline)).via(&executor));
    }
    // Bounded settle window so late completions can release their state.
    const TimePoint settle_deadline =
        std::min(drain_deadline + std::chrono::seconds(2), Now() + std::chrono::seconds(5));
    bool quiesced = false;
    while (Now() < settle_deadline) {
        quiesced = true;
        for (ClientBehavior *behavior : behaviors) {
            if (!behavior->Quiesced()) {
                quiesced = false;
            }
        }
        if (quiesced) {
            break;
        }
        async_simple::coro::syncAwait(
            std::move(SleepFor(executor, std::chrono::milliseconds(20), StopToken())).via(&executor));
    }

    return finish(ExitCode::kOk, "completed", true, quiesced);
}

} // namespace kvcm_swarm
