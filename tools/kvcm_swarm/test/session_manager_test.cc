// SessionManager tests: deployment-wide arrival rate, affinity and migration,
// one turn at a time, skipped slots that do not consume a logical turn, and
// admission rejection at the active-session limit.
#include <atomic>
#include <gtest/gtest.h>
#include <map>
#include <mutex>
#include <thread>

#include "tools/kvcm_swarm/clients/v6d/session_manager.h"

namespace kvcm_swarm {
namespace {

// A deployment context that provides only what SessionManager needs; no
// transport and no process is created.
class StubDeployment : public V6dDeploymentContext {
public:
    StubDeployment(V6dConfig config, RuntimeServices services)
        : config_(std::move(config)), services_(services), checks_(expected_, services.evidence, "inst") {}

    const V6dConfig &config() const override { return config_; }
    const std::string &behavior_id() const override { return behavior_id_; }
    RuntimeServices &services() override { return services_; }
    ExpectedLocations &expected() override { return expected_; }
    V6dChecks &checks() override { return checks_; }
    std::vector<meta::StorageType> cold_backends() const override { return {}; }
    void SetStorageConfigs(const std::string &) override {}

private:
    V6dConfig config_;
    RuntimeServices services_;
    std::string behavior_id_ = "v6d-a";
    ExpectedLocations expected_;
    V6dChecks checks_;
};

class RecordingRunner : public TurnRunner {
public:
    RecordingRunner(SwarmExecutor &executor, uint32_t process_count)
        : executor_(executor), process_count_(process_count) {}

    std::vector<uint32_t> ReadyProcesses() const override {
        std::vector<uint32_t> ready;
        for (uint32_t i = 0; i < process_count_; ++i) {
            ready.push_back(i);
        }
        return ready;
    }

    Task<bool>
    RunTurn(SessionId session_id, uint32_t process_index, SessionWorkload &workload, TimePoint /*deadline*/) override {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++turns_;
            ++per_process_[process_index];
            const int concurrent = ++active_[session_id];
            max_concurrent_per_session_ = std::max(max_concurrent_per_session_, concurrent);
            tokens_[session_id] = workload.token_count();
        }
        if (turn_delay_ > Duration::zero()) {
            co_await SleepFor(executor_, turn_delay_, StopToken());
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --active_[session_id];
        }
        co_return true;
    }

    void set_turn_delay(Duration delay) { turn_delay_ = delay; }
    uint64_t turns() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return turns_;
    }
    int max_concurrent_per_session() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return max_concurrent_per_session_;
    }
    std::map<uint32_t, uint64_t> per_process() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return per_process_;
    }
    std::map<SessionId, size_t> tokens() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tokens_;
    }

private:
    SwarmExecutor &executor_;
    uint32_t process_count_;
    Duration turn_delay_ = Duration::zero();
    mutable std::mutex mutex_;
    uint64_t turns_ = 0;
    int max_concurrent_per_session_ = 0;
    std::map<SessionId, int> active_;
    std::map<uint32_t, uint64_t> per_process_;
    std::map<SessionId, size_t> tokens_;
};

V6dConfig MakeConfig(uint32_t process_count, double rate, ArrivalMode mode, uint32_t max_active, uint32_t turns) {
    V6dConfig config;
    config.process_count = process_count;
    config.instance_group = "grp";
    config.instance_id = "inst";
    config.local_cache_capacity_bytes = 1 << 20;
    config.session_arrival_rate = rate;
    config.arrival_mode = mode;
    config.session_affinity = 0.0;
    config.max_active_sessions = max_active;
    config.turn_deadline = std::chrono::seconds(5);

    CacheGroupSpec group;
    group.group_id = "full-0";
    group.kind = CacheGroupKind::kFullAttention;
    group.block_size_tokens = 4;
    group.object_size_bytes = 4096;
    group.spec_name = "v6d_4096";
    group.lookup_selector = FullSelector::kPrefix;
    config.groups.push_back(group);

    SessionClass session_class;
    session_class.name = "chat";
    session_class.weight = 1.0;
    session_class.turns = IntSpec(turns);
    session_class.turn_interval = DurationSpec(Duration(std::chrono::milliseconds(5)));
    session_class.initial_tokens = IntSpec(16);
    session_class.new_tokens_per_turn = IntSpec(4);
    session_class.rewrite_tail_tokens = IntSpec(0);
    session_class.shared_prefix_probability = 0.0;
    config.session_classes.push_back(session_class);
    return config;
}

struct Harness {
    explicit Harness(V6dConfig config, uint32_t workers = 4)
        : executor(workers)
        , admission(executor, AdmissionLimits{})
        , transports(executor, admission, evidence, phase, MakeEndpoints(), TransportLimits{}, 1, 1)
        , services{executor, admission, transports, evidence, seeds, stop.Token(), phase}
        , deployment(std::move(config), services)
        , runner(executor, deployment.config().process_count)
        , sessions(deployment, runner) {}

    static EndpointSet MakeEndpoints() {
        EndpointSet endpoints;
        endpoints.meta_http = "http://127.0.0.1:1";
        endpoints.meta_grpc = "127.0.0.1:2";
        endpoints.admin_http = "http://127.0.0.1:3";
        endpoints.admin_grpc = "127.0.0.1:2";
        return endpoints;
    }

    ~Harness() {
        sessions.StopAdmission();
        stop.RequestStop();
        transports.Shutdown();
        executor.Shutdown();
    }

    SwarmExecutor executor;
    AdmissionController admission;
    EvidenceSink evidence;
    PhaseSource phase;
    StopSource stop;
    SeedDeriver seeds{11};
    TransportProvider transports;
    RuntimeServices services;
    StubDeployment deployment;
    RecordingRunner runner;
    SessionManager sessions;
};

TEST(SessionManagerTest, ArrivalRateIsForTheWholeDeploymentAndTurnsRun) {
    Harness harness(MakeConfig(4, 200.0, ArrivalMode::kEven, 10000, 2));
    harness.sessions.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    harness.sessions.StopAdmission();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    const SessionStats stats = harness.sessions.Stats();
    // 200 sessions/second for the whole deployment, not per process.
    EXPECT_GE(stats.admitted, 20u);
    EXPECT_LE(stats.admitted, 120u);
    EXPECT_EQ(stats.rejected_admission, 0u);
    EXPECT_GT(stats.turns_started, 0u);
    // Every session runs at most one turn at a time.
    EXPECT_LE(harness.runner.max_concurrent_per_session(), 1);
    // The first turn spreads uniformly over the ready processes.
    EXPECT_GE(harness.runner.per_process().size(), 2u);
}

TEST(SessionManagerTest, ActiveLimitRejectsPlannedArrivalsWithoutDeferringThem) {
    V6dConfig config = MakeConfig(1, 500.0, ArrivalMode::kEven, 2, 50);
    // Long turns keep sessions active so the limit is reached.
    Harness harness(std::move(config));
    harness.runner.set_turn_delay(std::chrono::milliseconds(50));
    harness.sessions.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    harness.sessions.StopAdmission();
    const SessionStats stats = harness.sessions.Stats();
    EXPECT_GT(stats.rejected_admission, 0u);
    EXPECT_LE(stats.active_peak, 2u);
    EXPECT_EQ(stats.planned_arrivals, stats.admitted + stats.rejected_admission);
    // Admission rejection marks the generator saturated, not the server.
    EXPECT_TRUE(harness.admission.saturated());
    const auto reasons = harness.admission.saturation_reasons();
    EXPECT_NE(std::find(reasons.begin(), reasons.end(), "session_admission_rejected"), reasons.end());
}

TEST(SessionManagerTest, SlowTurnsProduceSkippedSlotsThatDoNotConsumeLogicalTurns) {
    V6dConfig config = MakeConfig(1, 20.0, ArrivalMode::kEven, 4, 4);
    Harness harness(std::move(config));
    // A turn far longer than the 5ms interval forces expired slots.
    harness.runner.set_turn_delay(std::chrono::milliseconds(40));
    harness.sessions.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(600));
    harness.sessions.StopAdmission();
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    const SessionStats stats = harness.sessions.Stats();
    EXPECT_GT(stats.skipped_slots, 0u) << "expired slots must be recorded";
    // A skipped slot must not consume a logical turn: no session may exceed the
    // 4 turns it sampled, and every completed session ran all four.
    EXPECT_GE(stats.turns_completed, stats.completed * 4u);
    for (const auto &entry : harness.runner.tokens()) {
        EXPECT_LE(entry.second, 16u + 4u * 4u) << "a session ran more turns than it sampled";
    }
    EXPECT_LE(harness.runner.max_concurrent_per_session(), 1);
}

TEST(SessionManagerTest, AffinityKeepsTheSameProcessAndZeroAffinityMigrates) {
    V6dConfig sticky = MakeConfig(3, 100.0, ArrivalMode::kPoisson, 1000, 6);
    sticky.session_affinity = 1.0;
    {
        Harness harness(std::move(sticky));
        harness.sessions.Start();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        harness.sessions.StopAdmission();
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        const SessionStats stats = harness.sessions.Stats();
        EXPECT_EQ(stats.migrations, 0u);
        EXPECT_GT(stats.affinity_retained, 0u);
    }
    V6dConfig migrating = MakeConfig(3, 100.0, ArrivalMode::kPoisson, 1000, 6);
    migrating.session_affinity = 0.0;
    {
        Harness harness(std::move(migrating));
        harness.sessions.Start();
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        harness.sessions.StopAdmission();
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        const SessionStats stats = harness.sessions.Stats();
        EXPECT_GT(stats.migrations, 0u);
        EXPECT_EQ(stats.affinity_retained, 0u);
    }
}

TEST(SessionManagerTest, LogicalHistoryGrowsExactlyOncePerTurn) {
    Harness harness(MakeConfig(1, 30.0, ArrivalMode::kEven, 8, 3));
    harness.sessions.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    harness.sessions.StopAdmission();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // initial 16 tokens + 4 per turn; the observed token count of the last turn
    // of a session must be 16 + 4 * turn_index.
    for (const auto &entry : harness.runner.tokens()) {
        EXPECT_EQ((entry.second - 16u) % 4u, 0u) << "history advanced more than once per turn";
        EXPECT_LE(entry.second, 16u + 4u * 3u);
        EXPECT_GE(entry.second, 20u);
    }
}

TEST(SessionManagerTest, StopAdmissionPreventsNewSessionsAndNewTurns) {
    Harness harness(MakeConfig(1, 100.0, ArrivalMode::kEven, 1000, 20));
    harness.sessions.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    harness.sessions.StopAdmission();
    const SessionStats after_stop = harness.sessions.Stats();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const SessionStats later = harness.sessions.Stats();
    EXPECT_EQ(later.admitted, after_stop.admitted) << "no session may be admitted after steady ends";
    EXPECT_LE(later.turns_started - after_stop.turns_started, 2u);
}

} // namespace
} // namespace kvcm_swarm
