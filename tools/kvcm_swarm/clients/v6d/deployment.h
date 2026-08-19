// V6dDeployment: one model deployment made of several simulated V6D processes
// that share one KVCM instance_id.
//
// The deployment is the only module that understands sessions, the token
// workload, shared prefixes, process caches, V6D selectors and expected
// locations. The common runtime knows none of them.
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/client_behavior.h"
#include "tools/kvcm_swarm/clients/v6d/checks.h"
#include "tools/kvcm_swarm/clients/v6d/config.h"
#include "tools/kvcm_swarm/clients/v6d/expected_locations.h"
#include "tools/kvcm_swarm/clients/v6d/process.h"
#include "tools/kvcm_swarm/clients/v6d/session_manager.h"

namespace kvcm_swarm {

struct TurnStats {
    uint64_t turns = 0;
    uint64_t objects_considered = 0;
    uint64_t local_hits = 0;
    uint64_t remote_hot_hits = 0;
    uint64_t cold_hits = 0;
    uint64_t materialized = 0;
    uint64_t sealed = 0;
    uint64_t insert_failed_backpressure = 0;
    uint64_t insert_skipped_evicting = 0;
    uint64_t turn_capacity_timeouts = 0;
    uint64_t insert_cancelled_at_drain = 0;
    uint64_t lookups_cancelled_at_drain = 0;
    uint64_t turns_cancelled_at_drain = 0;
    uint64_t hot_lookup_batches = 0;
    uint64_t hot_lookup_keys = 0;
    uint64_t hot_lookup_failures = 0;
    uint64_t cold_lookup_batches = 0;
    uint64_t cold_lookup_keys = 0;
    uint64_t cold_lookup_failures = 0;
    uint64_t mamba_coverage_batches = 0;
    uint64_t mamba_candidates = 0;
    uint64_t block_add_batches = 0;
    uint64_t reusable_tokens_total = 0;
    uint64_t context_tokens_total = 0;
};

struct GroupWorkloadStats {
    uint64_t objects = 0;
    uint64_t complete_blocks = 0;
    uint64_t local_hits = 0;
    uint64_t remote_hot_hits = 0;
    uint64_t cold_hits = 0;
    uint64_t sealed = 0;
    uint64_t lookup_keys = 0;
};

class V6dDeployment : public ClientBehavior, public V6dDeploymentContext, public TurnRunner {
public:
    V6dDeployment(BehaviorSpec spec, V6dConfig config, RuntimeServices services);
    ~V6dDeployment() override;

    // ---- ClientBehavior ----
    Task<bool> Initialize(TimePoint deadline) override;
    void StartTraffic() override;
    Task<> Drain(TimePoint deadline) override;
    std::string_view TypeName() const override { return "v6d_deployment"; }
    const std::string &Id() const override { return spec_.id; }
    void WriteReport(JsonWriter &writer) const override;
    void WriteEffectiveConfig(JsonWriter &writer) const override;
    std::vector<InvariantObservation> Invariants() const override;
    bool WriteCacheReport(JsonWriter &writer) const override;
    bool WriteWorkloadShape(JsonWriter &writer) const override;
    bool WriteCleanupReport(JsonWriter &writer) const override;
    bool Quiesced() const override;

    // ---- V6dDeploymentContext ----
    const V6dConfig &config() const override { return config_; }
    const std::string &behavior_id() const override { return spec_.id; }
    RuntimeServices &services() override { return services_; }
    ExpectedLocations &expected() override { return expected_; }
    V6dChecks &checks() override { return checks_; }
    std::vector<meta::StorageType> cold_backends() const override;
    void SetStorageConfigs(const std::string &json) override;

    // ---- TurnRunner ----
    std::vector<uint32_t> ReadyProcesses() const override;
    Task<bool>
    RunTurn(SessionId session_id, uint32_t process_index, SessionWorkload &workload, TimePoint deadline) override;

private:
    struct GroupLookupState {
        const CacheGroupSpec *group = nullptr;
        std::vector<GroupObject> objects;
        std::vector<bool> local;
        std::vector<bool> hot_remote;
        std::vector<bool> cold;
    };

    Task<bool> RunHotLookup(V6dProcess &process,
                            const CacheGroupSpec &group,
                            const std::vector<GroupObject> &objects,
                            const std::vector<bool> &local,
                            std::vector<bool> *hot_remote,
                            std::vector<bool> *cold,
                            TimePoint deadline);
    Task<bool> RunColdLookup(V6dProcess &process,
                             const std::vector<GroupObject> &objects,
                             const std::vector<size_t> &indices,
                             std::vector<bool> *cold,
                             TimePoint deadline);
    Task<bool> RunMambaCoverageLookup(V6dProcess &process,
                                      std::vector<GroupLookupState> &states,
                                      const std::vector<size_t> &mamba_state_indices,
                                      uint64_t full_boundary,
                                      TimePoint deadline);

    BehaviorSpec spec_;
    V6dConfig config_;
    RuntimeServices services_;
    ExpectedLocations expected_;
    V6dChecks checks_;
    std::vector<std::unique_ptr<V6dProcess>> processes_;
    std::unique_ptr<SessionManager> sessions_;

    mutable std::mutex mutex_;
    std::string storage_configs_json_;
    std::vector<meta::StorageType> cold_backends_;
    TurnStats turn_stats_;
    std::vector<GroupWorkloadStats> group_stats_;
    // Cancels in-flight turns at the drain deadline so every short lease is
    // released before the shutdown flush selects victims.
    StopSource turn_stop_;
    std::atomic<uint32_t> active_turns_{0};
    std::atomic<bool> drained_{false};
    Duration initialize_duration_{};
    uint64_t register_failures_ = 0;
};

std::unique_ptr<BehaviorFactory> MakeV6dDeploymentFactory();

} // namespace kvcm_swarm
