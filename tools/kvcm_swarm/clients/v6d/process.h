// One simulated V6D process: a vineyard daemon plus its inference node.
//
// A process is a domain object, not a thread and not a single runtime task.
// It owns its reporter identity, its transport context, its local cache and
// its single eviction pipeline. It never owns a workload session.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/v6d/checks.h"
#include "tools/kvcm_swarm/clients/v6d/config.h"
#include "tools/kvcm_swarm/clients/v6d/expected_locations.h"
#include "tools/kvcm_swarm/clients/v6d/local_cache.h"
#include "tools/kvcm_swarm/clients/v6d/workload.h"
#include "tools/kvcm_swarm/evidence/json_writer.h"
#include "tools/kvcm_swarm/protocol/proto_alias.h"
#include "tools/kvcm_swarm/transport/transport.h"

namespace kvcm_swarm {

class V6dProcess;

// Deployment-scoped services a process needs, without a header cycle.
class V6dDeploymentContext {
public:
    virtual ~V6dDeploymentContext() = default;
    virtual const V6dConfig &config() const = 0;
    virtual const std::string &behavior_id() const = 0;
    virtual RuntimeServices &services() = 0;
    virtual ExpectedLocations &expected() = 0;
    virtual V6dChecks &checks() = 0;
    // Cold-tier backends, derived from the storage_configs returned by
    // RegisterInstance rather than hard-coded.
    virtual std::vector<meta::StorageType> cold_backends() const = 0;
    virtual void SetStorageConfigs(const std::string &json) = 0;
};

struct ReporterStats {
    uint64_t generation = 0;
    uint64_t heartbeats_sent = 0;
    uint64_t heartbeats_failed = 0;
    uint64_t node_registers = 0;
    uint64_t leader_polls = 0;
    uint64_t leader_poll_failures = 0;
    uint64_t leader_endpoint_changes = 0;
    uint64_t not_leader_retries = 0;
    uint64_t not_leader_retry_failures = 0;
    uint64_t block_add_batches = 0;
    uint64_t block_add_items = 0;
    uint64_t block_add_confirmed = 0;
    uint64_t block_add_failed = 0;
    uint64_t block_add_unknown = 0;
    uint64_t block_delete_batches = 0;
    uint64_t block_delete_items = 0;
    uint64_t block_delete_confirmed = 0;
    uint64_t block_delete_failed = 0;
    uint64_t block_delete_unknown = 0;
    uint64_t host_down_attempted = 0;
    uint64_t host_down_succeeded = 0;
    bool registered = false;
};

struct EvictionStats {
    uint64_t batches = 0;
    uint64_t objects_selected = 0;
    uint64_t start_write_ok = 0;
    uint64_t start_write_failed = 0;
    uint64_t start_write_unknown = 0;
    uint64_t writable_items = 0;
    uint64_t masked_items = 0;
    uint64_t finish_write_ok = 0;
    uint64_t finish_write_failed = 0;
    uint64_t finish_write_unknown = 0;
    uint64_t cold_allocations_confirmed = 0;
    uint64_t cold_allocation_bytes = 0;
    uint64_t local_removed = 0;
    uint64_t restored_resident = 0;
    uint64_t protected_uncertain = 0;
    uint64_t shutdown_flush_objects = 0;
    uint64_t shutdown_flush_batches = 0;
};

class V6dProcess {
public:
    V6dProcess(V6dDeploymentContext &deployment, V6dProcessIdentity identity, TransportKind transport_kind);
    ~V6dProcess();

    const V6dProcessIdentity &identity() const { return identity_; }
    const ReporterIdentity &reporter() const { return reporter_; }
    LocalCache &cache() { return cache_; }
    const LocalCache &cache() const { return cache_; }
    ClientTransportContext &transport() { return *transport_; }
    const std::string &self_uri() const { return self_uri_; }

    // initialize: RegisterInstance, then NODE_REGISTER + HEARTBEAT in one batch.
    Task<bool> Register(TimePoint planned_start, TimePoint deadline);
    // Starts heartbeat and leader-discovery deadlines plus the eviction loop.
    void StartMaintenance();
    // Drain: finish in-flight evictions, shutdown flush, then HOST_DOWN.
    Task<> Drain(TimePoint deadline);

    // ---- business operations used by a turn ----
    Task<RpcResult> Lookup(const meta::GetCacheLocationsByBackendRequest &request,
                           meta::GetCacheLocationsByBackendResponse *response,
                           TimePoint planned_at,
                           TimePoint deadline,
                           StopToken stop);
    // BLOCK_ADD for the objects that entered this process's cache in one
    // materialisation/seal batch.
    Task<> ReportBlockAdd(const std::vector<GroupObject> &objects);

    // Wakes the single eviction pipeline of this process.
    void WakeEvictor();

    // A turn reserves its actual object bytes for its lifetime. Concurrent
    // turns may share a process only while their simple byte sum fits in the
    // process cache; shared object keys are deliberately not deduplicated.
    Task<AsyncCapacityBudget::Guard>
    AcquireTurnCapacity(uint64_t working_set_bytes, TimePoint deadline, StopToken stop);
    const AsyncCapacityBudget &turn_capacity_stats() const { return turn_capacity_; }

    ReporterStats reporter_stats() const;
    EvictionStats eviction_stats() const;
    bool ready() const { return ready_.load(std::memory_order_acquire); }
    bool quiesced() const { return active_operations_.load(std::memory_order_acquire) == 0; }

    void WriteReport(JsonWriter &writer) const;

private:
    struct OperationGuard {
        explicit OperationGuard(V6dProcess *process) : process_(process) {
            process_->active_operations_.fetch_add(1, std::memory_order_release);
        }
        ~OperationGuard() { process_->active_operations_.fetch_sub(1, std::memory_order_release); }
        V6dProcess *process_;
    };

    Task<> HeartbeatLoop();
    Task<> LeaderDiscoveryLoop();
    Task<> EvictionLoop();
    Task<bool> RefreshLeaderEndpoint(TimePoint deadline);
    // Issues one call, and on SERVER_NOT_LEADER refreshes the endpoint and
    // retries exactly once. Both the original failure and the retry are
    // recorded; nothing is hidden inside the transport.
    Task<RpcResult> CallWithLeaderRefresh(Api api,
                                          const google::protobuf::Message &request,
                                          google::protobuf::Message *response,
                                          CallOptions options);
    Task<bool> RunEvictionBatch(std::vector<GroupObject> batch, bool shutdown_flush);
    Task<> ReportBlockDelete(const std::vector<GroupObject> &objects);
    Task<> SendHostDown(TimePoint deadline);
    std::string NextTraceId(const char *prefix);
    meta::ReportEventRequest MakeReportEventRequest(const char *trace_prefix);
    V6dDeploymentContext &deployment_;
    V6dProcessIdentity identity_;
    ReporterIdentity reporter_;
    std::string self_uri_;
    TransportKind transport_kind_;
    ClientTransportContext *transport_ = nullptr;
    LocalCache cache_;
    AsyncCapacityBudget turn_capacity_;
    StopSource own_stop_;

    std::atomic<bool> ready_{false};
    std::atomic<bool> draining_{false};
    std::atomic<uint64_t> trace_counter_{0};
    std::atomic<uint32_t> active_operations_{0};
    std::atomic<uint32_t> maintenance_loops_{0};

    // Eviction pipeline: exactly one logical pipeline per process.
    std::mutex evictor_mutex_;
    std::shared_ptr<AsyncSlot<bool>> evictor_wake_;
    bool evictor_running_ = false;
    std::atomic<uint32_t> eviction_batches_in_flight_{0};

    mutable std::mutex stats_mutex_;
    ReporterStats reporter_stats_;
    EvictionStats eviction_stats_;
    uint64_t heartbeat_sequence_ = 0;
};

} // namespace kvcm_swarm
