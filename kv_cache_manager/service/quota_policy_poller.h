#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "kv_cache_manager/protocol/protobuf/optimizer_service.grpc.pb.h"

namespace kv_cache_manager {

class RegistryManager;
class ServiceDiscovery;
class MetaIndexerManager;
class MetricsRegistry;
class RequestContext;

struct QuotaPolicyPollerConfig {
    bool enable = false;
    bool enable_hard_resize = false;
    std::string optimizer_service_discovery_url;
    std::string pool_id;
    std::string quota_target_id;
    std::string instance_group;
    std::string state_file;
    int64_t poll_interval_seconds = 30;
    int64_t rpc_timeout_ms = 1000;
};

class QuotaPolicyPoller {
public:
    using IsLeaderFunction = std::function<bool()>;

    QuotaPolicyPoller(QuotaPolicyPollerConfig config,
                      std::shared_ptr<RegistryManager> registry_manager,
                      IsLeaderFunction is_leader,
                      std::shared_ptr<MetaIndexerManager> meta_indexer_manager = nullptr,
                      std::shared_ptr<MetricsRegistry> metrics_registry = nullptr);
    ~QuotaPolicyPoller();

    bool Init();
    bool Start();
    void Stop();
    bool PollOnce();

    uint64_t last_leader_epoch() const { return last_leader_epoch_; }
    uint64_t last_allocation_epoch() const { return last_allocation_epoch_; }
    uint64_t last_execution_revision() const { return last_execution_revision_; }

private:
    void Loop();
    bool LoadState();
    bool SaveState(uint64_t leader_epoch,
                   uint64_t allocation_epoch,
                   uint64_t execution_revision,
                   const std::string &plan_hash);
    bool EnsureStub();
    std::optional<int64_t> GetGroupUsedBytes(RequestContext *context) const;
    bool ApplyHardQuota(const proto::optimizer::PullQuotaAllocationResponse &plan,
                        int64_t *observed_quota_bytes,
                        int64_t *instance_group_version,
                        std::string *reason);
    void ReportMetrics(const proto::optimizer::PullQuotaAllocationResponse &plan,
                       int64_t used_bytes,
                       const std::string &status);
    bool ReportResult(const proto::optimizer::PullQuotaAllocationResponse &plan,
                      const std::string &phase,
                      const std::string &status,
                      const std::string &reason,
                      int64_t current_quota_bytes,
                      int64_t used_bytes,
                      int64_t instance_group_version);

    QuotaPolicyPollerConfig config_;
    std::shared_ptr<RegistryManager> registry_manager_;
    std::shared_ptr<MetaIndexerManager> meta_indexer_manager_;
    std::shared_ptr<MetricsRegistry> metrics_registry_;
    IsLeaderFunction is_leader_;
    std::unique_ptr<ServiceDiscovery> discovery_;
    std::unique_ptr<proto::optimizer::OptimizerService::Stub> stub_;
    std::string connected_endpoint_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    uint64_t last_leader_epoch_ = 0;
    uint64_t last_allocation_epoch_ = 0;
    uint64_t last_execution_revision_ = 0;
    std::string last_plan_hash_;
    std::string release_plan_id_;
    int64_t consecutive_release_samples_ = 0;
};

} // namespace kv_cache_manager
