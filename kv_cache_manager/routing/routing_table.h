#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace kv_cache_manager {

using BucketId = uint32_t;
using PartitionId = uint32_t;

// Lightweight views into a decoded request. Register, query, finish and event
// calls for the same instance must agree on its owner.
struct RouteKey {
    std::string_view method;
    std::string_view instance_id;
};

class BucketPolicy {
public:
    virtual ~BucketPolicy() = default;
    virtual std::optional<BucketId> GetBucket(RouteKey key, uint32_t bucket_count) const = 0;
};

class InstanceBucketPolicy final : public BucketPolicy {
public:
    std::optional<BucketId> GetBucket(RouteKey key, uint32_t bucket_count) const override;
};

// Only the control plane activates an assignment. Every replica of one
// partition receives the same target; liveness reports cannot move buckets.
struct BucketAssignment {
    uint64_t version = 0;
    std::vector<PartitionId> owners; // index is the logical bucket
};

struct ReplicaReport {
    PartitionId partition = 0;
    std::string process_id; // unique for the lifetime of a running replica
    uint64_t version = 0;
    uint64_t sequence = 0;
    std::string endpoint;
    bool serving_leader = false;
};

struct Route {
    uint64_t version;
    BucketId bucket;
    PartitionId partition;
    std::string endpoint;
};

class RoutingTable {
public:
    using Clock = std::chrono::steady_clock;
    explicit RoutingTable(std::shared_ptr<const BucketPolicy> policy);

    bool ApplyAssignment(BucketAssignment assignment);
    bool Report(ReplicaReport report, Clock::time_point now);
    std::optional<Route> Resolve(RouteKey key, Clock::time_point now, Clock::duration lease) const;
    void Expire(Clock::time_point now, Clock::duration lease);

private:
    struct Entry {
        ReplicaReport report;
        Clock::time_point received;
    };
    std::shared_ptr<const BucketPolicy> policy_;
    mutable std::shared_mutex mutex_;
    BucketAssignment assignment_;
    std::unordered_map<PartitionId, std::unordered_map<std::string, Entry>> replicas_;
};
} // namespace kv_cache_manager
