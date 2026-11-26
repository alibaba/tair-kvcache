#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kv_cache_manager {

using BucketId = uint32_t;
using PartitionId = uint32_t;

// A placement key contains only request data that selects a destination.
enum class Placement {
    Hash,
    All,
    RoleZero,
    RandomReady
};
struct RouteKey {
    static RouteKey Session(std::string session_id) { return {Placement::Hash, std::move(session_id)}; }
    static RouteKey Embedding(std::string emb_key) { return {Placement::Hash, std::move(emb_key)}; }
    static RouteKey PaceAllocate() { return {Placement::RandomReady, {}}; }

    Placement placement;
    std::string key;
    bool global_event = false; // the same reporter lifecycle is also maintained on role zero
};

class BucketPolicy {
public:
    virtual ~BucketPolicy() = default;
    virtual std::optional<BucketId> GetBucket(std::string_view key, uint32_t bucket_count) const = 0;
};

class HashBucketPolicy final : public BucketPolicy {
public:
    std::optional<BucketId> GetBucket(std::string_view key, uint32_t bucket_count) const override;
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
    PartitionId partition;
    std::string endpoint;
};

class RoutingTable {
public:
    using Clock = std::chrono::steady_clock;
    explicit RoutingTable(std::shared_ptr<const BucketPolicy> policy);

    bool Configure(uint64_t version, uint32_t columns);
    bool Report(ReplicaReport report, Clock::time_point now);
    std::vector<Route> Resolve(const RouteKey &key, Clock::time_point now, Clock::duration lease) const;
    void Expire(Clock::time_point now, Clock::duration lease);

private:
    std::optional<Route> Leader(PartitionId partition, Clock::time_point now, Clock::duration lease) const;
    struct Entry {
        ReplicaReport report;
        Clock::time_point received;
    };
    std::shared_ptr<const BucketPolicy> policy_;
    mutable std::shared_mutex mutex_;
    uint64_t version_ = 0;
    uint32_t columns_ = 0;
    std::unordered_map<PartitionId, std::unordered_map<std::string, Entry>> replicas_;
};
} // namespace kv_cache_manager
