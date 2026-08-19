// EvidenceSink: the single place that accumulates run facts.
//
// The generator only records facts. Scenario PASS/FAIL decisions are made
// out-of-process by the evaluator from the emitted report.
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/evidence/histogram.h"
#include "tools/kvcm_swarm/evidence/observation.h"

namespace kvcm_swarm {

struct RpcAggregateKey {
    std::string behavior_type;
    std::string behavior_id;
    std::string api;
    Phase phase = Phase::kSteady;
    TrafficLane lane = TrafficLane::kBusiness;

    bool operator<(const RpcAggregateKey &other) const;
};

struct RpcAggregate {
    uint64_t total = 0;
    uint64_t success = 0;
    uint64_t transport_failures = 0;
    uint64_t service_failures = 0;
    uint64_t uncertain = 0;
    std::map<std::string, uint64_t> transport_errors;
    std::map<int, uint64_t> service_statuses;
    Histogram latency;
    Histogram permit_wait;
    Histogram queue_delay;

    void Merge(const RpcAggregate &other);
};

// Bounded, append-only violation log. Details stream to JSONL; only a small
// preview stays in memory.
class ViolationLog {
public:
    ViolationLog() = default;
    ~ViolationLog();

    // Returns false when the file could not be opened; the run then fails.
    bool Open(const std::string &path);
    void Record(const std::string &check_name, const std::string &detail_json);
    void Close();

    std::vector<std::string> Preview(const std::string &check_name) const;
    uint64_t Count(const std::string &check_name) const;
    uint64_t total() const;
    const std::string &path() const { return path_; }
    bool failed() const { return failed_; }

private:
    static constexpr size_t kPreviewLimit = 8;

    mutable std::mutex mutex_;
    std::string path_;
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> file_{nullptr, nullptr};
    std::map<std::string, std::vector<std::string>> preview_;
    std::map<std::string, uint64_t> counts_;
    uint64_t total_ = 0;
    bool failed_ = false;
};

class EvidenceSink {
public:
    EvidenceSink();

    void RecordRpc(const RpcObservation &observation);
    std::map<RpcAggregateKey, RpcAggregate> RpcSnapshot() const;

    ViolationLog &violations() { return violations_; }
    const ViolationLog &violations() const { return violations_; }

private:
    static constexpr size_t kShards = 8;

    struct Shard {
        mutable std::mutex mutex;
        std::map<RpcAggregateKey, RpcAggregate> aggregates;
    };

    Shard &ShardFor(const RpcAggregateKey &key);

    std::array<Shard, kShards> shards_;
    ViolationLog violations_;
};

} // namespace kvcm_swarm
