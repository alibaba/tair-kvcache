#include "tools/kvcm_swarm/evidence/sink.h"

#include <algorithm>
#include <cstdio>

#include "tools/kvcm_swarm/evidence/json_writer.h"
#include "tools/kvcm_swarm/runtime/rng.h"

namespace kvcm_swarm {

bool RpcAggregateKey::operator<(const RpcAggregateKey &other) const {
    if (behavior_type != other.behavior_type)
        return behavior_type < other.behavior_type;
    if (behavior_id != other.behavior_id)
        return behavior_id < other.behavior_id;
    if (api != other.api)
        return api < other.api;
    if (phase != other.phase)
        return phase < other.phase;
    return lane < other.lane;
}

void RpcAggregate::Merge(const RpcAggregate &other) {
    total += other.total;
    success += other.success;
    transport_failures += other.transport_failures;
    service_failures += other.service_failures;
    uncertain += other.uncertain;
    for (const auto &entry : other.transport_errors) {
        transport_errors[entry.first] += entry.second;
    }
    for (const auto &entry : other.service_statuses) {
        service_statuses[entry.first] += entry.second;
    }
    latency.Merge(other.latency);
    permit_wait.Merge(other.permit_wait);
    queue_delay.Merge(other.queue_delay);
}

ViolationLog::~ViolationLog() { Close(); }

bool ViolationLog::Open(const std::string &path) {
    std::lock_guard<std::mutex> lock(mutex_);
    path_ = path;
    if (path.empty()) {
        return true;
    }
    std::FILE *raw = std::fopen(path.c_str(), "w");
    if (raw == nullptr) {
        failed_ = true;
        return false;
    }
    file_ = std::unique_ptr<std::FILE, int (*)(std::FILE *)>(raw, &std::fclose);
    return true;
}

void ViolationLog::Record(const std::string &check_name, const std::string &detail_json) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++total_;
    ++counts_[check_name];
    auto &preview = preview_[check_name];
    if (preview.size() < kPreviewLimit) {
        preview.push_back(detail_json);
    }
    if (file_) {
        const std::string line = "{\"check\":" + JsonQuote(check_name) + ",\"detail\":" + detail_json + "}\n";
        if (std::fwrite(line.data(), 1, line.size(), file_.get()) != line.size()) {
            failed_ = true;
        }
    }
}

void ViolationLog::Close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_) {
        if (std::fflush(file_.get()) != 0) {
            failed_ = true;
        }
        file_.reset();
    }
}

std::vector<std::string> ViolationLog::Preview(const std::string &check_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = preview_.find(check_name);
    return it == preview_.end() ? std::vector<std::string>{} : it->second;
}

uint64_t ViolationLog::Count(const std::string &check_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = counts_.find(check_name);
    return it == counts_.end() ? 0 : it->second;
}

uint64_t ViolationLog::total() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_;
}

EvidenceSink::EvidenceSink() = default;

EvidenceSink::Shard &EvidenceSink::ShardFor(const RpcAggregateKey &key) {
    const uint64_t hash = HashString(key.behavior_id) ^ HashString(key.api);
    return shards_[hash % kShards];
}

void EvidenceSink::RecordRpc(const RpcObservation &observation) {
    RpcAggregateKey key{
        observation.behavior_type, observation.behavior_id, observation.api, observation.phase, observation.lane};
    Shard &shard = ShardFor(key);
    std::lock_guard<std::mutex> lock(shard.mutex);
    RpcAggregate &agg = shard.aggregates[key];
    ++agg.total;
    if (observation.result.ok) {
        ++agg.success;
    }
    if (observation.result.transport_error != TransportError::kNone) {
        ++agg.transport_failures;
        ++agg.transport_errors[TransportErrorName(observation.result.transport_error)];
        if (IsUncertain(observation.result.transport_error)) {
            ++agg.uncertain;
        }
    } else if (!observation.result.ok) {
        ++agg.service_failures;
    }
    if (observation.result.service_status != 0) {
        ++agg.service_statuses[observation.result.service_status];
    }
    agg.latency.Add(ToMillis(observation.result.rpc_latency));
    agg.permit_wait.Add(ToMillis(observation.permit_wait));
    agg.queue_delay.Add(ToMillis(observation.queue_delay));
}

std::map<RpcAggregateKey, RpcAggregate> EvidenceSink::RpcSnapshot() const {
    std::map<RpcAggregateKey, RpcAggregate> merged;
    for (const auto &shard : shards_) {
        std::lock_guard<std::mutex> lock(shard.mutex);
        for (const auto &entry : shard.aggregates) {
            merged[entry.first].Merge(entry.second);
        }
    }
    return merged;
}

} // namespace kvcm_swarm
