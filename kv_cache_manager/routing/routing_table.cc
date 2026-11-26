#include "kv_cache_manager/routing/routing_table.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "kv_cache_manager/common/hash/hash.h"

namespace kv_cache_manager {

std::optional<BucketId> InstanceBucketPolicy::GetBucket(RouteKey key, uint32_t bucket_count) const {
    if (key.instance_id.empty() || bucket_count == 0) {
        return std::nullopt;
    }
    return Hash64(key.instance_id.data(), key.instance_id.size(), 0) % bucket_count;
}

RoutingTable::RoutingTable(std::shared_ptr<const BucketPolicy> policy) : policy_(std::move(policy)) {
    if (!policy_) {
        throw std::invalid_argument("missing bucket policy");
    }
}

bool RoutingTable::ApplyAssignment(BucketAssignment assignment) {
    if (assignment.version == 0 || assignment.owners.empty() ||
        assignment.owners.size() > std::numeric_limits<BucketId>::max()) {
        return false;
    }
    std::unique_lock lock(mutex_);
    if (assignment.version == assignment_.version) {
        return assignment.owners == assignment_.owners;
    }
    if (assignment.version < assignment_.version ||
        (!assignment_.owners.empty() && assignment.owners.size() != assignment_.owners.size())) {
        return false;
    }
    assignment_ = std::move(assignment);
    replicas_.clear(); // each new target requires a fresh readiness report
    for (const auto partition : assignment_.owners) {
        replicas_.try_emplace(partition);
    }
    return true;
}

bool RoutingTable::Report(ReplicaReport report, Clock::time_point now) {
    if (report.process_id.empty() || report.endpoint.empty() || report.sequence == 0) {
        return false;
    }
    std::unique_lock lock(mutex_);
    const auto partition = replicas_.find(report.partition);
    if (partition == replicas_.end() || report.version != assignment_.version) {
        return false;
    }
    auto &entries = partition->second;
    const auto current = entries.find(report.process_id);
    if (current != entries.end() && report.sequence <= current->second.report.sequence) {
        return false;
    }
    const auto id = report.process_id;
    entries.insert_or_assign(id, Entry{std::move(report), now});
    return true;
}

std::optional<Route> RoutingTable::Resolve(RouteKey key, Clock::time_point now, Clock::duration lease) const {
    std::shared_lock lock(mutex_);
    const auto bucket = policy_->GetBucket(key, static_cast<uint32_t>(assignment_.owners.size()));
    if (!bucket || *bucket >= assignment_.owners.size()) {
        return std::nullopt;
    }
    const auto partition = assignment_.owners[*bucket];
    const Entry *leader = nullptr;
    for (const auto &item : replicas_.at(partition)) {
        const auto &entry = item.second;
        if (entry.report.serving_leader && now - entry.received < lease) {
            if (leader) {
                return std::nullopt; // conflicting live claims: wait for demotion or expiry
            }
            leader = &entry;
        }
    }
    if (!leader) {
        return std::nullopt;
    }
    return Route{assignment_.version, *bucket, partition, leader->report.endpoint};
}

void RoutingTable::Expire(Clock::time_point now, Clock::duration lease) {
    std::unique_lock lock(mutex_);
    for (auto &partition : replicas_) {
        auto &entries = partition.second;
        for (auto it = entries.begin(); it != entries.end();) {
            if (now - it->second.received >= lease) {
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
    }
}
} // namespace kv_cache_manager
