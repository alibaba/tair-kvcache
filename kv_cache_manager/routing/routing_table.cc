#include "kv_cache_manager/routing/routing_table.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>

#include "kv_cache_manager/common/hash/hash.h"

namespace kv_cache_manager {

std::optional<BucketId> HashBucketPolicy::GetBucket(std::string_view key, uint32_t bucket_count) const {
    if (key.empty() || bucket_count == 0) {
        return std::nullopt;
    }
    return Hash64(key.data(), key.size(), 0) % bucket_count;
}

RoutingTable::RoutingTable(std::shared_ptr<const BucketPolicy> policy) : policy_(std::move(policy)) {
    if (!policy_) {
        throw std::invalid_argument("missing bucket policy");
    }
}

bool RoutingTable::Configure(uint64_t version, uint32_t columns) {
    if (!version || !columns) {
        return false;
    }
    std::unique_lock lock(mutex_);
    if (version == version_) {
        return columns == columns_;
    }
    if (version < version_) {
        return false;
    }
    version_ = version;
    columns_ = columns;
    replicas_.clear();
    return true;
}

bool RoutingTable::Report(ReplicaReport report, Clock::time_point now) {
    if (report.process_id.empty() || report.endpoint.empty() || report.sequence == 0) {
        return false;
    }
    std::unique_lock lock(mutex_);
    if (report.partition >= columns_ || report.version != version_) {
        return false;
    }
    auto &entries = replicas_[report.partition];
    const auto current = entries.find(report.process_id);
    if (current != entries.end() && report.sequence <= current->second.report.sequence) {
        return false;
    }
    const auto id = report.process_id;
    entries.insert_or_assign(id, Entry{std::move(report), now});
    return true;
}

std::vector<Route> RoutingTable::Resolve(const RouteKey &key, Clock::time_point now, Clock::duration lease) const {
    std::shared_lock lock(mutex_);
    std::vector<Route> routes;
    if (key.placement == Placement::Hash || key.placement == Placement::RoleZero) {
        PartitionId partition = 0;
        if (key.placement == Placement::Hash) {
            auto bucket = policy_->GetBucket(key.key, columns_);
            if (!bucket || *bucket >= columns_) {
                return {};
            }
            partition = *bucket;
        }
        if (auto route = Leader(partition, now, lease)) {
            routes.push_back(std::move(*route));
        }
        return routes;
    }
    if (key.placement == Placement::All && replicas_.size() != columns_) {
        return {};
    }
    for (const auto &partition : replicas_) {
        auto route = Leader(partition.first, now, lease);
        if (route) {
            routes.push_back(std::move(*route));
        } else if (key.placement == Placement::All) {
            return {}; // A missing member cannot turn a broadcast into a successful partial write.
        }
    }
    if (key.placement == Placement::RandomReady && !routes.empty()) {
        thread_local std::mt19937 random(std::random_device{}());
        return {routes[std::uniform_int_distribution<size_t>(0, routes.size() - 1)(random)]};
    }
    std::sort(routes.begin(), routes.end(), [](const Route &a, const Route &b) { return a.partition < b.partition; });
    return routes;
}

std::optional<Route> RoutingTable::Leader(PartitionId partition, Clock::time_point now, Clock::duration lease) const {
    const auto part = replicas_.find(partition);
    if (part == replicas_.end()) {
        return std::nullopt;
    }
    const Entry *leader = nullptr;
    for (const auto &item : part->second) {
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
    return Route{version_, partition, leader->report.endpoint};
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
