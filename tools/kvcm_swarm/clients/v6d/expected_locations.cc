#include "tools/kvcm_swarm/clients/v6d/expected_locations.h"

namespace kvcm_swarm {

const char *LocationStateName(LocationState state) {
    switch (state) {
    case LocationState::kPendingCreate:
        return "pending-create";
    case LocationState::kConfirmed:
        return "confirmed";
    case LocationState::kPendingDelete:
        return "pending-delete";
    case LocationState::kUnknown:
        return "unknown";
    case LocationState::kRemoved:
        return "removed";
    }
    return "unknown";
}

void ExpectedLocations::HotPendingCreate(const HotLocationKey &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    HotLocationRecord &record = hot_[key];
    record.state = LocationState::kPendingCreate;
    record.pending_delete_direction = false;
    BumpCandidateLocked();
}

void ExpectedLocations::HotConfirm(const HotLocationKey &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = hot_.find(key);
    if (it == hot_.end()) {
        return;
    }
    it->second.state = LocationState::kConfirmed;
    it->second.pending_delete_direction = false;
    BumpCandidateLocked();
}

void ExpectedLocations::HotPendingDelete(const HotLocationKey &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = hot_.find(key);
    if (it == hot_.end()) {
        return;
    }
    it->second.state = LocationState::kPendingDelete;
    it->second.pending_delete_direction = true;
    BumpCandidateLocked();
}

void ExpectedLocations::HotRemove(const HotLocationKey &key) {
    // An explicitly successful BLOCK_DELETE removes the location immediately:
    // there is no client-side delete-visibility grace period.
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = hot_.find(key);
    if (it == hot_.end()) {
        return;
    }
    it->second.state = LocationState::kRemoved;
    it->second.removed_at = Now();
    BumpCandidateLocked();
}

void ExpectedLocations::HotUnknown(const HotLocationKey &key, bool delete_direction) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = hot_.find(key);
    if (it == hot_.end()) {
        return;
    }
    it->second.state = LocationState::kUnknown;
    it->second.pending_delete_direction = delete_direction;
    BumpCandidateLocked();
}

void ExpectedLocations::HotNotExecuted(const HotLocationKey &key, bool was_delete) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = hot_.find(key);
    if (it == hot_.end()) {
        return;
    }
    it->second.state = was_delete ? LocationState::kConfirmed : LocationState::kRemoved;
    if (!was_delete) {
        it->second.removed_at = Now();
    }
    it->second.pending_delete_direction = false;
    BumpCandidateLocked();
}

void ExpectedLocations::RetireReporter(const ReporterIdentity &reporter) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &entry : hot_) {
        if (entry.first.reporter == reporter && entry.second.state != LocationState::kRemoved) {
            entry.second.state = LocationState::kRemoved;
            entry.second.removed_at = Now();
        }
    }
    auto liveness = reporters_.find(reporter);
    if (liveness != reporters_.end()) {
        liveness->second.available = false;
        liveness->second.retired = true;
    }
    ++liveness_revision_;
    BumpCandidateLocked();
}

void ExpectedLocations::ColdPendingCreate(const ColdLocationKey &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    ColdLocationRecord &record = cold_[key];
    record.state = LocationState::kPendingCreate;
}

void ExpectedLocations::ColdConfirm(const ColdLocationKey &key, uint64_t size_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = cold_.find(key);
    if (it == cold_.end()) {
        return;
    }
    if (it->second.state != LocationState::kConfirmed) {
        cold_confirmed_bytes_ += size_bytes;
    }
    it->second.state = LocationState::kConfirmed;
}

void ExpectedLocations::ColdNotExecuted(const ColdLocationKey &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = cold_.find(key);
    if (it == cold_.end()) {
        return;
    }
    it->second.state = LocationState::kRemoved;
}

void ExpectedLocations::ColdUnknown(const ColdLocationKey &key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = cold_.find(key);
    if (it == cold_.end()) {
        return;
    }
    it->second.state = LocationState::kUnknown;
}

void ExpectedLocations::SetReporterLive(const ReporterIdentity &reporter) {
    std::lock_guard<std::mutex> lock(mutex_);
    ReporterLiveness &liveness = reporters_[reporter];
    liveness.available = true;
    liveness.retired = false;
    ++liveness_revision_;
}

void ExpectedLocations::SetReporterUnavailable(const ReporterIdentity &reporter, bool retired) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = reporters_.find(reporter);
    if (it == reporters_.end()) {
        return;
    }
    it->second.available = false;
    it->second.retired = retired;
    ++liveness_revision_;
}

ExpectedLocations::CandidateSnapshot ExpectedLocations::SnapshotCandidates(int64_t block_key,
                                                                           const std::string &spec_name,
                                                                           const ReporterIdentity &requester) const {
    CandidateSnapshot snapshot;
    std::lock_guard<std::mutex> lock(mutex_);
    HotLocationKey lower;
    lower.block_key = block_key;
    lower.spec_name = spec_name;
    for (auto it = hot_.lower_bound(lower); it != hot_.end(); ++it) {
        if (it->first.block_key != block_key || it->first.spec_name != spec_name) {
            break;
        }
        const bool is_requester = it->first.reporter == requester;
        const LocationState state = it->second.state;
        if (is_requester) {
            if (state != LocationState::kRemoved) {
                snapshot.requester_has_possible_local = true;
            }
            continue;
        }
        const auto liveness = reporters_.find(it->first.reporter);
        const bool live = liveness != reporters_.end() && liveness->second.available && !liveness->second.retired;
        if (!live) {
            continue;
        }
        if (state == LocationState::kConfirmed) {
            snapshot.has_confirmed_remote = true;
        }
    }
    return snapshot;
}

ExpectedLocations::HotAcceptance ExpectedLocations::CheckHotAcceptable(const HotLocationKey &key,
                                                                       TimePoint query_issued_at) const {
    HotAcceptance acceptance;
    std::lock_guard<std::mutex> lock(mutex_);
    const auto liveness = reporters_.find(key.reporter);
    acceptance.known_reporter = liveness != reporters_.end();
    acceptance.retired_reporter = liveness != reporters_.end() && liveness->second.retired;
    const auto it = hot_.find(key);
    if (it == hot_.end()) {
        acceptance.state = LocationState::kRemoved;
        acceptance.state_allows = false;
        return acceptance;
    }
    acceptance.state = it->second.state;
    // Soundness accepts every state that could still legitimately exist server
    // side, including pending and unknown. A location removed only after the
    // query was issued was still legitimately visible to that query, so it is
    // accepted as well; a location removed strictly before the query was issued
    // is a real violation.
    acceptance.removed_after_query =
        it->second.state == LocationState::kRemoved && it->second.removed_at >= query_issued_at;
    acceptance.state_allows = it->second.state != LocationState::kRemoved || acceptance.removed_after_query;
    return acceptance;
}

uint64_t ExpectedLocations::candidate_revision() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return candidate_revision_;
}

uint64_t ExpectedLocations::liveness_revision() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return liveness_revision_;
}

ExpectedLocationsStats ExpectedLocations::Stats() const {
    ExpectedLocationsStats stats;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &entry : hot_) {
        switch (entry.second.state) {
        case LocationState::kPendingCreate:
            ++stats.hot_pending_create;
            break;
        case LocationState::kConfirmed:
            ++stats.hot_confirmed;
            break;
        case LocationState::kPendingDelete:
            ++stats.hot_pending_delete;
            break;
        case LocationState::kUnknown:
            ++stats.hot_unknown;
            break;
        case LocationState::kRemoved:
            ++stats.hot_removed;
            break;
        }
    }
    for (const auto &entry : cold_) {
        switch (entry.second.state) {
        case LocationState::kPendingCreate:
            ++stats.cold_pending_create;
            break;
        case LocationState::kConfirmed:
            ++stats.cold_confirmed;
            break;
        case LocationState::kUnknown:
            ++stats.cold_unknown;
            break;
        case LocationState::kPendingDelete:
            ++stats.cold_unknown;
            break;
        case LocationState::kRemoved:
            ++stats.cold_removed;
            break;
        }
    }
    stats.cold_confirmed_bytes = cold_confirmed_bytes_;
    return stats;
}

std::vector<std::string> ExpectedLocations::UnresolvedSummary(size_t limit) const {
    std::vector<std::string> summary;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &entry : hot_) {
        if (summary.size() >= limit) {
            break;
        }
        if (entry.second.state == LocationState::kUnknown || entry.second.state == LocationState::kPendingCreate ||
            entry.second.state == LocationState::kPendingDelete) {
            summary.push_back(std::string("hot block_key=") + std::to_string(entry.first.block_key) +
                              " spec=" + entry.first.spec_name + " reporter=" + entry.first.reporter.host_ip_port +
                              " state=" + LocationStateName(entry.second.state) +
                              (entry.second.pending_delete_direction ? " direction=delete" : " direction=create"));
        }
    }
    for (const auto &entry : cold_) {
        if (summary.size() >= limit) {
            break;
        }
        if (entry.second.state == LocationState::kUnknown || entry.second.state == LocationState::kPendingCreate) {
            summary.push_back(std::string("cold block_key=") + std::to_string(entry.first.block_key) +
                              " spec=" + entry.first.spec_name + " uri=" + entry.first.storage_uri +
                              " state=" + LocationStateName(entry.second.state));
        }
    }
    return summary;
}

} // namespace kvcm_swarm
