// ExpectedLocations: the minimum client-observable state needed to classify
// whether a location returned by KVCM could legitimately exist.
//
// This is client-owned state used to decide behavior and to classify what the
// client observes. It is not a copy of KVCM's authoritative metadata and never
// substitutes for KVCM's own location/allocation decisions.
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/runtime/clock.h"

namespace kvcm_swarm {

enum class LocationState {
    kPendingCreate,
    kConfirmed,
    kPendingDelete,
    kUnknown,
    kRemoved,
};

const char *LocationStateName(LocationState state);

// Full reporter identity: instance_id + ST_EVENT_REPORT_L2 + host_ip_port.
struct ReporterIdentity {
    std::string instance_id;
    std::string host_ip_port;

    bool operator<(const ReporterIdentity &other) const {
        if (instance_id != other.instance_id) {
            return instance_id < other.instance_id;
        }
        return host_ip_port < other.host_ip_port;
    }
    bool operator==(const ReporterIdentity &other) const {
        return instance_id == other.instance_id && host_ip_port == other.host_ip_port;
    }
};

struct HotLocationKey {
    int64_t block_key = 0;
    std::string spec_name;
    ReporterIdentity reporter;

    bool operator<(const HotLocationKey &other) const {
        if (block_key != other.block_key)
            return block_key < other.block_key;
        if (spec_name != other.spec_name)
            return spec_name < other.spec_name;
        return reporter < other.reporter;
    }
};

struct ColdLocationKey {
    int64_t block_key = 0;
    std::string spec_name;
    std::string storage_uri;

    bool operator<(const ColdLocationKey &other) const {
        if (block_key != other.block_key)
            return block_key < other.block_key;
        if (spec_name != other.spec_name)
            return spec_name < other.spec_name;
        return storage_uri < other.storage_uri;
    }
};

struct ExpectedLocationsStats {
    uint64_t hot_pending_create = 0;
    uint64_t hot_confirmed = 0;
    uint64_t hot_pending_delete = 0;
    uint64_t hot_unknown = 0;
    uint64_t hot_removed = 0;
    uint64_t cold_pending_create = 0;
    uint64_t cold_confirmed = 0;
    uint64_t cold_unknown = 0;
    uint64_t cold_removed = 0;
    uint64_t cold_confirmed_bytes = 0;
};

class ExpectedLocations {
public:
    // ---- hot tier (per-reporter provenance) ----
    void HotPendingCreate(const HotLocationKey &key);
    void HotConfirm(const HotLocationKey &key);
    void HotPendingDelete(const HotLocationKey &key);
    void HotRemove(const HotLocationKey &key);
    void HotUnknown(const HotLocationKey &key, bool delete_direction);
    // Explicitly not executed: revert to the previous stable state.
    void HotNotExecuted(const HotLocationKey &key, bool was_delete);
    // Reporter lifecycle retires every hot location it owns.
    void RetireReporter(const ReporterIdentity &reporter);

    // ---- cold tier (allocation provenance, no machine owner) ----
    void ColdPendingCreate(const ColdLocationKey &key);
    void ColdConfirm(const ColdLocationKey &key, uint64_t size_bytes);
    void ColdNotExecuted(const ColdLocationKey &key);
    void ColdUnknown(const ColdLocationKey &key);

    // ---- reporter liveness ----
    void SetReporterLive(const ReporterIdentity &reporter);
    void SetReporterUnavailable(const ReporterIdentity &reporter, bool retired);

    // A remote candidate exists when some other live reporter has a confirmed
    // location for this key/spec.
    struct CandidateSnapshot {
        bool has_confirmed_remote = false;
        bool requester_has_possible_local = false;
    };
    CandidateSnapshot
    SnapshotCandidates(int64_t block_key, const std::string &spec_name, const ReporterIdentity &requester) const;

    // Is a returned hot location acceptable at all? (C1a soundness)
    struct HotAcceptance {
        bool known_reporter = false;
        bool retired_reporter = false;
        bool state_allows = false; // confirmed / pending-create / pending-delete / unknown
        LocationState state = LocationState::kRemoved;
        // True when the location was removed only after `query_issued_at`, so
        // the server answer was still legitimate when the query was issued.
        bool removed_after_query = false;
    };
    HotAcceptance CheckHotAcceptable(const HotLocationKey &key, TimePoint query_issued_at) const;

    // Monotonic revision of the candidate set and of reporter liveness.
    uint64_t candidate_revision() const;
    uint64_t liveness_revision() const;

    ExpectedLocationsStats Stats() const;
    // Hot locations left in an unresolved state at the end of the run.
    std::vector<std::string> UnresolvedSummary(size_t limit) const;

private:
    struct HotLocationRecord {
        LocationState state = LocationState::kPendingCreate;
        // Direction is retained only to make unresolved end-of-run evidence useful.
        bool pending_delete_direction = false;
        // A lookup issued before removal may still legitimately observe it.
        TimePoint removed_at{};
    };

    struct ColdLocationRecord {
        LocationState state = LocationState::kPendingCreate;
    };

    struct ReporterLiveness {
        bool available = true;
        bool retired = false;
    };

    void BumpCandidateLocked() { ++candidate_revision_; }

    mutable std::mutex mutex_;
    std::map<HotLocationKey, HotLocationRecord> hot_;
    std::map<ColdLocationKey, ColdLocationRecord> cold_;
    std::map<ReporterIdentity, ReporterLiveness> reporters_;
    uint64_t candidate_revision_ = 0;
    uint64_t liveness_revision_ = 0;
    uint64_t cold_confirmed_bytes_ = 0;
};

} // namespace kvcm_swarm
