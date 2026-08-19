// V6D correctness contracts C1a, C1b, C2, C3 and C4.
//
// Every check is a passive classification of the traffic the normal workload
// already produces: no correctness-only RPC, no pinning, no forced migration
// and no correctness-only phase.
#pragma once

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "tools/kvcm_swarm/clients/v6d/config.h"
#include "tools/kvcm_swarm/clients/v6d/expected_locations.h"
#include "tools/kvcm_swarm/evidence/sink.h"
#include "tools/kvcm_swarm/protocol/proto_alias.h"

namespace kvcm_swarm {

// Extracts "host:port" from a vineyard://host:port/medium[?params] URI.
// Returns an empty string when the URI is not shaped that way.
std::string ParseVineyardHostPort(const std::string &uri);
// Extracts the medium path element of a vineyard URI.
std::string ParseVineyardMedium(const std::string &uri);

enum class LookupTier {
    kHot,
    kCold
};

struct LookupItem {
    int64_t block_key = 0;
    std::string spec_name;
    std::string object_key;
    // Masked items are already available to the caller and are excluded from
    // the backend query while keeping the positional response shape.
    bool masked = false;
};

// Read-only snapshot captured before a lookup is submitted.
struct LookupExpectation {
    LookupTier tier = LookupTier::kHot;
    // PREFIX contains at most its first unmasked index; COVERAGE contains every
    // unmasked index for which a confirmed remote candidate is known.
    std::vector<size_t> remote_eligible_indices;
    uint64_t candidate_revision = 0;
    uint64_t liveness_revision = 0;
    size_t unmasked_keys = 0;
    // False for cold-only queries: an event-report location returned there
    // would break the backend selector contract.
    bool hot_backend_requested = true;
    // When the query was issued. A candidate deleted after this instant may
    // still legitimately appear in the answer.
    TimePoint issued_at{};
};

struct V6dCheckCounters {
    // C1a
    uint64_t hot_locations_accepted = 0;
    uint64_t c1a_violations = 0;
    // C1b
    uint64_t eligible_queries = 0;
    uint64_t stable_eligible_queries = 0;
    uint64_t invalidated_queries = 0;
    uint64_t stable_eligible_returned = 0;
    uint64_t c1b_violations = 0;
    // C2
    uint64_t shape_checks = 0;
    uint64_t c2_violations = 0;
    // C3
    uint64_t eviction_operations = 0;
    uint64_t eviction_writable_items = 0;
    uint64_t eviction_masked_items = 0;
    uint64_t eviction_writable_completed = 0;
    uint64_t eviction_masked_completed = 0;
    uint64_t cold_allocations_confirmed = 0;
    uint64_t write_sessions_closed = 0;
    uint64_t local_removals_after_write_close = 0;
    uint64_t deletes_after_local_removal = 0;
    uint64_t c3_violations = 0;
};

class V6dChecks {
public:
    V6dChecks(ExpectedLocations &expected, EvidenceSink &evidence, std::string instance_id)
        : expected_(expected), evidence_(evidence), instance_id_(std::move(instance_id)) {}

    // Captures the eligibility snapshot for one lookup batch.
    LookupExpectation BeforeLookup(LookupTier tier,
                                   const std::vector<LookupItem> &items,
                                   FullSelector selector,
                                   const ReporterIdentity &requester) const;

    // Classifies one lookup response: C1a soundness, C1b availability and C2
    // response shape. `resolved_remote` is filled with the per-index remote
    // hot hit result so the caller can drive materialisation.
    void OnLookupResult(const std::vector<LookupItem> &items,
                        const LookupExpectation &expectation,
                        const meta::GetCacheLocationsByBackendResponse &response,
                        const ReporterIdentity &requester,
                        bool response_ok,
                        std::vector<std::string> *hot_hit_host_ports,
                        std::vector<std::string> *cold_hit_uris);

    // C2 for StartWriteCache: the returned location count must equal the
    // number of writable (unmasked) blocks and stay positionally correlated.
    void CheckStartWriteShape(size_t requested_keys,
                              size_t writable_keys,
                              size_t returned_locations,
                              const std::string &context);
    // C2 for ReportEvent: item_results is either empty or one per event.
    void CheckReportEventShape(size_t events, size_t item_results, const std::string &context);

    // C3 is recorded once, at the end of a completed eviction pipeline. The
    // call site establishes the operation order; this method checks its item
    // accounting instead of maintaining a duplicate eviction state machine.
    void RecordCompletedEviction(uint64_t writable_items,
                                 uint64_t masked_items,
                                 uint64_t cold_allocations,
                                 uint64_t local_removals,
                                 uint64_t delete_requests);

    V6dCheckCounters Counters() const;
    std::vector<InvariantObservation> Snapshot(const std::string &behavior_type) const;

private:
    void RecordViolation(const char *check_name, const std::string &detail_json);

    ExpectedLocations &expected_;
    EvidenceSink &evidence_;
    std::string instance_id_;

    mutable std::mutex mutex_;
    V6dCheckCounters counters_;
};

} // namespace kvcm_swarm
