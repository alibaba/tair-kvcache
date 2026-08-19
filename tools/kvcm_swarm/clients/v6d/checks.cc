#include "tools/kvcm_swarm/clients/v6d/checks.h"

#include <algorithm>

#include "tools/kvcm_swarm/evidence/json_writer.h"

namespace kvcm_swarm {
namespace {
constexpr const char *kC1a = "C1a_selector_soundness";
constexpr const char *kC1b = "C1b_remote_availability";
constexpr const char *kC2 = "C2_batch_response_shape";
constexpr const char *kC3 = "C3_capacity_pressure_eviction";
constexpr const char *kC4 = "C4_server_metric_cross_check";
} // namespace

std::string ParseVineyardHostPort(const std::string &uri) {
    const std::string scheme = "vineyard://";
    if (uri.rfind(scheme, 0) != 0) {
        return {};
    }
    const size_t authority_begin = scheme.size();
    const size_t authority_end = uri.find('/', authority_begin);
    const std::string authority = authority_end == std::string::npos
                                      ? uri.substr(authority_begin)
                                      : uri.substr(authority_begin, authority_end - authority_begin);
    if (authority.find(':') == std::string::npos) {
        return {};
    }
    return authority;
}

std::string ParseVineyardMedium(const std::string &uri) {
    const std::string scheme = "vineyard://";
    if (uri.rfind(scheme, 0) != 0) {
        return {};
    }
    const size_t path_begin = uri.find('/', scheme.size());
    if (path_begin == std::string::npos) {
        return {};
    }
    size_t path_end = uri.find('?', path_begin + 1);
    if (path_end == std::string::npos) {
        path_end = uri.size();
    }
    return uri.substr(path_begin + 1, path_end - path_begin - 1);
}

LookupExpectation V6dChecks::BeforeLookup(LookupTier tier,
                                          const std::vector<LookupItem> &items,
                                          FullSelector selector,
                                          const ReporterIdentity &requester) const {
    LookupExpectation expectation;
    expectation.tier = tier;
    expectation.hot_backend_requested = tier == LookupTier::kHot;
    expectation.issued_at = Now();
    expectation.candidate_revision = expected_.candidate_revision();
    expectation.liveness_revision = expected_.liveness_revision();

    std::vector<size_t> unmasked;
    for (size_t i = 0; i < items.size(); ++i) {
        if (!items[i].masked) {
            unmasked.push_back(i);
        }
    }
    expectation.unmasked_keys = unmasked.size();
    if (tier != LookupTier::kHot || unmasked.empty()) {
        return expectation;
    }

    // PREFIX looks at the first unmasked key; COVERAGE at any unmasked key.
    if (selector == FullSelector::kPrefix) {
        const size_t index = unmasked.front();
        const auto snapshot = expected_.SnapshotCandidates(items[index].block_key, items[index].spec_name, requester);
        if (!snapshot.requester_has_possible_local && snapshot.has_confirmed_remote) {
            expectation.remote_eligible_indices.push_back(index);
        }
        return expectation;
    }
    for (const size_t index : unmasked) {
        const auto snapshot = expected_.SnapshotCandidates(items[index].block_key, items[index].spec_name, requester);
        if (!snapshot.requester_has_possible_local && snapshot.has_confirmed_remote) {
            expectation.remote_eligible_indices.push_back(index);
        }
    }
    return expectation;
}

void V6dChecks::OnLookupResult(const std::vector<LookupItem> &items,
                               const LookupExpectation &expectation,
                               const meta::GetCacheLocationsByBackendResponse &response,
                               const ReporterIdentity &requester,
                               bool response_ok,
                               std::vector<std::string> *hot_hit_host_ports,
                               std::vector<std::string> *cold_hit_uris) {
    hot_hit_host_ports->assign(items.size(), std::string());
    cold_hit_uris->assign(items.size(), std::string());

    // ---- C2: response shape ----
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++counters_.shape_checks;
    }
    if (response_ok && static_cast<size_t>(response.key_locations_size()) != items.size()) {
        JsonWriter writer(false);
        writer.BeginObject();
        writer.KeyString("reason", "key_locations length does not match block_keys length");
        writer.KeyUint("requested_keys", items.size());
        writer.KeyInt("returned_key_locations", response.key_locations_size());
        writer.EndObject();
        RecordViolation(kC2, writer.Take());
        std::lock_guard<std::mutex> lock(mutex_);
        ++counters_.c2_violations;
        return;
    }
    if (!response_ok) {
        return;
    }

    for (size_t index = 0; index < items.size(); ++index) {
        const auto &vector = response.key_locations(static_cast<int>(index));
        for (const auto &location : vector.locations()) {
            // A masked index is never sent to the backend, so any location
            // returned at that position breaks positional correlation.
            if (items[index].masked && location.location_specs_size() > 0) {
                JsonWriter writer(false);
                writer.BeginObject();
                writer.KeyString("reason", "location returned for a masked index");
                writer.KeyUint("index", index);
                writer.KeyInt("block_key", items[index].block_key);
                writer.EndObject();
                RecordViolation(kC2, writer.Take());
                std::lock_guard<std::mutex> lock(mutex_);
                ++counters_.c2_violations;
                continue;
            }
            const meta::LocationSpec *matching_spec = nullptr;
            for (const auto &spec : location.location_specs()) {
                if (spec.name() == items[index].spec_name) {
                    matching_spec = &spec;
                    break;
                }
            }
            if (matching_spec == nullptr) {
                // Every location must carry the spec requested for this key.
                JsonWriter writer(false);
                writer.BeginObject();
                writer.KeyString("reason", "returned location does not carry the requested spec for this key");
                writer.KeyUint("index", index);
                writer.KeyInt("block_key", items[index].block_key);
                writer.KeyString("requested_spec", items[index].spec_name);
                writer.KeyInt("storage_type", static_cast<int>(location.type()));
                writer.EndObject();
                RecordViolation(kC2, writer.Take());
                std::lock_guard<std::mutex> lock(mutex_);
                ++counters_.c2_violations;
                continue;
            }

            if (location.type() != meta::ST_EVENT_REPORT_L2) {
                // Cold-tier allocation: no machine owner, only URI provenance.
                if ((*cold_hit_uris)[index].empty()) {
                    (*cold_hit_uris)[index] = matching_spec->uri();
                }
                continue;
            }
            if (!expectation.hot_backend_requested) {
                // The query asked for cold backends only.
                JsonWriter writer(false);
                writer.BeginObject();
                writer.KeyString("reason", "event-report location returned for a cold-only backend selector");
                writer.KeyUint("index", index);
                writer.KeyInt("block_key", items[index].block_key);
                writer.KeyString("uri", matching_spec->uri());
                writer.EndObject();
                RecordViolation(kC1a, writer.Take());
                std::lock_guard<std::mutex> lock(mutex_);
                ++counters_.c1a_violations;
                continue;
            }

            // ---- C1a: hot selector soundness ----
            const std::string host_port = ParseVineyardHostPort(matching_spec->uri());
            HotLocationKey key;
            key.block_key = items[index].block_key;
            key.spec_name = items[index].spec_name;
            key.reporter = ReporterIdentity{instance_id_, host_port};
            const auto acceptance = expected_.CheckHotAcceptable(key, expectation.issued_at);
            const bool is_self = host_port == requester.host_ip_port;
            const std::string medium = ParseVineyardMedium(matching_spec->uri());
            const bool sound = !host_port.empty() && medium == "mem" && acceptance.known_reporter &&
                               !acceptance.retired_reporter && acceptance.state_allows;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++counters_.hot_locations_accepted;
            }
            if (!sound) {
                JsonWriter writer(false);
                writer.BeginObject();
                writer.KeyString("reason", "hot location is not attributable to a live, non-retired reporter");
                writer.KeyUint("index", index);
                writer.KeyInt("block_key", items[index].block_key);
                writer.KeyString("spec", items[index].spec_name);
                writer.KeyString("uri", matching_spec->uri());
                writer.KeyString("host_ip_port", host_port);
                writer.KeyString("medium", medium);
                writer.KeyBool("known_reporter", acceptance.known_reporter);
                writer.KeyBool("retired_reporter", acceptance.retired_reporter);
                writer.KeyString("expected_state", LocationStateName(acceptance.state));
                writer.KeyBool("removed_after_query", acceptance.removed_after_query);
                writer.EndObject();
                RecordViolation(kC1a, writer.Take());
                std::lock_guard<std::mutex> lock(mutex_);
                ++counters_.c1a_violations;
                continue;
            }
            if (!is_self) {
                (*hot_hit_host_ports)[index] = host_port;
            } else {
                // Self locations are excluded from remote reuse but are a legal
                // server answer; keep the raw fact for reporting.
                (*hot_hit_host_ports)[index] = std::string();
            }
        }
    }

    // ---- C1b: remote availability ----
    if (expectation.tier != LookupTier::kHot || expectation.remote_eligible_indices.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ++counters_.eligible_queries;
    const bool stable = expected_.candidate_revision() == expectation.candidate_revision &&
                        expected_.liveness_revision() == expectation.liveness_revision;
    if (!stable) {
        ++counters_.invalidated_queries;
        return;
    }
    ++counters_.stable_eligible_queries;
    const auto returned_eligible_remote =
        std::find_if(expectation.remote_eligible_indices.begin(),
                     expectation.remote_eligible_indices.end(),
                     [hot_hit_host_ports](size_t index) {
                         return index < hot_hit_host_ports->size() && !(*hot_hit_host_ports)[index].empty();
                     });
    if (returned_eligible_remote != expectation.remote_eligible_indices.end()) {
        ++counters_.stable_eligible_returned;
        return;
    }
    ++counters_.c1b_violations;
    JsonWriter writer(false);
    writer.BeginObject();
    writer.KeyString("reason", "stable remote-eligible query returned no legal remote hot location");
    writer.KeyUint("eligible_keys", expectation.remote_eligible_indices.size());
    writer.KeyString("requester", requester.host_ip_port);
    writer.KeyUint("unmasked_keys", expectation.unmasked_keys);
    writer.EndObject();
    evidence_.violations().Record(kC1b, writer.Take());
}

void V6dChecks::CheckStartWriteShape(size_t requested_keys,
                                     size_t writable_keys,
                                     size_t returned_locations,
                                     const std::string &context) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++counters_.shape_checks;
    }
    if (returned_locations == writable_keys) {
        return;
    }
    JsonWriter writer(false);
    writer.BeginObject();
    writer.KeyString("reason", "StartWriteCache returned a location count that does not match the writable blocks");
    writer.KeyString("context", context);
    writer.KeyUint("requested_keys", requested_keys);
    writer.KeyUint("writable_keys", writable_keys);
    writer.KeyUint("returned_locations", returned_locations);
    writer.EndObject();
    RecordViolation(kC2, writer.Take());
    std::lock_guard<std::mutex> lock(mutex_);
    ++counters_.c2_violations;
}

void V6dChecks::CheckReportEventShape(size_t events, size_t item_results, const std::string &context) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++counters_.shape_checks;
    }
    if (item_results == 0 || item_results == events) {
        return;
    }
    JsonWriter writer(false);
    writer.BeginObject();
    writer.KeyString("reason", "ReportEvent item_results must be empty or one per event");
    writer.KeyString("context", context);
    writer.KeyUint("events", events);
    writer.KeyUint("item_results", item_results);
    writer.EndObject();
    RecordViolation(kC2, writer.Take());
    std::lock_guard<std::mutex> lock(mutex_);
    ++counters_.c2_violations;
}

void V6dChecks::RecordCompletedEviction(uint64_t writable_items,
                                        uint64_t masked_items,
                                        uint64_t cold_allocations,
                                        uint64_t local_removals,
                                        uint64_t delete_requests) {
    const uint64_t total_items = writable_items + masked_items;
    const bool accounting_matches =
        cold_allocations == writable_items && local_removals == total_items && delete_requests == local_removals;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++counters_.eviction_operations;
        counters_.eviction_writable_items += writable_items;
        counters_.eviction_masked_items += masked_items;
        counters_.eviction_writable_completed += writable_items;
        counters_.eviction_masked_completed += masked_items;
        counters_.cold_allocations_confirmed += cold_allocations;
        ++counters_.write_sessions_closed;
        counters_.local_removals_after_write_close += local_removals;
        counters_.deletes_after_local_removal += delete_requests;
        if (!accounting_matches) {
            ++counters_.c3_violations;
        }
    }
    if (accounting_matches) {
        return;
    }
    JsonWriter writer(false);
    writer.BeginObject();
    writer.KeyString("reason", "completed eviction has inconsistent writable, allocation, removal or delete counts");
    writer.KeyUint("writable_items", writable_items);
    writer.KeyUint("masked_items", masked_items);
    writer.KeyUint("cold_allocations", cold_allocations);
    writer.KeyUint("local_removals", local_removals);
    writer.KeyUint("delete_requests", delete_requests);
    writer.EndObject();
    RecordViolation(kC3, writer.Take());
}

void V6dChecks::RecordViolation(const char *check_name, const std::string &detail_json) {
    evidence_.violations().Record(check_name, detail_json);
}

V6dCheckCounters V6dChecks::Counters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return counters_;
}

std::vector<InvariantObservation> V6dChecks::Snapshot(const std::string &behavior_type) const {
    V6dCheckCounters counters = Counters();
    std::vector<InvariantObservation> observations;

    {
        InvariantObservation o;
        o.behavior_type = behavior_type;
        o.check_name = kC1a;
        o.checked = counters.hot_locations_accepted;
        o.violations = counters.c1a_violations;
        o.counters["hot_locations_accepted"] = static_cast<int64_t>(counters.hot_locations_accepted);
        if (counters.hot_locations_accepted == 0) {
            o.status = CheckStatus::kNotRun;
            o.reason = "no remote hot location was returned by any lookup";
        } else if (counters.c1a_violations > 0) {
            o.status = CheckStatus::kFail;
            o.reason = "a returned hot location did not match instance/key/spec/backend or a live reporter";
        } else {
            o.status = CheckStatus::kPass;
            o.reason = "every accepted hot location matched key, spec, backend and a live reporter";
        }
        o.detail_preview = evidence_.violations().Preview(kC1a);
        observations.push_back(std::move(o));
    }
    {
        InvariantObservation o;
        o.behavior_type = behavior_type;
        o.check_name = kC1b;
        o.checked = counters.stable_eligible_queries;
        o.violations = counters.c1b_violations;
        o.counters["eligible_queries"] = static_cast<int64_t>(counters.eligible_queries);
        o.counters["stable_remote_eligible_queries"] = static_cast<int64_t>(counters.stable_eligible_queries);
        o.counters["invalidated_queries"] = static_cast<int64_t>(counters.invalidated_queries);
        o.counters["stable_eligible_returned"] = static_cast<int64_t>(counters.stable_eligible_returned);
        if (counters.stable_eligible_queries == 0) {
            o.status = CheckStatus::kNotRun;
            o.reason = "no stable remote-eligible lookup occurred; scenario coverage is insufficient";
        } else if (counters.c1b_violations > 0) {
            o.status = CheckStatus::kFail;
            o.reason = "a stable remote-eligible lookup returned no legal remote hot location";
        } else {
            o.status = CheckStatus::kPass;
            o.reason = "every stable remote-eligible lookup returned at least one legal remote hot location";
        }
        o.detail_preview = evidence_.violations().Preview(kC1b);
        observations.push_back(std::move(o));
    }
    {
        InvariantObservation o;
        o.behavior_type = behavior_type;
        o.check_name = kC2;
        o.checked = counters.shape_checks;
        o.violations = counters.c2_violations;
        o.counters["shape_checks"] = static_cast<int64_t>(counters.shape_checks);
        if (counters.shape_checks == 0) {
            o.status = CheckStatus::kNotRun;
            o.reason = "no batch response was inspected";
        } else if (counters.c2_violations > 0) {
            o.status = CheckStatus::kFail;
            o.reason = "a batch response violated length, ordering or per-item correlation";
        } else {
            o.status = CheckStatus::kPass;
            o.reason = "every batch response preserved length, ordering and per-item correlation";
        }
        o.detail_preview = evidence_.violations().Preview(kC2);
        observations.push_back(std::move(o));
    }
    {
        InvariantObservation o;
        o.behavior_type = behavior_type;
        o.check_name = kC3;
        o.checked = counters.eviction_operations;
        o.violations = counters.c3_violations;
        o.counters["eviction_operations"] = static_cast<int64_t>(counters.eviction_operations);
        o.counters["writable_items"] = static_cast<int64_t>(counters.eviction_writable_items);
        o.counters["masked_items"] = static_cast<int64_t>(counters.eviction_masked_items);
        o.counters["writable_completed"] = static_cast<int64_t>(counters.eviction_writable_completed);
        o.counters["masked_completed"] = static_cast<int64_t>(counters.eviction_masked_completed);
        o.counters["cold_allocations_confirmed"] = static_cast<int64_t>(counters.cold_allocations_confirmed);
        o.counters["write_sessions_closed"] = static_cast<int64_t>(counters.write_sessions_closed);
        o.counters["local_removals_after_write_close"] =
            static_cast<int64_t>(counters.local_removals_after_write_close);
        o.counters["deletes_after_local_removal"] = static_cast<int64_t>(counters.deletes_after_local_removal);
        if (counters.eviction_operations == 0) {
            o.status = CheckStatus::kNotRun;
            o.reason = "no capacity-driven eviction happened";
        } else if (counters.c3_violations > 0) {
            o.status = CheckStatus::kFail;
            o.reason = "an eviction violated the write-session / local-removal / BLOCK_DELETE order or the "
                       "writable/masked contract";
        } else {
            o.status = CheckStatus::kPass;
            o.reason = "every eviction closed its write session before local removal and BLOCK_DELETE";
        }
        o.detail_preview = evidence_.violations().Preview(kC3);
        observations.push_back(std::move(o));
    }
    {
        InvariantObservation o;
        o.behavior_type = behavior_type;
        o.check_name = kC4;
        o.status = CheckStatus::kInconclusive;
        o.checked = 0;
        o.violations = 0;
        o.reason = "TODO: KVCM exposes no server API with the same accounting basis, so a cross-check is not "
                   "possible in this version; this check is non-gating by design";
        observations.push_back(std::move(o));
    }
    return observations;
}

} // namespace kvcm_swarm
