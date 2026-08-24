#include "kv_cache_manager/manager/meta_searcher.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/config/instance_info.h"
#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

namespace {

void LogErrorCodes(const std::string &operation_name,
                   const std::vector<ErrorCode> &error_codes,
                   const kv_cache_manager::MetaSearcher::KeyVector &keys) {
    for (size_t i = 0; i < keys.size(); i++) {
        if (i >= error_codes.size()) {
            KVCM_LOG_WARN(
                "error_codes size %ld < keys size %ld in %s", error_codes.size(), keys.size(), operation_name.c_str());
            break;
        }
        if (error_codes[i] != ErrorCode::EC_OK && error_codes[i] != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("%s failed, keys[%lu](%lu) return %d", operation_name.c_str(), i, keys[i], error_codes[i]);
        }
    }
}

bool TryGetLocationSpecSize(const LocationSpec &spec, std::uint64_t &size) {
    size = 0;
    DataStorageUri uri(spec.uri());
    if (!uri.Valid()) {
        return false;
    }
    uri.GetParamAs<std::uint64_t>("size", size);
    return true;
}

std::uint64_t GetLocationSpecSize(const LocationSpec &spec) {
    std::uint64_t size = 0;
    (void)TryGetLocationSpecSize(spec, size);
    return size;
}

std::uint64_t GetLocationSpecsSize(const std::vector<LocationSpec> &specs) {
    std::uint64_t total_size = 0;
    for (const auto &loc_spec : specs) {
        total_size += GetLocationSpecSize(loc_spec);
    }
    return total_size;
}

void ApplyStorageUsageChange(MetaIndexer *meta_indexer,
                             DataStorageType type,
                             std::uint64_t old_size,
                             std::uint64_t new_size) {
    if (new_size >= old_size) {
        meta_indexer->AddStorageUsageByType(type, new_size - old_size);
    } else {
        meta_indexer->SubStorageUsageByType(type, old_size - new_size);
    }
}

struct StorageUsageChange {
    std::uint64_t old_size = 0;
    std::uint64_t new_size = 0;
    bool has_old = false;
};

template <typename SpecAccessor>
ErrorCode
ValidateConsistentSnapshotVersion(size_t spec_count, SpecAccessor spec_at, std::uint64_t *out_total_size = nullptr) {
    if (spec_count == 0) {
        return EC_BADARGS;
    }
    bool has_snapshot_version = false;
    std::string snapshot_version;
    bool has_unversioned_spec = false;
    std::uint64_t total_size = 0;
    std::unordered_set<std::string_view> spec_names;
    if (spec_count > 1) {
        spec_names.reserve(spec_count);
    }
    for (size_t spec_index = 0; spec_index < spec_count; ++spec_index) {
        const auto &spec = spec_at(spec_index);
        const DataStorageUri uri(spec.uri());
        if (spec.name().empty() || (spec_count > 1 && !spec_names.insert(std::string_view(spec.name())).second) ||
            !uri.Valid()) {
            return EC_BADARGS;
        }
        std::uint64_t spec_size = 0;
        uri.GetParamAs<std::uint64_t>("size", spec_size);
        if (spec_size > std::numeric_limits<std::uint64_t>::max() - total_size) {
            return EC_BADARGS;
        }
        total_size += spec_size;
        const size_t version_param_count =
            SnapshotUriUtils::CountUriParam(spec.uri(), SnapshotUriUtils::kSnapshotVersionParam);
        if (version_param_count == 0) {
            if (has_snapshot_version) {
                return EC_BADARGS;
            }
            has_unversioned_spec = true;
            continue;
        }
        SnapshotUriInfo info;
        if (has_unversioned_spec || version_param_count != 1 || !SnapshotUriUtils::ParseSnapshotUriInfo(uri, info)) {
            return EC_BADARGS;
        }
        if (!has_snapshot_version) {
            snapshot_version = std::move(info.version);
            has_snapshot_version = true;
        } else if (info.version != snapshot_version) {
            return EC_BADARGS;
        }
    }
    if (out_total_size) {
        *out_total_size = total_size;
    }
    return EC_OK;
}

ErrorCode ValidateConsistentSnapshotVersion(const std::vector<LocationSpec> &specs,
                                            std::uint64_t *out_total_size = nullptr) {
    return ValidateConsistentSnapshotVersion(
        specs.size(), [&specs](size_t index) -> const LocationSpec & { return specs[index]; }, out_total_size);
}

ErrorCode ValidateConsistentSnapshotVersion(const MetaSearcher::MergeLocationSpecsTask &task,
                                            std::uint64_t *out_total_size = nullptr) {
    return ValidateConsistentSnapshotVersion(
        task.SpecCount(), [&task](size_t index) -> const LocationSpec & { return task.SpecAt(index); }, out_total_size);
}

void MergeLocationSpecsByName(std::vector<LocationSpec> &merged_specs,
                              const MetaSearcher::MergeLocationSpecsTask &task) {
    // Snapshot generations are reconciliation/cleanup tags, not a visibility
    // fence. After a KVCM restart, a new delta may use a fresh generation while
    // untouched specs still carry an older one. Preserve those specs and
    // overwrite only names present in this delta. Normalize any legacy
    // duplicate names in place with the same last-value-wins behavior the old
    // std::map implementation provided.
    size_t unique_count = 0;
    for (size_t i = 0; i < merged_specs.size(); ++i) {
        auto duplicate =
            std::find_if(merged_specs.begin(),
                         merged_specs.begin() + unique_count,
                         [&merged_specs, i](const auto &spec) { return spec.name() == merged_specs[i].name(); });
        if (duplicate != merged_specs.begin() + unique_count) {
            *duplicate = std::move(merged_specs[i]);
            continue;
        }
        if (i != unique_count) {
            merged_specs[unique_count] = std::move(merged_specs[i]);
        }
        ++unique_count;
    }
    merged_specs.resize(unique_count);

    for (size_t spec_index = 0; spec_index < task.SpecCount(); ++spec_index) {
        const auto &spec = task.SpecAt(spec_index);
        auto existing = std::find_if(merged_specs.begin(), merged_specs.end(), [&spec](const auto &candidate) {
            return candidate.name() == spec.name();
        });
        if (existing == merged_specs.end()) {
            merged_specs.push_back(spec);
        } else {
            *existing = spec;
        }
    }
    std::sort(merged_specs.begin(), merged_specs.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.name() < rhs.name();
    });
}

CacheLocationConstPtr SelectAndMergeForMatch(SelectLocationPolicy *policy,
                                             CacheLocationMap &location_map,
                                             CheckLocDataExistFunc check_loc_data_exist,
                                             std::vector<std::string> &out_prune_loc_ids) {
    // Filter valid locations into a shared map.
    CacheLocationMap valid_map;
    for (auto &[id, loc_ptr] : location_map) {
        if (!loc_ptr) {
            continue;
        }
        if (loc_ptr->status() != CacheLocationStatus::CLS_SERVING) {
            continue;
        }
        if (check_loc_data_exist && !check_loc_data_exist(*loc_ptr)) {
            if (!IsEventReportStorageType(loc_ptr->type())) {
                out_prune_loc_ids.push_back(id);
            }
            continue;
        }
        valid_map.try_emplace(id, loc_ptr);
    }
    if (valid_map.empty()) {
        return std::make_shared<CacheLocation>();
    }

    // Use the policy to select one winning location, which determines the
    // target storage backend instance.
    std::vector<std::string> unused_prune_ids;
    CacheLocationConstPtr winner = policy->SelectForMatch(valid_map, nullptr, unused_prune_ids);
    if (!winner || winner->location_specs().empty()) {
        return std::make_shared<CacheLocation>();
    }

    // Collect all specs from every valid location that belongs to the same
    // storage backend as the winner, dedup by spec name.
    std::map<std::string, LocationSpec> merged_specs;
    for (const auto &[id, loc_ptr] : valid_map) {
        if (!loc_ptr || !policy->IsSameDataStorage(*loc_ptr, *winner)) {
            continue;
        }
        for (const auto &spec : loc_ptr->location_specs()) {
            merged_specs.try_emplace(spec.name(), spec);
        }
    }

    if (merged_specs.empty()) {
        return std::make_shared<CacheLocation>();
    }

    // NOTE: this is an aggregated view merging
    // specs from multiple locations, not a real stored entity. Downstream
    // CacheLocationView / proto serialization never accesses id either.
    std::string representative_id = winner->id() + "_merged";
    auto result = std::make_shared<CacheLocation>();
    result->set_id(std::move(representative_id));
    result->set_status(CacheLocationStatus::CLS_SERVING);
    result->set_type(winner->type());
    std::vector<LocationSpec> specs;
    specs.reserve(merged_specs.size());
    for (auto &[name, spec] : merged_specs) {
        specs.push_back(std::move(spec));
    }
    result->set_spec_size(specs.size());
    result->set_location_specs(std::move(specs));
    return result;
}

const LocationSpec *FindRequestedSpec(const CacheLocation &loc, std::string_view requested_spec_name) {
    if (requested_spec_name.empty()) {
        return loc.location_specs().empty() ? nullptr : &loc.location_specs().front();
    }
    const auto it =
        std::find_if(loc.location_specs().begin(),
                     loc.location_specs().end(),
                     [requested_spec_name](const LocationSpec &spec) { return spec.name() == requested_spec_name; });
    return it == loc.location_specs().end() ? nullptr : &*it;
}

bool MatchesRequestedSpec(const CacheLocation &loc, std::string_view requested_spec_name) {
    return requested_spec_name.empty() || FindRequestedSpec(loc, requested_spec_name) != nullptr;
}

std::string ExtractPeerAddrFromLocation(const CacheLocation &loc, std::string_view requested_spec_name) {
    const auto *spec = FindRequestedSpec(loc, requested_spec_name);
    if (spec == nullptr) {
        return {};
    }
    StandardUri uri(spec->uri());
    if (!uri.Valid() || uri.GetHostName().empty()) {
        return {};
    }
    return uri.GetHostName() + ":" + std::to_string(uri.GetPort());
}

struct V6DPeerSelection {
    std::string peer_addr;
    std::vector<size_t> covered_indices;
};

V6DPeerSelection SelectV6DByPrefix(const std::vector<size_t> &candidate_indices,
                                   const std::unordered_map<size_t, std::vector<std::string>> &remote_peer_candidates) {
    if (candidate_indices.empty()) {
        return {};
    }
    size_t first_idx = candidate_indices[0];
    auto it = remote_peer_candidates.find(first_idx);
    if (it == remote_peer_candidates.end()) {
        return {};
    }

    V6DPeerSelection best;
    for (const auto &addr : it->second) {
        std::vector<size_t> prefix_covered;
        for (size_t ci : candidate_indices) {
            auto ci_it = remote_peer_candidates.find(ci);
            if (ci_it == remote_peer_candidates.end()) {
                break;
            }
            const auto &addrs = ci_it->second;
            if (std::find(addrs.begin(), addrs.end(), addr) == addrs.end()) {
                break;
            }
            prefix_covered.push_back(ci);
        }
        if (prefix_covered.size() > best.covered_indices.size() ||
            (prefix_covered.size() == best.covered_indices.size() &&
             (best.peer_addr.empty() || addr < best.peer_addr))) {
            best.peer_addr = addr;
            best.covered_indices = std::move(prefix_covered);
        }
    }
    return best;
}

V6DPeerSelection
SelectV6DByCoverage(const std::vector<size_t> &candidate_indices,
                    const std::unordered_map<size_t, std::vector<std::string>> &remote_peer_candidates) {
    std::unordered_map<std::string, std::vector<size_t>> addr_to_indices;
    for (size_t ci : candidate_indices) {
        auto it = remote_peer_candidates.find(ci);
        if (it == remote_peer_candidates.end()) {
            continue;
        }
        for (const auto &addr : it->second) {
            addr_to_indices[addr].push_back(ci);
        }
    }
    if (addr_to_indices.empty()) {
        return {};
    }
    V6DPeerSelection best;
    for (auto &[addr, indices] : addr_to_indices) {
        if (indices.size() > best.covered_indices.size() ||
            (indices.size() == best.covered_indices.size() && (best.peer_addr.empty() || addr < best.peer_addr))) {
            best.peer_addr = addr;
            best.covered_indices = indices;
        }
    }
    return best;
}

CacheLocationMap FilterValidLocations(const CacheLocationMap &location_map,
                                      CheckLocDataExistFunc check_loc_data_exist,
                                      std::vector<std::string> &out_prune_loc_ids) {
    out_prune_loc_ids.clear();
    CacheLocationMap valid;
    for (const auto &[id, loc] : location_map) {
        if (!loc)
            continue;
        if (loc->status() != CacheLocationStatus::CLS_SERVING)
            continue;
        if (check_loc_data_exist && !check_loc_data_exist(*loc)) {
            if (!IsEventReportStorageType(loc->type())) {
                out_prune_loc_ids.push_back(id);
            }
            continue;
        }
        valid.try_emplace(id, loc);
    }
    return valid;
}

using MediumViewSet = std::unordered_set<std::string_view>;

constexpr size_t WordCountForBits(size_t bit_count) noexcept { return bit_count == 0 ? 0 : 1 + (bit_count - 1) / 64; }

// Measures the union of concurrently executing callback intervals. Summing
// per-worker durations would turn the existing wall-time metric into CPU time
// and could make it exceed the enclosing request latency.
class ConcurrentWallTimer {
public:
    class Scope {
    public:
        explicit Scope(ConcurrentWallTimer *timer) noexcept : timer_(timer) { timer_->Begin(); }
        ~Scope() { timer_->End(); }

        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;

    private:
        ConcurrentWallTimer *timer_;
    };

    [[nodiscard]] Scope Measure() noexcept { return Scope(this); }
    [[nodiscard]] int64_t elapsed_us() const noexcept { return elapsed_us_.load(std::memory_order_relaxed); }

private:
    void Begin() noexcept {
        const int64_t now = TimestampUtil::GetCurrentTimeUs();
        if (active_.fetch_add(1, std::memory_order_acq_rel) == 0) {
            interval_start_us_.store(now, std::memory_order_release);
        }
    }

    void End() noexcept {
        const int64_t now = TimestampUtil::GetCurrentTimeUs();
        const int64_t interval_start = interval_start_us_.load(std::memory_order_acquire);
        if (active_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            elapsed_us_.fetch_add(std::max<int64_t>(now - interval_start, 0), std::memory_order_relaxed);
        }
    }

private:
    std::atomic<size_t> active_{0};
    std::atomic<int64_t> interval_start_us_{0};
    std::atomic<int64_t> elapsed_us_{0};
};

MediumViewSet BuildMediumViewSet(const std::vector<std::string> &medium_filter) {
    MediumViewSet mediums;
    mediums.reserve(medium_filter.size());
    for (const auto &medium : medium_filter) {
        mediums.emplace(medium);
    }
    return mediums;
}

bool IsMediumMatched(const StandardUri &uri, const MediumViewSet &medium_set) {
    if (medium_set.empty()) {
        return true;
    }
    std::string_view medium(uri.GetPath());
    if (!medium.empty() && medium.front() == '/') {
        medium.remove_prefix(1);
    }
    return medium_set.find(medium) != medium_set.end();
}

bool IsReporterMediumMatched(std::string_view medium, const MediumViewSet &medium_set) {
    return medium_set.empty() || medium_set.find(medium) != medium_set.end();
}

template <typename LocationRange, typename Visitor>
void VisitHostSpecsForOneKey(const LocationRange &locations,
                             const CheckLocDataExistFunc &check_loc_data_exist,
                             const MetaSearcher::CheckHostCacheLocationFunc *request_check_location,
                             const MediumViewSet &medium_set,
                             bool visit_spec_names,
                             Visitor &&visitor) {
    for (const auto &loc : locations) {
        // GetHostCacheState reports readable data, not metadata intent. A URI
        // attached to WRITING/DELETING/NEW must never become a cache hit.
        if (!loc || loc->status() != CacheLocationStatus::CLS_SERVING || loc->location_specs().empty()) {
            continue;
        }

        MetaSearcher::HostCacheLocationInfo location_info;
        if (request_check_location) {
            if (!(*request_check_location)(*loc, location_info)) {
                continue;
            }
        } else if (check_loc_data_exist && !check_loc_data_exist(*loc)) {
            continue;
        }

        const bool is_event_report = IsEventReportStorageType(loc->type());
        bool has_reporter_identity = is_event_report && location_info.has_reporter_identity;
        if (!has_reporter_identity && is_event_report) {
            std::string_view storage_type;
            has_reporter_identity = SnapshotUriUtils::ParseEventReportLocationIdView(
                loc->id(), storage_type, location_info.reporter_medium, location_info.reporter_host);
        }
        const bool is_vineyard = loc->type() == DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2;
        if (has_reporter_identity) {
            if (!IsReporterMediumMatched(location_info.reporter_medium, medium_set) ||
                location_info.reporter_host.empty()) {
                continue;
            }
            // The request checker already scans every EventReport URI while
            // applying the reporter generation fence. Reuse that result.
            const bool specs_already_validated =
                request_check_location != nullptr && is_event_report && location_info.has_reporter_identity;
            if (!visit_spec_names) {
                if (specs_already_validated) {
                    visitor(location_info.reporter_host, std::string_view{}, is_vineyard);
                    continue;
                }
                for (const auto &spec : loc->location_specs()) {
                    if (StandardUri(spec.uri()).Valid()) {
                        visitor(location_info.reporter_host, std::string_view{}, is_vineyard);
                        break;
                    }
                }
                continue;
            }
            for (const auto &spec : loc->location_specs()) {
                if (!specs_already_validated && !StandardUri(spec.uri()).Valid()) {
                    continue;
                }
                visitor(location_info.reporter_host, std::string_view(spec.name()), is_vineyard);
            }
            continue;
        }

        for (const auto &spec : loc->location_specs()) {
            const StandardUri uri(spec.uri());
            if (!uri.Valid() || !IsMediumMatched(uri, medium_set)) {
                continue;
            }
            const std::string host = uri.GetHostPort();
            if (!host.empty()) {
                visitor(std::string_view(host), std::string_view(spec.name()), false);
            }
        }
    }
}

template <typename LocationRange>
void BuildHostsForOneKey(const LocationRange &locations,
                         const CheckLocDataExistFunc &check_loc_data_exist,
                         const MetaSearcher::CheckHostCacheLocationFunc *request_check_location,
                         const MediumViewSet &medium_set,
                         std::vector<std::string> &hosts) {
    VisitHostSpecsForOneKey(locations,
                            check_loc_data_exist,
                            request_check_location,
                            medium_set,
                            false,
                            [&hosts](std::string_view host, std::string_view, bool) { hosts.emplace_back(host); });
    std::sort(hosts.begin(), hosts.end());
    hosts.erase(std::unique(hosts.begin(), hosts.end()), hosts.end());
}

template <typename LocationRange>
void BuildCandidatePresenceForOneKey(const LocationRange &locations,
                                     const CheckLocDataExistFunc &check_loc_data_exist,
                                     const MetaSearcher::CheckHostCacheLocationFunc *request_check_location,
                                     const MediumViewSet &medium_set,
                                     const std::vector<std::string> &candidate_hosts,
                                     std::uint64_t *presence_words) {
    VisitHostSpecsForOneKey(locations,
                            check_loc_data_exist,
                            request_check_location,
                            medium_set,
                            false,
                            [&candidate_hosts, presence_words](std::string_view host, std::string_view, bool) {
                                const auto it =
                                    std::lower_bound(candidate_hosts.begin(),
                                                     candidate_hosts.end(),
                                                     host,
                                                     [](const std::string &candidate, std::string_view value) {
                                                         return std::string_view(candidate) < value;
                                                     });
                                if (it != candidate_hosts.end() && std::string_view(*it) == host) {
                                    const size_t index = static_cast<size_t>(it - candidate_hosts.begin());
                                    presence_words[index / 64] |= std::uint64_t{1} << (index % 64);
                                }
                            });
}

using HostToSpecNames = std::map<std::string, std::set<std::string>>;
using KeyToHostSpecNames = std::vector<HostToSpecNames>; // key -> host -> spec names

std::vector<size_t> SelectTopHostIndicesByLocal(const std::vector<MetaSearcher::HostCacheMatch> &host_matches,
                                                size_t p2p_host_count) {
    if (p2p_host_count == 0) {
        return {};
    }
    std::vector<size_t> host_indices;
    host_indices.reserve(host_matches.size());
    for (size_t i = 0; i < host_matches.size(); ++i) {
        if (host_matches[i].local > 0) {
            host_indices.push_back(i);
        }
    }
    const size_t selected_count = std::min(p2p_host_count, host_indices.size());
    std::partial_sort(host_indices.begin(),
                      host_indices.begin() + selected_count,
                      host_indices.end(),
                      [&host_matches](size_t lhs, size_t rhs) {
                          if (host_matches[lhs].local != host_matches[rhs].local) {
                              return host_matches[lhs].local > host_matches[rhs].local;
                          }
                          return host_matches[lhs].host_ip_port < host_matches[rhs].host_ip_port;
                      });
    host_indices.resize(selected_count);
    return host_indices;
}

template <typename LocationRange>
void BuildHostSpecNamesForOneKey(const LocationRange &locations,
                                 const CheckLocDataExistFunc &check_loc_data_exist,
                                 const MetaSearcher::CheckHostCacheLocationFunc *request_check_location,
                                 const MediumViewSet &medium_set,
                                 HostToSpecNames &host_specs,
                                 HostToSpecNames &vineyard_host_specs) {
    VisitHostSpecsForOneKey(
        locations,
        check_loc_data_exist,
        request_check_location,
        medium_set,
        true,
        [&host_specs, &vineyard_host_specs](std::string_view host, std::string_view spec_name, bool is_vineyard) {
            auto &local_names = host_specs[std::string(host)];
            local_names.emplace(spec_name);
            if (is_vineyard) {
                vineyard_host_specs[std::string(host)].emplace(spec_name);
            }
        });
}

bool HasAllLocationSpecGroups(const std::set<std::string> &spec_names,
                              const std::vector<const LocationSpecGroup *> &groups);

V6DPeerSelection SelectP2PByPrefix(const std::string &target_host,
                                   const KeyToHostSpecNames &local_specs,
                                   const KeyToHostSpecNames &vineyard_specs,
                                   const std::vector<const LocationSpecGroup *> &required_groups = {}) {
    assert(local_specs.size() == vineyard_specs.size());
    std::vector<size_t> candidate_indices;
    std::unordered_map<size_t, std::vector<std::string>> remote_peer_candidates;
    for (size_t i = 0; i < local_specs.size(); ++i) {
        auto local_it = local_specs[i].find(target_host);
        const bool has_local_block = local_it != local_specs[i].end();
        const bool local_hit = required_groups.empty()
                                   ? has_local_block
                                   : has_local_block && HasAllLocationSpecGroups(local_it->second, required_groups);
        if (local_hit) {
            continue;
        }

        std::vector<std::string> candidates;
        for (const auto &[peer, specs] : vineyard_specs[i]) {
            if (peer == target_host || specs.empty()) {
                continue;
            }
            if (!required_groups.empty()) {
                std::set<std::string> merged_specs = specs;
                if (has_local_block) {
                    merged_specs.insert(local_it->second.begin(), local_it->second.end());
                }
                if (!HasAllLocationSpecGroups(merged_specs, required_groups)) {
                    continue;
                }
            }
            candidates.push_back(peer);
        }
        if (candidates.empty()) {
            break;
        }
        candidate_indices.push_back(i);
        remote_peer_candidates.emplace(i, std::move(candidates));
    }
    return SelectV6DByPrefix(candidate_indices, remote_peer_candidates);
}

bool HasLocationSpecGroup(const std::set<std::string> &local_specs,
                          const std::set<std::string> &peer_specs,
                          const LocationSpecGroup &group) {
    return std::all_of(group.spec_names().begin(), group.spec_names().end(), [&](const std::string &spec_name) {
        return local_specs.find(spec_name) != local_specs.end() || peer_specs.find(spec_name) != peer_specs.end();
    });
}

V6DPeerSelection SelectP2PGroupByPrefix(const std::string &target_host,
                                        const KeyToHostSpecNames &local_specs,
                                        const KeyToHostSpecNames &vineyard_specs,
                                        const LocationSpecGroup &required_group) {
    assert(local_specs.size() == vineyard_specs.size());
    const std::vector<const LocationSpecGroup *> required_groups{&required_group};
    std::vector<size_t> candidate_indices;
    std::unordered_map<size_t, std::vector<std::string>> remote_peer_candidates;
    const std::set<std::string> empty_specs;
    for (size_t i = 0; i < local_specs.size(); ++i) {
        const auto local_it = local_specs[i].find(target_host);
        const auto &target_specs = local_it == local_specs[i].end() ? empty_specs : local_it->second;
        if (HasAllLocationSpecGroups(target_specs, required_groups)) {
            continue;
        }

        std::vector<std::string> candidates;
        for (const auto &[peer, peer_specs] : vineyard_specs[i]) {
            if (peer == target_host) {
                continue;
            }
            if (HasLocationSpecGroup(target_specs, peer_specs, required_group)) {
                candidates.push_back(peer);
            }
        }
        if (candidates.empty()) {
            break;
        }
        candidate_indices.push_back(i);
        remote_peer_candidates.emplace(i, std::move(candidates));
    }
    return SelectV6DByPrefix(candidate_indices, remote_peer_candidates);
}

struct GroupAwareP2PSelection {
    V6DPeerSelection selection;
    std::vector<size_t> query_block_indices;
    std::vector<const LocationSpecGroup *> query_groups;
};

GroupAwareP2PSelection SelectP2PGroupsByCoverage(const std::string &target_host,
                                                 const KeyToHostSpecNames &local_specs,
                                                 const KeyToHostSpecNames &vineyard_specs,
                                                 size_t block_count,
                                                 const std::vector<const LocationSpecGroup *> &required_groups) {
    assert(local_specs.size() == vineyard_specs.size());
    std::vector<size_t> candidate_indices;
    std::unordered_map<size_t, std::vector<std::string>> remote_peer_candidates;
    std::vector<size_t> query_block_indices;
    std::vector<const LocationSpecGroup *> query_groups;
    const std::set<std::string> empty_specs;
    for (const auto *group : required_groups) {
        const std::vector<const LocationSpecGroup *> one_group{group};
        for (size_t block_index = 0; block_index < block_count; ++block_index) {
            const auto local_it = local_specs[block_index].find(target_host);
            const auto &target_specs = local_it == local_specs[block_index].end() ? empty_specs : local_it->second;
            if (HasAllLocationSpecGroups(target_specs, one_group)) {
                continue;
            }

            std::vector<std::string> candidates;
            for (const auto &[peer, peer_specs] : vineyard_specs[block_index]) {
                if (peer == target_host) {
                    continue;
                }
                if (HasLocationSpecGroup(target_specs, peer_specs, *group)) {
                    candidates.push_back(peer);
                }
            }
            if (candidates.empty()) {
                continue;
            }
            const size_t query_index = query_block_indices.size();
            query_block_indices.push_back(block_index);
            query_groups.push_back(group);
            candidate_indices.push_back(query_index);
            remote_peer_candidates.emplace(query_index, std::move(candidates));
        }
    }
    return GroupAwareP2PSelection{SelectV6DByCoverage(candidate_indices, remote_peer_candidates),
                                  std::move(query_block_indices),
                                  std::move(query_groups)};
}

void MergeLocationSpecGroup(std::set<std::string> &target_specs,
                            const std::set<std::string> &source_specs,
                            const LocationSpecGroup &group) {
    for (const auto &spec_name : group.spec_names()) {
        if (source_specs.find(spec_name) != source_specs.end()) {
            target_specs.insert(spec_name);
        }
    }
}

std::vector<std::set<std::string>> MergeHostAndP2PSpecs(const std::string &target_host,
                                                        const KeyToHostSpecNames &local_specs,
                                                        const KeyToHostSpecNames &vineyard_specs,
                                                        const V6DPeerSelection &selection) {
    assert(local_specs.size() == vineyard_specs.size());
    std::vector<std::set<std::string>> merged_specs(local_specs.size());
    for (size_t i = 0; i < local_specs.size(); ++i) {
        if (auto local_it = local_specs[i].find(target_host); local_it != local_specs[i].end()) {
            merged_specs[i] = local_it->second;
        }
    }
    for (size_t i : selection.covered_indices) {
        auto peer_it = vineyard_specs[i].find(selection.peer_addr);
        if (peer_it != vineyard_specs[i].end()) {
            merged_specs[i].insert(peer_it->second.begin(), peer_it->second.end());
        }
    }
    return merged_specs;
}

int64_t ComputePrefixMatchBlocks(const std::vector<std::set<std::string>> &specs_by_key, bool use_eagle_pop) {
    int64_t prefix_len = 0;
    for (const auto &specs : specs_by_key) {
        if (specs.empty()) {
            break;
        }
        ++prefix_len;
    }
    if (use_eagle_pop) {
        prefix_len = std::max<int64_t>(prefix_len - 1, 0);
    }
    return prefix_len;
}

ErrorCode ClassifySpecGroups(RequestContext *request_context,
                             const std::vector<LocationSpecGroup> &location_spec_groups,
                             std::vector<const LocationSpecGroup *> &full_groups,
                             std::vector<const LocationSpecGroup *> &mamba_state_groups) {
    full_groups.clear();
    mamba_state_groups.clear();
    for (const auto &group : location_spec_groups) {
        const auto &group_name = group.name();
        const char category = group_name.empty() ? '?' : group_name.front();
        switch (category) {
        case 'F':
            full_groups.push_back(&group);
            break;
        case 'L':
            mamba_state_groups.push_back(&group);
            break;
        case 'W':
        case 'C':
        case 'E':
        case 'X': {
            std::string error_msg =
                "unsupported location spec category for QT_PREFIX_MATCH_WITH_MAMBA, group: " + group_name +
                ", category: " + std::string(1, category);
            request_context->error_tracer()->AddErrorMsg(error_msg);
            KVCM_LOG_WARN("%s", error_msg.c_str());
            return EC_BADARGS;
        }
        default: {
            std::string error_msg =
                "invalid location spec category for QT_PREFIX_MATCH_WITH_MAMBA, group: " + group_name +
                ", category: " + std::string(1, category);
            request_context->error_tracer()->AddErrorMsg(error_msg);
            KVCM_LOG_WARN("%s", error_msg.c_str());
            return EC_BADARGS;
        }
        }
    }

    if (full_groups.empty() || mamba_state_groups.empty()) {
        std::string group_names = Jsonizable::ToJsonString(location_spec_groups);
        std::string error_msg =
            full_groups.empty() ? "no full location spec group" : "no mamba state location spec group";
        error_msg += ", location_spec_groups: " + group_names;
        request_context->error_tracer()->AddErrorMsg(error_msg);
        KVCM_LOG_WARN("%s", error_msg.c_str());
        return EC_BADARGS;
    }
    return EC_OK;
}

bool HasAllLocationSpecGroups(const std::set<std::string> &spec_names,
                              const std::vector<const LocationSpecGroup *> &groups) {
    for (const auto *group : groups) {
        for (const auto &spec_name : group->spec_names()) {
            if (spec_names.find(spec_name) == spec_names.end()) {
                return false;
            }
        }
    }
    return true;
}

int64_t ComputeMambaPrefixMatchBlocks(const std::vector<std::set<std::string>> &specs_by_key,
                                      bool use_eagle_pop,
                                      const std::vector<const LocationSpecGroup *> &full_groups,
                                      const std::vector<const LocationSpecGroup *> &mamba_state_groups) {
    size_t full_prefix_len = 0;
    for (; full_prefix_len < specs_by_key.size(); ++full_prefix_len) {
        if (!HasAllLocationSpecGroups(specs_by_key[full_prefix_len], full_groups)) {
            break;
        }
    }
    if (use_eagle_pop && full_prefix_len > 0) {
        --full_prefix_len;
    }
    for (size_t offset = full_prefix_len; offset > 0; --offset) {
        const size_t index = offset - 1;
        if (HasAllLocationSpecGroups(specs_by_key[index], mamba_state_groups)) {
            return static_cast<int64_t>(index + 1);
        }
    }
    return 0;
}

ErrorCode PrefixMatchByHostWithoutP2P(MetaIndexer *meta_indexer,
                                      const CheckLocDataExistFunc &check_loc_data_exist,
                                      RequestContext *request_context,
                                      const MetaSearcher::KeyVector &keys,
                                      bool use_eagle_pop,
                                      const std::vector<std::string> &medium_filter,
                                      std::vector<MetaSearcher::HostCacheMatch> &out_matches,
                                      const MetaSearcher::CheckHostCacheLocationFunc *request_check_location) {
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    const MediumViewSet medium_set = BuildMediumViewSet(medium_filter);
    std::vector<std::string> candidate_hosts;
    std::unique_ptr<std::atomic<std::size_t>[]> prefix_stops;
    std::size_t presence_word_count = 0;
    ConcurrentWallTimer projection_wall_timer;

    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    const auto visitor =
        [&keys,
         &check_loc_data_exist,
         request_check_location,
         &medium_set,
         &candidate_hosts,
         &prefix_stops,
         &presence_word_count,
         &projection_wall_timer](std::size_t begin, const CompactLocationsPerKey &locations, std::size_t valid_count) {
            auto projection_scope = projection_wall_timer.Measure();
            std::size_t first_local_index = 0;
            if (begin == 0) {
                BuildHostsForOneKey(
                    locations[0], check_loc_data_exist, request_check_location, medium_set, candidate_hosts);
                if (candidate_hosts.empty()) {
                    return size_t{0};
                }
                presence_word_count = WordCountForBits(candidate_hosts.size());
                prefix_stops = std::make_unique<std::atomic<std::size_t>[]>(candidate_hosts.size());
                for (std::size_t i = 0; i < candidate_hosts.size(); ++i) {
                    prefix_stops[i].store(keys.size(), std::memory_order_relaxed);
                }
                first_local_index = 1;
            }

            std::vector<std::uint64_t> presence_words(presence_word_count, 0);
            for (std::size_t local_index = first_local_index; local_index < valid_count; ++local_index) {
                const std::size_t key_index = begin + local_index;
                bool any_host_needs_key = false;
                for (std::size_t host_index = 0; host_index < candidate_hosts.size(); ++host_index) {
                    if (key_index < prefix_stops[host_index].load(std::memory_order_relaxed)) {
                        any_host_needs_key = true;
                        break;
                    }
                }
                if (!any_host_needs_key) {
                    // Prefix stops are monotonic. Once every candidate has
                    // stopped at or before this key, no later key in this
                    // ordered callback range can change the result.
                    break;
                }

                std::fill(presence_words.begin(), presence_words.end(), 0);
                BuildCandidatePresenceForOneKey(locations[local_index],
                                                check_loc_data_exist,
                                                request_check_location,
                                                medium_set,
                                                candidate_hosts,
                                                presence_words.data());
                for (std::size_t host_index = 0; host_index < candidate_hosts.size(); ++host_index) {
                    std::size_t current = prefix_stops[host_index].load(std::memory_order_relaxed);
                    if (key_index >= current) {
                        continue;
                    }
                    const std::uint64_t bit = std::uint64_t{1} << (host_index % 64);
                    if ((presence_words[host_index / 64] & bit) != 0) {
                        continue;
                    }
                    while (key_index < current &&
                           !prefix_stops[host_index].compare_exchange_weak(
                               current, key_index, std::memory_order_relaxed, std::memory_order_relaxed)) {}
                }
            }

            bool every_host_stopped = true;
            std::size_t last_required_key = 0;
            for (std::size_t host_index = 0; host_index < candidate_hosts.size(); ++host_index) {
                const std::size_t stop = prefix_stops[host_index].load(std::memory_order_relaxed);
                if (stop == keys.size()) {
                    every_host_stopped = false;
                    break;
                }
                last_required_key = std::max(last_required_key, stop);
            }
            return every_host_stopped ? last_required_key : keys.size();
        };
    const auto result = meta_indexer->VisitLocationValuesForPrefix(request_context, keys, visitor);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, host_projection_time_us, projection_wall_timer.elapsed_us());
    const std::size_t valid_key_count = result.valid_key_count;
    if (valid_key_count < keys.size() && result.terminal_ec != EC_OK) {
        KVCM_LOG_DEBUG("prefix match by host end because Get keys[%lu](%lu) return %d",
                       valid_key_count,
                       keys[valid_key_count],
                       result.terminal_ec);
        if (result.terminal_ec != ErrorCode::EC_NOENT) {
            request_context->error_tracer()->AddErrorMsg("prefix match metadata read failed");
            return result.terminal_ec;
        }
    }
    if (candidate_hosts.empty() || valid_key_count == 0) {
        return EC_OK;
    }

    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherHostPrefixReduce);
    out_matches.reserve(candidate_hosts.size());
    for (std::size_t host_index = 0; host_index < candidate_hosts.size(); ++host_index) {
        int64_t prefix_len =
            static_cast<int64_t>(std::min(valid_key_count, prefix_stops[host_index].load(std::memory_order_relaxed)));
        if (use_eagle_pop) {
            prefix_len = std::max<int64_t>(prefix_len - 1, 0);
        }
        if (prefix_len > 0) {
            out_matches.push_back(
                MetaSearcher::HostCacheMatch{std::move(candidate_hosts[host_index]), prefix_len, 0, prefix_len});
        }
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostPrefixReduce);
    return EC_OK;
}

ErrorCode PrefixMatchWithMambaByHostWithoutP2P(MetaIndexer *meta_indexer,
                                               const CheckLocDataExistFunc &check_loc_data_exist,
                                               RequestContext *request_context,
                                               const MetaSearcher::KeyVector &keys,
                                               bool use_eagle_pop,
                                               const std::vector<std::string> &medium_filter,
                                               const std::vector<const LocationSpecGroup *> &full_groups,
                                               const std::vector<const LocationSpecGroup *> &mamba_state_groups,
                                               std::vector<MetaSearcher::HostCacheMatch> &out_matches,
                                               const MetaSearcher::CheckHostCacheLocationFunc *request_check_location) {
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    const MediumViewSet medium_set = BuildMediumViewSet(medium_filter);

    std::vector<std::string> required_spec_names;
    auto append_required_names = [&required_spec_names](const std::vector<const LocationSpecGroup *> &groups) {
        for (const auto *group : groups) {
            required_spec_names.insert(
                required_spec_names.end(), group->spec_names().begin(), group->spec_names().end());
        }
    };
    append_required_names(full_groups);
    append_required_names(mamba_state_groups);
    std::sort(required_spec_names.begin(), required_spec_names.end());
    required_spec_names.erase(std::unique(required_spec_names.begin(), required_spec_names.end()),
                              required_spec_names.end());
    const size_t spec_word_count = WordCountForBits(required_spec_names.size());
    std::vector<std::uint64_t> full_required(spec_word_count, 0);
    std::vector<std::uint64_t> state_required(spec_word_count, 0);
    auto build_required_mask = [&required_spec_names](const std::vector<const LocationSpecGroup *> &groups,
                                                      std::vector<std::uint64_t> &mask) {
        for (const auto *group : groups) {
            for (const auto &spec_name : group->spec_names()) {
                const auto it = std::lower_bound(required_spec_names.begin(), required_spec_names.end(), spec_name);
                assert(it != required_spec_names.end() && *it == spec_name);
                const size_t index = static_cast<size_t>(it - required_spec_names.begin());
                mask[index / 64] |= std::uint64_t{1} << (index % 64);
            }
        }
    };
    build_required_mask(full_groups, full_required);
    build_required_mask(mamba_state_groups, state_required);

    std::vector<std::string> candidate_hosts;
    std::unique_ptr<std::atomic<std::size_t>[]> full_prefix_stops;
    std::vector<std::uint64_t> state_present_words;
    std::size_t state_key_word_count = 0;
    std::atomic<bool> projection_invalid(false);
    ConcurrentWallTimer projection_wall_timer;

    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    const auto visitor = [&keys,
                          &check_loc_data_exist,
                          request_check_location,
                          &medium_set,
                          &candidate_hosts,
                          &required_spec_names,
                          &full_required,
                          &state_required,
                          &full_prefix_stops,
                          &state_present_words,
                          &state_key_word_count,
                          &projection_invalid,
                          &projection_wall_timer,
                          spec_word_count](
                             std::size_t begin, const CompactLocationsPerKey &locations, std::size_t valid_count) {
        auto projection_scope = projection_wall_timer.Measure();
        if (begin == 0) {
            BuildHostsForOneKey(
                locations[0], check_loc_data_exist, request_check_location, medium_set, candidate_hosts);
            if (candidate_hosts.empty()) {
                return size_t{0};
            }
            if (spec_word_count > std::numeric_limits<size_t>::max() / candidate_hosts.size()) {
                projection_invalid.store(true, std::memory_order_relaxed);
                return size_t{0};
            }
            full_prefix_stops = std::make_unique<std::atomic<std::size_t>[]>(candidate_hosts.size());
            for (std::size_t i = 0; i < candidate_hosts.size(); ++i) {
                full_prefix_stops[i].store(keys.size(), std::memory_order_relaxed);
            }
            state_key_word_count = WordCountForBits(keys.size());
            if (state_key_word_count > std::numeric_limits<size_t>::max() / candidate_hosts.size()) {
                projection_invalid.store(true, std::memory_order_relaxed);
                return size_t{0};
            }
            state_present_words.assign(state_key_word_count * candidate_hosts.size(), 0);
        }

        std::vector<std::uint64_t> seen_words(candidate_hosts.size() * spec_word_count, 0);
        std::vector<std::uint8_t> host_present(candidate_hosts.size(), 0);
        for (std::size_t local_index = 0; local_index < valid_count; ++local_index) {
            const std::size_t key_index = begin + local_index;
            bool any_host_needs_key = false;
            for (size_t host_index = 0; host_index < candidate_hosts.size(); ++host_index) {
                if (key_index < full_prefix_stops[host_index].load(std::memory_order_relaxed)) {
                    any_host_needs_key = true;
                    break;
                }
            }
            if (!any_host_needs_key) {
                break;
            }
            std::fill(seen_words.begin(), seen_words.end(), 0);
            std::fill(host_present.begin(), host_present.end(), 0);
            VisitHostSpecsForOneKey(
                locations[local_index],
                check_loc_data_exist,
                request_check_location,
                medium_set,
                true,
                [&candidate_hosts, &required_spec_names, &seen_words, &host_present, spec_word_count](
                    std::string_view host, std::string_view spec_name, bool) {
                    const auto host_it = std::lower_bound(candidate_hosts.begin(),
                                                          candidate_hosts.end(),
                                                          host,
                                                          [](const std::string &candidate, std::string_view value) {
                                                              return std::string_view(candidate) < value;
                                                          });
                    if (host_it == candidate_hosts.end() || std::string_view(*host_it) != host) {
                        return;
                    }
                    const size_t host_index = static_cast<size_t>(host_it - candidate_hosts.begin());
                    host_present[host_index] = 1;
                    const auto spec_it = std::lower_bound(required_spec_names.begin(),
                                                          required_spec_names.end(),
                                                          spec_name,
                                                          [](const std::string &candidate, std::string_view value) {
                                                              return std::string_view(candidate) < value;
                                                          });
                    if (spec_it == required_spec_names.end() || std::string_view(*spec_it) != spec_name) {
                        return;
                    }
                    const size_t spec_index = static_cast<size_t>(spec_it - required_spec_names.begin());
                    seen_words[host_index * spec_word_count + spec_index / 64] |= std::uint64_t{1} << (spec_index % 64);
                });

            for (size_t host_index = 0; host_index < candidate_hosts.size(); ++host_index) {
                bool has_full = host_present[host_index] != 0;
                bool has_state = has_full;
                for (size_t word = 0; word < spec_word_count && (has_full || has_state); ++word) {
                    const auto seen = seen_words[host_index * spec_word_count + word];
                    has_full = has_full && (seen & full_required[word]) == full_required[word];
                    has_state = has_state && (seen & state_required[word]) == state_required[word];
                }
                if (has_state) {
                    state_present_words[host_index * state_key_word_count + key_index / 64] |= std::uint64_t{1}
                                                                                               << (key_index % 64);
                }
                if (!has_full) {
                    std::size_t current = full_prefix_stops[host_index].load(std::memory_order_relaxed);
                    while (key_index < current &&
                           !full_prefix_stops[host_index].compare_exchange_weak(
                               current, key_index, std::memory_order_relaxed, std::memory_order_relaxed)) {}
                }
            }
        }

        bool every_host_stopped = true;
        std::size_t last_required_key = 0;
        for (std::size_t host_index = 0; host_index < candidate_hosts.size(); ++host_index) {
            const std::size_t stop = full_prefix_stops[host_index].load(std::memory_order_relaxed);
            if (stop == keys.size()) {
                every_host_stopped = false;
                break;
            }
            last_required_key = std::max(last_required_key, stop);
        }
        return every_host_stopped ? last_required_key : keys.size();
    };
    const auto result = meta_indexer->VisitLocationValuesForPrefix(request_context, keys, visitor);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, host_projection_time_us, projection_wall_timer.elapsed_us());
    if (projection_invalid.load(std::memory_order_relaxed)) {
        request_context->error_tracer()->AddErrorMsg("mamba host/spec flag matrix size overflow");
        return EC_ERROR;
    }
    const std::size_t valid_key_count = result.valid_key_count;
    if (valid_key_count < keys.size() && result.terminal_ec != EC_OK) {
        KVCM_LOG_DEBUG("prefix match with mamba by host end because Get keys[%lu](%lu) return %d",
                       valid_key_count,
                       keys[valid_key_count],
                       result.terminal_ec);
        if (result.terminal_ec != ErrorCode::EC_NOENT) {
            request_context->error_tracer()->AddErrorMsg("mamba prefix match metadata read failed");
            return result.terminal_ec;
        }
    }
    if (candidate_hosts.empty() || valid_key_count == 0) {
        return EC_OK;
    }

    std::vector<int64_t> prefix_lengths(candidate_hosts.size(), 0);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherHostPrefixReduce);
    const bool reduce_ok = meta_indexer->ParallelForQuery(
        candidate_hosts.size(),
        [&full_prefix_stops,
         &state_present_words,
         state_key_word_count,
         valid_key_count,
         &prefix_lengths,
         use_eagle_pop](std::size_t begin, std::size_t end) {
            for (std::size_t host_index = begin; host_index < end; ++host_index) {
                size_t full_prefix_len =
                    std::min(valid_key_count, full_prefix_stops[host_index].load(std::memory_order_relaxed));
                if (use_eagle_pop && full_prefix_len > 0) {
                    --full_prefix_len;
                }
                while (full_prefix_len > 0) {
                    const size_t last_index = full_prefix_len - 1;
                    const size_t word_index = last_index / 64;
                    const unsigned used_bits = static_cast<unsigned>(last_index % 64) + 1;
                    const std::uint64_t prefix_mask = used_bits == 64 ? std::numeric_limits<std::uint64_t>::max()
                                                                      : (std::uint64_t{1} << used_bits) - 1;
                    const std::uint64_t present =
                        state_present_words[host_index * state_key_word_count + word_index] & prefix_mask;
                    if (present != 0) {
                        const size_t highest_bit = 63U - static_cast<size_t>(__builtin_clzll(present));
                        prefix_lengths[host_index] = static_cast<int64_t>(word_index * 64 + highest_bit + 1);
                        break;
                    }
                    full_prefix_len = word_index * 64;
                }
            }
        });
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostPrefixReduce);
    if (!reduce_ok) {
        request_context->error_tracer()->AddErrorMsg("parallel mamba host prefix reduction failed");
        return EC_ERROR;
    }
    out_matches.reserve(candidate_hosts.size());
    for (std::size_t i = 0; i < candidate_hosts.size(); ++i) {
        if (prefix_lengths[i] > 0) {
            out_matches.push_back(
                MetaSearcher::HostCacheMatch{std::move(candidate_hosts[i]), prefix_lengths[i], 0, prefix_lengths[i]});
        }
    }
    return EC_OK;
}

} // namespace

MetaSearcher::MetaSearcher(const std::shared_ptr<MetaIndexer> &meta_indexer) : meta_indexer_(meta_indexer) {}

MetaSearcher::MetaSearcher(const std::shared_ptr<MetaIndexer> &meta_indexer,
                           CheckLocDataExistFunc check_loc_data_exist,
                           SubmitDelReqFunc submit_del_req)
    : meta_indexer_(meta_indexer)
    , check_loc_data_exist_func_(check_loc_data_exist)
    , submit_del_req_func_(submit_del_req) {}

MetaSearcher::~MetaSearcher() = default;

std::string MetaSearcher::BatchErrorCodeToStr(const std::vector<std::vector<ErrorCode>> &batch_results) {
    std::stringstream result_stream;

    result_stream << "[";
    for (size_t idx = 0; idx < batch_results.size(); idx++) {
        if (idx > 0) {
            result_stream << ", ";
        }
        result_stream << "[";
        for (size_t j = 0; j < batch_results[idx].size(); j++) {
            if (j > 0) {
                result_stream << ", ";
            }
            result_stream << batch_results[idx][j];
        }
        result_stream << "]";
    }
    result_stream << "]";

    return result_stream.str();
}

ErrorCode MetaSearcher::PrefixMatchBestLocationImpl(RequestContext *request_context,
                                                    const KeyVector &keys,
                                                    CacheLocationVector &out_locations,
                                                    SelectLocationPolicy *policy) const {
    out_locations.clear();

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    CacheLocationMapVector location_maps;
    auto result = meta_indexer_->GetLocations(request_context, keys, location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);

    KeyVector prune_keys;
    std::vector<std::vector<std::string>> prune_loc_ids_vec;
    std::size_t i = 0;
    for (; i != keys.size(); ++i) {
        if (result.error_codes[i] != ErrorCode::EC_OK) {
            KVCM_LOG_DEBUG("prefix match end because Get keys[%lu](%lu) return %d", i, keys[i], result.error_codes[i]);
            break;
        }

        auto &location_map = location_maps[i];
        if (location_map.empty()) {
            KVCM_LOG_DEBUG("prefix match end because keys[%lu](%lu) no location", i, keys[i]);
            break;
        }
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr merged =
            SelectAndMergeForMatch(policy, location_map, check_loc_data_exist_func_, prune_loc_ids);
        if (!prune_loc_ids.empty()) {
            prune_keys.emplace_back(keys[i]);
            prune_loc_ids_vec.emplace_back(prune_loc_ids);
        }
        if (merged->location_specs().empty()) {
            KVCM_LOG_DEBUG("prefix match end because keys[%lu] no serving location", i);
            break;
        }
        out_locations.push_back(std::move(merged));
    }

    if (!prune_keys.empty()) {
        for (i == keys.size() ? /* do nothing */ i : ++i; i != keys.size(); ++i) {
            if (result.error_codes[i] != ErrorCode::EC_OK) {
                continue;
            }
            auto &location_map = location_maps[i];
            if (location_map.empty()) {
                continue;
            }
            std::vector<std::string> prune_loc_ids;
            policy->SelectForMatch(location_map, check_loc_data_exist_func_, prune_loc_ids);
            if (!prune_loc_ids.empty()) {
                prune_keys.emplace_back(keys[i]);
                prune_loc_ids_vec.emplace_back(prune_loc_ids);
            }
        }
    }

    if (!prune_keys.empty() && submit_del_req_func_) {
        submit_del_req_func_(prune_keys, prune_loc_ids_vec, {}, false);
    }

    return EC_OK;
}

ErrorCode MetaSearcher::PrefixMatch(RequestContext *request_context,
                                    const KeyVector &keys,
                                    const BlockMask &input_mask,
                                    CacheLocationVector &out_locations,
                                    SelectLocationPolicy *policy) const {
    assert(policy != nullptr);
    SPAN_TRACER(request_context);
    KeyVector query_keys;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!IsIndexInMaskRange(input_mask, i)) {
            query_keys.push_back(keys[i]);
        }
    }

    if (query_keys.empty()) {
        KVCM_LOG_DEBUG("prefix match end because query_keys is empty");
        return EC_OK;
    }
    // TODO: need to confirm shard lock range
    // TODO: use smaller batch if many prefix missed a lot
    ErrorCode ec = PrefixMatchBestLocationImpl(request_context, query_keys, out_locations, policy);
    if (ec != EC_OK) {
        KVCM_LOG_DEBUG("PrefixMatchBestLocationImpl failed");
    }
    return EC_OK;
}

ErrorCode MetaSearcher::BatchGetBestLocation(RequestContext *request_context,
                                             const KeyVector &keys,
                                             CacheLocationVector &out_locations,
                                             SelectLocationPolicy *policy) const {
    assert(policy != nullptr);
    SPAN_TRACER(request_context);
    out_locations.clear();
    out_locations.reserve(keys.size());
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    CacheLocationMapVector location_maps;
    auto result = meta_indexer_->GetLocations(request_context, keys, location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    KeyVector prune_keys;
    std::vector<std::vector<std::string>> prune_loc_ids_vec;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (result.error_codes[i] == ErrorCode::EC_NOENT) {
            out_locations.push_back(std::make_shared<CacheLocation>());
            continue;
        }
        if (result.error_codes[i] != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("get key failed, key[%lu](%lu), error_code: %d", i, keys[i], result.error_codes[i]);
            break;
        }

        auto &location_map = location_maps[i];
        if (location_map.empty()) {
            out_locations.push_back(std::make_shared<CacheLocation>());
            continue;
        }
        std::vector<std::string> prune_loc_ids;
        CacheLocationConstPtr merged =
            SelectAndMergeForMatch(policy, location_map, check_loc_data_exist_func_, prune_loc_ids);
        if (!prune_loc_ids.empty()) {
            prune_keys.emplace_back(keys[i]);
            prune_loc_ids_vec.emplace_back(prune_loc_ids);
        }
        if (merged->location_specs().empty()) {
            out_locations.push_back(std::make_shared<CacheLocation>());
            continue;
        }
        out_locations.push_back(std::move(merged));
    }

    if (!prune_keys.empty() && submit_del_req_func_) {
        submit_del_req_func_(prune_keys, prune_loc_ids_vec, {}, false);
    }

    return out_locations.size() == keys.size() ? EC_OK : EC_ERROR;
}

ErrorCode MetaSearcher::BatchGetBestLocationByBackend(RequestContext *request_context,
                                                      const KeyVector &keys,
                                                      LocationsPerKey &out_locations,
                                                      SelectLocationPolicy *policy,
                                                      const std::vector<BackendSelector> &selectors,
                                                      const std::vector<std::string> &requested_spec_names,
                                                      const BlockMask &input_mask) const {
    assert(policy != nullptr);
    SPAN_TRACER(request_context);
    out_locations.clear();
    out_locations.resize(keys.size());
    if (!requested_spec_names.empty() &&
        (requested_spec_names.size() != keys.size() ||
         std::any_of(requested_spec_names.begin(), requested_spec_names.end(), [](const std::string &name) {
             return name.empty();
         }))) {
        request_context->error_tracer()->AddErrorMsg(
            "requested_spec_names must be empty or contain one non-empty name per key");
        return EC_BADARGS;
    }
    const bool has_implicit_empty_mask =
        std::holds_alternative<BlockMaskVector>(input_mask) && std::get<BlockMaskVector>(input_mask).empty();
    if (!has_implicit_empty_mask && !IsBlockMaskValid(input_mask, keys.size())) {
        return EC_BADARGS;
    }

    // A masked block is already available to the caller.  Do not send it to
    // the metadata backend, but preserve the request's positional response
    // shape so callers can safely correlate each entry with the original key.
    KeyVector query_keys;
    std::vector<size_t> query_to_output_index;
    query_keys.reserve(keys.size());
    query_to_output_index.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (IsIndexInMaskRange(input_mask, i)) {
            continue;
        }
        query_keys.push_back(keys[i]);
        query_to_output_index.push_back(i);
    }
    if (query_keys.empty()) {
        return EC_OK;
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    CacheLocationMapVector location_maps;
    auto result = meta_indexer_->GetLocations(request_context, query_keys, location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    KeyVector prune_keys;
    std::vector<std::vector<std::string>> prune_loc_ids_vec;
    std::vector<CacheLocationMap> valid_maps(query_keys.size());
    bool has_error = false;

    for (size_t i = 0; i < query_keys.size(); ++i) {
        if (result.error_codes[i] == ErrorCode::EC_NOENT) {
            continue;
        }
        if (result.error_codes[i] != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("get key failed, key[%lu](%lu), error_code: %d", i, query_keys[i], result.error_codes[i]);
            has_error = true;
            break;
        }
        if (location_maps[i].empty()) {
            continue;
        }
        std::vector<std::string> prune_loc_ids;
        valid_maps[i] = FilterValidLocations(location_maps[i], check_loc_data_exist_func_, prune_loc_ids);
        if (!prune_loc_ids.empty()) {
            prune_keys.emplace_back(query_keys[i]);
            prune_loc_ids_vec.emplace_back(std::move(prune_loc_ids));
        }
    }

    for (const auto &selector : selectors) {
        DataStorageType target_type = selector.backend_type;

        if (target_type == DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2 &&
            (selector.strategy == LocationSelectStrategy::LSS_V6D_PREFIX ||
             selector.strategy == LocationSelectStrategy::LSS_V6D_COVERAGE)) {
            // --- event report cross-key selection ---
            bool is_prefix = (selector.strategy == LocationSelectStrategy::LSS_V6D_PREFIX);

            // Candidate enumeration
            std::vector<size_t> candidate_indices;
            std::unordered_map<size_t, std::vector<std::string>> remote_peer_candidates;
            // key_idx -> peer_addr -> CacheLocationConstPtr (for reverse lookup)
            std::unordered_map<size_t, std::unordered_map<std::string, CacheLocationConstPtr>> key_peer_to_location;

            bool stop_vineyard = false;
            for (size_t i = 0; i < query_keys.size(); ++i) {
                if (stop_vineyard)
                    break;

                const auto &vmap = valid_maps[i];
                const std::string_view requested_spec_name =
                    requested_spec_names.empty() ? std::string_view{} : requested_spec_names[query_to_output_index[i]];
                std::vector<std::string> vineyard_addrs;

                for (const auto &[id, loc] : vmap) {
                    if (loc->type() != target_type)
                        continue;
                    if (!MatchesRequestedSpec(*loc, requested_spec_name))
                        continue;
                    std::string addr = ExtractPeerAddrFromLocation(*loc, requested_spec_name);
                    if (addr.empty())
                        continue;
                    // Dedup: only add if not already present
                    if (key_peer_to_location[i].find(addr) == key_peer_to_location[i].end()) {
                        vineyard_addrs.push_back(addr);
                        key_peer_to_location[i][addr] = loc;
                    }
                }

                if (vineyard_addrs.empty()) {
                    if (is_prefix) {
                        stop_vineyard = true;
                    }
                    continue;
                }

                remote_peer_candidates[i] = std::move(vineyard_addrs);
                candidate_indices.push_back(i);
            }

            // Select best peer
            V6DPeerSelection selection;
            if (is_prefix) {
                selection = SelectV6DByPrefix(candidate_indices, remote_peer_candidates);
            } else {
                selection = SelectV6DByCoverage(candidate_indices, remote_peer_candidates);
            }

            // Populate results for covered keys
            if (!selection.peer_addr.empty()) {
                for (size_t idx : selection.covered_indices) {
                    auto peer_it = key_peer_to_location.find(idx);
                    if (peer_it == key_peer_to_location.end())
                        continue;
                    auto loc_it = peer_it->second.find(selection.peer_addr);
                    if (loc_it == peer_it->second.end())
                        continue;
                    out_locations[query_to_output_index[idx]].push_back(loc_it->second);
                }
            }

        } else {
            // --- Per-key independent selection (WEIGHTED_RANDOM or other non-event-report) ---
            for (size_t i = 0; i < query_keys.size(); ++i) {
                const std::string_view requested_spec_name =
                    requested_spec_names.empty() ? std::string_view{} : requested_spec_names[query_to_output_index[i]];
                const auto &vmap = valid_maps[i];
                CacheLocationMap filtered;
                for (const auto &[id, loc] : vmap) {
                    if (loc->type() == target_type && MatchesRequestedSpec(*loc, requested_spec_name)) {
                        filtered.try_emplace(id, loc);
                    }
                }
                if (filtered.empty())
                    continue;

                std::vector<std::string> unused_prune_ids;
                auto winner = policy->SelectForMatch(filtered, nullptr, unused_prune_ids);
                if (!winner || winner->id().empty() || winner->location_specs().empty())
                    continue;

                // Merge specs from same data storage (same logic as SelectAndMergeForMatch)
                std::map<std::string, LocationSpec> merged_specs;
                for (const auto &[id, loc] : filtered) {
                    if (!policy->IsSameDataStorage(*loc, *winner))
                        continue;
                    for (const auto &spec : loc->location_specs()) {
                        merged_specs.try_emplace(spec.name(), spec);
                    }
                }
                if (merged_specs.empty())
                    continue;

                auto merged = std::make_shared<CacheLocation>();
                merged->set_id(winner->id() + "_merged");
                merged->set_status(CacheLocationStatus::CLS_SERVING);
                merged->set_type(winner->type());
                std::vector<LocationSpec> specs;
                specs.reserve(merged_specs.size());
                for (auto &[name, spec] : merged_specs) {
                    specs.push_back(std::move(spec));
                }
                merged->set_spec_size(specs.size());
                merged->set_location_specs(std::move(specs));
                out_locations[query_to_output_index[i]].push_back(std::move(merged));
            }
        }
    }

    if (!prune_keys.empty() && submit_del_req_func_) {
        submit_del_req_func_(prune_keys, prune_loc_ids_vec, {}, false);
    }

    return has_error ? EC_ERROR : EC_OK;
}

ErrorCode MetaSearcher::ReverseRollSlideWindowMatch(RequestContext *request_context,
                                                    const KeyVector &keys,
                                                    int32_t sw_size,
                                                    CacheLocationVector &out_locations,
                                                    SelectLocationPolicy *policy) const {
    assert(policy != nullptr);
    SPAN_TRACER(request_context);
    assert(keys.size() >= sw_size);
    assert(sw_size > 0);
    // TODO: error handle
    out_locations.clear();
    out_locations.clear();
    out_locations.reserve(keys.size());
    for (size_t idx = 0; idx < keys.size(); ++idx) {
        out_locations.push_back(std::make_shared<CacheLocation>());
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    CacheLocationMapVector location_maps;
    auto result = meta_indexer_->GetLocations(request_context, keys, location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    bool is_match = false;
    CacheLocationVector temp_sw_locations;
    temp_sw_locations.reserve(sw_size);
    KeyVector prune_keys;
    std::vector<std::vector<std::string>> prune_loc_ids_vec;
    for (int base = keys.size() - sw_size; base >= 0;) {
        for (int offset = 0; offset < sw_size; ++offset) {
            if (result.error_codes[base + offset] != ErrorCode::EC_OK) {
                base -= sw_size - offset;
                is_match = false;
                break;
            }
            is_match = true;
        }
        if (!is_match) {
            continue;
        }
        for (size_t offset = 0; offset < sw_size; ++offset) {
            auto &location_map = location_maps[base + offset];
            if (location_map.empty()) {
                temp_sw_locations.clear();
                base -= sw_size - offset;
                is_match = false;
                break;
            }
            std::vector<std::string> prune_loc_ids;
            CacheLocationConstPtr merged =
                SelectAndMergeForMatch(policy, location_map, check_loc_data_exist_func_, prune_loc_ids);
            if (!prune_loc_ids.empty()) {
                prune_keys.emplace_back(keys[base + offset]);
                prune_loc_ids_vec.emplace_back(prune_loc_ids);
            }
            if (!merged || merged->location_specs().empty()) {
                temp_sw_locations.clear();
                base -= sw_size - offset;
                is_match = false;
                break;
            }
            temp_sw_locations.push_back(std::move(merged));
        }
        if (is_match) {
            std::move(temp_sw_locations.begin(), temp_sw_locations.end(), out_locations.begin() + base);
            break;
        }
    }

    if (!prune_keys.empty() && submit_del_req_func_) {
        submit_del_req_func_(prune_keys, prune_loc_ids_vec, {}, false);
    }

    return EC_OK;
}

ErrorCode MetaSearcher::PrefixMatchByHost(RequestContext *request_context,
                                          const KeyVector &keys,
                                          bool use_eagle_pop,
                                          const std::vector<std::string> &medium_filter,
                                          std::vector<HostCacheMatch> &out_matches,
                                          const CheckHostCacheLocationFunc *request_check_location,
                                          size_t p2p_host_count) const {
    SPAN_TRACER(request_context);
    out_matches.clear();
    if (keys.empty()) {
        return EC_OK;
    }
    if (p2p_host_count == 0) {
        return PrefixMatchByHostWithoutP2P(meta_indexer_.get(),
                                           check_loc_data_exist_func_,
                                           request_context,
                                           keys,
                                           use_eagle_pop,
                                           medium_filter,
                                           out_matches,
                                           request_check_location);
    }

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    LocationsPerKey location_values;
    auto result = meta_indexer_->GetLocationValues(request_context, keys, location_values);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    LogErrorCodes("PrefixMatchByHost", result.error_codes, keys);
    assert(keys.size() == location_values.size());

    std::size_t valid_key_count = 0;
    while (valid_key_count < keys.size() && result.error_codes[valid_key_count] == ErrorCode::EC_OK) {
        ++valid_key_count;
    }
    if (valid_key_count < keys.size()) {
        KVCM_LOG_DEBUG("prefix match by host end because Get keys[%lu](%lu) return %d",
                       valid_key_count,
                       keys[valid_key_count],
                       result.error_codes[valid_key_count]);
        if (result.error_codes[valid_key_count] != ErrorCode::EC_NOENT) {
            request_context->error_tracer()->AddErrorMsg("prefix match metadata read failed");
            return result.error_codes[valid_key_count];
        }
    }
    if (valid_key_count == 0) {
        return EC_OK;
    }

    KeyToHostSpecNames key_to_host_spec_names(valid_key_count);
    KeyToHostSpecNames key_to_vineyard_spec_names(valid_key_count);
    const MediumViewSet medium_set = BuildMediumViewSet(medium_filter);
    const auto &check_loc_data_exist = check_loc_data_exist_func_;
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherHostProjection);
    const bool projection_ok =
        meta_indexer_->ParallelForQuery(valid_key_count,
                                        [&location_values,
                                         &check_loc_data_exist,
                                         request_check_location,
                                         &medium_set,
                                         &key_to_host_spec_names,
                                         &key_to_vineyard_spec_names](std::size_t begin, std::size_t end) {
                                            for (std::size_t i = begin; i < end; ++i) {
                                                BuildHostSpecNamesForOneKey(location_values[i],
                                                                            check_loc_data_exist,
                                                                            request_check_location,
                                                                            medium_set,
                                                                            key_to_host_spec_names[i],
                                                                            key_to_vineyard_spec_names[i]);
                                            }
                                        });
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostProjection);
    if (!projection_ok) {
        request_context->error_tracer()->AddErrorMsg("parallel host cache projection failed");
        return EC_ERROR;
    }

    if (key_to_host_spec_names.front().empty()) {
        return EC_OK;
    }

    std::vector<std::string> candidate_hosts;
    candidate_hosts.reserve(key_to_host_spec_names.front().size());
    for (const auto &[host, spec_names] : key_to_host_spec_names.front()) {
        (void)spec_names;
        candidate_hosts.push_back(host);
    }
    std::vector<HostCacheMatch> host_matches(candidate_hosts.size());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherHostPrefixReduce);
    bool reduce_ok = meta_indexer_->ParallelForQuery(
        candidate_hosts.size(),
        [&candidate_hosts, &key_to_host_spec_names, &key_to_vineyard_spec_names, &host_matches, use_eagle_pop](
            std::size_t begin, std::size_t end) {
            for (std::size_t host_index = begin; host_index < end; ++host_index) {
                const auto &host = candidate_hosts[host_index];
                const V6DPeerSelection no_p2p;
                auto local_specs =
                    MergeHostAndP2PSpecs(host, key_to_host_spec_names, key_to_vineyard_spec_names, no_p2p);
                const int64_t local = ComputePrefixMatchBlocks(local_specs, use_eagle_pop);
                host_matches[host_index] = HostCacheMatch{host, local, 0, local};
            }
        });
    if (reduce_ok) {
        const auto top_host_indices = SelectTopHostIndicesByLocal(host_matches, p2p_host_count);
        reduce_ok = top_host_indices.empty() ||
                    meta_indexer_->ParallelForQuery(
                        top_host_indices.size(),
                        [&top_host_indices,
                         &candidate_hosts,
                         &keys,
                         &key_to_host_spec_names,
                         &key_to_vineyard_spec_names,
                         &host_matches,
                         use_eagle_pop](std::size_t begin, std::size_t end) {
                            for (std::size_t selected_index = begin; selected_index < end; ++selected_index) {
                                const size_t host_index = top_host_indices[selected_index];
                                const auto &host = candidate_hosts[host_index];
                                auto p2p_selection =
                                    SelectP2PByPrefix(host, key_to_host_spec_names, key_to_vineyard_spec_names);
                                std::unordered_set<int64_t> fetched_block_keys;
                                for (size_t index : p2p_selection.covered_indices) {
                                    fetched_block_keys.insert(keys[index]);
                                }
                                int64_t total_match = host_matches[host_index].local;
                                if (!p2p_selection.covered_indices.empty()) {
                                    auto p2p_specs = MergeHostAndP2PSpecs(
                                        host, key_to_host_spec_names, key_to_vineyard_spec_names, p2p_selection);
                                    total_match = ComputePrefixMatchBlocks(p2p_specs, use_eagle_pop);
                                }
                                host_matches[host_index].p2p_1_fetch = static_cast<int64_t>(fetched_block_keys.size());
                                host_matches[host_index].p2p_1_total_match = total_match;
                            }
                        });
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostPrefixReduce);
    if (!reduce_ok) {
        request_context->error_tracer()->AddErrorMsg("parallel host prefix reduction failed");
        return EC_ERROR;
    }
    out_matches.reserve(candidate_hosts.size());
    for (std::size_t i = 0; i < candidate_hosts.size(); ++i) {
        if (host_matches[i].local > 0) {
            out_matches.push_back(std::move(host_matches[i]));
        }
    }
    return EC_OK;
}

ErrorCode MetaSearcher::PrefixMatchWithMambaByHost(RequestContext *request_context,
                                                   const KeyVector &keys,
                                                   bool use_eagle_pop,
                                                   const std::vector<std::string> &medium_filter,
                                                   const std::vector<LocationSpecGroup> &location_spec_groups,
                                                   std::vector<HostCacheMatch> &out_matches,
                                                   const CheckHostCacheLocationFunc *request_check_location,
                                                   size_t p2p_host_count) const {
    SPAN_TRACER(request_context);
    out_matches.clear();
    if (keys.empty()) {
        return EC_OK;
    }

    std::vector<const LocationSpecGroup *> full_groups;
    std::vector<const LocationSpecGroup *> mamba_state_groups;
    auto ec = ClassifySpecGroups(request_context, location_spec_groups, full_groups, mamba_state_groups);
    if (ec != EC_OK) {
        return ec;
    }
    if (p2p_host_count == 0) {
        return PrefixMatchWithMambaByHostWithoutP2P(meta_indexer_.get(),
                                                    check_loc_data_exist_func_,
                                                    request_context,
                                                    keys,
                                                    use_eagle_pop,
                                                    medium_filter,
                                                    full_groups,
                                                    mamba_state_groups,
                                                    out_matches,
                                                    request_check_location);
    }
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    LocationsPerKey location_values;
    auto result = meta_indexer_->GetLocationValues(request_context, keys, location_values);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    LogErrorCodes("PrefixMatchWithMambaByHost", result.error_codes, keys);
    assert(keys.size() == location_values.size());

    std::size_t valid_key_count = 0;
    while (valid_key_count < keys.size() && result.error_codes[valid_key_count] == ErrorCode::EC_OK) {
        ++valid_key_count;
    }
    if (valid_key_count < keys.size()) {
        KVCM_LOG_DEBUG("prefix match with mamba by host end because Get keys[%lu](%lu) return %d",
                       valid_key_count,
                       keys[valid_key_count],
                       result.error_codes[valid_key_count]);
        if (result.error_codes[valid_key_count] != ErrorCode::EC_NOENT) {
            request_context->error_tracer()->AddErrorMsg("mamba prefix match metadata read failed");
            return result.error_codes[valid_key_count];
        }
    }
    if (valid_key_count == 0) {
        return EC_OK;
    }

    KeyToHostSpecNames key_to_host_spec_names(valid_key_count);
    KeyToHostSpecNames key_to_vineyard_spec_names(valid_key_count);
    const MediumViewSet medium_set = BuildMediumViewSet(medium_filter);
    const auto &check_loc_data_exist = check_loc_data_exist_func_;
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherHostProjection);
    const bool projection_ok =
        meta_indexer_->ParallelForQuery(valid_key_count,
                                        [&location_values,
                                         &check_loc_data_exist,
                                         request_check_location,
                                         &medium_set,
                                         &key_to_host_spec_names,
                                         &key_to_vineyard_spec_names](std::size_t begin, std::size_t end) {
                                            for (std::size_t i = begin; i < end; ++i) {
                                                BuildHostSpecNamesForOneKey(location_values[i],
                                                                            check_loc_data_exist,
                                                                            request_check_location,
                                                                            medium_set,
                                                                            key_to_host_spec_names[i],
                                                                            key_to_vineyard_spec_names[i]);
                                            }
                                        });
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostProjection);
    if (!projection_ok) {
        request_context->error_tracer()->AddErrorMsg("parallel mamba host cache projection failed");
        return EC_ERROR;
    }

    std::vector<std::string> candidate_hosts;
    candidate_hosts.reserve(key_to_host_spec_names.front().size());
    for (const auto &[host, spec_names] : key_to_host_spec_names.front()) {
        (void)spec_names;
        candidate_hosts.push_back(host);
    }
    std::vector<HostCacheMatch> host_matches(candidate_hosts.size());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherHostPrefixReduce);
    bool reduce_ok = meta_indexer_->ParallelForQuery(
        candidate_hosts.size(),
        [&candidate_hosts,
         &key_to_host_spec_names,
         &key_to_vineyard_spec_names,
         &full_groups,
         &mamba_state_groups,
         &host_matches,
         use_eagle_pop](std::size_t begin, std::size_t end) {
            for (std::size_t host_index = begin; host_index < end; ++host_index) {
                const auto &host = candidate_hosts[host_index];
                const V6DPeerSelection no_p2p;
                auto combined_specs =
                    MergeHostAndP2PSpecs(host, key_to_host_spec_names, key_to_vineyard_spec_names, no_p2p);
                const int64_t local =
                    ComputeMambaPrefixMatchBlocks(combined_specs, use_eagle_pop, full_groups, mamba_state_groups);
                host_matches[host_index] = HostCacheMatch{host, local, 0, local};
            }
        });
    if (reduce_ok) {
        const auto top_host_indices = SelectTopHostIndicesByLocal(host_matches, p2p_host_count);
        reduce_ok =
            top_host_indices.empty() ||
            meta_indexer_->ParallelForQuery(
                top_host_indices.size(),
                [&top_host_indices,
                 &candidate_hosts,
                 &keys,
                 &key_to_host_spec_names,
                 &key_to_vineyard_spec_names,
                 &full_groups,
                 &mamba_state_groups,
                 &host_matches,
                 use_eagle_pop](std::size_t begin, std::size_t end) {
                    for (std::size_t selected_index = begin; selected_index < end; ++selected_index) {
                        const size_t host_index = top_host_indices[selected_index];
                        const auto &host = candidate_hosts[host_index];
                        const V6DPeerSelection no_p2p;
                        auto combined_specs =
                            MergeHostAndP2PSpecs(host, key_to_host_spec_names, key_to_vineyard_spec_names, no_p2p);
                        std::unordered_set<int64_t> fetched_block_keys;

                        for (const auto *full_group : full_groups) {
                            auto full_selection = SelectP2PGroupByPrefix(
                                host, key_to_host_spec_names, key_to_vineyard_spec_names, *full_group);
                            for (size_t block_index : full_selection.covered_indices) {
                                const auto peer_it =
                                    key_to_vineyard_spec_names[block_index].find(full_selection.peer_addr);
                                if (peer_it == key_to_vineyard_spec_names[block_index].end()) {
                                    continue;
                                }
                                MergeLocationSpecGroup(combined_specs[block_index], peer_it->second, *full_group);
                                fetched_block_keys.insert(keys[block_index]);
                            }
                        }

                        size_t full_prefix_len = 0;
                        while (full_prefix_len < combined_specs.size() &&
                               HasAllLocationSpecGroups(combined_specs[full_prefix_len], full_groups)) {
                            ++full_prefix_len;
                        }
                        if (use_eagle_pop && full_prefix_len > 0) {
                            --full_prefix_len;
                        }

                        if (full_prefix_len > 0) {
                            auto mamba_selection = SelectP2PGroupsByCoverage(host,
                                                                             key_to_host_spec_names,
                                                                             key_to_vineyard_spec_names,
                                                                             full_prefix_len,
                                                                             mamba_state_groups);
                            for (size_t query_index : mamba_selection.selection.covered_indices) {
                                const size_t block_index = mamba_selection.query_block_indices[query_index];
                                const auto *group = mamba_selection.query_groups[query_index];
                                const auto peer_it =
                                    key_to_vineyard_spec_names[block_index].find(mamba_selection.selection.peer_addr);
                                if (peer_it == key_to_vineyard_spec_names[block_index].end()) {
                                    continue;
                                }
                                MergeLocationSpecGroup(combined_specs[block_index], peer_it->second, *group);
                                fetched_block_keys.insert(keys[block_index]);
                            }
                        }

                        const int64_t total_match = ComputeMambaPrefixMatchBlocks(
                            combined_specs, use_eagle_pop, full_groups, mamba_state_groups);
                        host_matches[host_index].p2p_1_fetch = static_cast<int64_t>(fetched_block_keys.size());
                        host_matches[host_index].p2p_1_total_match = total_match;
                    }
                });
    }
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostPrefixReduce);
    if (!reduce_ok) {
        request_context->error_tracer()->AddErrorMsg("parallel mamba host prefix reduction failed");
        return EC_ERROR;
    }
    out_matches.reserve(candidate_hosts.size());
    for (std::size_t i = 0; i < candidate_hosts.size(); ++i) {
        if (host_matches[i].local > 0) {
            out_matches.push_back(std::move(host_matches[i]));
        }
    }
    return EC_OK;
}

ErrorCode MetaSearcher::BatchGetLocation(RequestContext *request_context,
                                         const KeyVector &keys,
                                         const BlockMask &input_mask,
                                         std::vector<CacheLocationMap> &out_location_maps) {
    out_location_maps.clear();

    KeyVector query_keys;
    for (size_t idx = 0; idx < keys.size(); idx++) {
        if (IsIndexInMaskRange(input_mask, idx)) {
            continue;
        }
        query_keys.push_back(keys[idx]);
    }
    if (query_keys.empty()) {
        return EC_OK;
    }

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    auto result = meta_indexer_->GetLocations(request_context, query_keys, out_location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    for (size_t idx = 0; idx < query_keys.size(); idx++) {
        if (result.error_codes[idx] != ErrorCode::EC_OK && result.error_codes[idx] != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN(
                "get key failed, key[%lu](%lu), error_code: %d", idx, query_keys[idx], result.error_codes[idx]);
        }
    }
    return EC_OK;
}

ErrorCode MetaSearcher::BatchAddLocation(RequestContext *request_context,
                                         const KeyVector &keys,
                                         const CacheLocationVector &locations,
                                         std::vector<AddLocationResult> &out_results) {
    out_results.assign(keys.size(), AddLocationResult{});
    if (keys.size() != locations.size()) {
        for (auto &result : out_results) {
            result.ec = EC_BADARGS;
        }
        return EC_BADARGS;
    }
    std::vector<std::pair<DataStorageType, std::uint64_t>> loc_sz(keys.size());

    const int64_t batch_create_time = TimestampUtil::GetCurrentTimeUs();
    auto modifier = [&locations, &out_results, &keys, &loc_sz, batch_create_time](
                        const LocationIdVector &existing_location_ids,
                        ErrorCode get_ec,
                        size_t index,
                        PropertyMap &upsert_property_map,
                        CacheLocationMap &out_new_locations) -> ModifierResult {
        if (get_ec != ErrorCode::EC_OK && get_ec != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu) return %d", index, keys[index], get_ec);
            return {ModifierAction::MA_FAIL, get_ec};
        }

        // first time this block_key is created: record prev_key
        if (get_ec == EC_NOENT) {
            std::string prev_key = index > 0 ? std::to_string(keys[index - 1]) : std::string();
            upsert_property_map[PROPERTY_PREV_BLOCK_KEY] = prev_key;
        }

        // generate a unique location_id that does not collide with existing ones
        const std::unordered_set<std::string> existing_id_set(existing_location_ids.begin(),
                                                              existing_location_ids.end());
        std::string location_id;
        do {
            location_id = StringUtil::GenerateRandomString(8);
        } while (existing_id_set.count(location_id) > 0);

        // build the new CacheLocation with status = CLS_WRITING
        auto new_loc = std::make_shared<CacheLocation>(*locations[index]);
        new_loc->set_id(location_id);
        new_loc->set_status(CLS_WRITING);
        new_loc->set_create_time(batch_create_time);
        out_new_locations[location_id] = std::move(new_loc);

        // compute storage size for usage tracking
        std::uint64_t sz = 0;
        for (const auto &loc_spec : locations[index]->location_specs()) {
            if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
                std::uint64_t spec_sz;
                ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                sz += spec_sz;
            }
        }
        loc_sz[index] = std::make_pair(locations[index]->type(), sz);

        out_results[index].location_id = std::move(location_id);
        return {ModifierAction::MA_OK, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteBlock);
    auto result = meta_indexer_->ReadModifyWriteBlock(request_context, keys, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteBlock);

    ErrorCode aggregate_ec = result.ec;
    if (result.error_codes.size() != keys.size()) {
        KVCM_LOG_ERROR(
            "BatchAddLocation result size mismatch, expect: %lu, actual: %lu", keys.size(), result.error_codes.size());
        for (auto &add_result : out_results) {
            add_result.ec = ErrorCode::EC_MISMATCH;
        }
        aggregate_ec = ErrorCode::EC_MISMATCH;
    } else {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            out_results[i].ec = result.error_codes[i];
            if (out_results[i].ec == ErrorCode::EC_OK && out_results[i].location_id.empty()) {
                KVCM_LOG_ERROR("BatchAddLocation returned EC_OK with empty location id, key[%lu](%lu)", i, keys[i]);
                out_results[i].ec = ErrorCode::EC_MISMATCH;
                aggregate_ec = ErrorCode::EC_MISMATCH;
            }
        }
    }

    // update the usage of each storage type
    for (std::size_t i = 0; i < keys.size(); i++) {
        if (out_results[i].ec == ErrorCode::EC_OK) {
            meta_indexer_->AddStorageUsageByType(loc_sz[i].first, loc_sz[i].second);
        }
    }

    if (aggregate_ec != ErrorCode::EC_OK) {
        std::vector<ErrorCode> per_key_ec;
        per_key_ec.reserve(out_results.size());
        for (const auto &add_result : out_results) {
            per_key_ec.push_back(add_result.ec);
        }
        LogErrorCodes("meta_indexer_->ReadModifyWriteBlock", per_key_ec, keys);
    }
    return aggregate_ec;
}

namespace {

void ClassifyAddLocationRollbackItems(const KeyVector &keys,
                                      const std::vector<MetaSearcher::AddLocationResult> &add_results,
                                      MetaSearcher::AddLocationRollbackPlan &out_plan,
                                      KeyVector &uncertain_keys,
                                      std::vector<std::string> &uncertain_location_ids,
                                      std::vector<size_t> &uncertain_indices) {
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto &add_result = add_results[i];
        if (add_result.ec == EC_OK && !add_result.location_id.empty()) {
            out_plan.pipeline_keys.push_back(keys[i]);
            out_plan.pipeline_location_ids.push_back(add_result.location_id);
        } else if (!add_result.location_id.empty()) {
            // location id 已生成但写结果失败/未知：先做幂等元数据删除，确认无引用后才能删 URI。
            uncertain_keys.push_back(keys[i]);
            uncertain_location_ids.push_back(add_result.location_id);
            uncertain_indices.push_back(i);
        } else {
            out_plan.direct_delete_indices.push_back(i);
        }
    }
}

} // namespace

size_t MetaSearcher::ClassifyAddLocationRollback(const KeyVector &keys,
                                                 const std::vector<AddLocationResult> &add_results,
                                                 AddLocationRollbackPlan &out_plan) {
    out_plan = {};
    KeyVector uncertain_keys;
    std::vector<std::string> uncertain_location_ids;
    std::vector<size_t> uncertain_indices;
    ClassifyAddLocationRollbackItems(
        keys, add_results, out_plan, uncertain_keys, uncertain_location_ids, uncertain_indices);
    return uncertain_keys.size();
}

ErrorCode MetaSearcher::ReconcileAddLocationRollback(RequestContext *request_context,
                                                     const KeyVector &keys,
                                                     const std::vector<AddLocationResult> &add_results,
                                                     AddLocationRollbackPlan &out_plan) {
    out_plan = {};
    if (keys.size() != add_results.size()) {
        KVCM_LOG_ERROR("ReconcileAddLocationRollback input size mismatch, keys[%lu], results[%lu]",
                       keys.size(),
                       add_results.size());
        return EC_BADARGS;
    }

    KeyVector uncertain_keys;
    std::vector<std::string> uncertain_location_ids;
    std::vector<size_t> uncertain_indices;
    ClassifyAddLocationRollbackItems(
        keys, add_results, out_plan, uncertain_keys, uncertain_location_ids, uncertain_indices);

    if (uncertain_keys.empty()) {
        return EC_OK;
    }

    LocationIdsPerKey location_ids_per_key;
    location_ids_per_key.reserve(uncertain_location_ids.size());
    for (const auto &location_id : uncertain_location_ids) {
        location_ids_per_key.push_back({location_id});
    }
    std::vector<std::vector<ErrorCode>> delete_results;
    const ErrorCode delete_ec = BatchDeleteLocations(request_context,
                                                     uncertain_keys,
                                                     location_ids_per_key,
                                                     delete_results,
                                                     {},
                                                     false /* these failed adds were never included in usage */,
                                                     false /* failed adds were never counted as new keys */);
    const size_t invalid_result_count =
        std::count_if(delete_results.begin(), delete_results.end(), [](const auto &per_location_results) {
            return per_location_results.size() != 1;
        });
    if (delete_results.size() != uncertain_keys.size() || invalid_result_count != 0) {
        KVCM_LOG_WARN("ReconcileAddLocationRollback metadata delete returned unexpected result shape, expected %zu, "
                      "got %zu, invalid inner result count %zu, ec %d; retaining URIs",
                      uncertain_keys.size(),
                      delete_results.size(),
                      invalid_result_count,
                      delete_ec);
        return EC_OK;
    }

    KeyVector sync_keys;
    for (size_t i = 0; i < delete_results.size(); ++i) {
        if (delete_results[i].front() == EC_OK) {
            sync_keys.push_back(uncertain_keys[i]);
        }
    }
    bool metadata_synced = true;
    if (!sync_keys.empty()) {
        metadata_synced = meta_indexer_->Sync(sync_keys);
        if (!metadata_synced) {
            KVCM_LOG_WARN("ReconcileAddLocationRollback failed to sync deleted metadata, key_count[%zu]",
                          sync_keys.size());
        }
    }

    for (size_t i = 0; i < delete_results.size(); ++i) {
        const ErrorCode per_location_ec = delete_results[i].front();
        if (per_location_ec == EC_NOENT || (per_location_ec == EC_OK && metadata_synced)) {
            out_plan.direct_delete_indices.push_back(uncertain_indices[i]);
        } else if (per_location_ec != EC_OK) {
            KVCM_LOG_WARN("ReconcileAddLocationRollback metadata delete failed, key[%lu](%lu), location_id %s, ec %d",
                          i,
                          uncertain_keys[i],
                          uncertain_location_ids[i].c_str(),
                          per_location_ec);
        }
    }
    return EC_OK;
}

ErrorCode
MetaSearcher::BatchReplaceLocationSpecs(RequestContext *request_context,
                                        const KeyVector &keys,
                                        const std::vector<std::vector<ReplaceLocationSpecsTask>> &tasks_per_key,
                                        std::vector<ErrorCode> &out_per_key_ec,
                                        AcquireMetadataWriteLeaseFunc acquire_write_lease) {
    if (keys.size() != tasks_per_key.size()) {
        return EC_BADARGS;
    }
    out_per_key_ec.assign(keys.size(), ErrorCode::EC_OK);
    if (keys.empty()) {
        return EC_OK;
    }

    LocationIdsPerKey location_ids_per_key(keys.size());
    std::vector<std::vector<StorageUsageChange>> usage_changes(keys.size());
    std::unordered_set<int64_t> seen_keys;
    for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
        if (!seen_keys.insert(keys[key_index]).second) {
            std::fill(out_per_key_ec.begin(), out_per_key_ec.end(), EC_BADARGS);
            return EC_BADARGS;
        }
        auto &location_ids = location_ids_per_key[key_index];
        auto &key_usage_changes = usage_changes[key_index];
        location_ids.reserve(tasks_per_key[key_index].size());
        key_usage_changes.resize(tasks_per_key[key_index].size());
        std::unordered_set<std::string> seen_location_ids;
        for (size_t task_index = 0; task_index < tasks_per_key[key_index].size(); ++task_index) {
            const auto &task = tasks_per_key[key_index][task_index];
            const auto &location_id = task.ResolvedLocationId();
            std::uint64_t incoming_size = 0;
            if (location_id.empty() || !seen_location_ids.insert(location_id).second ||
                ValidateConsistentSnapshotVersion(task.specs, &incoming_size) != EC_OK) {
                std::fill(out_per_key_ec.begin(), out_per_key_ec.end(), EC_BADARGS);
                return EC_BADARGS;
            }
            location_ids.push_back(location_id);
            key_usage_changes[task_index].new_size = incoming_size;
        }
    }

    const int64_t batch_create_time = TimestampUtil::GetCurrentTimeUs();
    bool write_lease_attempted = false;
    ErrorCode write_lease_ec = EC_OK;
    MetadataWriteLease write_lease;
    auto modifier = [&keys,
                     &tasks_per_key,
                     &usage_changes,
                     &acquire_write_lease,
                     &write_lease_attempted,
                     &write_lease_ec,
                     &write_lease,
                     batch_create_time](const std::vector<ErrorCode> &get_ecs,
                                        const LocationIdVector &location_ids,
                                        size_t key_index,
                                        CacheLocationVector &locations,
                                        PropertyMap & /*upsert_property_map*/) -> LocationModifierResult {
        if (acquire_write_lease && !write_lease_attempted) {
            std::tie(write_lease_ec, write_lease) = acquire_write_lease();
            write_lease_attempted = true;
        }
        if (write_lease_ec != EC_OK) {
            return {ModifierAction::MA_FAIL, std::vector<ErrorCode>(location_ids.size(), write_lease_ec)};
        }
        const auto &tasks = tasks_per_key[key_index];
        std::vector<ErrorCode> modifier_ecs(location_ids.size(), ErrorCode::EC_OK);
        bool updated = false;
        for (size_t location_index = 0; location_index < location_ids.size(); ++location_index) {
            if (location_index >= tasks.size() || location_index >= get_ecs.size() ||
                location_index >= locations.size()) {
                modifier_ecs[location_index] = ErrorCode::EC_ERROR;
                continue;
            }
            const ErrorCode get_ec = get_ecs[location_index];
            if (get_ec != ErrorCode::EC_OK && get_ec != ErrorCode::EC_NOENT) {
                modifier_ecs[location_index] = get_ec;
                KVCM_LOG_WARN("load location failed, key[%lu](%lu), location_id: %s, return %d",
                              key_index,
                              keys[key_index],
                              location_ids[location_index].c_str(),
                              get_ec);
                continue;
            }

            const auto &task = tasks[location_index];
            auto &usage = usage_changes[key_index][location_index];
            std::shared_ptr<CacheLocation> new_location;
            if (get_ec == ErrorCode::EC_OK && locations[location_index]) {
                if (locations[location_index]->type() != task.type) {
                    modifier_ecs[location_index] = ErrorCode::EC_BADARGS;
                    continue;
                }
                usage.old_size = GetLocationSpecsSize(locations[location_index]->location_specs());
                usage.has_old = true;
                new_location = std::make_shared<CacheLocation>(*locations[location_index]);
            } else {
                new_location = std::make_shared<CacheLocation>();
            }
            if (const auto *interned_location_id = task.ResolvedInternedLocationId()) {
                // New locations and legacy owned-id locations converge to the
                // request's canonical reporter/medium id on the next write.
                new_location->set_id(*interned_location_id);
            } else if (get_ec != ErrorCode::EC_OK || !locations[location_index]) {
                new_location->set_id(task.location_id);
            }

            std::vector<LocationSpec> specs;
            specs.reserve(task.specs.size());
            for (const auto &spec : task.specs) {
                specs.emplace_back(spec.name(), spec.uri());
            }
            new_location->set_location_specs(std::move(specs));
            new_location->set_type(task.type);
            new_location->set_status(task.status);
            new_location->set_spec_size(new_location->location_specs().size());
            new_location->set_create_time(batch_create_time);
            new_location->set_validated_total_size(usage.new_size);
            locations[location_index] = std::move(new_location);
            updated = true;
        }
        if (!updated) {
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }
        return {ModifierAction::MA_OK, std::move(modifier_ecs)};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto result =
        meta_indexer_->ReadModifyWriteLocation(request_context, keys, location_ids_per_key, std::move(modifier));
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);

    bool malformed_result = result.per_location_error_codes.size() != keys.size();
    if (malformed_result) {
        KVCM_LOG_ERROR("BatchReplaceLocationSpecs result size mismatch, keys[%zu], results[%zu]",
                       keys.size(),
                       result.per_location_error_codes.size());
    }
    for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
        ErrorCode key_ec = ErrorCode::EC_OK;
        const size_t expected_location_count = tasks_per_key[key_index].size();
        if (key_index >= result.per_location_error_codes.size() ||
            result.per_location_error_codes[key_index].size() != expected_location_count) {
            key_ec = ErrorCode::EC_MISMATCH;
            malformed_result = true;
        } else {
            const auto &location_ecs = result.per_location_error_codes[key_index];
            for (size_t location_index = 0; location_index < expected_location_count; ++location_index) {
                const ErrorCode location_ec = location_ecs[location_index];
                if (location_ec != ErrorCode::EC_OK) {
                    if (key_ec == ErrorCode::EC_OK) {
                        key_ec = location_ec;
                    }
                    continue;
                }
                const auto &usage = usage_changes[key_index][location_index];
                if (usage.has_old) {
                    ApplyStorageUsageChange(meta_indexer_.get(),
                                            tasks_per_key[key_index][location_index].type,
                                            usage.old_size,
                                            usage.new_size);
                } else {
                    meta_indexer_->AddStorageUsageByType(tasks_per_key[key_index][location_index].type, usage.new_size);
                }
            }
        }
        out_per_key_ec[key_index] = key_ec;
    }
    if (result.ec != ErrorCode::EC_OK) {
        KVCM_LOG_WARN("meta_indexer_->ReadModifyWriteLocation failed, ec: %d", result.ec);
    }
    return malformed_result ? ErrorCode::EC_MISMATCH : result.ec;
}

class MetaSearcher::MergeLocationSpecsTaskView {
public:
    explicit MergeLocationSpecsTaskView(const std::vector<std::vector<MergeLocationSpecsTask>> &nested_tasks)
        : nested_tasks_(&nested_tasks) {}

    MergeLocationSpecsTaskView(const std::vector<size_t> &offsets, std::vector<MergeLocationSpecsTask> &flat_tasks)
        : offsets_(&offsets), flat_tasks_(&flat_tasks) {}

    [[nodiscard]] bool Valid(size_t key_count) const noexcept {
        if (nested_tasks_) {
            return nested_tasks_->size() == key_count;
        }
        if (!offsets_ || !flat_tasks_ || offsets_->size() != key_count + 1 || offsets_->front() != 0 ||
            offsets_->back() != flat_tasks_->size()) {
            return false;
        }
        return std::is_sorted(offsets_->begin(), offsets_->end());
    }

    [[nodiscard]] size_t Size(size_t key_index) const noexcept {
        if (nested_tasks_) {
            return (*nested_tasks_)[key_index].size();
        }
        return (*offsets_)[key_index + 1] - (*offsets_)[key_index];
    }

    [[nodiscard]] const MergeLocationSpecsTask &At(size_t key_index, size_t task_index) const noexcept {
        if (nested_tasks_) {
            return (*nested_tasks_)[key_index][task_index];
        }
        return (*flat_tasks_)[(*offsets_)[key_index] + task_index];
    }

    // The ReportEvent-only flat representation owns its input strings and is
    // dead after this call except for spec names used to map failures back to
    // request items. Transfer the potentially large URI into the immutable
    // CacheLocation while restoring the usually-SSO name in the task. Generic
    // nested callers retain their historical copy semantics.
    [[nodiscard]] LocationSpec CopyOrConsumeSpec(size_t key_index, size_t task_index, size_t spec_index) {
        if (!flat_tasks_) {
            return At(key_index, task_index).SpecAt(spec_index);
        }
        auto &task = (*flat_tasks_)[(*offsets_)[key_index] + task_index];
        if (!task.prevalidated_total_size.has_value()) {
            return task.SpecAt(spec_index);
        }
        auto &source = task.MutableSpecAt(spec_index);
        std::string retained_name = source.name();
        LocationSpec result = std::move(source);
        source.set_name(std::move(retained_name));
        return result;
    }

private:
    const std::vector<std::vector<MergeLocationSpecsTask>> *nested_tasks_ = nullptr;
    const std::vector<size_t> *offsets_ = nullptr;
    std::vector<MergeLocationSpecsTask> *flat_tasks_ = nullptr;
};

ErrorCode MetaSearcher::BatchMergeLocationSpecs(RequestContext *request_context,
                                                const KeyVector &keys,
                                                const std::vector<std::vector<MergeLocationSpecsTask>> &tasks_per_key,
                                                std::vector<ErrorCode> &out_per_key_ec,
                                                AcquireMetadataWriteLeaseFunc acquire_write_lease) {
    return BatchMergeLocationSpecsImpl(request_context,
                                       keys,
                                       MergeLocationSpecsTaskView(tasks_per_key),
                                       out_per_key_ec,
                                       std::move(acquire_write_lease));
}

ErrorCode MetaSearcher::BatchMergeLocationSpecsFlat(RequestContext *request_context,
                                                    const KeyVector &keys,
                                                    const std::vector<size_t> &task_offsets,
                                                    std::vector<MergeLocationSpecsTask> &flat_tasks,
                                                    std::vector<ErrorCode> &out_per_key_ec,
                                                    AcquireMetadataWriteLeaseFunc acquire_write_lease) {
    return BatchMergeLocationSpecsImpl(request_context,
                                       keys,
                                       MergeLocationSpecsTaskView(task_offsets, flat_tasks),
                                       out_per_key_ec,
                                       std::move(acquire_write_lease));
}

ErrorCode MetaSearcher::BatchMergeLocationSpecsImpl(RequestContext *request_context,
                                                    const KeyVector &keys,
                                                    MergeLocationSpecsTaskView tasks,
                                                    std::vector<ErrorCode> &out_per_key_ec,
                                                    AcquireMetadataWriteLeaseFunc acquire_write_lease) {
    if (!tasks.Valid(keys.size())) {
        return EC_BADARGS;
    }
    out_per_key_ec.assign(keys.size(), ErrorCode::EC_OK);
    if (keys.empty()) {
        return EC_OK;
    }

    bool has_duplicate_keys = false;
    if (std::is_sorted(keys.begin(), keys.end())) {
        has_duplicate_keys = std::adjacent_find(keys.begin(), keys.end()) != keys.end();
    } else {
        std::unordered_set<int64_t> seen_keys;
        seen_keys.reserve(keys.size());
        for (const int64_t key : keys) {
            if (!seen_keys.insert(key).second) {
                has_duplicate_keys = true;
                break;
            }
        }
    }
    if (has_duplicate_keys) {
        std::fill(out_per_key_ec.begin(), out_per_key_ec.end(), EC_BADARGS);
        return EC_BADARGS;
    }

    bool use_single_location_fast_path = meta_indexer_->SupportsSingleLocationRmw();
    for (size_t key_index = 0; key_index < keys.size() && use_single_location_fast_path; ++key_index) {
        use_single_location_fast_path = tasks.Size(key_index) == 1;
    }
    LocationIdsPerKey location_ids_per_key(use_single_location_fast_path ? 0 : keys.size());
    LocationIdRefVector single_location_ids;
    if (use_single_location_fast_path) {
        single_location_ids.reserve(keys.size());
    }
    // Keep the incoming and final usage in one request-shaped allocation.
    // The dominant single-location ReportEvent shape has exactly one location per
    // key, so its usage index is the key index and needs no offsets array.
    // Multi-location callers retain the generic flattened-offset layout.
    std::vector<size_t> usage_offsets(use_single_location_fast_path ? 0 : keys.size() + 1, 0);
    std::vector<StorageUsageChange> usage_changes;
    usage_changes.reserve(keys.size());
    for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
        const size_t task_count = tasks.Size(key_index);
        std::unordered_set<std::string_view> seen_location_ids;
        if (task_count > 1) {
            seen_location_ids.reserve(task_count);
        }
        if (!use_single_location_fast_path) {
            location_ids_per_key[key_index].reserve(task_count);
        }
        for (size_t task_index = 0; task_index < task_count; ++task_index) {
            const auto &task = tasks.At(key_index, task_index);
            const auto &task_location_id = task.ResolvedLocationId();
            std::uint64_t incoming_size = 0;
            const ErrorCode validation_ec = task.prevalidated_total_size.has_value()
                                                ? (task.SpecsEmpty() ? EC_BADARGS : EC_OK)
                                                : ValidateConsistentSnapshotVersion(task, &incoming_size);
            if (task.prevalidated_total_size.has_value()) {
                incoming_size = task.prevalidated_total_size->value();
            }
            if (task_location_id.empty() ||
                (task_count > 1 && !seen_location_ids.insert(std::string_view(task_location_id)).second) ||
                validation_ec != EC_OK) {
                std::fill(out_per_key_ec.begin(), out_per_key_ec.end(), EC_BADARGS);
                return EC_BADARGS;
            }
            if (use_single_location_fast_path) {
                single_location_ids.push_back(&task_location_id);
            } else {
                location_ids_per_key[key_index].push_back(task_location_id);
            }
            usage_changes.push_back(StorageUsageChange{0, incoming_size, false});
        }
        if (!use_single_location_fast_path) {
            usage_offsets[key_index + 1] = usage_offsets[key_index] + task_count;
        }
    }

    const int64_t batch_create_time = TimestampUtil::GetCurrentTimeUs();
    bool write_lease_attempted = false;
    ErrorCode write_lease_ec = EC_OK;
    MetadataWriteLease write_lease;
    auto ensure_write_lease = [&acquire_write_lease, &write_lease_attempted, &write_lease_ec, &write_lease]() {
        if (acquire_write_lease && !write_lease_attempted) {
            std::tie(write_lease_ec, write_lease) = acquire_write_lease();
            write_lease_attempted = true;
        }
        return write_lease_ec;
    };

    auto merge_one_location =
        [&keys, &tasks, &usage_offsets, &usage_changes, use_single_location_fast_path, batch_create_time](
            ErrorCode get_ec,
            const LocationId &location_id,
            size_t key_index,
            size_t location_index,
            const CacheLocation *existing_location,
            CacheLocationConstPtr &out_location) -> ErrorCode {
        if (key_index >= keys.size() || location_index >= tasks.Size(key_index)) {
            return EC_MISMATCH;
        }
        const auto &task = tasks.At(key_index, location_index);
        const auto &task_location_id = task.ResolvedLocationId();
        if (location_id != task_location_id) {
            return EC_MISMATCH;
        }

        if (get_ec != EC_OK && get_ec != EC_NOENT) {
            KVCM_LOG_WARN("load target location failed, key[%lu](%lu), location_id[%s], return[%d]",
                          key_index,
                          keys[key_index],
                          task_location_id.c_str(),
                          get_ec);
            return get_ec;
        }

        const size_t usage_index =
            use_single_location_fast_path ? key_index : usage_offsets[key_index] + location_index;
        auto &usage = usage_changes[usage_index];
        const std::uint64_t incoming_size = usage.new_size;
        std::shared_ptr<CacheLocation> new_location;
        bool final_specs_validated = true;
        if (get_ec == EC_OK) {
            if (!existing_location || existing_location->type() != task.type) {
                return existing_location ? ErrorCode::EC_BADARGS : ErrorCode::EC_MISMATCH;
            }
            std::uint64_t replaced_old_size = 0;
            bool has_legacy_duplicate_names = false;
            const auto &old_specs = existing_location->location_specs();
            std::uint64_t cached_single_spec_size = 0;
            const bool has_cached_single_spec_size =
                old_specs.size() == 1 && existing_location->GetValidatedTotalSize(cached_single_spec_size);
            bool old_size_overflow = false;
            for (size_t old_index = 0; old_index < old_specs.size(); ++old_index) {
                const auto &old_spec = old_specs[old_index];
                std::uint64_t old_spec_size = cached_single_spec_size;
                const bool old_spec_validated =
                    has_cached_single_spec_size || TryGetLocationSpecSize(old_spec, old_spec_size);
                if (old_spec_size > std::numeric_limits<std::uint64_t>::max() - usage.old_size) {
                    old_size_overflow = true;
                    break;
                }
                usage.old_size += old_spec_size;
                bool replaces_old_spec = false;
                for (size_t spec_index = 0; spec_index < task.SpecCount(); ++spec_index) {
                    if (task.SpecAt(spec_index).name() == old_spec.name()) {
                        replaces_old_spec = true;
                        break;
                    }
                }
                if (!replaces_old_spec && !old_spec_validated) {
                    final_specs_validated = false;
                }
                if (replaces_old_spec) {
                    if (old_spec_size > std::numeric_limits<std::uint64_t>::max() - replaced_old_size) {
                        old_size_overflow = true;
                        break;
                    }
                    replaced_old_size += old_spec_size;
                }
                has_legacy_duplicate_names =
                    has_legacy_duplicate_names ||
                    std::any_of(old_specs.begin(), old_specs.begin() + old_index, [&old_spec](const auto &prior) {
                        return prior.name() == old_spec.name();
                    });
            }
            if (old_size_overflow) {
                return EC_BADARGS;
            }
            usage.has_old = true;
            const bool replaces_only_spec =
                old_specs.size() == 1 && task.SpecCount() == 1 && old_specs.front().name() == task.SpecAt(0).name();
            if (replaces_only_spec) {
                // The dominant ReportEvent update replaces the only stored
                // spec with one same-named spec. Copying CacheLocation first
                // allocates/copies the old URI only to destroy it immediately
                // in MergeLocationSpecsByName. Build the final immutable value
                // directly for this exact case; every CacheLocation field is
                // assigned below and the incoming spec is still copied once.
                new_location = std::make_shared<CacheLocation>();
                if (const auto *interned_location_id = task.ResolvedInternedLocationId()) {
                    new_location->set_id(*interned_location_id);
                } else {
                    new_location->set_id(existing_location->id());
                }
                std::vector<LocationSpec> specs;
                specs.reserve(1);
                specs.push_back(tasks.CopyOrConsumeSpec(key_index, location_index, 0));
                new_location->set_location_specs(std::move(specs));
            } else {
                new_location = std::make_shared<CacheLocation>(*existing_location);
                if (const auto *interned_location_id = task.ResolvedInternedLocationId()) {
                    new_location->set_id(*interned_location_id);
                }
                MergeLocationSpecsByName(new_location->mutable_location_specs(), task);
            }
            if (has_legacy_duplicate_names) {
                usage.new_size = 0;
                final_specs_validated = true;
                for (const auto &merged_spec : new_location->location_specs()) {
                    std::uint64_t merged_spec_size = 0;
                    if (!TryGetLocationSpecSize(merged_spec, merged_spec_size)) {
                        final_specs_validated = false;
                    }
                    if (merged_spec_size > std::numeric_limits<std::uint64_t>::max() - usage.new_size) {
                        return EC_BADARGS;
                    }
                    usage.new_size += merged_spec_size;
                }
            } else {
                const std::uint64_t retained_size = usage.old_size - replaced_old_size;
                if (incoming_size > std::numeric_limits<std::uint64_t>::max() - retained_size) {
                    return EC_BADARGS;
                }
                usage.new_size = retained_size + incoming_size;
            }
        } else {
            new_location = std::make_shared<CacheLocation>();
            if (const auto *interned_location_id = task.ResolvedInternedLocationId()) {
                new_location->set_id(*interned_location_id);
            } else {
                new_location->set_id(task.location_id);
            }
            std::vector<LocationSpec> specs;
            specs.reserve(task.SpecCount());
            for (size_t spec_index = 0; spec_index < task.SpecCount(); ++spec_index) {
                specs.push_back(tasks.CopyOrConsumeSpec(key_index, location_index, spec_index));
            }
            new_location->set_location_specs(std::move(specs));
            usage.new_size = incoming_size;
        }
        new_location->set_type(task.type);
        new_location->set_status(task.status);
        new_location->set_create_time(batch_create_time);
        new_location->set_spec_size(new_location->location_specs().size());
        if (final_specs_validated) {
            new_location->set_validated_total_size(usage.new_size);
        }
        out_location = std::move(new_location);
        return EC_OK;
    };

    auto modifier = [&tasks, &keys, &ensure_write_lease, &merge_one_location](
                        const std::vector<ErrorCode> &get_ecs,
                        const LocationIdVector &location_ids,
                        size_t key_index,
                        CacheLocationVector &locations,
                        PropertyMap & /*upsert_property_map*/) -> LocationModifierResult {
        if (location_ids.empty()) {
            return {ModifierAction::MA_SKIP, {}};
        }
        const ErrorCode lease_ec = ensure_write_lease();
        if (lease_ec != EC_OK) {
            return {ModifierAction::MA_FAIL, std::vector<ErrorCode>(location_ids.size(), lease_ec)};
        }
        if (key_index >= keys.size() || get_ecs.size() != location_ids.size() ||
            locations.size() != location_ids.size() || tasks.Size(key_index) != location_ids.size()) {
            return {ModifierAction::MA_FAIL, std::vector<ErrorCode>(location_ids.size(), EC_MISMATCH)};
        }

        std::vector<ErrorCode> modifier_ecs(location_ids.size(), ErrorCode::EC_OK);
        bool updated = false;
        for (size_t location_index = 0; location_index < location_ids.size(); ++location_index) {
            modifier_ecs[location_index] = merge_one_location(get_ecs[location_index],
                                                              location_ids[location_index],
                                                              key_index,
                                                              location_index,
                                                              locations[location_index].get(),
                                                              locations[location_index]);
            updated = updated || modifier_ecs[location_index] == EC_OK;
        }
        if (!updated) {
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }
        return {ModifierAction::MA_OK, std::move(modifier_ecs)};
    };

    auto single_modifier = [&tasks, &keys, &ensure_write_lease, &merge_one_location](
                               ErrorCode get_ec,
                               const LocationId &location_id,
                               size_t key_index,
                               const CacheLocation *existing_location,
                               CacheLocationConstPtr &out_location) -> ModifierResult {
        const ErrorCode lease_ec = ensure_write_lease();
        if (lease_ec != EC_OK) {
            return {ModifierAction::MA_FAIL, lease_ec};
        }
        if (key_index >= keys.size() || tasks.Size(key_index) != 1) {
            return {ModifierAction::MA_FAIL, EC_MISMATCH};
        }
        const ErrorCode ec = merge_one_location(get_ec, location_id, key_index, 0, existing_location, out_location);
        return {ec == EC_OK ? ModifierAction::MA_OK : ModifierAction::MA_SKIP, ec};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    if (use_single_location_fast_path) {
        auto result = meta_indexer_->ReadModifyWriteSingleTargetLocations(
            request_context, keys, single_location_ids, single_modifier);
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);

        bool malformed_result = result.error_codes.size() != keys.size();
        if (malformed_result) {
            KVCM_LOG_ERROR("BatchMergeLocationSpecs flat result size mismatch, keys[%zu], results[%zu]",
                           keys.size(),
                           result.error_codes.size());
        }
        for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
            const ErrorCode key_ec =
                key_index < result.error_codes.size() ? result.error_codes[key_index] : EC_MISMATCH;
            out_per_key_ec[key_index] = key_ec;
            if (key_ec == EC_OK) {
                const auto &usage = usage_changes[key_index];
                const auto &task = tasks.At(key_index, 0);
                if (usage.has_old) {
                    ApplyStorageUsageChange(meta_indexer_.get(), task.type, usage.old_size, usage.new_size);
                } else {
                    meta_indexer_->AddStorageUsageByType(task.type, usage.new_size);
                }
            }
        }
        ErrorCode final_ec = result.ec;
        if (result.ec != EC_OK) {
            KVCM_LOG_WARN("meta_indexer_->ReadModifyWriteSingleTargetLocations failed, ec: %d", result.ec);
        }
        if (malformed_result) {
            final_ec = final_ec == EC_OK ? EC_MISMATCH : (final_ec == EC_MISMATCH ? EC_MISMATCH : EC_PARTIAL_OK);
        }
        return final_ec;
    }

    auto result = meta_indexer_->ReadModifyWriteTargetLocations(request_context, keys, location_ids_per_key, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);

    bool malformed_result = result.per_location_error_codes.size() != keys.size();
    if (malformed_result) {
        KVCM_LOG_ERROR("BatchMergeLocationSpecs fused result size mismatch, keys[%zu], results[%zu]",
                       keys.size(),
                       result.per_location_error_codes.size());
    }
    for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
        ErrorCode key_ec = EC_OK;
        const size_t expected_location_count = tasks.Size(key_index);
        if (key_index >= result.per_location_error_codes.size() ||
            result.per_location_error_codes[key_index].size() != expected_location_count) {
            key_ec = EC_MISMATCH;
            malformed_result = true;
        } else {
            for (size_t location_index = 0; location_index < expected_location_count; ++location_index) {
                const ErrorCode location_ec = result.per_location_error_codes[key_index][location_index];
                if (location_ec != EC_OK) {
                    if (key_ec == EC_OK) {
                        key_ec = location_ec;
                    }
                    continue;
                }
                const auto &usage = usage_changes[usage_offsets[key_index] + location_index];
                const auto &task = tasks.At(key_index, location_index);
                if (usage.has_old) {
                    ApplyStorageUsageChange(meta_indexer_.get(), task.type, usage.old_size, usage.new_size);
                } else {
                    meta_indexer_->AddStorageUsageByType(task.type, usage.new_size);
                }
            }
        }
        out_per_key_ec[key_index] = key_ec;
    }

    ErrorCode final_ec = result.ec;
    if (result.ec != EC_OK) {
        KVCM_LOG_WARN("meta_indexer_->ReadModifyWriteTargetLocations failed, ec: %d", result.ec);
    }
    if (malformed_result) {
        if (final_ec == EC_OK) {
            final_ec = EC_MISMATCH;
        } else if (final_ec != EC_MISMATCH) {
            final_ec = EC_PARTIAL_OK;
        }
    }
    return final_ec;
}

ErrorCode MetaSearcher::BatchDeleteLocationSpecs(RequestContext *request_context,
                                                 const KeyVector &keys,
                                                 const std::vector<std::vector<DeleteLocationSpecsTask>> &tasks_per_key,
                                                 std::vector<std::vector<ErrorCode>> &out_batch_results,
                                                 std::vector<std::vector<bool>> *out_missing_targets,
                                                 AcquireMetadataWriteLeaseFunc acquire_write_lease) {
    if (keys.size() != tasks_per_key.size()) {
        return EC_BADARGS;
    }
    out_batch_results.clear();
    out_batch_results.resize(keys.size());
    if (out_missing_targets) {
        out_missing_targets->clear();
        out_missing_targets->resize(keys.size());
    }

    std::unordered_set<int64_t> seen_keys;
    bool invalid_input = false;
    for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
        out_batch_results[key_index].assign(tasks_per_key[key_index].size(), ErrorCode::EC_OK);
        if (out_missing_targets) {
            (*out_missing_targets)[key_index].assign(tasks_per_key[key_index].size(), false);
        }
        if (!seen_keys.insert(keys[key_index]).second) {
            invalid_input = true;
        }
        std::unordered_set<std::string_view> seen_location_ids;
        if (tasks_per_key[key_index].size() > 1) {
            seen_location_ids.reserve(tasks_per_key[key_index].size());
        }
        for (const auto &task : tasks_per_key[key_index]) {
            const auto &location_id = task.ResolvedLocationId();
            if (location_id.empty() || !seen_location_ids.insert(location_id).second) {
                invalid_input = true;
            }
        }
    }
    if (invalid_input) {
        for (auto &per_key_results : out_batch_results) {
            std::fill(per_key_results.begin(), per_key_results.end(), EC_BADARGS);
        }
        return EC_BADARGS;
    }

    // 每个 DeleteLocationSpecsTask 需要独立返回结果。一个 key 下的 location
    // 必须唯一；否则多个 RMW 槽会从同一旧值计算并发生 last-write-wins 丢更新。
    KeyVector flat_keys;
    LocationIdsPerKey flat_location_ids;
    std::vector<std::pair<size_t, size_t>> flat_to_original_task_indices;
    std::vector<std::pair<DataStorageType, std::uint64_t>> flat_deleted_specs_size;
    std::vector<std::uint8_t> flat_missing_targets;
    for (size_t i = 0; i < keys.size(); ++i) {
        for (size_t task_index = 0; task_index < tasks_per_key[i].size(); ++task_index) {
            flat_keys.push_back(keys[i]);
            flat_location_ids.push_back({tasks_per_key[i][task_index].ResolvedLocationId()});
            flat_to_original_task_indices.push_back({i, task_index});
            flat_deleted_specs_size.emplace_back(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN, 0);
            flat_missing_targets.push_back(0);
        }
    }
    if (flat_keys.empty()) {
        return EC_OK;
    }

    // 每条 flat task 独立处理：NOENT 视为幂等成功；spec_names 为空返回 BADARGS；
    // 删除后无剩余 specs 则删除整个 location，否则 COW 更新 location_specs。
    bool write_lease_attempted = false;
    ErrorCode write_lease_ec = EC_OK;
    MetadataWriteLease write_lease;
    auto modifier = [&keys,
                     &tasks_per_key,
                     &flat_to_original_task_indices,
                     &flat_deleted_specs_size,
                     &flat_missing_targets,
                     &acquire_write_lease,
                     &write_lease_attempted,
                     &write_lease_ec,
                     &write_lease](const std::vector<ErrorCode> &get_ecs,
                                   const LocationIdVector &loc_ids,
                                   size_t key_index,
                                   CacheLocationVector &locs,
                                   PropertyMap &upsert_property_map) -> LocationModifierResult {
        (void)upsert_property_map;
        if (acquire_write_lease && !write_lease_attempted) {
            std::tie(write_lease_ec, write_lease) = acquire_write_lease();
            write_lease_attempted = true;
        }
        if (write_lease_ec != EC_OK) {
            return {ModifierAction::MA_FAIL, std::vector<ErrorCode>(loc_ids.size(), write_lease_ec)};
        }
        std::vector<ErrorCode> modifier_ecs(loc_ids.size(), ErrorCode::EC_OK);
        if (loc_ids.size() != 1 || key_index >= flat_to_original_task_indices.size()) {
            modifier_ecs.assign(loc_ids.size(), ErrorCode::EC_ERROR);
            return {ModifierAction::MA_FAIL, std::move(modifier_ecs)};
        }

        const auto [original_key_index, original_task_index] = flat_to_original_task_indices[key_index];
        const auto &task = tasks_per_key[original_key_index][original_task_index];
        if (task.spec_names.empty() || std::any_of(task.spec_names.begin(),
                                                   task.spec_names.end(),
                                                   [](const std::string &name) { return name.empty(); })) {
            modifier_ecs[0] = ErrorCode::EC_BADARGS;
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }
        const ErrorCode ec = get_ecs.empty() ? ErrorCode::EC_ERROR : get_ecs[0];
        const std::string &loc_id = loc_ids[0];

        if (ec != ErrorCode::EC_OK) {
            if (ec == ErrorCode::EC_NOENT) {
                flat_missing_targets[key_index] = 1;
            }
            modifier_ecs[0] = ec == ErrorCode::EC_NOENT ? ErrorCode::EC_OK : ec;
            if (ec != ErrorCode::EC_NOENT) {
                KVCM_LOG_WARN("load location failed, key[%lu](%lu), location_id: %s, return %d",
                              original_key_index,
                              keys[original_key_index],
                              loc_id.c_str(),
                              ec);
            }
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }

        if (locs.empty() || !locs[0]) {
            flat_missing_targets[key_index] = 1;
            modifier_ecs[0] = ErrorCode::EC_OK;
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }

        std::uint64_t validated_total_size = 0;
        const bool old_specs_validated = locs[0]->GetValidatedTotalSize(validated_total_size);
        std::unordered_set<std::string> delete_spec_names(task.spec_names.begin(), task.spec_names.end());
        std::vector<LocationSpec> kept_specs;
        std::vector<LocationSpec> deleted_specs;
        kept_specs.reserve(locs[0]->location_specs().size());
        deleted_specs.reserve(task.spec_names.size());
        for (const auto &spec : locs[0]->location_specs()) {
            if (delete_spec_names.count(spec.name()) == 0) {
                kept_specs.emplace_back(spec.name(), spec.uri());
            } else {
                deleted_specs.emplace_back(spec.name(), spec.uri());
            }
        }
        if (kept_specs.size() == locs[0]->location_specs().size()) {
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }

        const std::uint64_t deleted_specs_size = GetLocationSpecsSize(deleted_specs);
        flat_deleted_specs_size[key_index] = std::make_pair(locs[0]->type(), deleted_specs_size);
        if (kept_specs.empty()) {
            return {ModifierAction::MA_DELETE, std::move(modifier_ecs)};
        }

        auto new_loc = std::make_shared<CacheLocation>(*locs[0]);
        if (const auto *interned_location_id = task.ResolvedInternedLocationId()) {
            new_loc->set_id(*interned_location_id);
        }
        new_loc->set_location_specs(std::move(kept_specs));
        new_loc->set_spec_size(new_loc->location_specs().size());
        if (old_specs_validated && deleted_specs_size <= validated_total_size) {
            new_loc->set_validated_total_size(validated_total_size - deleted_specs_size);
        }
        locs[0] = std::move(new_loc);
        return {ModifierAction::MA_OK, std::move(modifier_ecs)};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto result = meta_indexer_->ReadModifyWriteLocation(request_context, flat_keys, flat_location_ids, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);

    // ReadModifyWriteLocation 返回 flat 维度结果，这里映射回原始 key/task 维度。
    bool malformed_result = result.per_location_error_codes.size() != flat_to_original_task_indices.size();
    if (malformed_result) {
        KVCM_LOG_ERROR("BatchDeleteLocationSpecs result size mismatch, tasks[%zu], results[%zu]",
                       flat_to_original_task_indices.size(),
                       result.per_location_error_codes.size());
    }
    for (size_t i = 0; i < flat_to_original_task_indices.size(); ++i) {
        const auto [original_key_index, original_task_index] = flat_to_original_task_indices[i];
        ErrorCode ec = ErrorCode::EC_MISMATCH;
        if (i < result.per_location_error_codes.size() && result.per_location_error_codes[i].size() == 1) {
            ec = result.per_location_error_codes[i][0];
        } else {
            malformed_result = true;
        }
        out_batch_results[original_key_index][original_task_index] = ec;
        if (out_missing_targets) {
            (*out_missing_targets)[original_key_index][original_task_index] = flat_missing_targets[i] != 0;
        }
    }

    // 只有实际删除了 specs 且 RMW 成功时，才扣减 storage usage。
    for (size_t i = 0; i < flat_to_original_task_indices.size(); ++i) {
        const auto [original_key_index, original_task_index] = flat_to_original_task_indices[i];
        if (out_batch_results[original_key_index][original_task_index] == ErrorCode::EC_OK &&
            flat_deleted_specs_size[i].first != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) {
            meta_indexer_->SubStorageUsageByType(flat_deleted_specs_size[i].first, flat_deleted_specs_size[i].second);
        }
    }

    if (result.ec != ErrorCode::EC_OK) {
        KVCM_LOG_WARN("meta_indexer_->ReadModifyWriteLocation failed, ec: %d", result.ec);
    }
    return malformed_result ? ErrorCode::EC_MISMATCH : result.ec;
}

ErrorCode MetaSearcher::BatchUpdateLocationStatus(RequestContext *request_context,
                                                  const KeyVector &keys,
                                                  const std::vector<std::vector<LocationUpdateTask>> &batch_tasks,
                                                  std::vector<std::vector<ErrorCode>> &out_batch_results) {

    if (keys.size() != batch_tasks.size()) {
        return EC_BADARGS;
    }
    out_batch_results.clear();
    out_batch_results.resize(keys.size());

    LocationIdsPerKey location_ids_per_key(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        location_ids_per_key[i].reserve(batch_tasks[i].size());
        for (const auto &task : batch_tasks[i]) {
            location_ids_per_key[i].push_back(task.location_id);
        }
    }

    // Per-key modifier: OK slots flip to new_status and report EC_OK so the
    // upsert ec eventually lands on them; NOENT slots are reported as EC_OK
    // (idempotent no-op); hard errors are surfaced verbatim per slot.
    auto modifier = [&keys, &batch_tasks](const std::vector<ErrorCode> &get_ecs,
                                          const LocationIdVector &loc_ids,
                                          size_t key_index,
                                          CacheLocationVector &locs,
                                          PropertyMap &upsert_property_map) -> LocationModifierResult {
        (void)upsert_property_map;
        std::vector<ErrorCode> modifier_ecs(loc_ids.size(), ErrorCode::EC_OK);
        bool updated = false;
        for (size_t loc_index = 0; loc_index < loc_ids.size(); ++loc_index) {
            const ErrorCode ec = get_ecs[loc_index];
            const std::string &loc_id = loc_ids[loc_index];
            if (ec != ErrorCode::EC_OK) {
                modifier_ecs[loc_index] = ec;
                if (ec != ErrorCode::EC_NOENT) {
                    KVCM_LOG_WARN("load location failed, key[%lu](%lu), location_id: %s, return %d",
                                  key_index,
                                  keys[key_index],
                                  loc_id.c_str(),
                                  ec);
                }
                continue;
            }
            updated = true;
            // COW: copy the location, modify the copy, replace the pointer
            auto new_loc = std::make_shared<CacheLocation>(*locs[loc_index]);
            new_loc->set_status(batch_tasks[key_index][loc_index].new_status);
            locs[loc_index] = std::move(new_loc);
        }
        if (!updated) {
            // do not need to update status, skip and return ok
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }
        return {ModifierAction::MA_OK, std::move(modifier_ecs)};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto result = meta_indexer_->ReadModifyWriteLocation(request_context, keys, location_ids_per_key, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    out_batch_results = std::move(result.per_location_error_codes);

    if (result.ec != ErrorCode::EC_OK) {
        KVCM_LOG_WARN("meta_indexer_->ReadModifyWriteLocation failed, ec: %d", result.ec);
    }
    return result.ec;
}

ErrorCode MetaSearcher::BatchCASLocationStatus(RequestContext *request_context,
                                               const KeyVector &keys,
                                               const std::vector<std::vector<LocationCASTask>> &batch_tasks,
                                               std::vector<std::vector<ErrorCode>> &out_batch_results,
                                               bool refresh_cache_from_persistent) {

    if (keys.size() != batch_tasks.size()) {
        return EC_BADARGS;
    }
    out_batch_results.clear();
    out_batch_results.resize(keys.size());

    LocationIdsPerKey location_ids_per_key(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        location_ids_per_key[i].reserve(batch_tasks[i].size());
        for (const auto &task : batch_tasks[i]) {
            location_ids_per_key[i].push_back(task.location_id);
        }
    }

    // Per-key CAS modifier: OK slot whose status matches old_status flips to
    // new_status (slot ec EC_OK -> participates in upsert); status mismatch
    // yields EC_MISMATCH; NOENT is idempotent EC_OK; hard errors surface as-is.
    auto modifier = [&keys, &batch_tasks](const std::vector<ErrorCode> &get_ecs,
                                          const LocationIdVector &loc_ids,
                                          size_t key_index,
                                          CacheLocationVector &locs,
                                          PropertyMap &upsert_property_map) -> LocationModifierResult {
        (void)upsert_property_map;
        std::vector<ErrorCode> modifier_ecs(loc_ids.size(), ErrorCode::EC_OK);
        bool updated = false;
        for (size_t loc_index = 0; loc_index < loc_ids.size(); ++loc_index) {
            const ErrorCode ec = get_ecs[loc_index];
            const std::string &loc_id = loc_ids[loc_index];
            if (ec != ErrorCode::EC_OK) {
                modifier_ecs[loc_index] = ec;
                if (ec != ErrorCode::EC_NOENT) {
                    KVCM_LOG_WARN("load location failed, key[%lu](%lu), location_id: %s, return %d",
                                  key_index,
                                  keys[key_index],
                                  loc_id.c_str(),
                                  ec);
                }
                continue;
            }
            const auto &task = batch_tasks[key_index][loc_index];
            if ((!task.expected_location_value.empty() &&
                 locs[loc_index]->ToJsonString() != task.expected_location_value) ||
                locs[loc_index]->status() != task.old_status) {
                modifier_ecs[loc_index] = ErrorCode::EC_MISMATCH;
            } else {
                updated = true;
                // COW: copy the location, modify the copy, replace the pointer
                auto new_loc = std::make_shared<CacheLocation>(*locs[loc_index]);
                new_loc->set_status(task.new_status);
                locs[loc_index] = std::move(new_loc);
            }
        }
        if (!updated) {
            // do not need to update status, skip and return ok
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }
        return {ModifierAction::MA_OK, std::move(modifier_ecs)};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto result = meta_indexer_->ReadModifyWriteLocation(
        request_context, keys, location_ids_per_key, modifier, true, refresh_cache_from_persistent);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    out_batch_results = std::move(result.per_location_error_codes);

    if (result.ec != ErrorCode::EC_OK) {
        KVCM_LOG_WARN("meta_indexer_->ReadModifyWriteLocation failed, ec: %d", result.ec);
    }
    return result.ec;
}

ErrorCode MetaSearcher::BatchCADLocationStatus(RequestContext *request_context,
                                               const KeyVector &keys,
                                               const std::vector<std::vector<LocationCADTask>> &batch_tasks,
                                               std::vector<std::vector<ErrorCode>> &out_batch_results) {
    if (keys.size() != batch_tasks.size()) {
        return EC_BADARGS;
    }
    out_batch_results.clear();
    out_batch_results.resize(keys.size());

    std::vector<std::vector<std::pair<DataStorageType, std::uint64_t>>> locs_sz(keys.size());
    LocationIdsPerKey location_ids_per_key(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        locs_sz[i].resize(batch_tasks[i].size());
        location_ids_per_key[i].reserve(batch_tasks[i].size());
        for (const auto &task : batch_tasks[i]) {
            location_ids_per_key[i].push_back(task.location_id);
        }
    }

    // Per-key CAD modifier: each slot is gated by a status match. Matching
    // OK slots stay EC_OK (delete will be dispatched, payload size captured
    // for usage replay); mismatches yield EC_MISMATCH; NOENT is EC_OK
    // (idempotent); hard errors surface verbatim.
    auto modifier = [&keys, &batch_tasks, &locs_sz](const std::vector<ErrorCode> &get_ecs,
                                                    const LocationIdVector &loc_ids,
                                                    size_t key_index,
                                                    CacheLocationVector &locs,
                                                    PropertyMap &upsert_property_map) -> LocationModifierResult {
        (void)upsert_property_map;
        std::vector<ErrorCode> modifier_ecs(loc_ids.size(), ErrorCode::EC_OK);
        bool updated = false;
        for (size_t loc_index = 0; loc_index < loc_ids.size(); ++loc_index) {
            const ErrorCode ec = get_ecs[loc_index];
            const std::string &loc_id = loc_ids[loc_index];
            if (ec != ErrorCode::EC_OK) {
                modifier_ecs[loc_index] = ec;
                if (ec != ErrorCode::EC_NOENT) {
                    KVCM_LOG_WARN("load location failed, key[%lu](%lu), location_id: %s, return %d",
                                  key_index,
                                  keys[key_index],
                                  loc_id.c_str(),
                                  ec);
                }
                continue;
            }
            if (!locs[loc_index] || locs[loc_index]->status() != batch_tasks[key_index][loc_index].expect_status) {
                modifier_ecs[loc_index] = ErrorCode::EC_MISMATCH;
                continue;
            }
            updated = true;
            // compute storage size before deletion for usage tracking
            std::uint64_t sz = 0;
            for (const auto &loc_spec : locs[loc_index]->location_specs()) {
                if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
                    std::uint64_t spec_sz = 0;
                    ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                    sz += spec_sz;
                }
            }
            locs_sz[key_index][loc_index] = std::make_pair(locs[loc_index]->type(), sz);
        }
        if (!updated) {
            // do not need to update status, skip and return ok
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }
        return {ModifierAction::MA_DELETE, std::move(modifier_ecs)};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto result = meta_indexer_->ReadModifyWriteLocation(request_context, keys, location_ids_per_key, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    out_batch_results = std::move(result.per_location_error_codes);

    // update the usage of each storage type
    for (std::size_t i = 0; i < keys.size(); ++i) {
        for (std::size_t j = 0; j < batch_tasks[i].size(); ++j) {
            if (j < out_batch_results[i].size() && out_batch_results[i][j] == ErrorCode::EC_OK) {
                meta_indexer_->SubStorageUsageByType(locs_sz[i][j].first, locs_sz[i][j].second);
            }
        }
    }

    if (result.ec != ErrorCode::EC_OK) {
        KVCM_LOG_WARN("meta_indexer_->ReadModifyWriteLocation failed, ec: %d", result.ec);
    }
    return result.ec;
}

ErrorCode MetaSearcher::BatchDeleteLocations(RequestContext *request_context,
                                             const KeyVector &keys,
                                             const LocationIdsPerKey &location_ids_per_key,
                                             std::vector<std::vector<ErrorCode>> &out_per_location_ec,
                                             const std::vector<std::vector<std::string>> &expected_location_values,
                                             bool adjust_storage_usage,
                                             bool adjust_reclaimed_key_count) {
    if (keys.size() != location_ids_per_key.size()) {
        return EC_BADARGS;
    }
    const bool check_expected_values = !expected_location_values.empty();
    if (check_expected_values) {
        if (expected_location_values.size() != location_ids_per_key.size()) {
            return EC_BADARGS;
        }
        for (size_t i = 0; i < location_ids_per_key.size(); ++i) {
            if (expected_location_values[i].size() != location_ids_per_key[i].size()) {
                return EC_BADARGS;
            }
        }
    }
    out_per_location_ec.clear();
    out_per_location_ec.resize(keys.size());

    std::vector<std::vector<std::pair<DataStorageType, std::uint64_t>>> locs_sz(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        locs_sz[i].resize(location_ids_per_key[i].size());
    }

    auto modifier = [&keys, &locs_sz, &expected_location_values, check_expected_values](
                        const std::vector<ErrorCode> &get_ecs,
                        const LocationIdVector &loc_ids,
                        size_t key_index,
                        CacheLocationVector &locs,
                        PropertyMap & /*upsert_property_map*/) -> LocationModifierResult {
        std::vector<ErrorCode> modifier_ecs(loc_ids.size(), ErrorCode::EC_OK);
        bool any_found = false;
        for (size_t k = 0; k < loc_ids.size(); ++k) {
            const ErrorCode ec = get_ecs[k];
            if (ec == ErrorCode::EC_NOENT || loc_ids[k].empty()) {
                modifier_ecs[k] = (ec != ErrorCode::EC_OK) ? ec : ErrorCode::EC_NOENT;
                continue;
            }
            if (ec != ErrorCode::EC_OK) {
                KVCM_LOG_WARN("location load failed, key[%lu](%lu), loc_id: %s, return %d",
                              key_index,
                              keys[key_index],
                              loc_ids[k].c_str(),
                              ec);
                modifier_ecs[k] = ec;
                continue;
            }
            if (check_expected_values &&
                (!locs[k] || locs[k]->ToJsonString() != expected_location_values[key_index][k])) {
                // The stable location id was refreshed after the cleanup
                // scan. Treat the old delete request as stale.
                modifier_ecs[k] = ErrorCode::EC_MISMATCH;
                continue;
            }
            any_found = true;
            std::uint64_t sz = 0;
            for (const auto &loc_spec : locs[k]->location_specs()) {
                if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
                    std::uint64_t spec_sz = 0;
                    ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                    sz += spec_sz;
                }
            }
            locs_sz[key_index][k] = std::make_pair(locs[k]->type(), sz);
        }
        if (!any_found) {
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }
        return {ModifierAction::MA_DELETE, std::move(modifier_ecs)};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto result = meta_indexer_->ReadModifyWriteLocation(
        request_context, keys, location_ids_per_key, modifier, adjust_reclaimed_key_count);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    out_per_location_ec = std::move(result.per_location_error_codes);

    if (adjust_storage_usage) {
        for (size_t i = 0; i < keys.size(); ++i) {
            if (i >= out_per_location_ec.size()) {
                continue;
            }
            for (size_t k = 0; k < location_ids_per_key[i].size(); ++k) {
                if (k >= out_per_location_ec[i].size()) {
                    continue;
                }
                if (out_per_location_ec[i][k] == ErrorCode::EC_OK && !location_ids_per_key[i][k].empty()) {
                    meta_indexer_->SubStorageUsageByType(locs_sz[i][k].first, locs_sz[i][k].second);
                }
            }
        }
    }

    if (result.ec != ErrorCode::EC_OK) {
        KVCM_LOG_WARN("meta_indexer_->ReadModifyWriteLocation failed, ec: %d", result.ec);
    }
    return result.ec;
}

ErrorCode
MetaSearcher::VisitAllLocations(RequestContext *request_context, size_t scan_batch_size, LocationVisitor visitor) {
    if (!visitor) {
        return EC_BADARGS;
    }
    if (scan_batch_size == 0) {
        scan_batch_size = 1000;
    }

    bool has_failure = false;
    std::string cursor = SCAN_BASE_CURSOR;
    do {
        std::string next_cursor;
        KeyVector keys;
        if (auto ec = meta_indexer_->Scan(request_context, cursor, scan_batch_size, next_cursor, keys); ec != EC_OK) {
            KVCM_LOG_WARN("VisitAllLocations: scan failed, ec %d", ec);
            return ec;
        }
        if (!keys.empty()) {
            CacheLocationMapVector location_maps;
            auto get_result = meta_indexer_->GetLocations(request_context, keys, location_maps);
            if (get_result.ec != EC_OK && get_result.ec != EC_PARTIAL_OK) {
                KVCM_LOG_WARN("VisitAllLocations: GetLocations failed, ec %d", get_result.ec);
                return get_result.ec;
            }
            if (get_result.ec == EC_PARTIAL_OK) {
                has_failure = true;
            }
            for (size_t i = 0; i < keys.size(); ++i) {
                if (i >= location_maps.size() || i >= get_result.error_codes.size() ||
                    get_result.error_codes[i] != EC_OK) {
                    has_failure = true;
                    continue;
                }
                for (const auto &[location_id, location] : location_maps[i]) {
                    if (location) {
                        visitor(keys[i], location_id, *location);
                    }
                }
            }
        }
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);

    return has_failure ? EC_PARTIAL_OK : EC_OK;
}

ErrorCode MetaSearcher::CleanupLocationsByPredicate(RequestContext *request_context,
                                                    DataStorageType storage_type,
                                                    size_t scan_batch_size,
                                                    LocationCleanupPredicate should_delete,
                                                    std::function<bool()> should_abort,
                                                    AcquireMetadataWriteLeaseFunc acquire_cleanup_lease) {
    if (!should_delete) {
        return EC_BADARGS;
    }
    if (scan_batch_size == 0) {
        scan_batch_size = 1000;
    }

    bool has_failure = false;
    std::string cursor = SCAN_BASE_CURSOR;
    do {
        if (should_abort && should_abort()) {
            KVCM_LOG_INFO("CleanupLocationsByPredicate: aborted by caller");
            return EC_OK;
        }
        std::string next_cursor;
        KeyVector keys;
        if (auto ec = meta_indexer_->Scan(request_context, cursor, scan_batch_size, next_cursor, keys); ec != EC_OK) {
            KVCM_LOG_WARN("CleanupLocationsByPredicate: scan failed, ec %d", ec);
            return ec;
        }
        if (!keys.empty()) {
            CacheLocationMapVector location_maps;
            auto get_result = meta_indexer_->GetLocations(request_context, keys, location_maps);
            if (get_result.ec != EC_OK && get_result.ec != EC_PARTIAL_OK) {
                KVCM_LOG_WARN("CleanupLocationsByPredicate: GetLocations failed, ec %d", get_result.ec);
                return get_result.ec;
            }
            if (get_result.ec == EC_PARTIAL_OK) {
                has_failure = true;
            }
            LocationIdsPerKey delete_location_ids(keys.size());
            std::vector<std::vector<std::string>> expected_location_values(keys.size());
            bool has_deletes = false;
            for (size_t i = 0; i < keys.size(); ++i) {
                if (i >= location_maps.size() || i >= get_result.error_codes.size() ||
                    get_result.error_codes[i] != EC_OK) {
                    has_failure = true;
                    continue;
                }
                for (const auto &[location_id, location] : location_maps[i]) {
                    if (!location || location->type() != storage_type) {
                        continue;
                    }
                    if (should_delete(keys[i], location_id, *location)) {
                        delete_location_ids[i].push_back(location_id);
                        expected_location_values[i].push_back(location->ToJsonString());
                        has_deletes = true;
                    }
                }
            }
            if (has_deletes) {
                // Callers without a lifecycle lease historically cancel only at
                // scan-batch boundaries.  Lease-aware cleanup rechecks here so
                // a lifecycle change during the scan cannot reach the delete.
                if (acquire_cleanup_lease && should_abort && should_abort()) {
                    KVCM_LOG_INFO("CleanupLocationsByPredicate: aborted before delete");
                    return EC_OK;
                }
                if (acquire_cleanup_lease) {
                    auto [lease_ec, cleanup_lease] = acquire_cleanup_lease();
                    if (lease_ec == EC_MISMATCH || lease_ec == EC_NODE_NOT_REGISTERED ||
                        lease_ec == EC_INSTANCE_NOT_EXIST) {
                        KVCM_LOG_INFO("CleanupLocationsByPredicate: lifecycle changed before delete, ec %d", lease_ec);
                        return EC_OK;
                    }
                    if (lease_ec != EC_OK) {
                        KVCM_LOG_WARN("CleanupLocationsByPredicate: failed to acquire cleanup lease, ec %d", lease_ec);
                        return lease_ec;
                    }

                    KeyVector delete_keys;
                    LocationIdsPerKey compact_location_ids;
                    std::vector<std::vector<std::string>> compact_expected_values;
                    delete_keys.reserve(keys.size());
                    compact_location_ids.reserve(keys.size());
                    compact_expected_values.reserve(keys.size());
                    for (size_t i = 0; i < keys.size(); ++i) {
                        if (delete_location_ids[i].empty()) {
                            continue;
                        }
                        delete_keys.push_back(keys[i]);
                        compact_location_ids.push_back(std::move(delete_location_ids[i]));
                        compact_expected_values.push_back(std::move(expected_location_values[i]));
                    }
                    std::vector<std::vector<ErrorCode>> per_location_ec;
                    const ErrorCode delete_ec = BatchDeleteLocations(
                        request_context, delete_keys, compact_location_ids, per_location_ec, compact_expected_values);
                    if (delete_ec != EC_OK) {
                        has_failure = true;
                    }
                    for (const auto &per_key_ec : per_location_ec) {
                        for (const ErrorCode location_ec : per_key_ec) {
                            if (location_ec != EC_OK && location_ec != EC_NOENT && location_ec != EC_MISMATCH) {
                                has_failure = true;
                            }
                        }
                    }
                } else {
                    if (!submit_del_req_func_) {
                        KVCM_LOG_WARN("CleanupLocationsByPredicate: reclaimer submit callback is unavailable");
                        return EC_ERROR;
                    }
                    submit_del_req_func_(keys, delete_location_ids, expected_location_values, true);
                }
            }
        }
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);

    return has_failure ? EC_PARTIAL_OK : EC_OK;
}

ErrorCode MetaSearcher::CleanupLocationsByHost(RequestContext *request_context,
                                               const std::string &host_suffix,
                                               DataStorageType storage_type,
                                               size_t scan_batch_size,
                                               std::function<bool()> should_abort,
                                               AcquireMetadataWriteLeaseFunc acquire_cleanup_lease) {
    if (host_suffix.empty()) {
        return EC_BADARGS;
    }
    if (scan_batch_size == 0) {
        scan_batch_size = 1000;
    }

    bool has_failure = false;
    std::string cursor = SCAN_BASE_CURSOR;
    do {
        if (should_abort && should_abort()) {
            KVCM_LOG_INFO("CleanupLocationsByHost: aborted by caller (host_suffix=%s)", host_suffix.c_str());
            return EC_OK;
        }
        std::string next_cursor;
        KeyVector keys;
        if (auto ec = meta_indexer_->Scan(request_context, cursor, scan_batch_size, next_cursor, keys); ec != EC_OK) {
            KVCM_LOG_WARN("CleanupLocationsByHost: scan failed, ec %d", ec);
            has_failure = true;
            break;
        }
        if (!keys.empty()) {
            CacheLocationMapVector location_maps;
            auto get_result = meta_indexer_->GetLocations(request_context, keys, location_maps);
            if (get_result.ec == EC_OK || get_result.ec == EC_PARTIAL_OK) {
                if (get_result.ec == EC_PARTIAL_OK) {
                    has_failure = true;
                }
                LocationIdsPerKey delete_loc_ids(keys.size());
                std::vector<std::vector<std::string>> expected_location_values(keys.size());
                bool has_any_location = false;
                for (size_t i = 0; i < keys.size(); ++i) {
                    if (get_result.ec == EC_PARTIAL_OK && get_result.error_codes[i] != EC_OK) {
                        continue;
                    }
                    for (const auto &kv : location_maps[i]) {
                        const std::string &loc_id = kv.first;
                        if (!kv.second) {
                            has_failure = true;
                            continue;
                        }
                        const CacheLocation &loc = *kv.second;
                        if (loc.type() == storage_type && loc_id.size() >= host_suffix.size() &&
                            loc_id.compare(loc_id.size() - host_suffix.size(), host_suffix.size(), host_suffix) == 0) {
                            delete_loc_ids[i].push_back(loc_id);
                            expected_location_values[i].push_back(loc.ToJsonString());
                            has_any_location = true;
                        }
                    }
                }
                if (has_any_location) {
                    MetadataWriteLease cleanup_lease;
                    if (acquire_cleanup_lease) {
                        auto [lease_ec, lease] = acquire_cleanup_lease();
                        if (lease_ec == EC_MISMATCH) {
                            KVCM_LOG_INFO("CleanupLocationsByHost: lifecycle changed before delete "
                                          "(host_suffix=%s)",
                                          host_suffix.c_str());
                            return EC_OK;
                        }
                        if (lease_ec != EC_OK) {
                            KVCM_LOG_WARN("CleanupLocationsByHost: failed to acquire cleanup lease, ec %d", lease_ec);
                            return lease_ec;
                        }
                        cleanup_lease = std::move(lease);
                    } else if (should_abort && should_abort()) {
                        KVCM_LOG_INFO("CleanupLocationsByHost: aborted before delete (host_suffix=%s)",
                                      host_suffix.c_str());
                        return EC_OK;
                    }
                    KeyVector delete_keys;
                    LocationIdsPerKey compact_delete_loc_ids;
                    std::vector<std::vector<std::string>> compact_expected_location_values;
                    delete_keys.reserve(keys.size());
                    compact_delete_loc_ids.reserve(keys.size());
                    compact_expected_location_values.reserve(keys.size());
                    for (size_t i = 0; i < keys.size(); ++i) {
                        if (delete_loc_ids[i].empty()) {
                            continue;
                        }
                        delete_keys.push_back(keys[i]);
                        compact_delete_loc_ids.push_back(std::move(delete_loc_ids[i]));
                        compact_expected_location_values.push_back(std::move(expected_location_values[i]));
                    }
                    std::vector<std::vector<ErrorCode>> per_location_ec;
                    auto del_ec = BatchDeleteLocations(request_context,
                                                       delete_keys,
                                                       compact_delete_loc_ids,
                                                       per_location_ec,
                                                       compact_expected_location_values);
                    if (del_ec != EC_OK) {
                        KVCM_LOG_WARN("CleanupLocationsByHost: BatchDeleteLocations failed, ec %d", del_ec);
                        has_failure = true;
                    } else {
                        for (size_t i = 0; i < per_location_ec.size(); ++i) {
                            for (const auto &loc_ec : per_location_ec[i]) {
                                if (loc_ec != EC_OK && loc_ec != EC_NOENT && loc_ec != EC_MISMATCH) {
                                    KVCM_LOG_WARN(
                                        "CleanupLocationsByHost: delete location failed for key index %zu, ec %d",
                                        i,
                                        loc_ec);
                                    has_failure = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            } else {
                KVCM_LOG_WARN("CleanupLocationsByHost: GetLocations failed, ec %d", get_result.ec);
                has_failure = true;
            }
        }
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);

    return has_failure ? EC_PARTIAL_OK : EC_OK;
}

} // namespace kv_cache_manager
