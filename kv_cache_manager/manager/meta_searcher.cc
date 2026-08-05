#include "kv_cache_manager/manager/meta_searcher.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
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

std::uint64_t GetLocationSpecsSize(const std::vector<LocationSpec> &specs) {
    std::uint64_t total_size = 0;
    for (const auto &loc_spec : specs) {
        if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
            std::uint64_t spec_size = 0;
            ds_uri.GetParamAs<std::uint64_t>("size", spec_size);
            total_size += spec_size;
        }
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

ErrorCode ValidateConsistentSnapshotVersion(const std::vector<LocationSpec> &specs) {
    if (specs.empty()) {
        return EC_BADARGS;
    }
    bool has_snapshot_version = false;
    std::string snapshot_version;
    bool has_unversioned_spec = false;
    std::unordered_set<std::string> spec_names;
    for (const auto &spec : specs) {
        const DataStorageUri uri(spec.uri());
        if (spec.name().empty() || !spec_names.insert(spec.name()).second || !uri.Valid()) {
            return EC_BADARGS;
        }
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
    return EC_OK;
}

void MergeLocationSpecsByName(std::vector<LocationSpec> &merged_specs, const std::vector<LocationSpec> &new_specs) {
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

    for (const auto &spec : new_specs) {
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

using RequestedSpecNameSet = std::unordered_set<std::string>;

bool MatchesRequestedSpec(const CacheLocation &loc, const RequestedSpecNameSet &requested_spec_names) {
    if (requested_spec_names.empty()) {
        return true;
    }
    return std::any_of(loc.location_specs().begin(), loc.location_specs().end(), [&](const LocationSpec &spec) {
        return requested_spec_names.count(spec.name()) > 0;
    });
}

std::string ExtractPeerAddrFromLocation(const CacheLocation &loc) {
    if (loc.location_specs().empty()) {
        return {};
    }
    StandardUri uri(loc.location_specs().front().uri());
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

bool IsMediumMatched(const StandardUri &uri, const std::unordered_set<std::string> &medium_set) {
    if (medium_set.empty()) {
        return true;
    }
    std::string medium = uri.GetPath();
    if (!medium.empty() && medium[0] == '/') {
        medium = medium.substr(1);
    }
    return medium_set.find(medium) != medium_set.end();
}

struct HostSpecNames {
    std::string host;
    std::vector<std::string> spec_names;
};

using HostSpecNamesVector = std::vector<HostSpecNames>;
using KeyToHostSpecNames = std::vector<HostSpecNamesVector>;

template <typename Visitor>
void VisitHostSpecsForOneKey(const CacheLocationVector &locations,
                             const CheckLocDataExistFunc &check_loc_data_exist,
                             const MetaSearcher::CheckHostCacheLocationFunc *request_check_location,
                             const std::unordered_set<std::string> &medium_set,
                             bool visit_spec_names,
                             Visitor &&visitor) {
    for (const auto &loc : locations) {
        // Host cache state is a read-availability API. WRITING, DELETING and
        // other transitional locations must not contribute a hit merely
        // because their URI is already present in metadata.
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

        bool has_reporter_identity = location_info.has_reporter_identity;
        if (!has_reporter_identity && IsEventReportStorageType(loc->type())) {
            has_reporter_identity = SnapshotUriUtils::ParseEventReportLocationId(
                loc->id(), location_info.reporter_medium, location_info.reporter_host);
        }
        if (has_reporter_identity) {
            if ((!medium_set.empty() && medium_set.count(location_info.reporter_medium) == 0) ||
                location_info.reporter_host.empty()) {
                continue;
            }
            // The request-specific EventReport checker validates every URI
            // while applying the reporter generation/liveness fence. Reuse
            // that work instead of reparsing each URI during host projection.
            const bool specs_already_validated =
                request_check_location != nullptr && location_info.has_reporter_identity;
            if (!visit_spec_names) {
                if (specs_already_validated) {
                    visitor(location_info.reporter_host, std::string{});
                    continue;
                }
                for (const auto &spec : loc->location_specs()) {
                    const StandardUri uri(spec.uri());
                    if (uri.Valid()) {
                        visitor(location_info.reporter_host, std::string{});
                        break;
                    }
                }
                continue;
            }
            for (const auto &spec : loc->location_specs()) {
                if (!specs_already_validated) {
                    const StandardUri uri(spec.uri());
                    if (!uri.Valid()) {
                        continue;
                    }
                }
                visitor(location_info.reporter_host, spec.name());
            }
            continue;
        }

        for (const auto &spec : loc->location_specs()) {
            const StandardUri uri(spec.uri());
            if (!uri.Valid() || !IsMediumMatched(uri, medium_set)) {
                continue;
            }
            std::string host = uri.GetHostPort();
            if (host.empty()) {
                continue;
            }
            visitor(host, spec.name());
        }
    }
}

void BuildHostsForOneKey(const CacheLocationVector &locations,
                         const CheckLocDataExistFunc &check_loc_data_exist,
                         const MetaSearcher::CheckHostCacheLocationFunc *request_check_location,
                         const std::unordered_set<std::string> &medium_set,
                         std::vector<std::string> &hosts) {
    VisitHostSpecsForOneKey(locations,
                            check_loc_data_exist,
                            request_check_location,
                            medium_set,
                            false,
                            [&hosts](const std::string &host, const std::string &) { hosts.push_back(host); });
    std::sort(hosts.begin(), hosts.end());
    hosts.erase(std::unique(hosts.begin(), hosts.end()), hosts.end());
}

void BuildCandidatePresenceForOneKey(const CacheLocationVector &locations,
                                     const CheckLocDataExistFunc &check_loc_data_exist,
                                     const MetaSearcher::CheckHostCacheLocationFunc *request_check_location,
                                     const std::unordered_set<std::string> &medium_set,
                                     const std::unordered_map<std::string, std::size_t> &candidate_indices,
                                     std::uint64_t *presence_words) {
    VisitHostSpecsForOneKey(locations,
                            check_loc_data_exist,
                            request_check_location,
                            medium_set,
                            false,
                            [&candidate_indices, presence_words](const std::string &host, const std::string &) {
                                const auto it = candidate_indices.find(host);
                                if (it != candidate_indices.end()) {
                                    presence_words[it->second / 64] |= std::uint64_t{1} << (it->second % 64);
                                }
                            });
}

void BuildHostSpecNamesForOneKey(const CacheLocationVector &locations,
                                 const CheckLocDataExistFunc &check_loc_data_exist,
                                 const MetaSearcher::CheckHostCacheLocationFunc *request_check_location,
                                 const std::unordered_set<std::string> &medium_set,
                                 HostSpecNamesVector &host_specs) {
    VisitHostSpecsForOneKey(locations,
                            check_loc_data_exist,
                            request_check_location,
                            medium_set,
                            true,
                            [&host_specs](const std::string &host, const std::string &spec_name) {
                                auto host_it = std::find_if(host_specs.begin(),
                                                            host_specs.end(),
                                                            [&host](const auto &entry) { return entry.host == host; });
                                if (host_it == host_specs.end()) {
                                    host_specs.push_back(HostSpecNames{host, {spec_name}});
                                    return;
                                }
                                if (std::find(host_it->spec_names.begin(), host_it->spec_names.end(), spec_name) ==
                                    host_it->spec_names.end()) {
                                    host_it->spec_names.push_back(spec_name);
                                }
                            });
    for (auto &host_spec : host_specs) {
        std::sort(host_spec.spec_names.begin(), host_spec.spec_names.end());
    }
    std::sort(
        host_specs.begin(), host_specs.end(), [](const auto &lhs, const auto &rhs) { return lhs.host < rhs.host; });
}

const std::vector<std::string> *FindHostSpecNames(const HostSpecNamesVector &host_specs, const std::string &host) {
    const auto it =
        std::lower_bound(host_specs.begin(), host_specs.end(), host, [](const auto &entry, const auto &name) {
            return entry.host < name;
        });
    return it != host_specs.end() && it->host == host ? &it->spec_names : nullptr;
}

bool IsFullLocationSpecGroup(const LocationSpecGroup &group) {
    const auto &name = group.name();
    return name.rfind("full", 0) == 0 || name.rfind("FULL", 0) == 0;
}

bool HasAllLocationSpecGroups(const std::vector<std::string> &spec_names,
                              const std::vector<const LocationSpecGroup *> &groups) {
    for (const auto *group : groups) {
        for (const auto &spec_name : group->spec_names()) {
            if (!std::binary_search(spec_names.begin(), spec_names.end(), spec_name)) {
                return false;
            }
        }
    }
    return true;
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

    const RequestedSpecNameSet requested_spec_name_set(requested_spec_names.begin(), requested_spec_names.end());
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
                std::vector<std::string> vineyard_addrs;

                for (const auto &[id, loc] : vmap) {
                    if (loc->type() != target_type)
                        continue;
                    if (!MatchesRequestedSpec(*loc, requested_spec_name_set))
                        continue;
                    std::string addr = ExtractPeerAddrFromLocation(*loc);
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
                const auto &vmap = valid_maps[i];
                CacheLocationMap filtered;
                for (const auto &[id, loc] : vmap) {
                    if (loc->type() == target_type && MatchesRequestedSpec(*loc, requested_spec_name_set)) {
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
                                          const CheckHostCacheLocationFunc *request_check_location) const {
    SPAN_TRACER(request_context);
    out_matches.clear();
    if (keys.empty()) {
        return EC_OK;
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
    }
    if (valid_key_count == 0) {
        return EC_OK;
    }

    const std::unordered_set<std::string> medium_set(medium_filter.begin(), medium_filter.end());
    const auto &check_loc_data_exist = check_loc_data_exist_func_;
    std::vector<std::string> candidate_hosts;
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherHostProjection);
    BuildHostsForOneKey(
        location_values.front(), check_loc_data_exist, request_check_location, medium_set, candidate_hosts);
    if (candidate_hosts.empty()) {
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostProjection);
        return EC_OK;
    }

    std::unordered_map<std::string, std::size_t> candidate_indices;
    candidate_indices.reserve(candidate_hosts.size());
    for (std::size_t i = 0; i < candidate_hosts.size(); ++i) {
        candidate_indices.emplace(candidate_hosts[i], i);
    }
    const std::size_t presence_word_count = (candidate_hosts.size() + 63) / 64;
    if (valid_key_count > std::numeric_limits<std::size_t>::max() / presence_word_count) {
        KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostProjection);
        request_context->error_tracer()->AddErrorMsg("host presence matrix size overflow");
        return EC_ERROR;
    }
    std::vector<std::uint64_t> candidate_presence(valid_key_count * presence_word_count, 0);
    const bool projection_ok = meta_indexer_->ParallelForQuery(
        valid_key_count - 1,
        [&location_values,
         &check_loc_data_exist,
         request_check_location,
         &medium_set,
         &candidate_indices,
         &candidate_presence,
         presence_word_count](std::size_t begin, std::size_t end) {
            for (std::size_t offset = begin; offset < end; ++offset) {
                const std::size_t key_index = offset + 1;
                BuildCandidatePresenceForOneKey(location_values[key_index],
                                                check_loc_data_exist,
                                                request_check_location,
                                                medium_set,
                                                candidate_indices,
                                                candidate_presence.data() + key_index * presence_word_count);
            }
        });
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostProjection);
    if (!projection_ok) {
        request_context->error_tracer()->AddErrorMsg("parallel host cache projection failed");
        return EC_ERROR;
    }

    std::vector<int64_t> prefix_lengths(candidate_hosts.size(), 0);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherHostPrefixReduce);
    const bool reduce_ok = meta_indexer_->ParallelForQuery(
        candidate_hosts.size(),
        [&candidate_presence, presence_word_count, valid_key_count, &prefix_lengths, use_eagle_pop](std::size_t begin,
                                                                                                    std::size_t end) {
            for (std::size_t host_index = begin; host_index < end; ++host_index) {
                int64_t prefix_len = 1;
                const std::size_t word_index = host_index / 64;
                const std::uint64_t bit = std::uint64_t{1} << (host_index % 64);
                for (std::size_t i = 1; i < valid_key_count; ++i) {
                    if ((candidate_presence[i * presence_word_count + word_index] & bit) == 0) {
                        break;
                    }
                    ++prefix_len;
                }
                if (use_eagle_pop) {
                    prefix_len = std::max<int64_t>(prefix_len - 1, 0);
                }
                prefix_lengths[host_index] = prefix_len;
            }
        });
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostPrefixReduce);
    if (!reduce_ok) {
        request_context->error_tracer()->AddErrorMsg("parallel host prefix reduction failed");
        return EC_ERROR;
    }
    out_matches.reserve(candidate_hosts.size());
    for (std::size_t i = 0; i < candidate_hosts.size(); ++i) {
        if (prefix_lengths[i] > 0) {
            out_matches.push_back(HostCacheMatch{std::move(candidate_hosts[i]), prefix_lengths[i]});
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
                                                   const CheckHostCacheLocationFunc *request_check_location) const {
    SPAN_TRACER(request_context);
    out_matches.clear();
    if (keys.empty()) {
        return EC_OK;
    }

    std::vector<const LocationSpecGroup *> full_groups;
    std::vector<const LocationSpecGroup *> mamba_state_groups;
    for (const auto &group : location_spec_groups) {
        if (IsFullLocationSpecGroup(group)) {
            full_groups.push_back(&group);
        } else {
            mamba_state_groups.push_back(&group);
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
    }
    if (valid_key_count == 0) {
        return EC_OK;
    }

    KeyToHostSpecNames key_to_host_spec_names(valid_key_count);
    const std::unordered_set<std::string> medium_set(medium_filter.begin(), medium_filter.end());
    const auto &check_loc_data_exist = check_loc_data_exist_func_;
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherHostProjection);
    const bool projection_ok = meta_indexer_->ParallelForQuery(
        valid_key_count,
        [&location_values, &check_loc_data_exist, request_check_location, &medium_set, &key_to_host_spec_names](
            std::size_t begin, std::size_t end) {
            for (std::size_t i = begin; i < end; ++i) {
                BuildHostSpecNamesForOneKey(location_values[i],
                                            check_loc_data_exist,
                                            request_check_location,
                                            medium_set,
                                            key_to_host_spec_names[i]);
            }
        });
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherHostProjection);
    if (!projection_ok) {
        request_context->error_tracer()->AddErrorMsg("parallel mamba host cache projection failed");
        return EC_ERROR;
    }

    std::vector<std::string> candidate_hosts;
    candidate_hosts.reserve(key_to_host_spec_names.front().size());
    for (const auto &host_spec_names : key_to_host_spec_names.front()) {
        candidate_hosts.push_back(host_spec_names.host);
    }
    std::vector<int64_t> prefix_lengths(candidate_hosts.size(), 0);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherHostPrefixReduce);
    const bool reduce_ok = meta_indexer_->ParallelForQuery(
        candidate_hosts.size(),
        [&candidate_hosts, &key_to_host_spec_names, &full_groups, &mamba_state_groups, &prefix_lengths, use_eagle_pop](
            std::size_t begin, std::size_t end) {
            for (std::size_t host_index = begin; host_index < end; ++host_index) {
                const auto &host = candidate_hosts[host_index];
                size_t full_prefix_len = 0;
                for (; full_prefix_len < key_to_host_spec_names.size(); ++full_prefix_len) {
                    const auto *spec_names = FindHostSpecNames(key_to_host_spec_names[full_prefix_len], host);
                    if (!spec_names || !HasAllLocationSpecGroups(*spec_names, full_groups)) {
                        break;
                    }
                }
                if (use_eagle_pop && full_prefix_len > 0) {
                    --full_prefix_len;
                }
                if (full_prefix_len == 0) {
                    continue;
                }

                // Mamba requires the last usable block in the full-prefix
                // range to also contain every state spec group.
                for (size_t offset = full_prefix_len; offset > 0; --offset) {
                    const size_t index = offset - 1;
                    const auto *spec_names = FindHostSpecNames(key_to_host_spec_names[index], host);
                    if (spec_names && HasAllLocationSpecGroups(*spec_names, mamba_state_groups)) {
                        prefix_lengths[host_index] = static_cast<int64_t>(index + 1);
                        break;
                    }
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
            out_matches.push_back(HostCacheMatch{std::move(candidate_hosts[i]), prefix_lengths[i]});
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
        for (const auto &task : tasks_per_key[key_index]) {
            if (task.location_id.empty() || !seen_location_ids.insert(task.location_id).second ||
                ValidateConsistentSnapshotVersion(task.specs) != EC_OK) {
                std::fill(out_per_key_ec.begin(), out_per_key_ec.end(), EC_BADARGS);
                return EC_BADARGS;
            }
            location_ids.push_back(task.location_id);
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
            usage.new_size = GetLocationSpecsSize(task.specs);
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

ErrorCode MetaSearcher::BatchMergeLocationSpecs(RequestContext *request_context,
                                                const KeyVector &keys,
                                                const std::vector<std::vector<MergeLocationSpecsTask>> &tasks_per_key,
                                                std::vector<ErrorCode> &out_per_key_ec,
                                                AcquireMetadataWriteLeaseFunc acquire_write_lease) {
    if (keys.size() != tasks_per_key.size()) {
        return EC_BADARGS;
    }
    out_per_key_ec.assign(keys.size(), ErrorCode::EC_OK);
    if (keys.empty()) {
        return EC_OK;
    }

    std::unordered_set<int64_t> seen_keys;
    for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
        if (!seen_keys.insert(keys[key_index]).second) {
            std::fill(out_per_key_ec.begin(), out_per_key_ec.end(), EC_BADARGS);
            return EC_BADARGS;
        }
        std::unordered_set<std::string> seen_location_ids;
        for (const auto &task : tasks_per_key[key_index]) {
            if (task.location_id.empty() || !seen_location_ids.insert(task.location_id).second ||
                ValidateConsistentSnapshotVersion(task.specs) != EC_OK) {
                std::fill(out_per_key_ec.begin(), out_per_key_ec.end(), EC_BADARGS);
                return EC_BADARGS;
            }
        }
    }

    std::vector<std::vector<std::pair<DataStorageType, std::uint64_t>>> created_locs_sz(keys.size());
    const int64_t batch_create_time = TimestampUtil::GetCurrentTimeUs();
    // Keep indices into the caller-owned task vectors. Copying each task here
    // duplicated every location id, spec name and URI before the second RMW
    // phase, which is especially expensive for large ReportEvent batches.
    std::vector<std::vector<size_t>> merge_task_indices_per_key(keys.size());
    bool create_write_lease_attempted = false;
    ErrorCode create_write_lease_ec = EC_OK;
    MetadataWriteLease create_write_lease;

    auto create_modifier = [&tasks_per_key,
                            &merge_task_indices_per_key,
                            &keys,
                            &created_locs_sz,
                            &acquire_write_lease,
                            &create_write_lease_attempted,
                            &create_write_lease_ec,
                            &create_write_lease,
                            batch_create_time](const LocationIdVector &existing_ids,
                                               ErrorCode get_ec,
                                               size_t index,
                                               PropertyMap & /*upsert_property_map*/,
                                               CacheLocationMap &out_new_locations) -> ModifierResult {
        if (acquire_write_lease && !create_write_lease_attempted) {
            std::tie(create_write_lease_ec, create_write_lease) = acquire_write_lease();
            create_write_lease_attempted = true;
        }
        if (create_write_lease_ec != EC_OK) {
            return {ModifierAction::MA_FAIL, create_write_lease_ec};
        }
        if (get_ec != ErrorCode::EC_OK && get_ec != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("load location ids failed, key[%lu](%lu) return %d", index, keys[index], get_ec);
            return {ModifierAction::MA_FAIL, get_ec};
        }

        bool created = false;
        for (size_t task_index = 0; task_index < tasks_per_key[index].size(); ++task_index) {
            const auto &entry = tasks_per_key[index][task_index];
            if (get_ec == ErrorCode::EC_OK &&
                std::find(existing_ids.begin(), existing_ids.end(), entry.location_id) != existing_ids.end()) {
                merge_task_indices_per_key[index].push_back(task_index);
                continue;
            }

            CacheLocation loc;
            loc.set_id(entry.location_id);
            loc.set_type(entry.type);
            loc.set_status(entry.status);
            loc.set_spec_size(entry.specs.size());
            loc.set_create_time(batch_create_time);
            for (const auto &ls : entry.specs) {
                loc.push_location_spec(LocationSpec(ls.name(), ls.uri()));
            }
            out_new_locations[entry.location_id] = std::make_shared<const CacheLocation>(std::move(loc));
            created_locs_sz[index].emplace_back(entry.type, GetLocationSpecsSize(entry.specs));
            created = true;
        }
        if (!created) {
            return {ModifierAction::MA_SKIP, ErrorCode::EC_OK};
        }
        return {ModifierAction::MA_OK, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteBlock);
    auto result = meta_indexer_->ReadModifyWriteBlock(request_context, keys, create_modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteBlock);
    ErrorCode final_ec = result.ec;

    if (result.error_codes.size() != keys.size()) {
        KVCM_LOG_ERROR("BatchMergeLocationSpecs create result size mismatch, keys[%zu], results[%zu]",
                       keys.size(),
                       result.error_codes.size());
        std::fill(out_per_key_ec.begin(), out_per_key_ec.end(), ErrorCode::EC_MISMATCH);
        return ErrorCode::EC_MISMATCH;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        ErrorCode key_ec = result.error_codes[i];
        out_per_key_ec[i] = key_ec;
        if (key_ec == ErrorCode::EC_OK) {
            for (const auto &[type, size] : created_locs_sz[i]) {
                meta_indexer_->AddStorageUsageByType(type, size);
            }
        }
    }

    if (result.ec != ErrorCode::EC_OK) {
        LogErrorCodes("meta_indexer_->ReadModifyWriteBlock", result.error_codes, keys);
    }

    KeyVector merge_keys;
    std::vector<size_t> merge_key_indices;
    LocationIdsPerKey merge_location_ids;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (out_per_key_ec[i] != ErrorCode::EC_OK || merge_task_indices_per_key[i].empty()) {
            continue;
        }
        merge_keys.push_back(keys[i]);
        merge_key_indices.push_back(i);
        auto &ids = merge_location_ids.emplace_back();
        ids.reserve(merge_task_indices_per_key[i].size());
        for (const size_t task_index : merge_task_indices_per_key[i]) {
            ids.push_back(tasks_per_key[i][task_index].location_id);
        }
    }

    if (merge_keys.empty()) {
        return final_ec;
    }

    std::vector<std::vector<StorageUsageChange>> merge_usage_changes(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        merge_usage_changes[i].resize(merge_task_indices_per_key[i].size());
    }
    create_write_lease.reset();
    bool merge_write_lease_attempted = false;
    ErrorCode merge_write_lease_ec = EC_OK;
    MetadataWriteLease merge_write_lease;
    auto merge_modifier = [&keys,
                           &tasks_per_key,
                           &merge_task_indices_per_key,
                           &merge_key_indices,
                           &merge_usage_changes,
                           &acquire_write_lease,
                           &merge_write_lease_attempted,
                           &merge_write_lease_ec,
                           &merge_write_lease,
                           batch_create_time](const std::vector<ErrorCode> &get_ecs,
                                              const LocationIdVector &loc_ids,
                                              size_t key_index,
                                              CacheLocationVector &locs,
                                              PropertyMap &upsert_property_map) -> LocationModifierResult {
        (void)upsert_property_map;
        if (acquire_write_lease && !merge_write_lease_attempted) {
            std::tie(merge_write_lease_ec, merge_write_lease) = acquire_write_lease();
            merge_write_lease_attempted = true;
        }
        if (merge_write_lease_ec != EC_OK) {
            return {ModifierAction::MA_FAIL, std::vector<ErrorCode>(loc_ids.size(), merge_write_lease_ec)};
        }
        const size_t original_key_index = merge_key_indices[key_index];
        const auto &task_indices = merge_task_indices_per_key[original_key_index];
        std::vector<ErrorCode> modifier_ecs(loc_ids.size(), ErrorCode::EC_OK);
        bool updated = false;
        for (size_t loc_index = 0; loc_index < loc_ids.size(); ++loc_index) {
            if (loc_index >= task_indices.size() || loc_index >= get_ecs.size() || loc_index >= locs.size()) {
                modifier_ecs[loc_index] = ErrorCode::EC_ERROR;
                continue;
            }
            const ErrorCode ec = get_ecs[loc_index];
            const auto &task = tasks_per_key[original_key_index][task_indices[loc_index]];
            if (ec != ErrorCode::EC_OK && ec != ErrorCode::EC_NOENT) {
                modifier_ecs[loc_index] = ec;
                KVCM_LOG_WARN("load location failed, key[%lu](%lu), location_id: %s, return %d",
                              original_key_index,
                              keys[original_key_index],
                              task.location_id.c_str(),
                              ec);
                continue;
            }

            auto &usage = merge_usage_changes[original_key_index][loc_index];
            std::shared_ptr<CacheLocation> new_loc;
            if (ec == ErrorCode::EC_OK && locs[loc_index]) {
                if (locs[loc_index]->type() != task.type) {
                    modifier_ecs[loc_index] = ErrorCode::EC_BADARGS;
                    continue;
                }
                usage.old_size = GetLocationSpecsSize(locs[loc_index]->location_specs());
                usage.has_old = true;
                new_loc = std::make_shared<CacheLocation>(*locs[loc_index]);
                MergeLocationSpecsByName(new_loc->mutable_location_specs(), task.specs);
            } else {
                new_loc = std::make_shared<CacheLocation>();
                new_loc->set_id(task.location_id);
                std::vector<LocationSpec> specs;
                specs.reserve(task.specs.size());
                for (const auto &spec : task.specs) {
                    specs.emplace_back(spec.name(), spec.uri());
                }
                new_loc->set_location_specs(std::move(specs));
            }
            new_loc->set_type(task.type);
            new_loc->set_status(task.status);
            new_loc->set_create_time(batch_create_time);
            new_loc->set_spec_size(new_loc->location_specs().size());
            usage.new_size = GetLocationSpecsSize(new_loc->location_specs());
            locs[loc_index] = std::move(new_loc);
            updated = true;
        }
        if (!updated) {
            return {ModifierAction::MA_SKIP, std::move(modifier_ecs)};
        }
        return {ModifierAction::MA_OK, std::move(modifier_ecs)};
    };

    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto merge_result =
        meta_indexer_->ReadModifyWriteLocation(request_context, merge_keys, merge_location_ids, merge_modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);

    bool malformed_merge_result = merge_result.per_location_error_codes.size() != merge_key_indices.size();
    if (malformed_merge_result) {
        KVCM_LOG_ERROR("BatchMergeLocationSpecs merge result size mismatch, keys[%zu], results[%zu]",
                       merge_key_indices.size(),
                       merge_result.per_location_error_codes.size());
    }
    for (size_t i = 0; i < merge_key_indices.size(); ++i) {
        const size_t original_key_index = merge_key_indices[i];
        ErrorCode key_ec = ErrorCode::EC_OK;
        const auto &task_indices = merge_task_indices_per_key[original_key_index];
        const size_t expected_location_count = task_indices.size();
        if (i >= merge_result.per_location_error_codes.size() ||
            merge_result.per_location_error_codes[i].size() != expected_location_count) {
            key_ec = ErrorCode::EC_MISMATCH;
            malformed_merge_result = true;
        } else {
            for (size_t loc_index = 0; loc_index < expected_location_count; ++loc_index) {
                const auto loc_ec = merge_result.per_location_error_codes[i][loc_index];
                if (loc_ec == ErrorCode::EC_OK) {
                    const auto &usage = merge_usage_changes[original_key_index][loc_index];
                    const auto &task = tasks_per_key[original_key_index][task_indices[loc_index]];
                    if (usage.has_old) {
                        ApplyStorageUsageChange(meta_indexer_.get(), task.type, usage.old_size, usage.new_size);
                    } else {
                        meta_indexer_->AddStorageUsageByType(task.type, usage.new_size);
                    }
                    continue;
                }
                if (key_ec == ErrorCode::EC_OK) {
                    key_ec = loc_ec;
                }
            }
        }
        if (key_ec != ErrorCode::EC_OK) {
            out_per_key_ec[original_key_index] = key_ec;
        }
    }

    if (merge_result.ec != ErrorCode::EC_OK) {
        KVCM_LOG_WARN("meta_indexer_->ReadModifyWriteLocation failed, ec: %d", merge_result.ec);
        if (final_ec == ErrorCode::EC_OK) {
            final_ec = merge_result.ec;
        } else if (merge_result.ec != final_ec) {
            final_ec = ErrorCode::EC_PARTIAL_OK;
        }
    }
    if (malformed_merge_result) {
        if (final_ec == ErrorCode::EC_OK) {
            final_ec = ErrorCode::EC_MISMATCH;
        } else if (final_ec != ErrorCode::EC_MISMATCH) {
            final_ec = ErrorCode::EC_PARTIAL_OK;
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
        std::unordered_set<std::string> seen_location_ids;
        for (const auto &task : tasks_per_key[key_index]) {
            if (task.location_id.empty() || !seen_location_ids.insert(task.location_id).second) {
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
            flat_location_ids.push_back({tasks_per_key[i][task_index].location_id});
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

        flat_deleted_specs_size[key_index] = std::make_pair(locs[0]->type(), GetLocationSpecsSize(deleted_specs));
        if (kept_specs.empty()) {
            return {ModifierAction::MA_DELETE, std::move(modifier_ecs)};
        }

        auto new_loc = std::make_shared<CacheLocation>(*locs[0]);
        new_loc->set_location_specs(std::move(kept_specs));
        new_loc->set_spec_size(new_loc->location_specs().size());
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
    auto result = meta_indexer_->ReadModifyWriteLocation(request_context, keys, location_ids_per_key, modifier);
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
