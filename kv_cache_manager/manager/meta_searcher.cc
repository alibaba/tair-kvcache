#include "kv_cache_manager/manager/meta_searcher.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
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
    bool has_snapshot_version = false;
    std::string snapshot_version;
    bool has_unversioned_spec = false;
    for (const auto &spec : specs) {
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
        if (has_unversioned_spec || version_param_count != 1 ||
            !SnapshotUriUtils::ParseSnapshotUriInfo(spec.uri(), info)) {
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

ErrorCode MergeLocationSpecsByName(const std::vector<LocationSpec> &old_specs,
                                   const std::vector<LocationSpec> &new_specs,
                                   std::vector<LocationSpec> &out_specs) {
    const ErrorCode parse_ec = ValidateConsistentSnapshotVersion(new_specs);
    if (parse_ec != EC_OK) {
        return parse_ec;
    }

    std::map<std::string, LocationSpec> merged_specs;
    for (const auto &spec : old_specs) {
        // Snapshot generations are reconciliation/cleanup tags, not a
        // visibility fence. After a KVCM restart, a new delta may use a fresh
        // generation while untouched specs still carry an older one. Preserve
        // those specs and overwrite only names present in this delta.
        merged_specs[spec.name()] = spec;
    }
    for (const auto &spec : new_specs) {
        merged_specs[spec.name()] = spec;
    }

    out_specs.clear();
    out_specs.reserve(merged_specs.size());
    for (auto &[name, spec] : merged_specs) {
        out_specs.push_back(std::move(spec));
    }
    return EC_OK;
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

using HostToSpecNames = std::map<std::string, std::set<std::string>>;
using KeyToHostSpecNames = std::vector<HostToSpecNames>; // key -> host -> spec names

void BuildHostSpecNamesForOneKey(const CacheLocationMap &location_map,
                                 CheckLocDataExistFunc check_loc_data_exist,
                                 const std::unordered_set<std::string> &medium_set,
                                 HostToSpecNames &host_specs) {
    for (const auto &kv : location_map) {
        const auto &loc = kv.second;
        if (!loc || loc->location_specs().empty()) {
            continue;
        }
        if (check_loc_data_exist && !check_loc_data_exist(*loc)) {
            continue;
        }
        std::string event_medium;
        std::string reporter_host;
        const bool has_reporter_identity =
            IsEventReportStorageType(loc->type()) &&
            SnapshotUriUtils::ParseEventReportLocationId(kv.first, event_medium, reporter_host);
        for (const auto &spec : loc->location_specs()) {
            StandardUri uri(spec.uri());
            const bool medium_matches = has_reporter_identity ? medium_set.empty() || medium_set.count(event_medium) > 0
                                                              : IsMediumMatched(uri, medium_set);
            if (!uri.Valid() || !medium_matches) {
                continue;
            }
            std::string host = has_reporter_identity ? reporter_host : uri.GetHostPort();
            if (host.empty()) {
                continue;
            }
            host_specs[std::move(host)].insert(spec.name());
        }
    }
}

void BuildVineyardHostSpecNamesForOneKey(const CacheLocationMap &location_map,
                                         CheckLocDataExistFunc check_loc_data_exist,
                                         const std::unordered_set<std::string> &medium_set,
                                         HostToSpecNames &host_specs) {
    for (const auto &[location_id, loc] : location_map) {
        if (!loc || loc->type() != DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT_L2 ||
            loc->location_specs().empty()) {
            continue;
        }
        if (check_loc_data_exist && !check_loc_data_exist(*loc)) {
            continue;
        }
        std::string event_medium;
        std::string reporter_host;
        if (!SnapshotUriUtils::ParseEventReportLocationId(location_id, event_medium, reporter_host) ||
            reporter_host.empty() || (!medium_set.empty() && medium_set.count(event_medium) == 0)) {
            continue;
        }
        for (const auto &spec : loc->location_specs()) {
            if (StandardUri(spec.uri()).Valid()) {
                host_specs[reporter_host].insert(spec.name());
            }
        }
    }
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

bool IsFullLocationSpecGroup(const LocationSpecGroup &group) {
    const auto &name = group.name();
    return name.rfind("full", 0) == 0 || name.rfind("FULL", 0) == 0;
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
                                                      const std::vector<std::string> &requested_spec_names) const {
    assert(policy != nullptr);
    SPAN_TRACER(request_context);
    out_locations.clear();
    out_locations.resize(keys.size());
    const RequestedSpecNameSet requested_spec_name_set(requested_spec_names.begin(), requested_spec_names.end());
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    CacheLocationMapVector location_maps;
    auto result = meta_indexer_->GetLocations(request_context, keys, location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    KeyVector prune_keys;
    std::vector<std::vector<std::string>> prune_loc_ids_vec;
    std::vector<CacheLocationMap> valid_maps(keys.size());
    bool has_error = false;

    for (size_t i = 0; i < keys.size(); ++i) {
        if (result.error_codes[i] == ErrorCode::EC_NOENT) {
            continue;
        }
        if (result.error_codes[i] != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("get key failed, key[%lu](%lu), error_code: %d", i, keys[i], result.error_codes[i]);
            has_error = true;
            break;
        }
        if (location_maps[i].empty()) {
            continue;
        }
        std::vector<std::string> prune_loc_ids;
        valid_maps[i] = FilterValidLocations(location_maps[i], check_loc_data_exist_func_, prune_loc_ids);
        if (!prune_loc_ids.empty()) {
            prune_keys.emplace_back(keys[i]);
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
            for (size_t i = 0; i < keys.size(); ++i) {
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
                    out_locations[idx].push_back(loc_it->second);
                }
            }

        } else {
            // --- Per-key independent selection (WEIGHTED_RANDOM or other non-event-report) ---
            for (size_t i = 0; i < keys.size(); ++i) {
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
                out_locations[i].push_back(std::move(merged));
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
                                          std::vector<HostCacheMatch> &out_matches) const {
    SPAN_TRACER(request_context);
    out_matches.clear();
    if (keys.empty()) {
        return EC_OK;
    }

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    CacheLocationMapVector location_maps;
    auto result = meta_indexer_->GetLocations(request_context, keys, location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    LogErrorCodes("PrefixMatchByHost", result.error_codes, keys);
    assert(keys.size() == location_maps.size());

    KeyToHostSpecNames key_to_host_spec_names;
    KeyToHostSpecNames key_to_vineyard_spec_names;
    key_to_host_spec_names.reserve(keys.size());
    key_to_vineyard_spec_names.reserve(keys.size());
    const std::unordered_set<std::string> medium_set(medium_filter.begin(), medium_filter.end());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (result.error_codes[i] != ErrorCode::EC_OK) {
            KVCM_LOG_DEBUG(
                "prefix match by host end because Get keys[%lu](%lu) return %d", i, keys[i], result.error_codes[i]);
            break;
        }
        HostToSpecNames host_to_spec_names;
        HostToSpecNames vineyard_spec_names;
        BuildHostSpecNamesForOneKey(location_maps[i], check_loc_data_exist_func_, medium_set, host_to_spec_names);
        BuildVineyardHostSpecNamesForOneKey(
            location_maps[i], check_loc_data_exist_func_, medium_set, vineyard_spec_names);
        // 只保留可参与前缀匹配的连续 key 前缀。
        key_to_host_spec_names.push_back(std::move(host_to_spec_names));
        key_to_vineyard_spec_names.push_back(std::move(vineyard_spec_names));
    }

    if (key_to_host_spec_names.empty() || key_to_host_spec_names.front().empty()) {
        return EC_OK;
    }

    out_matches.reserve(key_to_host_spec_names.front().size());
    // 只有命中第一个 key 的 host 才可能有大于 0 的前缀长度。
    for (const auto &[host, _] : key_to_host_spec_names.front()) {
        const V6DPeerSelection no_p2p;
        auto local_specs = MergeHostAndP2PSpecs(host, key_to_host_spec_names, key_to_vineyard_spec_names, no_p2p);
        const int64_t local = ComputePrefixMatchBlocks(local_specs, use_eagle_pop);
        if (local == 0) {
            continue;
        }

        auto p2p_selection = SelectP2PByPrefix(host, key_to_host_spec_names, key_to_vineyard_spec_names);
        int64_t p2p_1_total_match = local;
        if (!p2p_selection.covered_indices.empty()) {
            auto p2p_specs =
                MergeHostAndP2PSpecs(host, key_to_host_spec_names, key_to_vineyard_spec_names, p2p_selection);
            p2p_1_total_match = ComputePrefixMatchBlocks(p2p_specs, use_eagle_pop);
        }
        out_matches.push_back(
            HostCacheMatch{host, local, static_cast<int64_t>(p2p_selection.covered_indices.size()), p2p_1_total_match});
    }
    return EC_OK;
}

ErrorCode MetaSearcher::PrefixMatchWithMambaByHost(RequestContext *request_context,
                                                   const KeyVector &keys,
                                                   bool use_eagle_pop,
                                                   const std::vector<std::string> &medium_filter,
                                                   const std::vector<LocationSpecGroup> &location_spec_groups,
                                                   std::vector<HostCacheMatch> &out_matches) const {
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
    std::vector<const LocationSpecGroup *> required_groups = full_groups;
    required_groups.insert(required_groups.end(), mamba_state_groups.begin(), mamba_state_groups.end());

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    CacheLocationMapVector location_maps;
    auto result = meta_indexer_->GetLocations(request_context, keys, location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    LogErrorCodes("PrefixMatchWithMambaByHost", result.error_codes, keys);
    assert(keys.size() == location_maps.size());

    KeyToHostSpecNames key_to_host_spec_names(keys.size());
    KeyToHostSpecNames key_to_vineyard_spec_names(keys.size());
    const std::unordered_set<std::string> medium_set(medium_filter.begin(), medium_filter.end());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (result.error_codes[i] != ErrorCode::EC_OK) {
            KVCM_LOG_DEBUG("prefix match with mamba by host end because Get keys[%lu](%lu) return %d",
                           i,
                           keys[i],
                           result.error_codes[i]);
            break;
        }
        BuildHostSpecNamesForOneKey(
            location_maps[i], check_loc_data_exist_func_, medium_set, key_to_host_spec_names[i]);
        BuildVineyardHostSpecNamesForOneKey(
            location_maps[i], check_loc_data_exist_func_, medium_set, key_to_vineyard_spec_names[i]);
    }

    out_matches.reserve(key_to_host_spec_names.front().size());
    // 只有命中第一个 key 的 host 才可能有大于 0 的前缀长度。
    for (const auto &kv : key_to_host_spec_names.front()) {
        const auto &host = kv.first;
        const V6DPeerSelection no_p2p;
        auto local_specs = MergeHostAndP2PSpecs(host, key_to_host_spec_names, key_to_vineyard_spec_names, no_p2p);
        const int64_t local =
            ComputeMambaPrefixMatchBlocks(local_specs, use_eagle_pop, full_groups, mamba_state_groups);
        if (local == 0) {
            continue;
        }

        auto p2p_selection =
            SelectP2PByPrefix(host, key_to_host_spec_names, key_to_vineyard_spec_names, required_groups);
        int64_t p2p_1_total_match = local;
        if (!p2p_selection.covered_indices.empty()) {
            auto p2p_specs =
                MergeHostAndP2PSpecs(host, key_to_host_spec_names, key_to_vineyard_spec_names, p2p_selection);
            p2p_1_total_match =
                ComputeMambaPrefixMatchBlocks(p2p_specs, use_eagle_pop, full_groups, mamba_state_groups);
        }
        out_matches.push_back(
            HostCacheMatch{host, local, static_cast<int64_t>(p2p_selection.covered_indices.size()), p2p_1_total_match});
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
                                         std::vector<std::string> &out_location_ids) {
    if (keys.size() != locations.size()) {
        return EC_BADARGS;
    }
    out_location_ids.clear();
    out_location_ids.resize(keys.size());
    std::vector<std::pair<DataStorageType, std::uint64_t>> loc_sz(keys.size());

    const int64_t batch_create_time = TimestampUtil::GetCurrentTimeUs();
    auto modifier = [&locations, &out_location_ids, &keys, &loc_sz, batch_create_time](
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

        out_location_ids[index] = std::move(location_id);
        return {ModifierAction::MA_OK, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteBlock);
    auto result = meta_indexer_->ReadModifyWriteBlock(request_context, keys, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteBlock);

    // update the usage of each storage type
    for (std::size_t i = 0; i < keys.size(); i++) {
        if (result.error_codes[i] == ErrorCode::EC_OK) {
            meta_indexer_->AddStorageUsageByType(loc_sz[i].first, loc_sz[i].second);
        }
    }

    if (result.ec != ErrorCode::EC_OK) {
        LogErrorCodes("meta_indexer_->ReadModifyWriteBlock", result.error_codes, keys);
    }
    return result.ec;
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
    for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
        auto &location_ids = location_ids_per_key[key_index];
        auto &key_usage_changes = usage_changes[key_index];
        location_ids.reserve(tasks_per_key[key_index].size());
        key_usage_changes.resize(tasks_per_key[key_index].size());
        std::unordered_set<std::string> seen_location_ids;
        for (const auto &task : tasks_per_key[key_index]) {
            if (task.location_id.empty() || !seen_location_ids.insert(task.location_id).second) {
                out_per_key_ec[key_index] = EC_BADARGS;
                return EC_BADARGS;
            }
            location_ids.push_back(task.location_id);
        }
    }

    const int64_t batch_create_time = TimestampUtil::GetCurrentTimeUs();
    std::vector<MetadataWriteLease> write_leases;
    auto modifier = [&keys, &tasks_per_key, &usage_changes, &acquire_write_lease, &write_leases, batch_create_time](
                        const std::vector<ErrorCode> &get_ecs,
                        const LocationIdVector &location_ids,
                        size_t key_index,
                        CacheLocationVector &locations,
                        PropertyMap & /*upsert_property_map*/) -> LocationModifierResult {
        if (acquire_write_lease) {
            auto [lease_ec, lease] = acquire_write_lease();
            if (lease_ec != EC_OK) {
                return {ModifierAction::MA_FAIL, std::vector<ErrorCode>(location_ids.size(), lease_ec)};
            }
            write_leases.push_back(std::move(lease));
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

    for (size_t key_index = 0; key_index < keys.size(); ++key_index) {
        ErrorCode key_ec = ErrorCode::EC_OK;
        if (key_index >= result.per_location_error_codes.size()) {
            key_ec = result.ec == ErrorCode::EC_OK ? ErrorCode::EC_ERROR : result.ec;
        } else {
            const auto &location_ecs = result.per_location_error_codes[key_index];
            for (size_t location_index = 0; location_index < location_ecs.size(); ++location_index) {
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
    return result.ec;
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

    std::vector<std::vector<std::pair<DataStorageType, std::uint64_t>>> created_locs_sz(keys.size());
    const int64_t batch_create_time = TimestampUtil::GetCurrentTimeUs();
    std::vector<std::vector<MergeLocationSpecsTask>> merge_tasks_per_key(keys.size());
    std::vector<MetadataWriteLease> write_leases;

    auto create_modifier = [&tasks_per_key,
                            &merge_tasks_per_key,
                            &keys,
                            &created_locs_sz,
                            &acquire_write_lease,
                            &write_leases,
                            batch_create_time](const LocationIdVector &existing_ids,
                                               ErrorCode get_ec,
                                               size_t index,
                                               PropertyMap & /*upsert_property_map*/,
                                               CacheLocationMap &out_new_locations) -> ModifierResult {
        if (acquire_write_lease) {
            auto [lease_ec, lease] = acquire_write_lease();
            if (lease_ec != EC_OK) {
                return {ModifierAction::MA_FAIL, lease_ec};
            }
            write_leases.push_back(std::move(lease));
        }
        if (get_ec != ErrorCode::EC_OK && get_ec != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("load location ids failed, key[%lu](%lu) return %d", index, keys[index], get_ec);
            return {ModifierAction::MA_FAIL, get_ec};
        }

        const std::unordered_set<std::string> existing_id_set(existing_ids.begin(), existing_ids.end());
        bool created = false;
        for (const auto &entry : tasks_per_key[index]) {
            const ErrorCode validation_ec = ValidateConsistentSnapshotVersion(entry.specs);
            if (validation_ec != EC_OK) {
                return {ModifierAction::MA_FAIL, validation_ec};
            }
            if (get_ec == ErrorCode::EC_OK && existing_id_set.count(entry.location_id) > 0) {
                merge_tasks_per_key[index].push_back(entry);
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

    for (size_t i = 0; i < keys.size(); ++i) {
        ErrorCode key_ec = (i < result.error_codes.size()) ? result.error_codes[i] : result.ec;
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
        if (out_per_key_ec[i] != ErrorCode::EC_OK || merge_tasks_per_key[i].empty()) {
            continue;
        }
        merge_keys.push_back(keys[i]);
        merge_key_indices.push_back(i);
        auto &ids = merge_location_ids.emplace_back();
        ids.reserve(merge_tasks_per_key[i].size());
        for (const auto &task : merge_tasks_per_key[i]) {
            ids.push_back(task.location_id);
        }
    }

    if (merge_keys.empty()) {
        return final_ec;
    }

    std::vector<std::vector<StorageUsageChange>> merge_usage_changes(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        merge_usage_changes[i].resize(merge_tasks_per_key[i].size());
    }
    write_leases.clear();
    auto merge_modifier = [&keys,
                           &merge_tasks_per_key,
                           &merge_key_indices,
                           &merge_usage_changes,
                           &acquire_write_lease,
                           &write_leases,
                           batch_create_time](const std::vector<ErrorCode> &get_ecs,
                                              const LocationIdVector &loc_ids,
                                              size_t key_index,
                                              CacheLocationVector &locs,
                                              PropertyMap &upsert_property_map) -> LocationModifierResult {
        (void)upsert_property_map;
        if (acquire_write_lease) {
            auto [lease_ec, lease] = acquire_write_lease();
            if (lease_ec != EC_OK) {
                return {ModifierAction::MA_FAIL, std::vector<ErrorCode>(loc_ids.size(), lease_ec)};
            }
            write_leases.push_back(std::move(lease));
        }
        const size_t original_key_index = merge_key_indices[key_index];
        const auto &tasks = merge_tasks_per_key[original_key_index];
        std::vector<ErrorCode> modifier_ecs(loc_ids.size(), ErrorCode::EC_OK);
        bool updated = false;
        for (size_t loc_index = 0; loc_index < loc_ids.size(); ++loc_index) {
            const ErrorCode ec = get_ecs[loc_index];
            if (loc_index >= tasks.size()) {
                modifier_ecs[loc_index] = ErrorCode::EC_ERROR;
                continue;
            }
            const auto &task = tasks[loc_index];
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
                std::vector<LocationSpec> merged_specs;
                const ErrorCode merge_ec =
                    MergeLocationSpecsByName(new_loc->location_specs(), task.specs, merged_specs);
                if (merge_ec != EC_OK) {
                    modifier_ecs[loc_index] = merge_ec;
                    continue;
                }
                new_loc->set_location_specs(std::move(merged_specs));
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

    for (size_t i = 0; i < merge_key_indices.size(); ++i) {
        const size_t original_key_index = merge_key_indices[i];
        ErrorCode key_ec = ErrorCode::EC_OK;
        if (i >= merge_result.per_location_error_codes.size()) {
            key_ec = merge_result.ec == ErrorCode::EC_OK ? ErrorCode::EC_ERROR : merge_result.ec;
        } else {
            for (size_t loc_index = 0; loc_index < merge_result.per_location_error_codes[i].size(); ++loc_index) {
                const auto loc_ec = merge_result.per_location_error_codes[i][loc_index];
                if (loc_ec == ErrorCode::EC_OK) {
                    const auto &usage = merge_usage_changes[original_key_index][loc_index];
                    if (usage.has_old) {
                        ApplyStorageUsageChange(meta_indexer_.get(),
                                                merge_tasks_per_key[original_key_index][loc_index].type,
                                                usage.old_size,
                                                usage.new_size);
                    } else {
                        meta_indexer_->AddStorageUsageByType(merge_tasks_per_key[original_key_index][loc_index].type,
                                                             usage.new_size);
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

    // 每个 DeleteLocationSpecsTask 需要独立返回结果；同一个 key 下多个 task
    // 也可能删除同一个 location 的不同 specs，因此先展开成 task 维度执行。
    KeyVector flat_keys;
    LocationIdsPerKey flat_location_ids;
    std::vector<std::pair<size_t, size_t>> flat_to_original_task_indices;
    std::vector<std::pair<DataStorageType, std::uint64_t>> flat_deleted_specs_size;
    std::vector<std::uint8_t> flat_missing_targets;
    std::vector<MetadataWriteLease> write_leases;
    for (size_t i = 0; i < keys.size(); ++i) {
        out_batch_results[i].assign(tasks_per_key[i].size(), ErrorCode::EC_OK);
        if (out_missing_targets) {
            (*out_missing_targets)[i].assign(tasks_per_key[i].size(), false);
        }
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
    auto modifier = [&keys,
                     &tasks_per_key,
                     &flat_to_original_task_indices,
                     &flat_deleted_specs_size,
                     &flat_missing_targets,
                     &acquire_write_lease,
                     &write_leases](const std::vector<ErrorCode> &get_ecs,
                                    const LocationIdVector &loc_ids,
                                    size_t key_index,
                                    CacheLocationVector &locs,
                                    PropertyMap &upsert_property_map) -> LocationModifierResult {
        (void)upsert_property_map;
        if (acquire_write_lease) {
            auto [lease_ec, lease] = acquire_write_lease();
            if (lease_ec != EC_OK) {
                return {ModifierAction::MA_FAIL, std::vector<ErrorCode>(loc_ids.size(), lease_ec)};
            }
            write_leases.push_back(std::move(lease));
        }
        std::vector<ErrorCode> modifier_ecs(loc_ids.size(), ErrorCode::EC_OK);
        if (loc_ids.size() != 1 || key_index >= flat_to_original_task_indices.size()) {
            modifier_ecs.assign(loc_ids.size(), ErrorCode::EC_ERROR);
            return {ModifierAction::MA_FAIL, std::move(modifier_ecs)};
        }

        const auto [original_key_index, original_task_index] = flat_to_original_task_indices[key_index];
        const auto &task = tasks_per_key[original_key_index][original_task_index];
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

        if (task.spec_names.empty()) {
            modifier_ecs[0] = ErrorCode::EC_BADARGS;
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
    for (size_t i = 0; i < flat_to_original_task_indices.size(); ++i) {
        const auto [original_key_index, original_task_index] = flat_to_original_task_indices[i];
        ErrorCode ec = result.ec;
        if (i < result.per_location_error_codes.size() && !result.per_location_error_codes[i].empty()) {
            ec = result.per_location_error_codes[i][0];
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
    return result.ec;
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
                                             const std::vector<std::vector<std::string>> &expected_location_values) {
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
    auto result = meta_indexer_->ReadModifyWriteLocation(request_context, keys, location_ids_per_key, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    out_per_location_ec = std::move(result.per_location_error_codes);

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
                                                    std::function<bool()> should_abort) {
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
                if (!submit_del_req_func_) {
                    KVCM_LOG_WARN("CleanupLocationsByPredicate: reclaimer submit callback is unavailable");
                    return EC_ERROR;
                }
                submit_del_req_func_(keys, delete_location_ids, expected_location_values, true);
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
