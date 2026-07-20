#include "kv_cache_manager/manager/meta_searcher.h"

#include <algorithm>
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

std::vector<LocationSpec> MergeLocationSpecsByName(const std::vector<LocationSpec> &old_specs,
                                                   const std::vector<LocationSpec> &new_specs) {
    std::map<std::string, LocationSpec> merged_specs;
    for (const auto &spec : old_specs) {
        merged_specs[spec.name()] = spec;
    }
    for (const auto &spec : new_specs) {
        merged_specs[spec.name()] = spec;
    }

    std::vector<LocationSpec> specs;
    specs.reserve(merged_specs.size());
    for (auto &[name, spec] : merged_specs) {
        specs.push_back(std::move(spec));
    }
    return specs;
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
            if (loc_ptr->type() != DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT) {
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
        if (prefix_covered.size() > best.covered_indices.size()) {
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
        if (indices.size() > best.covered_indices.size()) {
            best.peer_addr = addr;
            best.covered_indices = indices;
        }
    }
    return best;
}

CacheLocationMap FilterValidLocations(const CacheLocationMap &location_map,
                                      CheckLocDataExistFunc check_loc_data_exist,
                                      std::vector<std::string> &out_prune_loc_ids) {
    CacheLocationMap valid;
    for (const auto &[id, loc] : location_map) {
        if (!loc)
            continue;
        if (loc->status() != CacheLocationStatus::CLS_SERVING)
            continue;
        if (check_loc_data_exist && !check_loc_data_exist(*loc)) {
            if (loc->type() != DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT) {
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
        for (const auto &spec : loc->location_specs()) {
            StandardUri uri(spec.uri());
            if (!uri.Valid() || !IsMediumMatched(uri, medium_set)) {
                continue;
            }
            std::string host = uri.GetHostPort();
            if (host.empty()) {
                continue;
            }
            host_specs[std::move(host)].insert(spec.name());
        }
    }
}

bool IsFullLocationSpecGroup(const LocationSpecGroup &group) { return group.name().rfind("full", 0) == 0; }

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
        submit_del_req_func_(prune_keys, prune_loc_ids_vec);
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
        submit_del_req_func_(prune_keys, prune_loc_ids_vec);
    }

    return out_locations.size() == keys.size() ? EC_OK : EC_ERROR;
}

ErrorCode MetaSearcher::BatchGetBestLocationByBackend(RequestContext *request_context,
                                                      const KeyVector &keys,
                                                      LocationsPerKey &out_locations,
                                                      SelectLocationPolicy *policy,
                                                      const std::vector<BackendSelector> &selectors) const {
    assert(policy != nullptr);
    SPAN_TRACER(request_context);
    out_locations.clear();
    out_locations.resize(keys.size());
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

        if (target_type == DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT &&
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
                    if (loc->type() != DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT)
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
                    if (loc->type() == target_type) {
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
        submit_del_req_func_(prune_keys, prune_loc_ids_vec);
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
        submit_del_req_func_(prune_keys, prune_loc_ids_vec);
    }

    return EC_OK;
}

ErrorCode MetaSearcher::PrefixMatchByHost(RequestContext *request_context,
                                          const KeyVector &keys,
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
    key_to_host_spec_names.reserve(keys.size());
    const std::unordered_set<std::string> medium_set(medium_filter.begin(), medium_filter.end());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (result.error_codes[i] != ErrorCode::EC_OK) {
            KVCM_LOG_DEBUG(
                "prefix match by host end because Get keys[%lu](%lu) return %d", i, keys[i], result.error_codes[i]);
            break;
        }
        HostToSpecNames host_to_spec_names;
        BuildHostSpecNamesForOneKey(location_maps[i], check_loc_data_exist_func_, medium_set, host_to_spec_names);
        // 只保留可参与前缀匹配的连续 key 前缀。
        key_to_host_spec_names.push_back(std::move(host_to_spec_names));
    }

    if (key_to_host_spec_names.empty() || key_to_host_spec_names.front().empty()) {
        return EC_OK;
    }

    out_matches.reserve(key_to_host_spec_names.front().size());
    // 只有命中第一个 key 的 host 才可能有大于 0 的前缀长度。
    for (const auto &[host, _] : key_to_host_spec_names.front()) {
        int64_t prefix_len = 1;
        for (size_t i = 1; i < key_to_host_spec_names.size(); ++i) {
            if (key_to_host_spec_names[i].find(host) == key_to_host_spec_names[i].end()) {
                break;
            }
            ++prefix_len;
        }
        out_matches.push_back(HostCacheMatch{host, prefix_len});
    }
    return EC_OK;
}

ErrorCode MetaSearcher::PrefixMatchWithMambaByHost(RequestContext *request_context,
                                                   const KeyVector &keys,
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

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    CacheLocationMapVector location_maps;
    auto result = meta_indexer_->GetLocations(request_context, keys, location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    LogErrorCodes("PrefixMatchWithMambaByHost", result.error_codes, keys);
    assert(keys.size() == location_maps.size());

    KeyToHostSpecNames key_to_host_spec_names(keys.size());
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
    }

    out_matches.reserve(key_to_host_spec_names.front().size());
    // 只有命中第一个 key 的 host 才可能有大于 0 的前缀长度。
    for (const auto &kv : key_to_host_spec_names.front()) {
        const auto &host = kv.first;
        size_t full_prefix_len = 0;
        for (; full_prefix_len < key_to_host_spec_names.size(); ++full_prefix_len) {
            auto host_it = key_to_host_spec_names[full_prefix_len].find(host);
            if (host_it == key_to_host_spec_names[full_prefix_len].end() ||
                !HasAllLocationSpecGroups(host_it->second, full_groups)) {
                break;
            }
        }
        if (full_prefix_len == 0) {
            continue;
        }

        int64_t prefix_len = 0;
        // Mamba 模式要求 full 前缀范围内最后一个可用 block 同时具备 mamba state。
        for (size_t offset = full_prefix_len; offset > 0; --offset) {
            const size_t index = offset - 1;
            auto host_it = key_to_host_spec_names[index].find(host);
            if (host_it != key_to_host_spec_names[index].end() &&
                HasAllLocationSpecGroups(host_it->second, mamba_state_groups)) {
                prefix_len = static_cast<int64_t>(index + 1);
                break;
            }
        }
        if (prefix_len > 0) {
            out_matches.push_back(HostCacheMatch{host, prefix_len});
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

ErrorCode MetaSearcher::BatchMergeLocationSpecs(RequestContext *request_context,
                                                const KeyVector &keys,
                                                const std::vector<std::vector<MergeLocationSpecsTask>> &tasks_per_key,
                                                std::vector<ErrorCode> &out_per_key_ec) {
    if (keys.size() != tasks_per_key.size()) {
        return EC_BADARGS;
    }
    out_per_key_ec.assign(keys.size(), ErrorCode::EC_OK);

    std::vector<std::pair<DataStorageType, std::uint64_t>> loc_sz(keys.size());
    const int64_t batch_create_time = TimestampUtil::GetCurrentTimeUs();
    std::vector<std::vector<MergeLocationSpecsTask>> merge_tasks_per_key(keys.size());

    auto create_modifier = [&tasks_per_key, &merge_tasks_per_key, &keys, &loc_sz, batch_create_time](
                               const LocationIdVector &existing_ids,
                               ErrorCode get_ec,
                               size_t index,
                               PropertyMap & /*upsert_property_map*/,
                               CacheLocationMap &out_new_locations) -> ModifierResult {
        if (get_ec != ErrorCode::EC_OK && get_ec != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("load location ids failed, key[%lu](%lu) return %d", index, keys[index], get_ec);
            return {ModifierAction::MA_FAIL, get_ec};
        }

        const std::unordered_set<std::string> existing_id_set(existing_ids.begin(), existing_ids.end());
        std::uint64_t key_total_sz = 0;
        DataStorageType key_type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
        bool created = false;
        for (const auto &entry : tasks_per_key[index]) {
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
            key_total_sz += GetLocationSpecsSize(entry.specs);
            if (key_type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) {
                key_type = entry.type;
            }
            created = true;
        }
        loc_sz[index] = std::make_pair(key_type, key_total_sz);
        if (!created) {
            return {ModifierAction::MA_SKIP, ErrorCode::EC_OK};
        }
        return {ModifierAction::MA_OK, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteBlock);
    auto result = meta_indexer_->ReadModifyWriteBlock(request_context, keys, create_modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteBlock);

    for (size_t i = 0; i < keys.size(); ++i) {
        ErrorCode key_ec = (i < result.error_codes.size()) ? result.error_codes[i] : result.ec;
        out_per_key_ec[i] = key_ec;
        if (key_ec == ErrorCode::EC_OK) {
            meta_indexer_->AddStorageUsageByType(loc_sz[i].first, loc_sz[i].second);
        }
    }

    if (result.ec != ErrorCode::EC_OK) {
        LogErrorCodes("meta_indexer_->ReadModifyWriteBlock", result.error_codes, keys);
        return result.ec;
    }

    KeyVector merge_keys;
    std::vector<size_t> merge_key_indexs;
    LocationIdsPerKey merge_location_ids;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (out_per_key_ec[i] != ErrorCode::EC_OK || merge_tasks_per_key[i].empty()) {
            continue;
        }
        merge_keys.push_back(keys[i]);
        merge_key_indexs.push_back(i);
        auto &ids = merge_location_ids.emplace_back();
        ids.reserve(merge_tasks_per_key[i].size());
        for (const auto &task : merge_tasks_per_key[i]) {
            ids.push_back(task.location_id);
        }
    }

    if (merge_keys.empty()) {
        return result.ec;
    }

    std::vector<std::vector<std::pair<DataStorageType, std::uint64_t>>> merge_locs_sz(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        merge_locs_sz[i].resize(merge_tasks_per_key[i].size());
    }
    auto merge_modifier = [&merge_tasks_per_key, &merge_key_indexs, &merge_locs_sz, batch_create_time](
                              const std::vector<ErrorCode> &get_ecs,
                              const LocationIdVector &loc_ids,
                              size_t key_index,
                              CacheLocationVector &locs,
                              PropertyMap &upsert_property_map) -> LocationModifierResult {
        (void)upsert_property_map;
        const size_t original_key_index = merge_key_indexs[key_index];
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
                continue;
            }

            std::shared_ptr<CacheLocation> new_loc;
            if (ec == ErrorCode::EC_OK && locs[loc_index]) {
                new_loc = std::make_shared<CacheLocation>(*locs[loc_index]);
                auto merged_specs = MergeLocationSpecsByName(new_loc->location_specs(), task.specs);
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
            locs[loc_index] = std::move(new_loc);
            merge_locs_sz[original_key_index][loc_index] = std::make_pair(task.type, GetLocationSpecsSize(task.specs));
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

    for (size_t i = 0; i < merge_key_indexs.size(); ++i) {
        const size_t original_key_index = merge_key_indexs[i];
        ErrorCode key_ec = ErrorCode::EC_OK;
        if (i >= merge_result.per_location_error_codes.size()) {
            key_ec = merge_result.ec == ErrorCode::EC_OK ? ErrorCode::EC_ERROR : merge_result.ec;
        } else {
            for (size_t loc_index = 0; loc_index < merge_result.per_location_error_codes[i].size(); ++loc_index) {
                const auto loc_ec = merge_result.per_location_error_codes[i][loc_index];
                if (loc_ec == ErrorCode::EC_OK) {
                    const auto &sz = merge_locs_sz[original_key_index][loc_index];
                    meta_indexer_->AddStorageUsageByType(sz.first, sz.second);
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
        return merge_result.ec;
    }
    return ErrorCode::EC_OK;
}

ErrorCode MetaSearcher::BatchDeleteLocationSpecs(RequestContext *request_context,
                                                 const KeyVector &keys,
                                                 const std::vector<std::vector<DeleteLocationSpecsTask>> &tasks_per_key,
                                                 std::vector<std::vector<ErrorCode>> &out_batch_results) {
    if (keys.size() != tasks_per_key.size()) {
        return EC_BADARGS;
    }
    out_batch_results.clear();
    out_batch_results.resize(keys.size());

    KeyVector flat_keys;
    LocationIdsPerKey flat_location_ids;
    std::vector<std::pair<size_t, size_t>> flat_to_original_index;
    std::vector<std::pair<DataStorageType, std::uint64_t>> flat_locs_sz;
    for (size_t i = 0; i < keys.size(); ++i) {
        out_batch_results[i].assign(tasks_per_key[i].size(), ErrorCode::EC_OK);
        for (size_t task_index = 0; task_index < tasks_per_key[i].size(); ++task_index) {
            flat_keys.push_back(keys[i]);
            flat_location_ids.push_back({tasks_per_key[i][task_index].location_id});
            flat_to_original_index.push_back({i, task_index});
            flat_locs_sz.emplace_back(DataStorageType::DATA_STORAGE_TYPE_UNKNOWN, 0);
        }
    }
    if (flat_keys.empty()) {
        return EC_OK;
    }

    auto modifier = [&keys, &tasks_per_key, &flat_to_original_index, &flat_locs_sz](
                        const std::vector<ErrorCode> &get_ecs,
                        const LocationIdVector &loc_ids,
                        size_t key_index,
                        CacheLocationVector &locs,
                        PropertyMap &upsert_property_map) -> LocationModifierResult {
        (void)upsert_property_map;
        std::vector<ErrorCode> modifier_ecs(loc_ids.size(), ErrorCode::EC_OK);
        if (loc_ids.size() != 1 || key_index >= flat_to_original_index.size()) {
            modifier_ecs.assign(loc_ids.size(), ErrorCode::EC_ERROR);
            return {ModifierAction::MA_FAIL, std::move(modifier_ecs)};
        }

        const auto [original_key_index, original_task_index] = flat_to_original_index[key_index];
        const auto &task = tasks_per_key[original_key_index][original_task_index];
        const ErrorCode ec = get_ecs.empty() ? ErrorCode::EC_ERROR : get_ecs[0];
        const std::string &loc_id = loc_ids[0];
        if (ec != ErrorCode::EC_OK) {
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
        flat_locs_sz[key_index] = std::make_pair(locs[0]->type(), GetLocationSpecsSize(deleted_specs));
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
    for (size_t i = 0; i < flat_to_original_index.size(); ++i) {
        const auto [original_key_index, original_task_index] = flat_to_original_index[i];
        ErrorCode ec = result.ec;
        if (i < result.per_location_error_codes.size() && !result.per_location_error_codes[i].empty()) {
            ec = result.per_location_error_codes[i][0];
        }
        out_batch_results[original_key_index][original_task_index] = ec;
    }

    for (size_t i = 0; i < flat_to_original_index.size(); ++i) {
        const auto [original_key_index, original_task_index] = flat_to_original_index[i];
        if (out_batch_results[original_key_index][original_task_index] == ErrorCode::EC_OK &&
            flat_locs_sz[i].first != DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) {
            meta_indexer_->SubStorageUsageByType(flat_locs_sz[i].first, flat_locs_sz[i].second);
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
            if (locs[loc_index]->status() != task.old_status) {
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
                                             std::vector<std::vector<ErrorCode>> &out_per_location_ec) {
    if (keys.size() != location_ids_per_key.size()) {
        return EC_BADARGS;
    }
    out_per_location_ec.clear();
    out_per_location_ec.resize(keys.size());

    std::vector<std::vector<std::pair<DataStorageType, std::uint64_t>>> locs_sz(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        locs_sz[i].resize(location_ids_per_key[i].size());
    }

    auto modifier = [&keys, &locs_sz](const std::vector<ErrorCode> &get_ecs,
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

ErrorCode MetaSearcher::CleanupLocationsByHost(RequestContext *request_context,
                                               const std::string &host_suffix,
                                               DataStorageType storage_type,
                                               size_t scan_batch_size,
                                               std::function<bool()> should_abort) {
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
                            has_any_location = true;
                        }
                    }
                }
                if (has_any_location) {
                    std::vector<std::vector<ErrorCode>> per_location_ec;
                    auto del_ec = BatchDeleteLocations(request_context, keys, delete_loc_ids, per_location_ec);
                    if (del_ec != EC_OK) {
                        KVCM_LOG_WARN("CleanupLocationsByHost: BatchDeleteLocations failed, ec %d", del_ec);
                        has_failure = true;
                    } else {
                        for (size_t i = 0; i < per_location_ec.size(); ++i) {
                            for (const auto &loc_ec : per_location_ec[i]) {
                                if (loc_ec != EC_OK && loc_ec != EC_NOENT) {
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
