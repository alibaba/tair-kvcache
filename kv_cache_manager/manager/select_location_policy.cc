#include "select_location_policy.h"

#include <random>
#include <string>
#include <vector>

#include "kv_cache_manager/common/standard_uri.h"

namespace kv_cache_manager {

CacheLocationConstPtr WeightSLPolicy::SelectForMatch(CacheLocationMap &location_map,
                                                     CheckLocDataExistFunc check_loc_data_exist,
                                                     std::vector<std::string> &out_prune_loc_ids) const {
    thread_local std::mt19937 rng(std::random_device{}());
    std::vector<CacheLocationConstPtr> serving_locations;
    std::vector<uint32_t> weights;
    serving_locations.reserve(location_map.size());
    weights.reserve(location_map.size());
    out_prune_loc_ids.clear();
    for (const auto &kv : location_map) {
        if (!kv.second) {
            continue;
        }
        if (kv.second->status() == CacheLocationStatus::CLS_SERVING) {
            if (check_loc_data_exist && !check_loc_data_exist(*kv.second)) {
                if (!IsEventReportStorageType(kv.second->type())) {
                    out_prune_loc_ids.emplace_back(kv.first);
                }
                continue;
            }
            if (int32_t weight = GetWeight(kv); weight > 0) {
                serving_locations.push_back(kv.second);
                weights.push_back(weight);
            }
        }
    }
    if (serving_locations.empty() || weights.empty()) {
        return std::make_shared<CacheLocation>();
    }
    std::discrete_distribution<uint32_t> dist(weights.begin(), weights.end());
    return serving_locations[dist(rng)];
}

bool WeightSLPolicy::ExistsForWrite(const CacheLocationMap &location_map,
                                    CheckLocDataExistFunc check_loc_data_exist,
                                    std::vector<std::string> &out_prune_loc_ids) const {
    bool exists = false;
    out_prune_loc_ids.clear();
    for (const auto &kv : location_map) {
        if (!kv.second) {
            continue;
        }
        if (kv.second->status() != CacheLocationStatus::CLS_NOT_FOUND) {
            if (kv.second->status() == CacheLocationStatus::CLS_SERVING && check_loc_data_exist &&
                !check_loc_data_exist(*kv.second)) {
                if (!IsEventReportStorageType(kv.second->type())) {
                    out_prune_loc_ids.emplace_back(kv.first);
                }
                continue;
            }
            if (!exists && GetWeight(kv) > 0) {
                exists = true;
            }
        }
    }
    return exists;
}

bool WeightSLPolicy::ExistsForWrite(const CacheLocationMap &location_map,
                                    const std::vector<std::string> &requested_spec_names,
                                    CheckLocDataExistFunc check_loc_data_exist,
                                    std::vector<std::string> &out_prune_loc_ids) const {
    // If no specific spec names requested, fall back to block-level check
    if (requested_spec_names.empty()) {
        return ExistsForWrite(location_map, check_loc_data_exist, out_prune_loc_ids);
    }
    // Check if any single serving location already covers ALL requested specs.
    // Use linear search on location_specs (typically 2-4 elements) instead of
    // building a hash set — avoids heap allocation and string copies.
    bool exists = false;
    out_prune_loc_ids.clear();
    for (const auto &kv : location_map) {
        if (!kv.second) {
            continue;
        }
        if (kv.second->status() == CacheLocationStatus::CLS_NOT_FOUND) {
            continue;
        }
        if (kv.second->status() == CacheLocationStatus::CLS_SERVING && check_loc_data_exist &&
            !check_loc_data_exist(*kv.second)) {
            if (!IsEventReportStorageType(kv.second->type())) {
                out_prune_loc_ids.emplace_back(kv.first);
            }
            continue;
        }
        if (exists) {
            continue;
        }
        if (GetWeight(kv) <= 0) {
            continue;
        }
        const auto &loc_specs = kv.second->location_specs();
        bool covers_all = std::all_of(
            requested_spec_names.begin(), requested_spec_names.end(), [&loc_specs](const std::string &name) {
                return std::any_of(
                    loc_specs.begin(), loc_specs.end(), [&name](const auto &spec) { return spec.name() == name; });
            });
        if (covers_all) {
            exists = true;
        }
    }
    return exists;
}

bool WeightSLPolicy::ExistsForWriteWithMinCount(const CacheLocationMap &location_map,
                                                int32_t min_count,
                                                const std::vector<std::string> &requested_spec_names,
                                                CheckLocDataExistFunc check_loc_data_exist,
                                                std::vector<std::string> &out_prune_loc_ids) const {
    out_prune_loc_ids.clear();
    int32_t count = 0;
    for (const auto &kv : location_map) {
        if (!kv.second) {
            continue;
        }
        if (kv.second->status() == CacheLocationStatus::CLS_NOT_FOUND) {
            continue;
        }
        if (check_loc_data_exist && kv.second->status() == CacheLocationStatus::CLS_SERVING &&
            !check_loc_data_exist(*kv.second)) {
            if (!IsEventReportStorageType(kv.second->type())) {
                out_prune_loc_ids.emplace_back(kv.first);
            }
            continue;
        }
        if (count >= min_count) {
            continue;
        }
        if (GetWeight(kv) == 0) {
            continue;
        }
        const auto &loc_specs = kv.second->location_specs();
        bool covers_all = std::all_of(
            requested_spec_names.begin(), requested_spec_names.end(), [&loc_specs](const std::string &name) {
                return std::any_of(
                    loc_specs.begin(), loc_specs.end(), [&name](const auto &spec) { return spec.name() == name; });
            });
        if (covers_all) {
            ++count;
        }
    }
    return count >= min_count;
}

uint32_t StaticWeightSLPolicy::GetWeight(CacheLocationMap::const_reference kv) const {
    if (!kv.second) {
        return 0;
    }
    DataStorageType type = kv.second->type();
    uint32_t weight = StorageTypeWeights::DEFAULT;
    // 如果枚举值对应数组索引可用，就直接取
    if (static_cast<size_t>(type) < std::size(storage_weights_)) {
        weight = storage_weights_[static_cast<size_t>(type)];
    }
    return weight;
}

uint32_t NamedStorageWeightedSLPolicy::GetWeight(CacheLocationMap::const_reference kv) const {
    if (!kv.second || kv.second->location_specs().empty()) {
        return 0;
    }
    // all location_specs in location have the same host name
    std::string host_name = StandardUri(kv.second->location_specs().front().uri()).GetHostPort();
    if (auto iter = weight_map_.find(host_name); iter != weight_map_.end()) {
        return iter->second;
    }
    return 0;
}

bool SelectLocationPolicy::IsSameDataStorage(const CacheLocation &candidate, const CacheLocation &reference) const {
    if (candidate.type() != reference.type()) {
        return false;
    }
    if (candidate.location_specs().empty() || reference.location_specs().empty()) {
        return candidate.location_specs().empty() == reference.location_specs().empty();
    }
    return StandardUri(candidate.location_specs().front().uri()).GetHostPort() ==
           StandardUri(reference.location_specs().front().uri()).GetHostPort();
}

} // namespace kv_cache_manager
