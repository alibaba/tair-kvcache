#include "select_location_policy.h"

#include <random>
#include <string>
#include <vector>

namespace {

// 从 URI 中提取 hostname，例如 "nfs://host01/path" → "host01"
std::string_view ExtractHostName(std::string_view uri) {
    static std::string empty_str;
    // 找协议分隔符 "://"
    auto pos_protocol_end = uri.find("://");
    if (pos_protocol_end == std::string::npos) {
        return empty_str;
    }
    // 提取 "[hostname][/path][?query]" 部分
    size_t host_start = pos_protocol_end + 3; // 跳过 "://"
    // 找第一个 '/' 或 '?'，那个是 hostname 结束的位置
    size_t pos_path_start = uri.find('/', host_start);
    size_t pos_query_start = uri.find('?', host_start);
    // 提取 hostname
    if (pos_path_start == std::string::npos && pos_query_start == std::string::npos) {
        // 没有 path 和 params
        return uri.substr(host_start);
    } else if (pos_path_start != std::string::npos &&
               (pos_query_start == std::string::npos || pos_path_start < pos_query_start)) {
        // 有 path（可能后面还有 query）
        return uri.substr(host_start, pos_path_start - host_start);

    } else if (pos_query_start != std::string::npos &&
               (pos_path_start == std::string::npos || pos_query_start < pos_path_start)) {
        // 没有 path，hostname 后直接是 query
        return uri.substr(host_start, pos_query_start - host_start);
    }
    return empty_str;
}

} // anonymous namespace

namespace kv_cache_manager {

CacheLocation *WeightSLPolicy::SelectForMatch(CacheLocationMap &location_map,
                                              CheckLocDataExistFunc check_loc_data_exist,
                                              std::vector<std::string> &out_prune_loc_ids) const {
    thread_local std::mt19937 rng(std::random_device{}());
    std::vector<CacheLocation *> serving_locations;
    std::vector<uint32_t> weights;
    serving_locations.reserve(location_map.size());
    weights.reserve(location_map.size());
    out_prune_loc_ids.clear();
    for (auto &kv : location_map) {
        if (kv.second.status() == CacheLocationStatus::CLS_SERVING) {
            if (check_loc_data_exist && !check_loc_data_exist(kv.second)) {
                out_prune_loc_ids.emplace_back(kv.first);
                continue;
            }
            if (int32_t weight = GetWeight(kv); weight > 0) {
                serving_locations.push_back(&kv.second);
                weights.push_back(weight);
            }
        }
    }
    if (serving_locations.empty() || weights.empty()) {
        return nullptr;
    }
    std::discrete_distribution<uint32_t> dist(weights.begin(), weights.end());
    return serving_locations[dist(rng)];
}

bool WeightSLPolicy::ExistsForWrite(const CacheLocationMap &location_map) const {
    for (auto &kv : location_map) {
        if (kv.second.status() != CacheLocationStatus::CLS_NOT_FOUND) {
            if (GetWeight(kv) > 0) {
                return true;
            }
        }
    }
    return false;
}

bool WeightSLPolicy::ExistsForWrite(const CacheLocationMap &location_map,
                                    const std::vector<std::string> &requested_spec_names) const {
    // If no specific spec names requested, fall back to block-level check
    if (requested_spec_names.empty()) {
        return ExistsForWrite(location_map);
    }
    // Check if any single serving location already covers ALL requested specs.
    // Use linear search on location_specs (typically 2-4 elements) instead of
    // building a hash set — avoids heap allocation and string copies.
    for (auto &kv : location_map) {
        if (kv.second.status() == CacheLocationStatus::CLS_NOT_FOUND) {
            continue;
        }
        if (GetWeight(kv) <= 0) {
            continue;
        }
        const auto &loc_specs = kv.second.location_specs();
        bool covers_all = std::all_of(
            requested_spec_names.begin(), requested_spec_names.end(), [&loc_specs](const std::string &name) {
                return std::any_of(
                    loc_specs.begin(), loc_specs.end(), [&name](const auto &spec) { return spec.name() == name; });
            });
        if (covers_all) {
            return true;
        }
    }
    return false;
}

uint32_t StaticWeightSLPolicy::GetWeight(CacheLocationMap::const_reference kv) const {
    DataStorageType type = kv.second.type();
    uint32_t weight = StorageTypeWeights::DEFAULT;
    // 如果枚举值对应数组索引可用，就直接取
    if (static_cast<size_t>(type) < std::size(storage_weights_)) {
        weight = storage_weights_[static_cast<size_t>(type)];
    }
    return weight;
}

uint32_t NamedStorageWeightedSLPolicy::GetWeight(CacheLocationMap::const_reference kv) const {
    // all location_specs in location have the same host name
    std::string_view host_name = ExtractHostName(kv.second.location_specs().front().uri());
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
    return ExtractHostName(candidate.location_specs().front().uri()) ==
           ExtractHostName(reference.location_specs().front().uri());
}

} // namespace kv_cache_manager