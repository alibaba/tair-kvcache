#include "select_location_policy.h"

#include <random>
#include <vector>

namespace kv_cache_manager {

CacheLocation *WeightSLPolicy::SelectForMatch(CacheLocationMap &location_map) const {
    thread_local std::mt19937 rng(std::random_device{}());
    std::vector<CacheLocation *> serving_locations;
    std::vector<uint32_t> weights;
    serving_locations.reserve(location_map.size());
    weights.reserve(location_map.size());
    for (auto &kv : location_map) {
        if (kv.second.status() == CacheLocationStatus::CLS_SERVING) {
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
    std::string_view host_name = ExtractHostName(kv.second.location_specs().front().uri());
    if (auto iter = weight_map_.find(host_name); iter != weight_map_.end()) {
        return iter->second;
    }
    return 0;
}

std::string_view NamedStorageWeightedSLPolicy::ExtractHostName(std::string_view uri) const {
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

} // namespace kv_cache_manager