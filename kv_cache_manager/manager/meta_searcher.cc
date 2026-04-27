#include "kv_cache_manager/manager/meta_searcher.h"

#include <map>
#include <set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/metrics/metrics_collector.h"

namespace kv_cache_manager {

const std::string MetaSearcher::PROPERTY_PREV_BLOCK_KEY = "_prev_key_";

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

CacheLocation SelectAndMergeForMatch(SelectLocationPolicy *policy,
                                     CacheLocationMap &location_map,
                                     CheckLocDataExistFunc check_loc_data_exist,
                                     std::vector<std::string> &out_prune_loc_ids) {
    // Filter valid locations into a shared map.
    CacheLocationMap valid_map;
    for (auto &[id, loc] : location_map) {
        if (loc.status() != CacheLocationStatus::CLS_SERVING) {
            continue;
        }
        if (check_loc_data_exist && !check_loc_data_exist(loc)) {
            out_prune_loc_ids.push_back(id);
            continue;
        }
        valid_map.try_emplace(id, loc);
    }
    if (valid_map.empty()) {
        return {};
    }

    // Use the policy to select one winning location, which determines the
    // target storage backend instance.
    std::vector<std::string> unused_prune_ids;
    CacheLocation *winner = policy->SelectForMatch(valid_map, nullptr, unused_prune_ids);
    if (!winner || winner->location_specs().empty()) {
        return {};
    }

    // Collect all specs from every valid location that belongs to the same
    // storage backend as the winner, dedup by spec name.
    std::map<std::string, LocationSpec> merged_specs;
    for (auto &[id, loc] : valid_map) {
        if (!policy->IsSameDataStorage(loc, *winner)) {
            continue;
        }
        for (const auto &spec : loc.location_specs()) {
            merged_specs.try_emplace(spec.name(), spec);
        }
    }

    if (merged_specs.empty()) {
        return {};
    }

    // NOTE: this is an aggregated view merging
    // specs from multiple locations, not a real stored entity. Downstream
    // CacheLocationView / proto serialization never accesses id either.
    std::string representative_id = winner->id() + "merged";
    CacheLocation result;
    result.set_id(std::move(representative_id));
    result.set_status(CacheLocationStatus::CLS_SERVING);
    result.set_type(winner->type());
    std::vector<LocationSpec> specs;
    specs.reserve(merged_specs.size());
    for (auto &[name, spec] : merged_specs) {
        specs.push_back(std::move(spec));
    }
    result.set_spec_size(specs.size());
    result.set_location_specs(std::move(specs));
    return result;
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
    MetaIndexer::LocationMapVector location_maps;
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
        CacheLocation merged = SelectAndMergeForMatch(policy, location_map, check_loc_data_exist_func_, prune_loc_ids);
        if (!prune_loc_ids.empty()) {
            prune_keys.emplace_back(keys[i]);
            prune_loc_ids_vec.emplace_back(prune_loc_ids);
        }
        if (merged.location_specs().empty()) {
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
    MetaIndexer::LocationMapVector location_maps;
    auto result = meta_indexer_->GetLocations(request_context, keys, location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    KeyVector prune_keys;
    std::vector<std::vector<std::string>> prune_loc_ids_vec;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (result.error_codes[i] == ErrorCode::EC_NOENT) {
            out_locations.push_back({});
            continue;
        }
        if (result.error_codes[i] != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("get key failed, key[%lu](%lu), error_code: %d", i, keys[i], result.error_codes[i]);
            break;
        }

        auto &location_map = location_maps[i];
        if (location_map.empty()) {
            out_locations.push_back({});
            continue;
        }
        std::vector<std::string> prune_loc_ids;
        CacheLocation merged = SelectAndMergeForMatch(policy, location_map, check_loc_data_exist_func_, prune_loc_ids);
        if (!prune_loc_ids.empty()) {
            prune_keys.emplace_back(keys[i]);
            prune_loc_ids_vec.emplace_back(prune_loc_ids);
        }
        if (merged.location_specs().empty()) {
            out_locations.push_back({});
            continue;
        }
        out_locations.push_back(std::move(merged));
    }

    if (!prune_keys.empty() && submit_del_req_func_) {
        submit_del_req_func_(prune_keys, prune_loc_ids_vec);
    }

    return out_locations.size() == keys.size() ? EC_OK : EC_ERROR;
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
    out_locations.assign(keys.size(), {});
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerGet);
    MetaIndexer::LocationMapVector location_maps;
    auto result = meta_indexer_->GetLocations(request_context, keys, location_maps);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    bool is_match = false;
    std::vector<CacheLocation> temp_sw_locations;
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
            CacheLocation merged =
                SelectAndMergeForMatch(policy, location_map, check_loc_data_exist_func_, prune_loc_ids);
            if (!prune_loc_ids.empty()) {
                prune_keys.emplace_back(keys[base + offset]);
                prune_loc_ids_vec.emplace_back(prune_loc_ids);
            }
            if (merged.location_specs().empty()) {
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

    auto modifier = [&locations, &out_location_ids, &keys, &loc_sz](
                        const MetaIndexer::LocationIdVector &existing_ids,
                        ErrorCode get_ec,
                        size_t index,
                        MetaIndexer::PropertyMap &upsert_property_map,
                        MetaIndexer::LocationMap &out_new_locations) -> MetaIndexer::ModifierResult {
        if (get_ec != ErrorCode::EC_OK && get_ec != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu) return %d", index, keys[index], get_ec);
            return {MetaIndexer::MA_FAIL, get_ec};
        }

        // first time this block_key is created: record prev_key
        if (get_ec == EC_NOENT) {
            std::string prev_key;
            if (index > 0) {
                prev_key = std::to_string(keys[index - 1]);
            }
            upsert_property_map[PROPERTY_PREV_BLOCK_KEY] = prev_key;
        }

        // generate a unique location_id that does not collide with existing ones
        std::set<std::string> existing_id_set(existing_ids.begin(), existing_ids.end());
        std::string location_id;
        do {
            location_id = StringUtil::GenerateRandomString(8);
        } while (existing_id_set.count(location_id) > 0);

        // build the new CacheLocation with status = CLS_WRITING
        CacheLocation new_loc = locations[index];
        new_loc.set_id(location_id);
        new_loc.set_status(CLS_WRITING);
        out_new_locations[location_id] = std::move(new_loc);

        // compute storage size for usage tracking
        std::uint64_t sz = 0;
        for (const auto &loc_spec : locations[index].location_specs()) {
            if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
                std::uint64_t spec_sz;
                ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                sz += spec_sz;
            }
        }
        loc_sz[index] = std::make_pair(locations[index].type(), sz);

        out_location_ids[index] = std::move(location_id);
        return {MetaIndexer::MA_OK, ErrorCode::EC_OK};
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

ErrorCode MetaSearcher::BatchUpdateLocationStatus(RequestContext *request_context,
                                                  const KeyVector &keys,
                                                  const std::vector<std::vector<LocationUpdateTask>> &batch_tasks,
                                                  std::vector<std::vector<ErrorCode>> &out_batch_results) {

    if (keys.size() != batch_tasks.size()) {
        return EC_BADARGS;
    }
    out_batch_results.clear();
    out_batch_results.resize(keys.size());

    // Build LocationIdsPerKey and a task lookup map from batch_tasks.
    MetaIndexer::LocationIdsPerKey location_ids_per_key(keys.size());
    // task_status_map[key_index][location_id] = new_status
    std::vector<std::map<std::string, CacheLocationStatus>> task_status_map(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        for (const auto &task : batch_tasks[i]) {
            location_ids_per_key[i].push_back(task.location_id);
            task_status_map[i][task.location_id] = task.new_status;
        }
    }

    auto modifier = [&keys,
                     &task_status_map](CacheLocation &loc,
                                       ErrorCode get_ec,
                                       size_t key_index,
                                       const MetaIndexer::LocationId &loc_id,
                                       MetaIndexer::PropertyMap &upsert_property_map) -> MetaIndexer::ModifierResult {
        if (get_ec == ErrorCode::EC_NOENT) {
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_NOENT};
        }
        if (get_ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu), location_id: %s, return %d",
                          key_index,
                          keys[key_index],
                          loc_id.c_str(),
                          get_ec);
            return {MetaIndexer::MA_FAIL, get_ec};
        }
        auto it = task_status_map[key_index].find(loc_id);
        if (it == task_status_map[key_index].end()) {
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_NOENT};
        }
        loc.set_status(it->second);
        return {MetaIndexer::MA_OK, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto result = meta_indexer_->ReadModifyWriteLocation(request_context, keys, location_ids_per_key, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i < result.per_location_error_codes.size()) {
            out_batch_results[i] = result.per_location_error_codes[i];
        } else {
            out_batch_results[i].assign(batch_tasks[i].size(), ErrorCode::EC_ERROR);
        }
    }

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

    // Build LocationIdsPerKey and a task lookup map from batch_tasks.
    MetaIndexer::LocationIdsPerKey location_ids_per_key(keys.size());
    // task_map[key_index][location_id] = {old_status, new_status}
    struct CASPair {
        CacheLocationStatus old_status;
        CacheLocationStatus new_status;
    };
    std::vector<std::map<std::string, CASPair>> task_map(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        for (const auto &task : batch_tasks[i]) {
            location_ids_per_key[i].push_back(task.location_id);
            task_map[i][task.location_id] = {task.old_status, task.new_status};
        }
    }

    auto modifier = [&keys, &task_map](CacheLocation &loc,
                                       ErrorCode get_ec,
                                       size_t key_index,
                                       const MetaIndexer::LocationId &loc_id,
                                       MetaIndexer::PropertyMap &upsert_property_map) -> MetaIndexer::ModifierResult {
        if (get_ec == ErrorCode::EC_NOENT) {
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_NOENT};
        }
        if (get_ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu), location_id: %s, return %d",
                          key_index,
                          keys[key_index],
                          loc_id.c_str(),
                          get_ec);
            return {MetaIndexer::MA_FAIL, get_ec};
        }
        auto it = task_map[key_index].find(loc_id);
        if (it == task_map[key_index].end()) {
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_NOENT};
        }
        if (loc.status() != it->second.old_status) {
            return {MetaIndexer::MA_FAIL, ErrorCode::EC_MISMATCH};
        }
        loc.set_status(it->second.new_status);
        return {MetaIndexer::MA_OK, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto result = meta_indexer_->ReadModifyWriteLocation(request_context, keys, location_ids_per_key, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i < result.per_location_error_codes.size()) {
            out_batch_results[i] = result.per_location_error_codes[i];
        } else {
            out_batch_results[i].assign(batch_tasks[i].size(), ErrorCode::EC_ERROR);
        }
    }

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

    // record multiple locations' storage type and total size for usage tracking
    std::vector<size_t> loc_counters(keys.size(), 0);
    std::vector<std::vector<std::pair<DataStorageType, std::uint64_t>>> locs_sz(keys.size());

    // Build LocationIdsPerKey and a task lookup map from batch_tasks.
    MetaIndexer::LocationIdsPerKey location_ids_per_key(keys.size());
    std::vector<std::map<std::string, CacheLocationStatus>> task_map(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        locs_sz[i].resize(batch_tasks[i].size());
        for (size_t j = 0; j < batch_tasks[i].size(); ++j) {
            location_ids_per_key[i].push_back(batch_tasks[i][j].location_id);
            task_map[i][batch_tasks[i][j].location_id] = batch_tasks[i][j].expect_status;
        }
    }

    auto modifier = [&keys, &task_map, &locs_sz, &loc_counters](
                        CacheLocation &loc,
                        ErrorCode get_ec,
                        size_t key_index,
                        const MetaIndexer::LocationId &loc_id,
                        MetaIndexer::PropertyMap &upsert_property_map) -> MetaIndexer::ModifierResult {
        size_t loc_idx = loc_counters[key_index]++;

        if (get_ec == ErrorCode::EC_NOENT) {
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_NOENT};
        }
        if (get_ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu), location_id: %s, return %d",
                          key_index,
                          keys[key_index],
                          loc_id.c_str(),
                          get_ec);
            return {MetaIndexer::MA_FAIL, get_ec};
        }
        auto it = task_map[key_index].find(loc_id);
        if (it == task_map[key_index].end()) {
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_NOENT};
        }

        if (loc.status() != it->second) {
            return {MetaIndexer::MA_FAIL, ErrorCode::EC_MISMATCH};
        }

        // compute storage size before deletion for usage tracking
        std::uint64_t sz = 0;
        for (const auto &loc_spec : loc.location_specs()) {
            if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
                std::uint64_t spec_sz;
                ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                sz += spec_sz;
            }
        }
        locs_sz[key_index][loc_idx] = std::make_pair(loc.type(), sz);
        return {MetaIndexer::MA_DELETE, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto result = meta_indexer_->ReadModifyWriteLocation(request_context, keys, location_ids_per_key, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i < result.per_location_error_codes.size()) {
            out_batch_results[i] = result.per_location_error_codes[i];
        } else {
            out_batch_results[i].assign(batch_tasks[i].size(), ErrorCode::EC_ERROR);
        }
    }

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

ErrorCode MetaSearcher::BatchDeleteLocation(RequestContext *request_context,
                                            const KeyVector &keys,
                                            const std::vector<std::string> &location_ids,
                                            std::vector<ErrorCode> &results) {

    if (keys.size() != location_ids.size()) {
        return EC_BADARGS;
    }
    results.clear();
    results.resize(keys.size(), ErrorCode::EC_OK);

    // record each location's storage type and total size for usage tracking
    std::vector<std::pair<DataStorageType, std::uint64_t>> loc_sz(keys.size());

    // Build LocationIdsPerKey: each key has exactly one location to delete
    MetaIndexer::LocationIdsPerKey location_ids_per_key(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        location_ids_per_key[i].push_back(location_ids[i]);
    }

    auto modifier = [&keys, &loc_sz](CacheLocation &loc,
                                     ErrorCode get_ec,
                                     size_t key_index,
                                     const MetaIndexer::LocationId &loc_id,
                                     MetaIndexer::PropertyMap &upsert_property_map) -> MetaIndexer::ModifierResult {
        if (get_ec == ErrorCode::EC_NOENT) {
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_NOENT};
        }
        if (get_ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu), location_id: %s, return %d",
                          key_index,
                          keys[key_index],
                          loc_id.c_str(),
                          get_ec);
            return {MetaIndexer::MA_FAIL, get_ec};
        }

        // compute storage size before deletion for usage tracking
        std::uint64_t sz = 0;
        for (const auto &loc_spec : loc.location_specs()) {
            if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
                std::uint64_t spec_sz;
                ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                sz += spec_sz;
            }
        }
        loc_sz[key_index] = std::make_pair(loc.type(), sz);
        return {MetaIndexer::MA_DELETE, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    auto result = meta_indexer_->ReadModifyWriteLocation(request_context, keys, location_ids_per_key, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWriteLocation);
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = ErrorCode::EC_OK;
        if (i >= result.per_location_error_codes.size() ||
            result.per_location_error_codes[i].size() != location_ids.size()) {
            results[i] = ErrorCode::EC_MISMATCH;
            continue;
        }
        size_t error_cnt = 0;
        for (const auto &ec : result.per_location_error_codes[i]) {
            error_cnt += (ec == ErrorCode::EC_OK);
        }
        results[i] =
            (error_cnt == 0 ? ErrorCode::EC_OK
                            : (error_cnt == location_ids.size() ? ErrorCode::EC_ERROR : ErrorCode::EC_PARTIAL_OK));
    }

    // update the usage of each storage type
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (results[i] == ErrorCode::EC_OK) {
            meta_indexer_->SubStorageUsageByType(loc_sz[i].first, loc_sz[i].second);
        }
    }

    if (result.ec != ErrorCode::EC_OK) {
        KVCM_LOG_WARN("meta_indexer_->ReadModifyWriteLocation failed, ec: %d", result.ec);
    }
    return result.ec;
}

} // namespace kv_cache_manager
