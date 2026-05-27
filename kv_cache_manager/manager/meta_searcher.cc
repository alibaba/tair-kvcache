#include "kv_cache_manager/manager/meta_searcher.h"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
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
            out_prune_loc_ids.push_back(id);
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
    std::string representative_id = winner->id() + "merged";
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
    auto modifier =
        [&locations, &out_location_ids, &keys, &loc_sz, batch_create_time](const LocationIdVector &existing_location_ids,
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

ErrorCode MetaSearcher::BatchUpsertLocations(RequestContext *request_context,
                                             const KeyVector &keys,
                                             const std::vector<std::vector<UpsertLocation>> &new_locations_per_key,
                                             std::vector<ErrorCode> &out_per_key_ec) {
    if (keys.size() != new_locations_per_key.size()) {
        return EC_BADARGS;
    }
    out_per_key_ec.assign(keys.size(), ErrorCode::EC_OK);

    std::vector<std::pair<DataStorageType, std::uint64_t>> loc_sz(keys.size());

    auto modifier = [&new_locations_per_key, &keys, &loc_sz](const LocationIdVector & /*existing_ids*/,
                                                             ErrorCode get_ec,
                                                             size_t index,
                                                             PropertyMap & /*upsert_property_map*/,
                                                             CacheLocationMap &out_new_locations) -> ModifierResult {
        if (get_ec != ErrorCode::EC_OK && get_ec != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu) return %d", index, keys[index], get_ec);
            return {ModifierAction::MA_FAIL, get_ec};
        }
        std::uint64_t key_total_sz = 0;
        DataStorageType key_type = DataStorageType::DATA_STORAGE_TYPE_UNKNOWN;
        for (const auto &entry : new_locations_per_key[index]) {
            CacheLocation loc;
            loc.set_id(entry.location_id);
            loc.set_type(entry.type);
            loc.set_status(entry.status);
            loc.set_spec_size(entry.specs.size());
            for (const auto &ls : entry.specs) {
                loc.push_location_spec(LocationSpec(ls.name(), ls.uri()));
                if (DataStorageUri ds_uri(ls.uri()); ds_uri.Valid()) {
                    std::uint64_t spec_sz = 0;
                    ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                    key_total_sz += spec_sz;
                }
            }
            out_new_locations[entry.location_id] = std::make_shared<const CacheLocation>(std::move(loc));
            if (key_type == DataStorageType::DATA_STORAGE_TYPE_UNKNOWN) {
                key_type = entry.type;
            }
        }
        loc_sz[index] = std::make_pair(key_type, key_total_sz);
        return {ModifierAction::MA_OK, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWriteBlock);
    auto result = meta_indexer_->ReadModifyWriteBlock(request_context, keys, modifier);
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

ErrorCode MetaSearcher::BatchDeleteLocation(RequestContext *request_context,
                                            const KeyVector &keys,
                                            const std::vector<std::string> &location_ids,
                                            std::vector<ErrorCode> &results) {

    if (keys.size() != location_ids.size()) {
        return EC_BADARGS;
    }
    results.clear();
    results.resize(keys.size(), ErrorCode::EC_OK);

    std::vector<std::pair<DataStorageType, std::uint64_t>> loc_sz(keys.size());
    LocationIdsPerKey location_ids_per_key(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        location_ids_per_key[i].push_back(location_ids[i]);
    }

    // Per-key modifier: each key carries exactly one target location. OK
    // slot's payload size is captured for usage replay and ec stays EC_OK
    // (delete will be dispatched); NOENT is idempotent; errors surface as-is.
    auto modifier = [&keys, &loc_sz](const std::vector<ErrorCode> &get_ecs,
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
                    KVCM_LOG_ERROR("load location failed, key[%lu](%lu), location_id: %s, return %d",
                                   key_index,
                                   keys[key_index],
                                   loc_id.c_str(),
                                   ec);
                }
                continue;
            }
            if (!locs[loc_index]) {
                modifier_ecs[loc_index] = ErrorCode::EC_NOENT;
                continue;
            }
            updated = true;
            std::uint64_t sz = 0;
            for (const auto &loc_spec : locs[loc_index]->location_specs()) {
                if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
                    std::uint64_t spec_sz = 0;
                    ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                    sz += spec_sz;
                }
            }
            loc_sz[key_index] = std::make_pair(locs[loc_index]->type(), sz);
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
    for (size_t i = 0; i < keys.size(); ++i) {
        if (i >= result.per_location_error_codes.size() || result.per_location_error_codes[i].size() != 1) {
            results[i] = ErrorCode::EC_MISMATCH;
            continue;
        }
        results[i] = result.per_location_error_codes[i].front();
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
                        if (loc.type() == DataStorageType::DATA_STORAGE_TYPE_VINEYARD &&
                            loc_id.size() >= host_suffix.size() &&
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
