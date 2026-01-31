#include "kv_cache_manager/manager/meta_searcher.h"

#include <set>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/request_context.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/common/timestamp_util.h"
#include "kv_cache_manager/meta/meta_indexer.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
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

bool safe_fetch_sub(std::atomic<std::uint64_t> &atom, const std::uint64_t sub_val) {
    std::uint64_t cur_val = atom.load();
    do {
        // check for underflow before subtraction
        if (cur_val < sub_val) {
            return false; // would underflow
        }
    } while (!atom.compare_exchange_strong(cur_val, cur_val - sub_val));
    return true;
}

} // namespace

MetaSearcher::MetaSearcher(const std::shared_ptr<MetaIndexer> &meta_indexer) : meta_indexer_(meta_indexer) {}

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
    UriVector uris;
    auto result = meta_indexer_->Get(request_context, keys, uris);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    int64_t index_deserialize_time_us = 0;

    for (size_t i = 0; i < keys.size(); ++i) {
        if (result.error_codes[i] != ErrorCode::EC_OK) {
            KVCM_LOG_DEBUG("prefix match end because Get keys[%lu](%lu) return %d", i, keys[i], result.error_codes[i]);
            break;
        }

        int64_t begin_deserialize_time = TimestampUtil::GetCurrentTimeUs();
        BlockCacheLocationsMeta meta;
        if (!meta.FromJsonString(uris[i])) {
            KVCM_LOG_WARN("location json parse failed, key[%lu](%lu), content: %s", i, keys[i], uris[i].c_str());
            break;
        }
        index_deserialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_deserialize_time);

        auto &location_map = meta.location_map();
        if (location_map.empty()) {
            KVCM_LOG_DEBUG("prefix match end because keys[%lu](%lu) no location", i, keys[i]);
            break;
        }
        const CacheLocation *best_location = policy->SelectForMatch(location_map);
        if (best_location == nullptr) {
            KVCM_LOG_DEBUG("prefix match end because keys[%lu] no serving location", i);
            break;
        }
        out_locations.push_back(*best_location);
    }

    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_deserialize_time_us, index_deserialize_time_us);

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
    UriVector uris;
    auto result = meta_indexer_->Get(request_context, keys, uris);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    int64_t index_deserialize_time_us = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (result.error_codes[i] == ErrorCode::EC_NOENT) {
            out_locations.push_back({});
            continue;
        }
        if (result.error_codes[i] != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("get key failed, key[%lu](%lu), error_code: %d", i, keys[i], result.error_codes[i]);
            break;
        }

        int64_t begin_deserialize_time = TimestampUtil::GetCurrentTimeUs();
        BlockCacheLocationsMeta meta;
        if (!meta.FromJsonString(uris[i])) {
            KVCM_LOG_WARN("location json parse failed, key[%lu](%lu), content: %s", i, keys[i], uris[i].c_str());
            break;
        }
        index_deserialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_deserialize_time);

        auto &location_map = meta.location_map();
        if (location_map.empty()) {
            out_locations.push_back({});
            continue;
        }
        const CacheLocation *best_location = policy->SelectForMatch(location_map);
        if (best_location == nullptr) {
            out_locations.push_back({});
            continue;
        }
        out_locations.push_back(*best_location);
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_deserialize_time_us, index_deserialize_time_us);
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
    UriVector uris;
    // TODO : optimize batch get
    auto result = meta_indexer_->Get(request_context, keys, uris);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);
    int64_t index_deserialize_time_us = 0;
    bool is_match = false;
    std::vector<CacheLocation> temp_sw_locations;
    temp_sw_locations.reserve(sw_size);
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
            int64_t begin_deserialize_time = TimestampUtil::GetCurrentTimeUs();
            BlockCacheLocationsMeta meta;
            if (!meta.FromJsonString(uris[base + offset])) {
                temp_sw_locations.clear();
                base -= sw_size - offset;
                is_match = false;
                break;
            }
            index_deserialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_deserialize_time);

            auto &location_map = meta.location_map();
            if (location_map.empty()) {
                temp_sw_locations.clear();
                base -= sw_size - offset;
                is_match = false;
                break;
            }
            CacheLocation *best_location = policy->SelectForMatch(location_map);
            if (best_location == nullptr) {
                temp_sw_locations.clear();
                base -= sw_size - offset;
                is_match = false;
                break;
            }
            temp_sw_locations.push_back(std::move(*best_location));
        }
        if (is_match) {
            std::move(temp_sw_locations.begin(), temp_sw_locations.end(), out_locations.begin() + base);
            break;
        }
    }
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_deserialize_time_us, index_deserialize_time_us);
    return EC_OK;
}

ErrorCode MetaSearcher::BatchGetLocation(RequestContext *request_context,
                                         const KeyVector &keys,
                                         const BlockMask &input_mask,
                                         std::vector<CacheLocationMap> &out_location_maps) {
    out_location_maps.clear();

    UriVector out_uris;
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
    auto result = meta_indexer_->Get(request_context, query_keys, out_uris);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerGet);

    for (size_t idx = 0; idx < query_keys.size(); idx++) {
        if (result.error_codes[idx] == ErrorCode::EC_NOENT) {
            out_location_maps.emplace_back();
            continue;
        }
        if (result.error_codes[idx] != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("get key failed, key[%lu](%lu), error_code: %d", idx, keys[idx], result.error_codes[idx]);
            out_location_maps.emplace_back();
            continue;
            ;
        }
        BlockCacheLocationsMeta meta;
        bool success = meta.FromJsonString(out_uris[idx]);
        if (!success) {
            out_location_maps.emplace_back();
            continue;
        }
        out_location_maps.push_back(meta.location_map());
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

    // record each location's storage type and total size between block
    // cache location metadata deserialize and serialize
    std::vector<std::pair<DataStorageType, std::uint64_t>> loc_sz(keys.size(),
                                                                  {DataStorageType::DATA_STORAGE_TYPE_UNKNOWN, 0});

    MetaSearcherMetrics metrics;
    auto modifier = [&locations, &out_location_ids, &keys, &metrics, &loc_sz](
                        std::string &uri,
                        ErrorCode get_ec,
                        size_t index,
                        MetaIndexer::PropertyMap &upsert_property_map) -> MetaIndexer::ModifierResult {
        BlockCacheLocationsMeta meta;
        std::string location_id;
        if (get_ec != ErrorCode::EC_OK && get_ec != ErrorCode::EC_NOENT) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu) return %d", index, keys[index], get_ec);
            return {MetaIndexer::MA_FAIL, get_ec};
        }
        int64_t begin_deserialize_time = TimestampUtil::GetCurrentTimeUs();
        if (get_ec == ErrorCode::EC_OK && !meta.FromJsonString(uri)) {
            KVCM_LOG_WARN("location json parse failed, key[%lu](%lu), content: %s", index, keys[index], uri.c_str());
            return {MetaIndexer::MA_FAIL, ErrorCode::EC_ERROR};
        }
        if (get_ec == EC_NOENT) {
            std::string prev_key;
            if (index > 0) {
                prev_key = std::to_string(keys[index - 1]);
            }
            upsert_property_map[PROPERTY_PREV_BLOCK_KEY] = prev_key;
        }
        metrics.index_deserialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_deserialize_time);
        meta.AddNewLocation(locations[index], location_id);
        ErrorCode ec = meta.UpdateLocationStatus(location_id, CLS_WRITING);
        if (ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("add location failed, keys[%lu](%lu), location id: %s, return: %d",
                          index,
                          keys[index],
                          location_id.c_str(),
                          ec);
            return {MetaIndexer::MA_FAIL, ErrorCode::EC_ERROR};
        }

        // save the storage type and size of this location
        std::uint64_t sz = 0;
        for (const auto &loc_spec : locations[index].location_specs()) {
            if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
                std::uint64_t spec_sz;
                ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                sz += spec_sz;
            }
        }
        loc_sz[index] = std::make_pair(locations[index].type() == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS
                                           ? DataStorageType::DATA_STORAGE_TYPE_HF3FS // treat vcns_hf3fs as hf3fs
                                           : locations[index].type(),
                                       sz);

        int64_t begin_serialize_time = TimestampUtil::GetCurrentTimeUs();
        uri = meta.ToJsonString();
        metrics.index_serialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_serialize_time);
        out_location_ids[index] = std::move(location_id);
        return {MetaIndexer::MA_OK, ErrorCode::EC_OK};
    };
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWrite);
    auto result = meta_indexer_->ReadModifyWrite(request_context, keys, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWrite);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_deserialize_time_us, metrics.index_deserialize_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_serialize_time_us, metrics.index_serialize_time_us);

    // update the usage of each storage type
    for (std::size_t i = 0; i < keys.size(); i++) {
        if (result.error_codes[i] == ErrorCode::EC_OK) {
            try {
                meta_indexer_->storage_usage_array_.at(static_cast<std::uint8_t>(loc_sz[i].first))
                    .fetch_add(loc_sz[i].second);
            } catch (const std::out_of_range &e) {
                KVCM_LOG_WARN("data storage type out of range: %d", static_cast<std::uint8_t>(loc_sz[i].first));
            }
        }
    }

    if (result.ec != ErrorCode::EC_OK) {
        LogErrorCodes("meta_indexer_->ReadModifyWrite", result.error_codes, keys);
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

    MetaSearcherMetrics metrics;
    auto modifier = [&batch_tasks, &keys, &metrics, &out_batch_results](
                        std::string &uri,
                        ErrorCode get_ec,
                        size_t index,
                        MetaIndexer::PropertyMap &upsert_property_map) -> MetaIndexer::ModifierResult {
        // 获取当前处理的键
        const int64_t current_key = keys[index];
        const auto &tasks = batch_tasks[index];
        auto &result = out_batch_results[index];

        if (get_ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu) return %d", index, current_key, get_ec);
            result.assign(tasks.size(), get_ec);
            return {MetaIndexer::MA_FAIL, get_ec};
        }

        BlockCacheLocationsMeta meta;
        int64_t begin_deserialize_time = TimestampUtil::GetCurrentTimeUs();
        if (!meta.FromJsonString(uri)) {
            KVCM_LOG_WARN("location json parse failed, key[%lu](%lu), content: %s", index, keys[index], uri.c_str());
            result.assign(tasks.size(), ErrorCode::EC_CORRUPTION);
            return {MetaIndexer::MA_FAIL, ErrorCode::EC_CORRUPTION};
        }
        metrics.index_deserialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_deserialize_time);
        bool updated = false;
        // 更新该键对应的所有location的status
        for (const auto &task : tasks) {
            ErrorCode ec = meta.UpdateLocationStatus(task.location_id, task.new_status);
            if (ec != ErrorCode::EC_OK) {
                KVCM_LOG_INFO(
                    "update location status failed, keys[%lu](%lu), location id: %s, new status: %d, return: %d",
                    index,
                    current_key,
                    task.location_id.c_str(),
                    task.new_status,
                    ec);
                result.push_back(ec);
            } else {
                updated = true;
                result.push_back(ErrorCode::EC_OK);
            }
        }

        if (!updated) {
            // do not need to update status, skip and return ok
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_OK};
        }
        int64_t begin_serialize_time = TimestampUtil::GetCurrentTimeUs();
        uri = meta.ToJsonString();
        metrics.index_serialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_serialize_time);
        return {MetaIndexer::MA_OK, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWrite);
    auto result = meta_indexer_->ReadModifyWrite(request_context, keys, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWrite);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_deserialize_time_us, metrics.index_deserialize_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_serialize_time_us, metrics.index_serialize_time_us);
    if (result.ec != ErrorCode::EC_OK) {
        LogErrorCodes("meta_indexer_->ReadModifyWrite", result.error_codes, keys);
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

    MetaSearcherMetrics metrics;
    auto modifier = [&keys, &batch_tasks, &metrics, &out_batch_results](
                        std::string &uri,
                        ErrorCode get_ec,
                        size_t index,
                        MetaIndexer::PropertyMap &upsert_property_map) -> MetaIndexer::ModifierResult {
        // 获取当前处理的键
        const int64_t current_key = keys[index];
        const auto &tasks = batch_tasks[index];
        auto &result = out_batch_results[index];

        if (get_ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu) return %d", index, current_key, get_ec);
            result.assign(tasks.size(), get_ec);
            return {MetaIndexer::MA_FAIL, ErrorCode::EC_IO_ERROR};
        }

        BlockCacheLocationsMeta meta;
        int64_t begin_deserialize_time = TimestampUtil::GetCurrentTimeUs();
        if (!meta.FromJsonString(uri)) {
            KVCM_LOG_WARN("location json parse failed, key[%lu](%lu), content: %s", index, keys[index], uri.c_str());
            result.assign(tasks.size(), ErrorCode::EC_CORRUPTION);
            return {MetaIndexer::MA_FAIL, ErrorCode::EC_CORRUPTION};
        }
        metrics.index_deserialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_deserialize_time);
        bool updated = false;
        // 更新该键对应的所有location的status
        for (const auto &task : tasks) {
            CacheLocationStatus status;
            ErrorCode ec = meta.GetLocationStatus(task.location_id, status);
            if (ec != ErrorCode::EC_OK) {
                KVCM_LOG_DEBUG("get location status failed, keys[%lu](%lu), location id: %s, return: %d",
                               index,
                               current_key,
                               task.location_id.c_str(),
                               ec);
                result.push_back(ec);
                continue;
            }
            if (status != task.old_status) {
                result.push_back(ErrorCode::EC_MISMATCH);
                continue;
            }
            ec = meta.UpdateLocationStatus(task.location_id, task.new_status);
            if (ec != ErrorCode::EC_OK) {
                KVCM_LOG_INFO("update location status failed, keys[%lu](%lu), location id: %s, old status: %d, new "
                              "status: %d, return: %d",
                              index,
                              current_key,
                              task.location_id.c_str(),
                              task.old_status,
                              task.new_status,
                              ec);
                result.push_back(ec);
            } else {
                updated = true;
                result.push_back(ErrorCode::EC_OK);
            }
        }

        if (!updated) {
            // do not need to update status, skip and return ok
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_OK};
        }
        int64_t begin_serialize_time = TimestampUtil::GetCurrentTimeUs();
        uri = meta.ToJsonString();
        metrics.index_serialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_serialize_time);
        return {MetaIndexer::MA_OK, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    auto result = meta_indexer_->ReadModifyWrite(request_context, keys, modifier);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_deserialize_time_us, metrics.index_deserialize_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_serialize_time_us, metrics.index_serialize_time_us);
    if (result.ec != ErrorCode::EC_OK) {
        LogErrorCodes("meta_indexer_->ReadModifyWrite", result.error_codes, keys);
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

    // record multiple locations' storage type and total size between
    // block cache location metadata deserialize and serialize
    std::vector<std::vector<std::pair<DataStorageType, std::uint64_t>>> locs_sz(keys.size());

    MetaSearcherMetrics metrics;
    auto modifier = [&batch_tasks, &keys, &metrics, &out_batch_results, &locs_sz](
                        std::string &uri,
                        ErrorCode get_ec,
                        size_t index,
                        MetaIndexer::PropertyMap &upsert_property_map) -> MetaIndexer::ModifierResult {
        // 获取当前处理的键
        const int64_t current_key = keys[index];
        const auto &tasks = batch_tasks[index];
        auto &result = out_batch_results[index];
        locs_sz[index].clear();

        if (get_ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu) return %d", index, current_key, get_ec);
            result.assign(tasks.size(), get_ec);
            return {MetaIndexer::MA_FAIL, ErrorCode::EC_ERROR};
        }
        BlockCacheLocationsMeta meta;
        int64_t begin_deserialize_time = TimestampUtil::GetCurrentTimeUs();
        if (!meta.FromJsonString(uri)) {
            KVCM_LOG_WARN("location json parse failed, key[%lu](%lu), content: %s", index, keys[index], uri.c_str());
            result.assign(tasks.size(), ErrorCode::EC_CORRUPTION);
            return {MetaIndexer::MA_FAIL, ErrorCode::EC_ERROR};
        }
        metrics.index_deserialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_deserialize_time);

        bool updated = false;
        for (const auto &task : tasks) {
            // save the storage type and size of this location
            std::uint64_t sz = 0;
            auto type{DataStorageType::DATA_STORAGE_TYPE_UNKNOWN};
            const auto &loc_map = meta.location_map();
            if (auto it = loc_map.find(task.location_id); it != loc_map.end()) {
                type = it->second.type();
                for (const auto &loc_spec : it->second.location_specs()) {
                    if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
                        std::uint64_t spec_sz;
                        ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                        sz += spec_sz;
                    }
                }
            }
            locs_sz[index].emplace_back(std::make_pair( // treat vcns_hf3fs as hf3fs
                type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS ? DataStorageType::DATA_STORAGE_TYPE_HF3FS : type,
                sz));

            CacheLocationStatus status;
            ErrorCode ec = meta.GetLocationStatus(task.location_id, status);
            if (ec != ErrorCode::EC_OK) {
                KVCM_LOG_WARN("get location status failed, keys[%lu](%lu), location id: %s, return: %d",
                              index,
                              current_key,
                              task.location_id.c_str(),
                              ec);
                result.push_back(ec);
                continue;
            }
            if (status != task.expect_status) {
                result.push_back(ErrorCode::EC_MISMATCH);
                continue;
            }
            ec = meta.DeleteLocation(task.location_id);
            if (ec != ErrorCode::EC_OK) {
                KVCM_LOG_WARN("delete location status failed, keys[%lu](%lu), location id: %s, return: %d",
                              index,
                              current_key,
                              task.location_id.c_str(),
                              ec);
                result.push_back(ec);
                continue;
            }
            updated = true;
            result.push_back(ErrorCode::EC_OK);
        }
        if (meta.GetLocationCount() == 0) {
            KVCM_LOG_DEBUG("all location deleted so delete keys[%lu](%lu)", index, current_key);
            return {MetaIndexer::MA_DELETE, ErrorCode::EC_OK};
        }
        if (!updated) {
            // do not need to update status, skip and return ok
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_OK};
        }
        int64_t begin_serialize_time = TimestampUtil::GetCurrentTimeUs();
        uri = meta.ToJsonString();
        metrics.index_serialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_serialize_time);

        return {MetaIndexer::MA_OK, ErrorCode::EC_OK};
    };

    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    auto result = meta_indexer_->ReadModifyWrite(request_context, keys, modifier);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_deserialize_time_us, metrics.index_deserialize_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_serialize_time_us, metrics.index_serialize_time_us);

    // update the usage of each storage type
    for (std::size_t i = 0; i != keys.size(); ++i) {
        if (result.error_codes[i] == ErrorCode::EC_OK) {
            for (std::size_t j = 0; j != batch_tasks[i].size(); ++j) {
                if (out_batch_results[i][j] == ErrorCode::EC_OK) {
                    try {
                        if (!safe_fetch_sub(
                                meta_indexer_->storage_usage_array_.at(static_cast<std::uint8_t>(locs_sz[i][j].first)),
                                locs_sz[i][j].second)) {
                            KVCM_LOG_DEBUG("would underflow");
                        }
                    } catch (const std::out_of_range &e) {
                        KVCM_LOG_WARN("data storage type out of range: %d",
                                      static_cast<std::uint8_t>(locs_sz[i][j].first));
                    }
                }
            }
        }
    }

    if (result.ec != ErrorCode::EC_OK) {
        LogErrorCodes("meta_indexer_->ReadModifyWrite", result.error_codes, keys);
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
    results.reserve(keys.size());

    // record each location's storage type and total size between block
    // cache location metadata deserialize and serialize
    std::vector<std::pair<DataStorageType, std::uint64_t>> loc_sz(keys.size(),
                                                                  {DataStorageType::DATA_STORAGE_TYPE_UNKNOWN, 0});

    MetaSearcherMetrics metrics;
    auto modifier = [&location_ids, &keys, &metrics, &loc_sz](
                        std::string &uri,
                        ErrorCode get_ec,
                        size_t index,
                        MetaIndexer::PropertyMap &upsert_property_map) -> MetaIndexer::ModifierResult {
        if (get_ec == ErrorCode::EC_NOENT) {
            // do not need to delete, skip and return ok
            return {MetaIndexer::MA_SKIP, ErrorCode::EC_OK};
        }
        if (get_ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("load location failed, key[%lu](%lu) return %d", index, keys[index], get_ec);
            return {MetaIndexer::MA_FAIL, get_ec};
        }
        BlockCacheLocationsMeta meta;
        int64_t begin_deserialize_time = TimestampUtil::GetCurrentTimeUs();
        if (!meta.FromJsonString(uri)) {
            KVCM_LOG_WARN("location json parse failed, key[%lu](%lu), content: %s", index, keys[index], uri.c_str());
            return {MetaIndexer::MA_FAIL, ErrorCode::EC_ERROR};
        }
        metrics.index_deserialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_deserialize_time);

        // save the storage type and size of this location
        std::uint64_t sz = 0;
        auto type{DataStorageType::DATA_STORAGE_TYPE_UNKNOWN};
        const auto &loc_map = meta.location_map();
        if (auto it = loc_map.find(location_ids[index]); it != loc_map.end()) {
            type = it->second.type();
            for (const auto &loc_spec : it->second.location_specs()) {
                if (DataStorageUri ds_uri(loc_spec.uri()); ds_uri.Valid()) {
                    std::uint64_t spec_sz;
                    ds_uri.GetParamAs<std::uint64_t>("size", spec_sz);
                    sz += spec_sz;
                }
            }
        }
        loc_sz[index] = std::make_pair( // treat vcns_hf3fs as hf3fs
            type == DataStorageType::DATA_STORAGE_TYPE_VCNS_HF3FS ? DataStorageType::DATA_STORAGE_TYPE_HF3FS : type,
            sz);

        ErrorCode ec = meta.DeleteLocation(location_ids[index]);
        if (ec != ErrorCode::EC_OK) {
            KVCM_LOG_WARN("delete location failed, keys[%lu](%lu), location id: %s, return: %d",
                          index,
                          keys[index],
                          location_ids[index].c_str(),
                          ec);
            return {MetaIndexer::MA_FAIL, ec};
        }

        int64_t begin_serialize_time = TimestampUtil::GetCurrentTimeUs();
        uri = meta.ToJsonString();
        metrics.index_serialize_time_us += (TimestampUtil::GetCurrentTimeUs() - begin_serialize_time);
        return {MetaIndexer::MA_OK, ErrorCode::EC_OK};
    };
    auto *service_metrics_collector = dynamic_cast<ServiceMetricsCollector *>(request_context->metrics_collector());
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_BEGIN(service_metrics_collector, MetaSearcherIndexerReadModifyWrite);
    auto result = meta_indexer_->ReadModifyWrite(request_context, keys, modifier);
    KVCM_METRICS_COLLECTOR_CHRONO_MARK_END(service_metrics_collector, MetaSearcherIndexerReadModifyWrite);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_deserialize_time_us, metrics.index_deserialize_time_us);
    KVCM_METRICS_COLLECTOR_SET_METRICS(
        service_metrics_collector, meta_searcher, index_serialize_time_us, metrics.index_serialize_time_us);

    // update the usage of each storage type
    for (std::size_t i = 0; i < keys.size(); i++) {
        if (result.error_codes[i] == ErrorCode::EC_OK) {
            try {
                if (!safe_fetch_sub(meta_indexer_->storage_usage_array_.at(static_cast<std::uint8_t>(loc_sz[i].first)),
                                    loc_sz[i].second)) {
                    KVCM_LOG_DEBUG("would underflow");
                }
            } catch (const std::out_of_range &e) {
                KVCM_LOG_WARN("data storage type out of range: %d", static_cast<std::uint8_t>(loc_sz[i].first));
            }
        }
    }

    if (result.ec != ErrorCode::EC_OK) {
        LogErrorCodes("meta_indexer_->ReadModifyWrite", result.error_codes, keys);
    }
    results = std::move(result.error_codes);
    return result.ec;
}

} // namespace kv_cache_manager
