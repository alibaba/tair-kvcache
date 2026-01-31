#include "kv_cache_manager/optimizer/trace_converter/publisher_log_converter.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/manager/cache_location.h"
namespace kv_cache_manager {
std::vector<std::shared_ptr<OptimizerSchemaTrace>>
PublisherLogConverter::ConvertLogFileToTraces(const std::string &log_file_path) {
    std::vector<std::shared_ptr<OptimizerSchemaTrace>> traces;
    std::ifstream file(log_file_path);

    if (!file.is_open()) {
        KVCM_LOG_ERROR("Failed to open log file: %s", log_file_path.c_str());
        return traces;
    }

    std::string line;
    while (std::getline(file, line)) {
        auto trace = ConvertLogLineToTrace(line);
        if (trace) {
            traces.push_back(trace);
        }
    }
    CheckAndConvertPendingWriteEvent(traces);
    ConvertPendingGetLocationEvent(traces);
    file.close();
    return traces;
}

std::shared_ptr<OptimizerSchemaTrace> PublisherLogConverter::ConvertLogLineToTrace(const std::string &log_line) {
    // 从日志行中提取JSON部分
    // 日志格式是: "JSON_CONTENT"
    std::string json_str = log_line;

    std::string event_type;
    rapidjson::Document event_doc;

    if (ParseEventTypeFromJson(json_str, event_type, event_doc)) {
        if (event_type == "GetCacheLocation") {
            return ConvertGetLocationEvent(event_doc);
        } else if (event_type == "StartWriteCache") {
            return ConvertStartWriteCacheEvent(event_doc);
        } else if (event_type == "FinishWriteCache") {
            return ConvertFinishWriteCacheEvent(event_doc);
        }
    }
    return nullptr;
}

bool PublisherLogConverter::ParseEventTypeFromJson(const std::string &json_str,
                                                   std::string &event_type,
                                                   rapidjson::Document &event_doc) {
    rapidjson::Document doc;
    std::string json_prefix = json_str.length() > 100 ? json_str.substr(0, 100) + "..." : json_str;

    if (doc.Parse(json_str.c_str()).HasParseError() || !doc.IsObject()) {
        KVCM_LOG_WARN("Failed to parse JSON: %s", json_prefix.c_str());
        return false;
    }
    event_doc.Swap(doc);
    if (event_doc.HasMember("type") && event_doc["type"].IsString()) {
        event_type = event_doc["type"].GetString();
    } else {
        KVCM_LOG_WARN("JSON missing 'type' field: %s", json_prefix.c_str());
        return false;
    }

    return true;
}

std::shared_ptr<GetLocationSchemaTrace>
PublisherLogConverter::ConvertGetLocationEvent(const rapidjson::Document &event_doc) {
    auto trace = std::make_shared<GetLocationSchemaTrace>();
    if (event_doc.HasMember("trigger_time_us") && event_doc["trigger_time_us"].IsInt64()) {
        trace->set_timestamp_us(event_doc["trigger_time_us"].GetInt64());
    }
    if (event_doc.HasMember("source") && event_doc["source"].IsString()) {
        std::string instance_id = event_doc["source"].GetString();
        trace->set_instance_id(instance_id);
        auto it = block_size_map_.find(instance_id);
        if (it == block_size_map_.end()) {
            KVCM_LOG_WARN("Instance ID %s not found in optimizer config", instance_id.c_str());
            return nullptr;
        }
    }

    if (event_doc.HasMember("keys") && event_doc["keys"].IsArray()) {
        std::vector<int64_t> keys;
        for (const auto &key : event_doc["keys"].GetArray()) {
            if (key.IsInt64()) {
                keys.push_back(key.GetInt64());
            }
        }
        trace->set_keys(keys);
    }

    if (event_doc.HasMember("query_type") && event_doc["query_type"].IsString()) {
        trace->set_query_type(event_doc["query_type"].GetString());
    }
    if (event_doc.HasMember("location_spec_names") && event_doc["location_spec_names"].IsArray()) {
        std::vector<std::string> location_spec_names;
        for (const auto &name : event_doc["location_spec_names"].GetArray()) {
            if (name.IsString()) {
                location_spec_names.push_back(name.GetString());
            }
        }
        trace->set_location_spec_names(location_spec_names);
    }
    if (event_doc.HasMember("block_mask")) {
        BlockMask block_mask;
        if (event_doc["block_mask"].IsArray()) {
            BlockMaskVector block_mask_vector;
            for (const auto &val : event_doc["block_mask"].GetArray()) {
                if (val.IsBool()) {
                    block_mask_vector.push_back(val.GetBool());
                }
            }
            block_mask = block_mask_vector;
        } else if (event_doc["block_mask"].IsInt64()) {
            block_mask = BlockMaskOffset(event_doc["block_mask"].GetInt64());
        }
        trace->set_block_mask(block_mask);
    }
    if (event_doc.HasMember("sw_size") && event_doc["sw_size"].IsInt()) {
        trace->set_sw_size(event_doc["sw_size"].GetInt());
    }

    pending_get_location_traces_.push_back(trace);
    return trace;
}

std::shared_ptr<DialogTurnSchemaTrace>
PublisherLogConverter::ConvertStartWriteCacheEvent(const rapidjson::Document &event_doc) {
    std::string write_session_id;
    if (!event_doc.HasMember("write_session_id") || !event_doc["write_session_id"].IsString()) {
        KVCM_LOG_WARN("StartWriteCache event missing write_session_id");
    } else {
        write_session_id = event_doc["write_session_id"].GetString();
    }
    auto trace = std::make_shared<WriteCacheSchemaTrace>();

    if (event_doc.HasMember("source") && event_doc["source"].IsString()) {
        std::string instance_id = event_doc["source"].GetString();
        trace->set_instance_id(instance_id);
        auto it = block_size_map_.find(instance_id);
        if (it == block_size_map_.end()) {
            KVCM_LOG_WARN("Instance ID %s not found in optimizer config", instance_id.c_str());
            return nullptr;
        }
    }
    if (event_doc.HasMember("trigger_time_us") && event_doc["trigger_time_us"].IsInt64()) {
        trace->set_timestamp_us(event_doc["trigger_time_us"].GetInt64());
    }
    if (event_doc.HasMember("keys") && event_doc["keys"].IsArray()) {
        std::vector<int64_t> keys;
        for (const auto &key : event_doc["keys"].GetArray()) {
            if (key.IsInt64()) {
                keys.push_back(key.GetInt64());
            }
        }
        trace->set_keys(keys);
    }
    if (!write_session_id.empty())
        pending_write_sessions_[write_session_id] = trace;

    return FindMatchingGetLocationTrace(trace);
}

std::shared_ptr<WriteCacheSchemaTrace>
PublisherLogConverter::ConvertFinishWriteCacheEvent(const rapidjson::Document &event_doc) {

    if (!event_doc.HasMember("write_session_id") || !event_doc["write_session_id"].IsString()) {
        KVCM_LOG_WARN("FinishWriteCache event missing write_session_id");
        return nullptr;
    }
    std::string write_session_id = event_doc["write_session_id"].GetString();
    auto it = pending_write_sessions_.find(write_session_id);
    if (it == pending_write_sessions_.end()) {
        KVCM_LOG_WARN("No matching StartWriteCache event for write_session_id: %s", write_session_id.c_str());
        return nullptr;
    }
    auto write_trace = it->second;
    if (event_doc.HasMember("source") && event_doc["source"].IsString()) {
        std::string finish_instance_id = event_doc["source"].GetString();
        std::string instance_id = write_trace->instance_id();
        if (finish_instance_id != instance_id) {
            KVCM_LOG_WARN("Mismatched instance_id for write_session_id: %s", write_session_id.c_str());
            return nullptr;
        }
        auto instance_it = block_size_map_.find(instance_id);
        if (instance_it == block_size_map_.end()) {
            KVCM_LOG_WARN("Instance ID %s not found in optimizer config", instance_id.c_str());
            return nullptr;
        }
    }
    pending_write_sessions_.erase(it);

    if (event_doc.HasMember("trigger_time_us") && event_doc["trigger_time_us"].IsInt64()) {
        write_trace->set_timestamp_us(event_doc["trigger_time_us"].GetInt64());
    }
    return write_trace;
}
// 处理只有startWriteCache没有对应finish的情况，此时假设结束时间比开始时间晚100毫秒
void PublisherLogConverter::CheckAndConvertPendingWriteEvent(
    std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
    if (!pending_write_sessions_.empty()) {
        for (const auto &pair : pending_write_sessions_) {
            auto trace = pair.second;
            int64_t start_write_timestamp_us = trace->timestamp_us();
            int64_t finish_write_timestamp_us = start_write_timestamp_us + 100000; // 假设结束时间比开始时间晚100毫秒
            trace->set_timestamp_us(finish_write_timestamp_us);
            traces.push_back(trace);
        }
        pending_write_sessions_.clear();
    }
}

std::shared_ptr<DialogTurnSchemaTrace>
PublisherLogConverter::FindMatchingGetLocationTrace(std::shared_ptr<WriteCacheSchemaTrace> write_trace) {
    const auto &write_keys = write_trace->keys();
    int64_t write_timestamp = write_trace->timestamp_us();
    std::shared_ptr<DialogTurnSchemaTrace> matched_trace = nullptr;
    for (auto it = pending_get_location_traces_.rbegin(); it != pending_get_location_traces_.rend(); ++it) {
        auto &trace_get = *it;
        auto get_instance_id = trace_get->instance_id();
        auto write_instance_id = write_trace->instance_id();
        if (get_instance_id != write_instance_id) {
            continue; // 实例ID不匹配，跳过
        }
        if (trace_get->timestamp_us() > write_timestamp) {
            continue; // GetLocation时间戳晚于WriteCache，跳过
        }
        const auto &get_keys = trace_get->keys();
        if (get_keys.empty()) {
            continue; // GetLocation没有keys，跳过
        }
        auto block_size = block_size_map_.find(get_instance_id)->second;
        size_t match_len = get_keys.size() - 1;
        if (write_keys.size() >= match_len && std::equal(get_keys.begin(), get_keys.end() - 1, write_keys.begin())) {

            matched_trace = std::make_shared<DialogTurnSchemaTrace>(trace_get);
            matched_trace->set_input_len(static_cast<int64_t>(get_keys.size() * block_size));
            matched_trace->set_output_len(static_cast<int64_t>((write_keys.size() - match_len) * block_size));
            matched_trace->set_total_keys(write_keys);

            pending_get_location_traces_.erase(std::next(it).base());
            return matched_trace;
        }
    }
    return matched_trace;
}

void PublisherLogConverter::ConvertPendingGetLocationEvent(std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces) {
    for (const auto &trace_get : pending_get_location_traces_) {
        auto trace_turn = std::make_shared<DialogTurnSchemaTrace>(*trace_get);
        auto block_size = block_size_map_.find(trace_get->instance_id())->second;
        trace_turn->set_input_len(trace_get->keys().size() * block_size);
        trace_turn->set_output_len(0);
        trace_turn->set_total_keys(trace_get->keys());
        traces.push_back(trace_turn);
    }
    pending_get_location_traces_.clear();
}
} // namespace kv_cache_manager