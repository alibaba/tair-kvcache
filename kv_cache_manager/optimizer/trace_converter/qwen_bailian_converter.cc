#include "kv_cache_manager/optimizer/trace_converter/qwen_bailian_converter.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/trace_converter/trace_util.h"
namespace kv_cache_manager {
std::vector<std::shared_ptr<OptimizerSchemaTrace>>
QwenBailianConverter::ConvertLogFileToTraces(const std::string &log_file_path) {
    std::vector<std::shared_ptr<OptimizerSchemaTrace>> traces;
    std::ifstream file(log_file_path);
    if (!file.is_open()) {
        KVCM_LOG_ERROR("Failed to open log file: %s", log_file_path.c_str());
        return traces;
    }
    std::string line;
    while (std::getline(file, line)) {
        // Qwen Bailian 格式每行都是对话轮次，只转换为 DialogTurnSchemaTrace
        auto trace = ConvertLogLineToTrace(line);
        if (trace) {
            traces.push_back(trace);
        }
    }
    file.close();
    return traces;
}
std::shared_ptr<OptimizerSchemaTrace> QwenBailianConverter::ConvertLogLineToTrace(const std::string &log_line) {
    // 从日志行中提取JSON部分
    // 日志格式是: "JSON_CONTENT"
    std::string json_str = log_line;

    rapidjson::Document event_doc;
    std::string json_prefix = json_str.length() > 100 ? json_str.substr(0, 100) + "..." : json_str;
    if (event_doc.Parse(json_str.c_str()).HasParseError() || !event_doc.IsObject()) {
        KVCM_LOG_WARN("Failed to parse JSON: %s", json_prefix.c_str());
        return nullptr;
    }

    return ConvertDialogTurnEvent(event_doc);
}

std::shared_ptr<OptimizerSchemaTrace> QwenBailianConverter::ConvertLogLineToTraceByType(const std::string &log_line,
                                                                                        const std::string &event_type) {
    // 从日志行中提取JSON部分
    // 日志格式是: "JSON_CONTENT"
    std::string json_str = log_line;

    rapidjson::Document event_doc;
    std::string json_prefix = json_str.length() > 100 ? json_str.substr(0, 100) + "..." : json_str;
    if (event_doc.Parse(json_str.c_str()).HasParseError() || !event_doc.IsObject()) {
        KVCM_LOG_WARN("Failed to parse JSON: %s", json_prefix.c_str());
        return nullptr;
    }

    if (event_type == "get_cache_location") {
        return ConvertGetCacheLocationEvent(event_doc);
    } else if (event_type == "write_cache") {
        return ConvertWriteCacheEvent(event_doc);
    } else if (event_type == "dialog_turn") {
        return ConvertDialogTurnEvent(event_doc);
    } else {
        KVCM_LOG_WARN("Unknown event type: %s", event_type.c_str());
        return nullptr;
    }
}

std::shared_ptr<GetLocationSchemaTrace>
QwenBailianConverter::ConvertGetCacheLocationEvent(const rapidjson::Document &event_doc) {
    auto trace = std::make_shared<GetLocationSchemaTrace>();
    if (event_doc.HasMember("timestamp") && event_doc["timestamp"].IsDouble()) {
        trace->set_timestamp_us(event_doc["timestamp"].GetDouble() * 1000000); // 转换为微秒
    }
    std::vector<int64_t> keys;
    if (event_doc.HasMember("hash_ids") && event_doc["hash_ids"].IsArray()) {
        for (const auto &key : event_doc["hash_ids"].GetArray()) {
            if (key.IsInt64()) {
                keys.push_back(key.GetInt64());
            }
        }
    }
    auto block_keys = ApplyPrefixHash(keys); // 应用前缀哈希转换
    trace->set_instance_id("instance");      // Qwen Bailian日志中没有instance_id信息，默认设置
    trace->set_keys(block_keys);
    trace->set_tokens({});                 // Qwen Bailian日志中没有token信息，设置为空
    trace->set_query_type("prefix_match"); // 默认前缀匹配
    trace->set_location_spec_names({});    // Qwen Bailian日志中没有location_spec_names信息，设置为空
    trace->set_sw_size(0);                 // Qwen Bailian日志中没有sw_size信息，设置为0
    trace->set_block_mask(BlockMask{});    // Qwen Bailian日志中没有block_mask信息，设置为空
    return trace;
}

std::shared_ptr<WriteCacheSchemaTrace>
QwenBailianConverter::ConvertWriteCacheEvent(const rapidjson::Document &event_doc) {
    auto trace = std::make_shared<WriteCacheSchemaTrace>();
    if (event_doc.HasMember("timestamp") && event_doc["timestamp"].IsDouble()) {
        trace->set_timestamp_us(event_doc["timestamp"].GetDouble() * 1000000); // 转换为微秒
    }
    std::vector<int64_t> keys;
    if (event_doc.HasMember("hash_ids") && event_doc["hash_ids"].IsArray()) {
        for (const auto &key : event_doc["hash_ids"].GetArray()) {
            if (key.IsInt64()) {
                keys.push_back(key.GetInt64());
            }
        }
    }
    auto block_keys = ApplyPrefixHash(keys); // 应用前缀哈希转换
    trace->set_instance_id("instance");      // Qwen Bailian日志中没有instance_id信息，默认设置
    trace->set_keys(block_keys);
    trace->set_tokens({}); // Qwen Bailian日志中没有token信息，设置为空
    return trace;
}

std::shared_ptr<DialogTurnSchemaTrace>
QwenBailianConverter::ConvertDialogTurnEvent(const rapidjson::Document &event_doc) {
    auto trace = std::make_shared<DialogTurnSchemaTrace>();
    if (event_doc.HasMember("timestamp") && event_doc["timestamp"].IsDouble()) {
        trace->set_timestamp_us(event_doc["timestamp"].GetDouble() * 1000000); // 转换为微秒
    }
    std::vector<int64_t> keys;
    if (event_doc.HasMember("hash_ids") && event_doc["hash_ids"].IsArray()) {
        for (const auto &key : event_doc["hash_ids"].GetArray()) {
            if (key.IsInt64()) {
                keys.push_back(key.GetInt64());
            }
        }
    }
    auto block_keys = ApplyPrefixHash(keys); // 应用前缀哈希转换，trace中的hash ids并不包含前缀部分
    trace->set_instance_id("instance");      // Qwen Bailian日志中没有instance_id信息，默认设置
    trace->set_keys(block_keys);
    trace->set_tokens({});                 // Qwen Bailian日志中没有token信息，设置为空
    trace->set_query_type("prefix_match"); // 默认前缀匹配
    trace->set_location_spec_names({});    // Qwen Bailian日志中没有location_spec_names信息，设置为空
    trace->set_sw_size(0);                 // Qwen Bailian日志中没有sw_size信息，设置为0
    trace->set_block_mask(BlockMask{});    // Qwen Bailian日志中没有block_mask信息，设置为空
    trace->set_total_keys(block_keys); // Qwen Bailian日志中没有decode keys信息，设置为prefill keys,也就是当前的keys
    if (event_doc.HasMember("input_length") && event_doc["input_length"].IsInt64()) {
        trace->set_input_len(event_doc["input_length"].GetInt64());
    }
    if (event_doc.HasMember("output_length") && event_doc["output_length"].IsInt64()) {
        trace->set_output_len(event_doc["output_length"].GetInt64());
    }
    return trace;
}
} // namespace kv_cache_manager