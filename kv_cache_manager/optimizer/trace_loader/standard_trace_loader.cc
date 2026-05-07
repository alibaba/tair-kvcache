#include "kv_cache_manager/optimizer/trace_loader/standard_trace_loader.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

std::vector<std::shared_ptr<OptimizerSchemaTrace>>
StandardTraceLoader::LoadFromFile(const std::string &trace_file_path) {
    std::vector<std::shared_ptr<OptimizerSchemaTrace>> traces;
    std::ifstream file(trace_file_path);

    if (!file.is_open()) {
        KVCM_LOG_ERROR("Failed to open trace file: %s", trace_file_path.c_str());
        return traces;
    }

    std::string line;
    int64_t line_number = 0;
    while (std::getline(file, line)) {
        line_number++;

        // 跳过空行
        if (line.empty()) {
            continue;
        }

        // 解析 JSON 以检查字段
        rapidjson::Document doc;
        if (doc.Parse(line.c_str()).HasParseError() || !doc.IsObject()) {
            std::string line_preview = line.length() > 100 ? line.substr(0, 100) + "..." : line;
            KVCM_LOG_WARN("Failed to parse JSON at line %ld in file %s: %s",
                          line_number,
                          trace_file_path.c_str(),
                          line_preview.c_str());
            continue;
        }

        // 根据type字段或字段推断来识别trace类型
        bool has_type_field = doc.HasMember("type") && doc["type"].IsString();
        std::string type_str = has_type_field ? doc["type"].GetString() : "";

        std::shared_ptr<OptimizerSchemaTrace> trace = nullptr;

        // 优先使用type字段
        if (has_type_field) {
            if (type_str == "get") {
                auto get_trace = std::make_shared<GetLocationSchemaTrace>();
                if (get_trace->FromJsonString(line)) {
                    trace = get_trace;
                }
            } else if (type_str == "write") {
                auto write_trace = std::make_shared<WriteCacheSchemaTrace>();
                if (write_trace->FromJsonString(line)) {
                    trace = write_trace;
                }
            } else {
                KVCM_LOG_WARN("Unknown trace type '%s' at line %ld", type_str.c_str(), line_number);
            }
        } else {
            // Fallback: 使用字段推断
            bool has_query_type = doc.HasMember("query_type");
            bool has_block_mask = doc.HasMember("block_mask");

            // GetLocationSchemaTrace: 有 query_type 和 block_mask
            if (has_query_type && has_block_mask) {
                auto get_trace = std::make_shared<GetLocationSchemaTrace>();
                if (get_trace->FromJsonString(line)) {
                    trace = get_trace;
                }
            }
            // WriteCacheSchemaTrace: 只有基础字段
            else {
                auto write_trace = std::make_shared<WriteCacheSchemaTrace>();
                if (write_trace->FromJsonString(line)) {
                    trace = write_trace;
                }
            }
        }

        // 验证并添加trace
        if (trace && ValidateTrace(*trace)) {
            traces.push_back(trace);
        } else {
            std::string line_preview = line.length() > 100 ? line.substr(0, 100) + "..." : line;
            KVCM_LOG_WARN("Failed to parse or validate trace at line %ld: %s", line_number, line_preview.c_str());
        }
    }

    file.close();
    KVCM_LOG_INFO("Loaded %zu traces from file: %s", traces.size(), trace_file_path.c_str());
    return traces;
}

bool StandardTraceLoader::ValidateTrace(const OptimizerSchemaTrace &trace) {
    // 验证必需字段
    if (trace.instance_id().empty()) {
        KVCM_LOG_ERROR("Validation failed: empty instance_id");
        return false;
    }
    if (trace.timestamp_ns() <= 0) {
        KVCM_LOG_ERROR("Validation failed: invalid timestamp_ns");
        return false;
    }
    if (trace.keys().empty()) {
        KVCM_LOG_ERROR("Validation failed: empty keys");
        return false;
    }

    return true;
}

} // namespace kv_cache_manager
