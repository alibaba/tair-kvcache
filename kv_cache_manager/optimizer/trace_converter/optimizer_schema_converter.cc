#include "kv_cache_manager/optimizer/trace_converter/optimizer_schema_converter.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "kv_cache_manager/common/jsonizable.h"
#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

std::vector<std::shared_ptr<OptimizerSchemaTrace>>
OptimizerSchemaConverter::ConvertLogFileToTraces(const std::string &log_file_path) {
    std::vector<std::shared_ptr<OptimizerSchemaTrace>> traces;
    std::ifstream file(log_file_path);

    if (!file.is_open()) {
        KVCM_LOG_ERROR("Failed to open trace file: %s", log_file_path.c_str());
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
                          log_file_path.c_str(),
                          line_preview.c_str());
            continue;
        }

        // 根据字段判断类型
        bool has_input_len = doc.HasMember("input_len");
        bool has_output_len = doc.HasMember("output_len");
        bool has_total_keys = doc.HasMember("total_keys");
        bool has_query_type = doc.HasMember("query_type");
        bool has_block_mask = doc.HasMember("block_mask");

        // DialogTurnSchemaTrace: 有 input_len, output_len, total_keys
        if (has_input_len && has_output_len && has_total_keys) {
            auto dialog_trace = std::make_shared<DialogTurnSchemaTrace>();
            if (dialog_trace->FromJsonString(line)) {
                traces.push_back(dialog_trace);
                continue;
            }
        }
        // GetLocationSchemaTrace: 有 query_type 和 block_mask，但没有 input_len
        else if (has_query_type && has_block_mask && !has_input_len) {
            auto get_trace = std::make_shared<GetLocationSchemaTrace>();
            if (get_trace->FromJsonString(line)) {
                traces.push_back(get_trace);
                continue;
            }
        }
        // WriteCacheSchemaTrace: 只有基础字段
        else {
            auto write_trace = std::make_shared<WriteCacheSchemaTrace>();
            if (write_trace->FromJsonString(line)) {
                traces.push_back(write_trace);
                continue;
            }
        }

        // 如果都失败，记录警告
        std::string line_preview = line.length() > 100 ? line.substr(0, 100) + "..." : line;
        KVCM_LOG_WARN(
            "Failed to parse line %ld in file %s: %s", line_number, log_file_path.c_str(), line_preview.c_str());
    }

    file.close();
    KVCM_LOG_INFO("Loaded %zu traces from file: %s", traces.size(), log_file_path.c_str());
    return traces;
}

std::shared_ptr<OptimizerSchemaTrace> OptimizerSchemaConverter::ConvertLogLineToTrace(const std::string &log_line) {
    // OptimizerSchemaConverter不需要转换单行，因为输入已经是标准格式
    // 这里返回nullptr表示不支持
    KVCM_LOG_WARN("OptimizerSchemaConverter does not support single line conversion");
    return nullptr;
}

std::shared_ptr<OptimizerSchemaTrace>
OptimizerSchemaConverter::ConvertLogLineToTraceByType(const std::string &log_line, const std::string &event_type) {
    // OptimizerSchemaConverter不需要按类型转换，因为输入已经是标准格式
    // 这里返回nullptr表示不支持
    KVCM_LOG_WARN("OptimizerSchemaConverter does not support type-based conversion");
    return nullptr;
}

} // namespace kv_cache_manager