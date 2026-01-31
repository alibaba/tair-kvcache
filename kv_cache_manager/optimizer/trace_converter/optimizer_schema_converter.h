#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/trace_converter/base_converter.h"
#include "kv_cache_manager/optimizer/trace_converter/optimizer_schema_trace.h"

namespace kv_cache_manager {

// OptimizerSchemaConverter直接加载已转换的标准JSONL格式trace文件
// 这种格式由anonymizer.py或其他工具生成，已经是OptimizerSchemaTrace的标准格式
class OptimizerSchemaConverter : public BaseConverter {
public:
    OptimizerSchemaConverter() = default;
    ~OptimizerSchemaConverter() override = default;

    // 从JSONL格式的标准trace文件加载traces
    std::vector<std::shared_ptr<OptimizerSchemaTrace>>
    ConvertLogFileToTraces(const std::string &log_file_path) override;

    // 不支持单行转换（因为已经是标准格式，不需要转换）
    std::shared_ptr<OptimizerSchemaTrace> ConvertLogLineToTrace(const std::string &log_line) override;

    // 不支持按类型转换（因为已经是标准格式）
    std::shared_ptr<OptimizerSchemaTrace> ConvertLogLineToTraceByType(const std::string &log_line,
                                                                      const std::string &event_type) override;
};

} // namespace kv_cache_manager