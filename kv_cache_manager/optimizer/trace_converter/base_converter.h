#pragma once
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/trace_converter/optimizer_schema_trace.h"
namespace kv_cache_manager {

class BaseConverter {
public:
    virtual ~BaseConverter() = default;
    virtual std::vector<std::shared_ptr<OptimizerSchemaTrace>>
    ConvertLogFileToTraces(const std::string &log_file_path) = 0;
    virtual std::shared_ptr<OptimizerSchemaTrace> ConvertLogLineToTrace(const std::string &log_line) = 0;
    virtual std::shared_ptr<OptimizerSchemaTrace> ConvertLogLineToTraceByType(const std::string &log_line,
                                                                              const std::string &event_type) = 0;
};
} // namespace kv_cache_manager