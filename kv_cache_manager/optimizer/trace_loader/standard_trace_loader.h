#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {

// StandardTraceLoader 加载标准格式的trace文件
// 支持三种trace类型的自动识别: GetLocationSchemaTrace, WriteCacheSchemaTrace, DialogTurnSchemaTrace
class StandardTraceLoader {
public:
    StandardTraceLoader() = default;
    ~StandardTraceLoader() = default;

    // 从标准格式JSONL文件加载traces
    // 自动识别每行的trace类型
    static std::vector<std::shared_ptr<OptimizerSchemaTrace>> LoadFromFile(const std::string &trace_file_path);

private:
    // 验证trace合法性
    static bool ValidateTrace(const OptimizerSchemaTrace &trace);
};

} // namespace kv_cache_manager
