#include "kv_cache_manager/optimizer/manager/optimizer_loader.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/trace_loader/standard_trace_loader.h"
#include "kv_cache_manager/optimizer/trace_loader/trace_util.h"

namespace kv_cache_manager {
std::vector<std::shared_ptr<OptimizerSchemaTrace>> OptimizerLoader::LoadTrace(OptimizerConfig &config) {
    // 直接加载标准格式trace文件
    auto traces = StandardTraceLoader::LoadFromFile(config.trace_file_path());

    // 后处理: 排序和生成trace_id
    TraceTimeSorter::SortTracesByTimestamp(traces);
    AddTraceId(traces);

    return traces;
}
} // namespace kv_cache_manager