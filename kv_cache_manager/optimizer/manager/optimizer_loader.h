#pragma once
#include <memory>
#include <string>

#include "kv_cache_manager/optimizer/config/optimizer_config.h"
#include "kv_cache_manager/optimizer/trace_converter/optimizer_schema_trace.h"

namespace kv_cache_manager {
class OptimizerLoader {
public:
    OptimizerLoader() = default;
    ~OptimizerLoader() = default;

    static std::vector<std::shared_ptr<OptimizerSchemaTrace>> LoadTrace(OptimizerConfig &config);
    static void DumpSchemaTracesToFile(OptimizerConfig &config);
};
} // namespace kv_cache_manager