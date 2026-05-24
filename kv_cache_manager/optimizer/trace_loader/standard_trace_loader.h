#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {

class StandardTraceLoader {
public:
    StandardTraceLoader() = default;
    ~StandardTraceLoader() = default;

    static std::vector<std::shared_ptr<OptimizerSchemaTrace>> LoadFromFile(const std::string &trace_file_path);

private:
    static bool ValidateTrace(const OptimizerSchemaTrace &trace);
};

} // namespace kv_cache_manager
