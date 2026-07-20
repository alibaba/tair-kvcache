#pragma once

#include <functional>
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

    // Streaming variant: parses the file line by line and invokes on_trace for each
    // valid trace without materializing the whole file. It shares the exact parsing
    // and validation rules as LoadFromFile. Callers that need global time ordering
    // must guarantee the input is already sorted by timestamp_ns.
    using TraceCallback = std::function<void(const std::shared_ptr<OptimizerSchemaTrace> &)>;
    static void StreamFromFile(const std::string &trace_file_path, const TraceCallback &on_trace);

private:
    static bool ValidateTrace(const OptimizerSchemaTrace &trace);
    // Parses one non-blank line into a validated trace. Reports errors through fail
    // (which is expected to throw) and never returns nullptr on the happy path.
    static std::shared_ptr<OptimizerSchemaTrace> ParseLine(const std::string &line,
                                                           const std::function<void(const std::string &)> &fail);
};

} // namespace kv_cache_manager
