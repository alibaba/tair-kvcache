#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/optimizer/trace_loader/optimizer_schema_trace.h"

namespace kv_cache_manager {

struct InferEngineActiveWindow {
    std::string infer_id;
    int64_t start_ns = 0;
    int64_t end_ns = 0;
};

class InferActiveWindowSet {
public:
    void Clear();
    void SetConfiguredWindows(const std::vector<InferEngineActiveWindow> &active_windows,
                              const std::unordered_set<std::string> &known_infer_ids);
    void BuildFromTrace(const std::vector<std::shared_ptr<OptimizerSchemaTrace>> &traces,
                        int64_t write_delay_ns,
                        bool require_known_infer_id,
                        const std::unordered_set<std::string> &known_infer_ids);

    [[nodiscard]] std::vector<std::string> FilterActive(const std::vector<std::string> &infer_ids,
                                                        int64_t timestamp_ns) const;
    [[nodiscard]] bool IsActiveAt(const std::string &infer_id, int64_t timestamp_ns) const;
    [[nodiscard]] bool empty() const { return active_windows_.empty(); }

private:
    struct ActiveWindow {
        int64_t first_timestamp_ns = 0;
        int64_t last_timestamp_ns = 0;
    };

    static void ValidateKnownInferId(const std::string &infer_id,
                                     bool require_known_infer_id,
                                     const std::unordered_set<std::string> &known_infer_ids,
                                     const std::string &error_prefix);
    void RecordActivity(const std::string &infer_id,
                        int64_t timestamp_ns,
                        const std::unordered_set<std::string> &known_infer_ids);

    std::unordered_map<std::string, std::vector<ActiveWindow>> active_windows_;
};

} // namespace kv_cache_manager
