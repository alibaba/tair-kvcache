#pragma once

#include "kv_cache_manager/optimizer/config/optimizer_lite_hit_config.h"

namespace kv_cache_manager {

// Offline entry point for LiteHit. It validates instance registration through
// the online stack (OnlineOptimizerManager), then replays the trace's read
// requests through per-instance LiteHit lanes and atomically publishes ONE
// capacity-independent facts CSV: ${output_result_path}/litehit_facts.csv.
// Capacities are applied afterwards by the facts query tool; the replay never
// materializes per-capacity results.
//
// Only read events ("get"/"request") produce facts and state updates; each is
// one complete atomic access. Write events are recognized and ignored.
//
// The replay is fail-fast: out-of-order timestamps, unknown instances,
// malformed requests, or IO errors abort the whole run and no facts file is
// published (the temporary file is removed). A run without any valid request
// also fails.
class LiteHitOfflineRunner {
public:
    explicit LiteHitOfflineRunner(const OptimizerLiteHitConfig &config) : config_(config) {}

    bool Run();

private:
    OptimizerLiteHitConfig config_;
};

} // namespace kv_cache_manager
