#pragma once

#include "kv_cache_manager/optimizer/config/optimizer_lite_hit_config.h"

namespace kv_cache_manager {

// Offline entry point for LiteHit. It reuses the online stack verbatim: it builds
// an in-memory OnlineOptimizerManager, registers the configured instance groups and
// instances, replays the trace's read requests through TraceQuery (full-attention
// instances go through LiteHit exactly like online), and writes a per-instance
// "capacity(GB) vs prefix hit rate" CSV from ListInstances.
//
// Only read events ("get"/"request") are consumed: TraceQuery commits every full
// block of a request to the analyzer, so explicit write events are not needed.
class LiteHitOfflineRunner {
public:
    explicit LiteHitOfflineRunner(const OptimizerLiteHitConfig &config) : config_(config) {}

    // Returns true on success. Registration / IO failures return false and are
    // logged; a trace request whose instance is not configured or whose shape is
    // rejected is skipped, not fatal.
    bool Run();

private:
    OptimizerLiteHitConfig config_;
};

} // namespace kv_cache_manager
