#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kv_cache_manager {

// Post-hoc capacity query over a facts CSV published by the offline replay.
// Facts are read line by line: every valid row is projected to the caller's
// capacity slots with the shared HitCurveProjector and appended to the output
// log immediately; only O(capacity slots) cumulative integers are kept.
//
// The output is JSONL: one line per request
//   {"trace_id":...,"instance_id":...,"timestamp_ns":...,
//    "input_token_len":...,"hit_blocks":[...],"hit_rates":[...]}
// then one summary line per instance_id (deterministic order; a fanout run
// over several block sizes reads as one line per granularity)
//   {"summary":true,"instance_id":...,"requests":...,...}
// and one final overall summary line
//   {"summary":true,"requests":N,"total_input_tokens":T,
//    "capacity_gb":[...],"total_hit_blocks":[...],"total_hit_tokens":[...],
//    "hit_rates":[...]}
//
// capacity_gb slots keep the caller's order, duplicates and zeros included.
// A negative capacity means infinite. Any malformed facts row fails the whole
// query (facts files are all-or-nothing artifacts).
bool RunLiteHitFactsQuery(const std::string &facts_csv_path,
                          const std::vector<double> &capacity_gb,
                          const std::string &output_log_path,
                          std::string &error);

} // namespace kv_cache_manager
