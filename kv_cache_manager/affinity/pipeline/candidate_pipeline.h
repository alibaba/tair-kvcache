#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kv_cache_manager/affinity/node_metrics.h"
#include "kv_cache_manager/affinity/pipeline/filter_cond.h"
#include "kv_cache_manager/common/affinity_types.h"
#include "rapidjson/document.h"

namespace kv_cache_manager {

// Top-level affinity strategy, parsed from JSON.
//
// A CandidatePipeline is five named, optional slots evaluated in a fixed order:
//
//     filter -> prefer_local -> sample -> sort -> limit
//
// Any slot that is not present is skipped. The order is hard-coded by
// design (see plan D4 / docs); configuration cannot reorder slots.
//
// Top-level JSON schema:
//
//   {
//     "filter":       <Cond>,                   // optional, FilterCond tree
//     "prefer_local": { "on_miss": "passthrough" | "abort" }, // optional
//     "sample":       { "n": <int>,
//                       "node_pattern"?: <regex>,
//                       "seed"?: "random" | "trace_id" },     // optional
//     "sort":         [ { "metric": <name>,
//                         "weight": <number> }, ... ],        // optional
//     "limit":        <int>                                    // optional
//   }
//
// The wrapper `{ "strategy": { ... } }` is also accepted; the unwrapping is
// done by CacheAffinityManager before calling CandidatePipeline::Parse.

struct PreferLocalSpec {
    enum class OnMiss {
        kPassthrough, // pass input through unchanged
        kAbort,       // strategy aborts (Resolve returns EC_ERROR)
    };
    OnMiss on_miss = OnMiss::kPassthrough;
};

struct SampleSpec {
    enum class Seed {
        kRandom,
        kTraceId,
    };
    int n = 0;
    std::optional<std::regex> node_pattern;
    std::string node_pattern_src; // diagnostic / debug only
    Seed seed = Seed::kRandom;
};

struct SortTerm {
    std::string metric;
    double weight = 0.0; // negative weight = ascending sort
};

struct CandidatePipeline {
    std::optional<std::unique_ptr<FilterCond>> filter;
    std::optional<PreferLocalSpec> prefer_local;
    std::optional<SampleSpec> sample;
    std::optional<std::vector<SortTerm>> sort;
    std::optional<int> limit;

    // Parse a CandidatePipeline from a rapidjson value (must be an object with any
    // subset of the five known keys). Returns nullptr on parse failure with
    // `error_msg` populated when non-null.
    static std::unique_ptr<CandidatePipeline> Parse(const rapidjson::Value &value, std::string *error_msg);

    // Convenience: parse a top-level JSON string. Accepts either bare
    // `{ ... }` or wrapped `{ "strategy": { ... } }`.
    static std::unique_ptr<CandidatePipeline> ParseJsonString(const std::string &json, std::string *error_msg);

    // Result of Apply(): kAbort signals CandidatePipeline aborted (e.g. prefer_local
    // with on_miss=abort missed). kOk -> nodes is the surviving candidate
    // ordering (may be empty, meaning "no preference").
    enum class Status {
        kOk,
        kAbort,
    };
    struct ApplyResult {
        Status status = Status::kOk;
        std::vector<std::string> nodes;
    };

    // Run the 5-slot pipeline against `candidates` using `find_metrics` to
    // look up per-candidate NodeMetrics (returning nullptr for unknowns).
    // `caller` is consulted by prefer_local (uses caller.node_id);
    // `trace_id` seeds sampling when seed == kTraceId.
    ApplyResult Apply(const std::vector<std::string> &candidates,
                      const std::function<const NodeMetrics *(const std::string &)> &find_metrics,
                      const CallerNode &caller,
                      const std::string &trace_id) const;
};

} // namespace kv_cache_manager
