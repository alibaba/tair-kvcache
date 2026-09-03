#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"

namespace kv_cache_manager {

// One row of the facts CSV. The row is a capacity-independent, recomputable
// fact: any capacity can be projected from hit_curve afterwards without
// replaying the trace. The default fact is an explicit byte-step curve;
// Full-only rows use the smaller arithmetic-run RLE on the block axis.
struct LiteHitFactRecord {
    std::string trace_id;
    std::string instance_id;
    int64_t timestamp_ns = 0;
    uint64_t input_token_len = 0;
    uint64_t block_size_tokens = 0;
    // Per-block byte charge used at projection boundaries. Recording it per
    // row keeps facts self-describing so a corrected charge estimate can
    // still reproject historical facts. For byte-step rows this is the Full
    // object charge for observability; projection already uses the byte axis.
    uint64_t block_bytes = 0;
    bool is_full_rle = false;
    RequestFact fact;              // default byte-step rows
    FullRequestFact full_rle_fact; // Full-only rows
};

inline constexpr const char *kLiteHitFactsCsvHeader =
    "trace_id,instance_id,timestamp_ns,input_token_len,block_size_tokens,block_bytes,hit_curve";

inline constexpr const char *kLiteHitFactsFileName = "litehit_facts.csv";

// Serializes one record to a CSV line (without trailing newline). String
// fields are quoted and escaped when needed; hit_curve is always a quoted
// JSON array. New rows use "bytes:" for byte-step facts and "rle:" for
// Full-only facts. The parser also accepts legacy "mamba:" byte steps and
// unprefixed Full RLE rows.
std::string SerializeLiteHitFactRow(const LiteHitFactRecord &record);

// Parses one CSV line produced by SerializeLiteHitFactRow. Returns false on
// any malformed field; error receives a human-readable reason.
bool ParseLiteHitFactRow(const std::string &line, LiteHitFactRecord &record, std::string &error);

} // namespace kv_cache_manager
