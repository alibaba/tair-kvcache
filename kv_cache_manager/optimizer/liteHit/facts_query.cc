#include "kv_cache_manager/optimizer/liteHit/facts_query.h"

#include <cmath>
#include <fstream>
#include <map>
#include <sstream>

#include "kv_cache_manager/optimizer/liteHit/facts_csv.h"
#include "kv_cache_manager/optimizer/liteHit/hit_curve.h"

namespace kv_cache_manager {

namespace {

constexpr long double kBytesPerGb = 1024.0L * 1024.0L * 1024.0L;

std::string EscapeJsonString(const std::string &value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        switch (c) {
        case '"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped.push_back(c);
        }
    }
    return escaped;
}

template <typename T>
void WriteJsonArray(std::ostream &out, const std::vector<T> &values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}

} // namespace

bool RunLiteHitFactsQuery(const std::string &facts_csv_path,
                          const std::vector<double> &capacity_gb,
                          const std::string &output_log_path,
                          std::string &error) {
    std::ifstream in(facts_csv_path);
    if (!in.is_open()) {
        error = "failed to open facts file: " + facts_csv_path;
        return false;
    }
    std::ofstream out(output_log_path, std::ios::trunc);
    if (!out.is_open()) {
        error = "failed to open output log: " + output_log_path;
        return false;
    }

    // Normalize caller slots once: finite capacities floor to bytes, negative
    // means infinite.
    struct Slot {
        bool infinite = false;
        uint64_t capacity_bytes = 0;
    };
    std::vector<Slot> slots;
    slots.reserve(capacity_gb.size());
    for (double gb : capacity_gb) {
        Slot slot;
        if (gb < 0) {
            slot.infinite = true;
        } else {
            const long double bytes = static_cast<long double>(gb) * kBytesPerGb;
            slot.capacity_bytes = static_cast<uint64_t>(std::floor(bytes));
        }
        slots.push_back(slot);
    }

    std::string line;
    if (!std::getline(in, line)) {
        error = "facts file is empty";
        return false;
    }
    if (line != kLiteHitFactsCsvHeader) {
        error = "facts file header mismatch";
        return false;
    }

    uint64_t requests = 0;
    uint64_t total_input_tokens = 0;
    std::vector<uint64_t> total_hit_blocks(slots.size(), 0);
    std::vector<uint64_t> total_hit_tokens(slots.size(), 0);

    struct InstanceTotals {
        uint64_t requests = 0;
        uint64_t total_input_tokens = 0;
        std::vector<uint64_t> hit_blocks;
        std::vector<uint64_t> hit_tokens;
    };
    std::map<std::string, InstanceTotals> instance_totals;

    std::size_t line_number = 1;
    while (std::getline(in, line)) {
        ++line_number;
        if (line.empty()) {
            continue;
        }
        LiteHitFactRecord record;
        std::string parse_error;
        if (!ParseLiteHitFactRow(line, record, parse_error)) {
            error = "facts line " + std::to_string(line_number) + ": " + parse_error;
            return false;
        }
        if (record.block_bytes == 0 || record.block_size_tokens == 0) {
            error = "facts line " + std::to_string(line_number) + ": block_bytes/block_size_tokens must be positive";
            return false;
        }

        std::vector<uint64_t> hit_blocks(slots.size(), 0);
        std::vector<double> hit_rates(slots.size(), 0.0);
        InstanceTotals &totals = instance_totals[record.instance_id];
        if (totals.hit_blocks.empty()) {
            totals.hit_blocks.assign(slots.size(), 0);
            totals.hit_tokens.assign(slots.size(), 0);
        }
        for (std::size_t i = 0; i < slots.size(); ++i) {
            uint64_t hits = 0;
            if (record.is_full_rle) {
                hits = slots[i].infinite ? HitCurveProjector::ProjectFullInfinite(record.full_rle_fact)
                                         : HitCurveProjector::ProjectFullBytes(
                                               record.full_rle_fact, slots[i].capacity_bytes, record.block_bytes);
            } else {
                hits = slots[i].infinite ? HitCurveProjector::ProjectInfinite(record.fact)
                                         : HitCurveProjector::ProjectBytes(record.fact, slots[i].capacity_bytes);
            }
            hit_blocks[i] = hits;
            const uint64_t hit_tokens = hits * record.block_size_tokens;
            hit_rates[i] = record.input_token_len == 0
                               ? 0.0
                               : static_cast<double>(hit_tokens) / static_cast<double>(record.input_token_len);
            total_hit_blocks[i] += hits;
            total_hit_tokens[i] += hit_tokens;
            totals.hit_blocks[i] += hits;
            totals.hit_tokens[i] += hit_tokens;
        }

        out << "{\"trace_id\":\"" << EscapeJsonString(record.trace_id) << "\",\"instance_id\":\""
            << EscapeJsonString(record.instance_id) << "\",\"timestamp_ns\":" << record.timestamp_ns
            << ",\"input_token_len\":" << record.input_token_len << ",\"hit_blocks\":";
        WriteJsonArray(out, hit_blocks);
        out << ",\"hit_rates\":";
        WriteJsonArray(out, hit_rates);
        out << "}\n";

        ++requests;
        total_input_tokens += record.input_token_len;
        ++totals.requests;
        totals.total_input_tokens += record.input_token_len;
    }
    if (in.bad()) {
        error = "I/O error while reading facts file: " + facts_csv_path;
        return false;
    }

    // One summary per instance (std::map keeps them in deterministic order).
    // A fanout run over several block sizes reads as one line per granularity.
    for (const auto &[instance_id, totals] : instance_totals) {
        std::vector<double> instance_rates(slots.size(), 0.0);
        for (std::size_t i = 0; i < slots.size(); ++i) {
            instance_rates[i] = totals.total_input_tokens == 0 ? 0.0
                                                               : static_cast<double>(totals.hit_tokens[i]) /
                                                                     static_cast<double>(totals.total_input_tokens);
        }
        out << "{\"summary\":true,\"instance_id\":\"" << EscapeJsonString(instance_id)
            << "\",\"requests\":" << totals.requests << ",\"total_input_tokens\":" << totals.total_input_tokens
            << ",\"capacity_gb\":";
        WriteJsonArray(out, capacity_gb);
        out << ",\"total_hit_blocks\":";
        WriteJsonArray(out, totals.hit_blocks);
        out << ",\"total_hit_tokens\":";
        WriteJsonArray(out, totals.hit_tokens);
        out << ",\"hit_rates\":";
        WriteJsonArray(out, instance_rates);
        out << "}\n";
    }

    std::vector<double> cumulative_rates(slots.size(), 0.0);
    for (std::size_t i = 0; i < slots.size(); ++i) {
        cumulative_rates[i] = total_input_tokens == 0
                                  ? 0.0
                                  : static_cast<double>(total_hit_tokens[i]) / static_cast<double>(total_input_tokens);
    }

    out << "{\"summary\":true,\"requests\":" << requests << ",\"total_input_tokens\":" << total_input_tokens
        << ",\"capacity_gb\":";
    WriteJsonArray(out, capacity_gb);
    out << ",\"total_hit_blocks\":";
    WriteJsonArray(out, total_hit_blocks);
    out << ",\"total_hit_tokens\":";
    WriteJsonArray(out, total_hit_tokens);
    out << ",\"hit_rates\":";
    WriteJsonArray(out, cumulative_rates);
    out << "}\n";

    out.flush();
    if (!out.good()) {
        error = "write to output log failed";
        return false;
    }
    return true;
}

} // namespace kv_cache_manager
