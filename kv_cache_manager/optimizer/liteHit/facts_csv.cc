#include "kv_cache_manager/optimizer/liteHit/facts_csv.h"

#include <cctype>
#include <cstdlib>
#include <sstream>

namespace kv_cache_manager {

namespace {

bool NeedsQuoting(const std::string &field) { return field.find_first_of(",\"\n\r") != std::string::npos; }

std::string QuoteCsvField(const std::string &field) {
    if (!NeedsQuoting(field)) {
        return field;
    }
    std::string quoted;
    quoted.reserve(field.size() + 2);
    quoted.push_back('"');
    for (char c : field) {
        if (c == '"') {
            quoted.push_back('"');
        }
        quoted.push_back(c);
    }
    quoted.push_back('"');
    return quoted;
}

// Splits one CSV line into fields, honoring double-quote escaping.
bool SplitCsvLine(const std::string &line, std::vector<std::string> &fields, std::string &error) {
    fields.clear();
    std::string current;
    bool in_quotes = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    current.push_back('"');
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                current.push_back(c);
            }
        } else if (c == '"') {
            in_quotes = true;
        } else if (c == ',') {
            fields.push_back(std::move(current));
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (in_quotes) {
        error = "unterminated quoted field";
        return false;
    }
    fields.push_back(std::move(current));
    return true;
}

bool ParseUint64Field(const std::string &field, uint64_t &value, const char *name, std::string &error) {
    if (field.empty()) {
        error = std::string(name) + " is empty";
        return false;
    }
    char *end = nullptr;
    errno = 0;
    value = std::strtoull(field.c_str(), &end, 10);
    if (errno != 0 || end != field.c_str() + field.size()) {
        error = std::string(name) + " is not a valid unsigned integer: " + field;
        return false;
    }
    return true;
}

bool ParseInt64Field(const std::string &field, int64_t &value, const char *name, std::string &error) {
    if (field.empty()) {
        error = std::string(name) + " is empty";
        return false;
    }
    char *end = nullptr;
    errno = 0;
    value = std::strtoll(field.c_str(), &end, 10);
    if (errno != 0 || end != field.c_str() + field.size()) {
        error = std::string(name) + " is not a valid integer: " + field;
        return false;
    }
    return true;
}

// Parses a JSON array of [uint, uint] pairs, e.g. "[[1,2],[4,1]]".
bool ParseHitCurve(const std::string &field, std::vector<HitCurveSegment> &segments, std::string &error) {
    segments.clear();
    std::size_t i = 0;
    const auto skip_spaces = [&] {
        while (i < field.size() && std::isspace(static_cast<unsigned char>(field[i]))) {
            ++i;
        }
    };
    const auto parse_uint = [&](uint64_t &value) -> bool {
        skip_spaces();
        const std::size_t start = i;
        while (i < field.size() && std::isdigit(static_cast<unsigned char>(field[i]))) {
            ++i;
        }
        if (start == i) {
            return false;
        }
        char *end = nullptr;
        errno = 0;
        value = std::strtoull(field.substr(start, i - start).c_str(), &end, 10);
        return errno == 0;
    };

    skip_spaces();
    if (i >= field.size() || field[i] != '[') {
        error = "hit_curve must start with '['";
        return false;
    }
    ++i;
    skip_spaces();
    if (i < field.size() && field[i] == ']') {
        ++i;
        skip_spaces();
        if (i != field.size()) {
            error = "hit_curve has trailing characters";
            return false;
        }
        return true;
    }

    while (true) {
        skip_spaces();
        if (i >= field.size() || field[i] != '[') {
            error = "hit_curve segment must start with '['";
            return false;
        }
        ++i;
        HitCurveSegment segment;
        if (!parse_uint(segment.start_required_blocks)) {
            error = "hit_curve segment start is not a valid integer";
            return false;
        }
        skip_spaces();
        if (i >= field.size() || field[i] != ',') {
            error = "hit_curve segment expects ','";
            return false;
        }
        ++i;
        if (!parse_uint(segment.run_length)) {
            error = "hit_curve segment length is not a valid integer";
            return false;
        }
        skip_spaces();
        if (i >= field.size() || field[i] != ']') {
            error = "hit_curve segment must end with ']'";
            return false;
        }
        ++i;
        segments.push_back(segment);
        skip_spaces();
        if (i < field.size() && field[i] == ',') {
            ++i;
            continue;
        }
        break;
    }
    skip_spaces();
    if (i >= field.size() || field[i] != ']') {
        error = "hit_curve must end with ']'";
        return false;
    }
    ++i;
    skip_spaces();
    if (i != field.size()) {
        error = "hit_curve has trailing characters";
        return false;
    }
    return true;
}

} // namespace

std::string SerializeLiteHitFactRow(const LiteHitFactRecord &record) {
    std::ostringstream curve;
    curve << '[';
    for (std::size_t i = 0; i < record.fact.hit_curve.size(); ++i) {
        const HitCurveSegment &segment = record.fact.hit_curve[i];
        if (i > 0) {
            curve << ',';
        }
        curve << '[' << segment.start_required_blocks << ',' << segment.run_length << ']';
    }
    curve << ']';

    std::ostringstream row;
    row << QuoteCsvField(record.trace_id) << ',' << QuoteCsvField(record.instance_id) << ',' << record.timestamp_ns
        << ',' << record.input_token_len << ',' << record.block_size_tokens << ',' << record.block_bytes << ",\""
        << curve.str() << '"';
    return row.str();
}

bool ParseLiteHitFactRow(const std::string &line, LiteHitFactRecord &record, std::string &error) {
    std::vector<std::string> fields;
    if (!SplitCsvLine(line, fields, error)) {
        return false;
    }
    if (fields.size() != 7) {
        error = "expected 7 fields, got " + std::to_string(fields.size());
        return false;
    }
    record.trace_id = fields[0];
    record.instance_id = fields[1];
    return ParseInt64Field(fields[2], record.timestamp_ns, "timestamp_ns", error) &&
           ParseUint64Field(fields[3], record.input_token_len, "input_token_len", error) &&
           ParseUint64Field(fields[4], record.block_size_tokens, "block_size_tokens", error) &&
           ParseUint64Field(fields[5], record.block_bytes, "block_bytes", error) &&
           ParseHitCurve(fields[6], record.fact.hit_curve, error);
}

} // namespace kv_cache_manager
