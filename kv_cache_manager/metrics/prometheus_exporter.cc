#include "kv_cache_manager/metrics/prometheus_exporter.h"

#include <cmath>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "kv_cache_manager/metrics/metrics_registry.h"

namespace kv_cache_manager {

namespace {

// replace characters that are invalid in a prometheus identifier
// with underscores; keeps only [a-zA-Z0-9_]
// a leading digit is prefixed with '_' since both metric names and
// label names require the first character to match [a-zA-Z_]
//
// WARN:
// distinct raw strings that differ only in characters replaced by '_'
// (e.g. "storage.type" vs "storage_type") will produce the same output
// callers must avoid collisions after sanitization in all contexts:
// - metric names: duplicates cause repeated HELP/TYPE lines
// - label keys: duplicates are invalid in prometheus text format
// - prefixes: same collision risk as metric names
std::string SanitizeIdentifier(const std::string &raw) {
    std::string out;
    out.reserve(raw.size() + 1);
    for (std::size_t i = 0; i < raw.size(); ++i) {
        char c = raw[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
            out += c;
        } else if (c >= '0' && c <= '9') {
            if (i == 0) {
                out += '_';
            }
            out += c;
        } else {
            out += '_';
        }
    }
    return out;
}

// build a prometheus metric name: prefix + '_' + sanitized(raw_name)
std::string SanitizeName(const std::string &prefix, const std::string &raw_name) {
    std::string out;
    out.reserve(prefix.size() + 1 + raw_name.size());
    out += SanitizeIdentifier(prefix);
    out += '_';
    out += SanitizeIdentifier(raw_name);
    return out;
}

// prometheus label values use double-quoted strings; backslash,
// double-quote and newline must be escaped
void EscapeLabelValue(std::ostringstream &ss, const std::string &v) {
    for (char c : v) {
        switch (c) {
        case '\\':
            ss << "\\\\";
            break;
        case '"':
            ss << "\\\"";
            break;
        case '\n':
            ss << "\\n";
            break;
        default:
            ss << c;
        }
    }
}

void WriteLabels(std::ostringstream &ss, const MetricsTags &tags) {
    if (tags.empty()) {
        return;
    }
    ss << '{';
    bool first = true;
    for (const auto &[k, v] : tags) {
        if (!first) {
            ss << ',';
        }
        first = false;
        ss << SanitizeIdentifier(k) << "=\"";
        EscapeLabelValue(ss, v);
        ss << '"';
    }
    ss << '}';
}

} // namespace

std::string PrometheusExporter::Expose(MetricsRegistry &registry, const std::string &prefix) {
    std::vector<MetricsRegistry::metrics_tuple_t> all_metrics;
    registry.GetAllMetrics(all_metrics);

    // Group metrics by name so we can emit TYPE / HELP lines once per
    // metric family.
    //
    // GetAllMetrics returns entries ordered by name (the underlying
    // map is std::map<string, ...>), so consecutive entries with the
    // same name belong to the same family.
    std::ostringstream ss;

    std::string prev_name;
    for (const auto &[name, tags, val] : all_metrics) {
        std::string prom_name = SanitizeName(prefix, name);

        if (name != prev_name) {
            const char *type_str = "untyped";
            if (std::holds_alternative<CounterValue>(*val)) {
                type_str = "counter";
            } else if (std::holds_alternative<GaugeValue>(*val)) {
                type_str = "gauge";
            }
            ss << "# HELP " << prom_name << ' ' << name << '\n';
            ss << "# TYPE " << prom_name << ' ' << type_str << '\n';
            prev_name = name;
        }

        ss << prom_name;
        WriteLabels(ss, tags);

        if (std::holds_alternative<CounterValue>(*val)) {
            ss << ' ' << std::get<CounterValue>(*val).load(std::memory_order_relaxed);
        } else if (std::holds_alternative<GaugeValue>(*val)) {
            double gv = std::get<GaugeValue>(*val).load(std::memory_order_relaxed);
            if (std::isnan(gv)) {
                ss << " NaN";
            } else if (std::isinf(gv)) {
                ss << ' ' << (gv > 0 ? "+Inf" : "-Inf");
            } else {
                ss << ' ' << gv;
            }
        }
        ss << '\n';
    }

    return ss.str();
}

} // namespace kv_cache_manager
