#include "kv_cache_manager/metrics/prometheus_exporter.h"

#include <cmath>
#include <set>
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
    //
    // Untouched series are skipped: the Service collector registers
    // every (manager.*, meta_searcher.*, meta_indexer.*) series for
    // every api_name, but most combinations are never written. Without
    // filtering, /metrics is dominated by zero-valued series that
    // mostly inflate Prometheus cardinality. A series becomes touched
    // the first time any write path on Counter/Gauge runs against it.
    // If every series in a family is untouched, the # HELP / # TYPE
    // header is omitted as well.
    //
    // Histogram families are identified via explicit metric→family mapping
    // registered by RevisitIntervalHistogram::Init(). The exporter queries
    // the registry — no suffix-based guessing.
    std::ostringstream ss;

    std::string prev_name;
    std::string current_family;             // histogram family for current metric (empty if regular)
    std::set<std::string> families_written; // histogram families whose TYPE header was emitted
    bool name_header_written = false;       // tracks TYPE/HELP for regular (non-histogram) metrics
    for (const auto &[name, tags, val] : all_metrics) {
        if (val == nullptr || !val->touched.load(std::memory_order_relaxed)) {
            continue;
        }

        std::string prom_name = SanitizeName(prefix, name);

        if (name != prev_name) {
            prev_name = name;
            name_header_written = false;
            current_family = registry.GetMetricFamily(name);
        }

        if (!name_header_written) {
            if (!current_family.empty()) {
                // Histogram metric: emit TYPE/HELP once per family
                if (families_written.insert(current_family).second) {
                    std::string family_prom_name = SanitizeName(prefix, current_family);
                    ss << "# HELP " << family_prom_name << ' ' << current_family << '\n';
                    ss << "# TYPE " << family_prom_name << " histogram\n";
                }
            } else {
                // Regular counter/gauge
                const char *type_str = "untyped";
                if (std::holds_alternative<CounterValue>(val->value)) {
                    type_str = "counter";
                } else if (std::holds_alternative<GaugeValue>(val->value)) {
                    type_str = "gauge";
                }
                ss << "# HELP " << prom_name << ' ' << name << '\n';
                ss << "# TYPE " << prom_name << ' ' << type_str << '\n';
            }
            name_header_written = true;
        }

        ss << prom_name;
        WriteLabels(ss, tags);

        if (std::holds_alternative<CounterValue>(val->value)) {
            uint64_t raw = std::get<CounterValue>(val->value).load(std::memory_order_relaxed);
            // Histogram _sum stores microseconds; convert to seconds for Prometheus
            static const std::string kSumSuffix = "_sum";
            if (!current_family.empty() && name.size() >= kSumSuffix.size() &&
                name.compare(name.size() - kSumSuffix.size(), kSumSuffix.size(), kSumSuffix) == 0) {
                ss << ' ' << static_cast<double>(raw) / 1e6;
            } else {
                ss << ' ' << raw;
            }
        } else if (std::holds_alternative<GaugeValue>(val->value)) {
            double gv = std::get<GaugeValue>(val->value).load(std::memory_order_relaxed);
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
