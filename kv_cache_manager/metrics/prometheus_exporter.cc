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

namespace {

// Check if a metric name is a histogram _bucket metric.
// Returns the family name (without _bucket suffix) if yes, empty string otherwise.
// Only _bucket is used for identification because:
// 1. It's unique to histograms (no regular counter ends with _bucket)
// 2. Alphabetically _bucket < _count < _sum, so _bucket is always seen first
std::string ExtractHistogramFamilyFromBucket(const std::string &name) {
    static const std::string kBucketSuffix = "_bucket";
    if (name.size() > kBucketSuffix.size() &&
        name.compare(name.size() - kBucketSuffix.size(), kBucketSuffix.size(), kBucketSuffix) == 0) {
        return name.substr(0, name.size() - kBucketSuffix.size());
    }
    return "";
}

// Check if a metric name belongs to a known histogram family (by _count or _sum suffix).
// Returns true if the name ends with _count or _sum AND the family is in the known set.
bool IsHistogramSubMetric(const std::string &name, const std::string &family) {
    if (family.empty())
        return false;
    return (name == family + "_count" || name == family + "_sum");
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
    std::ostringstream ss;

    std::string prev_name;
    std::string histogram_family; // set when we see a _bucket metric; used for _count/_sum
    bool family_header_written = false;
    for (const auto &[name, tags, val] : all_metrics) {
        if (val == nullptr || !val->touched.load(std::memory_order_relaxed)) {
            continue;
        }

        std::string prom_name = SanitizeName(prefix, name);

        // Detect histogram: _bucket sets the family; _count/_sum check against it
        std::string bucket_family = ExtractHistogramFamilyFromBucket(name);
        bool is_histogram_bucket = !bucket_family.empty();
        bool is_histogram_sub = IsHistogramSubMetric(name, histogram_family);

        if (name != prev_name) {
            family_header_written = false;
            prev_name = name;
        }

        if (!family_header_written) {
            if (is_histogram_bucket) {
                histogram_family = bucket_family;
                std::string family_prom_name = SanitizeName(prefix, histogram_family);
                ss << "# HELP " << family_prom_name << ' ' << histogram_family << '\n';
                ss << "# TYPE " << family_prom_name << " histogram\n";
            } else if (is_histogram_sub) {
                // _count or _sum of known histogram family — no header needed
            } else {
                histogram_family.clear();
                const char *type_str = "untyped";
                if (std::holds_alternative<CounterValue>(val->value)) {
                    type_str = "counter";
                } else if (std::holds_alternative<GaugeValue>(val->value)) {
                    type_str = "gauge";
                }
                ss << "# HELP " << prom_name << ' ' << name << '\n';
                ss << "# TYPE " << prom_name << ' ' << type_str << '\n';
            }
            family_header_written = true;
        }

        ss << prom_name;
        WriteLabels(ss, tags);

        if (std::holds_alternative<CounterValue>(val->value)) {
            uint64_t raw = std::get<CounterValue>(val->value).load(std::memory_order_relaxed);
            if (is_histogram_sub && name.size() > 4 && name.compare(name.size() - 4, 4, "_sum") == 0) {
                double seconds = static_cast<double>(raw) / 1e6;
                ss << ' ' << seconds;
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
