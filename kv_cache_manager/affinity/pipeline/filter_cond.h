#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "kv_cache_manager/affinity/node_metrics.h"
#include "rapidjson/document.h"

namespace kv_cache_manager {

// Recursive condition tree used by CandidatePipeline.filter.
//
// Grammar (one object per node, one of `and / or / metric / node_name` is
// the dispatch key):
//
//   Cond ::=
//     | { "and":       [Cond, Cond, ...] }
//     | { "or":        [Cond, Cond, ...] }
//     | { "metric":    "<name>", "min"?: <num>, "max"?: <num> }
//     | { "node_name": { "include"?: [<regex>...],
//                        "exclude"?: [<regex>...] } }
//
// Missing-metric semantics: a `metric` leaf evaluates to true when the
// candidate has no observation for the named metric. AND / OR therefore
// degrade gracefully when only some metrics are visible (a permissive
// filter that never punishes a candidate for lacking data).
//
// Parse errors (rejected by Parse(), surfaced via error_msg):
//   - object that has zero or more than one of and/or/metric/node_name
//   - and/or with empty array
//   - metric leaf with neither min nor max
//   - metric leaf naming an unregistered metric
//   - node_name leaf with neither include nor exclude
//   - non-string entry in include/exclude
//   - regex string that fails std::regex compilation
class FilterCond {
public:
    virtual ~FilterCond() = default;
    virtual bool Eval(const NodeMetrics *metrics) const = 0;

    // Apply this condition to `candidates`. `find_metrics` returns the metrics
    // for a given candidate, or nullptr if unavailable; passing nullptr is
    // legal and means "treat every candidate as missing".
    std::vector<std::string> Apply(const std::vector<std::string> &candidates,
                                   const std::function<const NodeMetrics *(const std::string &)> &find_metrics) const;

    // Build a FilterCond from a rapidjson value. Returns nullptr on parse
    // failure with `error_msg` populated when non-null.
    static std::unique_ptr<FilterCond> Parse(const rapidjson::Value &value, std::string *error_msg);
};

// AND of one or more children. Empty children => parse error (rejected by
// Parse).
class AndCond : public FilterCond {
public:
    explicit AndCond(std::vector<std::unique_ptr<FilterCond>> children) : children_(std::move(children)) {}
    bool Eval(const NodeMetrics *metrics) const override;

private:
    std::vector<std::unique_ptr<FilterCond>> children_;
};

class OrCond : public FilterCond {
public:
    explicit OrCond(std::vector<std::unique_ptr<FilterCond>> children) : children_(std::move(children)) {}
    bool Eval(const NodeMetrics *metrics) const override;

private:
    std::vector<std::unique_ptr<FilterCond>> children_;
};

// Range check on a registered metric. min / max are both optional but at
// least one must be present (enforced by Parse).
class MetricCond : public FilterCond {
public:
    MetricCond(std::string metric, std::optional<double> min_v, std::optional<double> max_v)
        : metric_(std::move(metric)), min_(min_v), max_(max_v) {}
    bool Eval(const NodeMetrics *metrics) const override;

private:
    std::string metric_;
    std::optional<double> min_;
    std::optional<double> max_;
};

// Regex include / exclude on NodeMetrics::node_name. include empty means
// "any name passes the include test"; exclude empty means "no name fails the
// exclude test". At least one of the two arrays must be non-empty (enforced
// by Parse).
class NodeNameCond : public FilterCond {
public:
    NodeNameCond(std::vector<std::regex> include, std::vector<std::regex> exclude)
        : include_(std::move(include)), exclude_(std::move(exclude)), has_include_(!include_.empty()) {}
    bool Eval(const NodeMetrics *metrics) const override;

private:
    std::vector<std::regex> include_;
    std::vector<std::regex> exclude_;
    bool has_include_ = false;
};

} // namespace kv_cache_manager
