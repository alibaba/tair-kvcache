#include "kv_cache_manager/affinity/filter_cond.h"

#include <utility>

#include "kv_cache_manager/affinity/metric_registry.h"

namespace kv_cache_manager {

namespace {

void SetError(std::string *out, std::string msg) {
    if (out != nullptr) {
        *out = std::move(msg);
    }
}

bool ReadOptionalNumber(const rapidjson::Value &obj, const char *key, std::optional<double> *out, std::string *err) {
    auto it = obj.FindMember(key);
    if (it == obj.MemberEnd()) {
        return true;
    }
    if (!it->value.IsNumber()) {
        SetError(err, std::string("filter.metric.") + key + " must be a number");
        return false;
    }
    *out = it->value.GetDouble();
    return true;
}

bool ParseRegexList(const rapidjson::Value &arr, const char *field, std::vector<std::regex> *out, std::string *err) {
    if (!arr.IsArray()) {
        SetError(err, std::string("filter.node_name.") + field + " must be an array");
        return false;
    }
    out->reserve(arr.Size());
    for (const auto &elem : arr.GetArray()) {
        if (!elem.IsString()) {
            SetError(err, std::string("filter.node_name.") + field + " entries must be strings");
            return false;
        }
        try {
            out->emplace_back(elem.GetString());
        } catch (const std::regex_error &e) {
            SetError(err,
                     std::string("filter.node_name.") + field + " has invalid regex \"" + elem.GetString() +
                         "\": " + e.what());
            return false;
        }
    }
    return true;
}

std::unique_ptr<FilterCond> ParseAndOr(const rapidjson::Value &arr, bool is_and, std::string *err) {
    if (!arr.IsArray()) {
        SetError(err, std::string("filter.") + (is_and ? "and" : "or") + " must be an array");
        return nullptr;
    }
    if (arr.Empty()) {
        SetError(err, std::string("filter.") + (is_and ? "and" : "or") + " must not be empty");
        return nullptr;
    }
    std::vector<std::unique_ptr<FilterCond>> children;
    children.reserve(arr.Size());
    for (const auto &elem : arr.GetArray()) {
        auto child = FilterCond::Parse(elem, err);
        if (!child) {
            return nullptr;
        }
        children.push_back(std::move(child));
    }
    if (is_and) {
        return std::make_unique<AndCond>(std::move(children));
    }
    return std::make_unique<OrCond>(std::move(children));
}

std::unique_ptr<FilterCond> ParseMetric(const rapidjson::Value &obj, std::string *err) {
    auto it = obj.FindMember("metric");
    if (!it->value.IsString()) {
        SetError(err, "filter.metric must be a string");
        return nullptr;
    }
    std::string name = it->value.GetString();
    if (!MetricRegistry::IsKnown(name)) {
        SetError(err, "filter.metric \"" + name + "\" is not a registered metric");
        return nullptr;
    }
    std::optional<double> min_v;
    std::optional<double> max_v;
    if (!ReadOptionalNumber(obj, "min", &min_v, err)) {
        return nullptr;
    }
    if (!ReadOptionalNumber(obj, "max", &max_v, err)) {
        return nullptr;
    }
    if (!min_v.has_value() && !max_v.has_value()) {
        SetError(err, "filter.metric \"" + name + "\" requires at least one of min / max");
        return nullptr;
    }
    return std::make_unique<MetricCond>(std::move(name), min_v, max_v);
}

std::unique_ptr<FilterCond> ParseNodeName(const rapidjson::Value &val, std::string *err) {
    if (!val.IsObject()) {
        SetError(err, "filter.node_name must be an object");
        return nullptr;
    }
    std::vector<std::regex> include;
    std::vector<std::regex> exclude;
    auto inc_it = val.FindMember("include");
    if (inc_it != val.MemberEnd()) {
        if (!ParseRegexList(inc_it->value, "include", &include, err)) {
            return nullptr;
        }
    }
    auto exc_it = val.FindMember("exclude");
    if (exc_it != val.MemberEnd()) {
        if (!ParseRegexList(exc_it->value, "exclude", &exclude, err)) {
            return nullptr;
        }
    }
    if (include.empty() && exclude.empty()) {
        SetError(err, "filter.node_name requires at least one of include / exclude");
        return nullptr;
    }
    return std::make_unique<NodeNameCond>(std::move(include), std::move(exclude));
}

} // namespace

std::vector<std::string>
FilterCond::Apply(const std::vector<std::string> &candidates,
                  const std::function<const NodeMetrics *(const std::string &)> &find_metrics) const {
    std::vector<std::string> kept;
    kept.reserve(candidates.size());
    for (const auto &id : candidates) {
        const NodeMetrics *m = find_metrics ? find_metrics(id) : nullptr;
        if (Eval(m)) {
            kept.push_back(id);
        }
    }
    return kept;
}

std::unique_ptr<FilterCond> FilterCond::Parse(const rapidjson::Value &value, std::string *err) {
    if (!value.IsObject()) {
        SetError(err, "filter condition must be a JSON object");
        return nullptr;
    }
    static constexpr const char *kKeys[] = {"and", "or", "metric", "node_name"};
    int hits = 0;
    const char *dispatch = nullptr;
    for (const char *k : kKeys) {
        if (value.HasMember(k)) {
            ++hits;
            dispatch = k;
        }
    }
    if (hits != 1) {
        SetError(err, "filter condition must contain exactly one of and / or / metric / node_name");
        return nullptr;
    }
    const rapidjson::Value &inner = value[dispatch];
    if (dispatch == std::string("and")) {
        return ParseAndOr(inner, /*is_and=*/true, err);
    }
    if (dispatch == std::string("or")) {
        return ParseAndOr(inner, /*is_and=*/false, err);
    }
    if (dispatch == std::string("metric")) {
        return ParseMetric(value, err);
    }
    return ParseNodeName(inner, err);
}

bool AndCond::Eval(const NodeMetrics *m) const {
    for (const auto &c : children_) {
        if (!c->Eval(m)) {
            return false;
        }
    }
    return true;
}

bool OrCond::Eval(const NodeMetrics *m) const {
    for (const auto &c : children_) {
        if (c->Eval(m)) {
            return true;
        }
    }
    return false;
}

bool MetricCond::Eval(const NodeMetrics *m) const {
    if (m == nullptr) {
        return true; // missing-metric => permissive
    }
    auto v = MetricRegistry::Extract(metric_, *m);
    if (!v.has_value()) {
        return true;
    }
    if (min_.has_value() && *v < *min_) {
        return false;
    }
    if (max_.has_value() && *v > *max_) {
        return false;
    }
    return true;
}

bool NodeNameCond::Eval(const NodeMetrics *m) const {
    // No metrics => no node_name observation. Treat as permissive (missing
    // semantics consistent with metric leaves).
    if (m == nullptr) {
        return true;
    }
    const std::string &name = m->node_name;
    if (has_include_) {
        bool any = false;
        for (const auto &re : include_) {
            if (std::regex_match(name, re)) {
                any = true;
                break;
            }
        }
        if (!any) {
            return false;
        }
    }
    for (const auto &re : exclude_) {
        if (std::regex_match(name, re)) {
            return false;
        }
    }
    return true;
}

} // namespace kv_cache_manager
