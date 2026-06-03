#include "kv_cache_manager/affinity/pipeline/candidate_pipeline.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "kv_cache_manager/affinity/pipeline/metric_catalog.h"

namespace kv_cache_manager {

namespace {

void SetError(std::string *out, std::string msg) {
    if (out != nullptr) {
        *out = std::move(msg);
    }
}

bool ParsePreferLocal(const rapidjson::Value &v, PreferLocalSpec *out, std::string *err) {
    if (!v.IsObject()) {
        SetError(err, "prefer_local must be a JSON object");
        return false;
    }
    out->on_miss = PreferLocalSpec::OnMiss::kPassthrough;
    auto it = v.FindMember("on_miss");
    if (it != v.MemberEnd()) {
        if (!it->value.IsString()) {
            SetError(err, "prefer_local.on_miss must be a string");
            return false;
        }
        std::string s = it->value.GetString();
        if (s == "passthrough") {
            out->on_miss = PreferLocalSpec::OnMiss::kPassthrough;
        } else if (s == "abort") {
            out->on_miss = PreferLocalSpec::OnMiss::kAbort;
        } else {
            SetError(err, "prefer_local.on_miss must be \"passthrough\" or \"abort\"");
            return false;
        }
    }
    return true;
}

bool ParseSample(const rapidjson::Value &v, SampleSpec *out, std::string *err) {
    if (!v.IsObject()) {
        SetError(err, "sample must be a JSON object");
        return false;
    }
    auto n_it = v.FindMember("n");
    if (n_it == v.MemberEnd() || !n_it->value.IsInt()) {
        SetError(err, "sample.n is required and must be an integer");
        return false;
    }
    int n = n_it->value.GetInt();
    if (n < 1) {
        SetError(err, "sample.n must be >= 1");
        return false;
    }
    out->n = n;

    auto p_it = v.FindMember("node_pattern");
    if (p_it != v.MemberEnd()) {
        if (!p_it->value.IsString()) {
            SetError(err, "sample.node_pattern must be a string");
            return false;
        }
        out->node_pattern_src = p_it->value.GetString();
        try {
            out->node_pattern = std::regex(out->node_pattern_src);
        } catch (const std::regex_error &e) {
            SetError(err, std::string("sample.node_pattern is not a valid regex: ") + e.what());
            return false;
        }
    }

    out->seed = SampleSpec::Seed::kRandom;
    auto s_it = v.FindMember("seed");
    if (s_it != v.MemberEnd()) {
        if (!s_it->value.IsString()) {
            SetError(err, "sample.seed must be a string");
            return false;
        }
        std::string s = s_it->value.GetString();
        if (s == "random") {
            out->seed = SampleSpec::Seed::kRandom;
        } else if (s == "trace_id") {
            out->seed = SampleSpec::Seed::kTraceId;
        } else {
            SetError(err, "sample.seed must be \"random\" or \"trace_id\"");
            return false;
        }
    }
    return true;
}

bool ParseSort(const rapidjson::Value &v, std::vector<SortTerm> *out, std::string *err) {
    if (!v.IsArray() || v.Empty()) {
        SetError(err, "sort must be a non-empty array");
        return false;
    }
    out->reserve(v.Size());
    for (const auto &elem : v.GetArray()) {
        if (!elem.IsObject()) {
            SetError(err, "sort entry must be an object");
            return false;
        }
        auto m_it = elem.FindMember("metric");
        auto w_it = elem.FindMember("weight");
        if (m_it == elem.MemberEnd() || !m_it->value.IsString()) {
            SetError(err, "sort.metric is required and must be a string");
            return false;
        }
        if (w_it == elem.MemberEnd() || !w_it->value.IsNumber()) {
            SetError(err, "sort.weight is required and must be a number");
            return false;
        }
        std::string metric = m_it->value.GetString();
        if (!MetricCatalog::IsKnown(metric)) {
            SetError(err, "sort.metric \"" + metric + "\" is not a registered metric");
            return false;
        }
        out->push_back(SortTerm{std::move(metric), w_it->value.GetDouble()});
    }
    return true;
}

uint64_t HashTraceId(const std::string &trace_id) {
    // Stable FNV-1a 64-bit; deterministic across runs and small.
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : trace_id) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

void ApplyPreferLocal(const PreferLocalSpec &spec,
                      const std::vector<std::string> &input,
                      const std::string &caller_node_id,
                      CandidatePipeline::ApplyResult *result) {
    bool found = false;
    std::vector<std::string> locals;
    locals.reserve(input.size());
    for (const auto &id : input) {
        if (!caller_node_id.empty() && id == caller_node_id) {
            locals.push_back(id);
            found = true;
        }
    }
    if (found) {
        result->nodes = std::move(locals);
        return;
    }
    if (spec.on_miss == PreferLocalSpec::OnMiss::kAbort) {
        result->status = CandidatePipeline::Status::kAbort;
        result->nodes.clear();
        return;
    }
    // passthrough: leave result->nodes as the input.
}

void ApplySample(const SampleSpec &spec,
                 const std::vector<std::string> &input,
                 const std::function<const NodeMetrics *(const std::string &)> &find_metrics,
                 const std::string &trace_id,
                 std::vector<std::string> *out) {
    std::vector<std::string> pool;
    pool.reserve(input.size());
    for (const auto &id : input) {
        if (spec.node_pattern.has_value()) {
            const NodeMetrics *m = find_metrics ? find_metrics(id) : nullptr;
            const std::string &name = (m != nullptr) ? m->node_name : std::string();
            if (!std::regex_match(name, *spec.node_pattern)) {
                continue;
            }
        }
        pool.push_back(id);
    }
    if (static_cast<int>(pool.size()) <= spec.n) {
        *out = std::move(pool);
        return;
    }
    std::mt19937_64 rng;
    if (spec.seed == SampleSpec::Seed::kTraceId) {
        rng.seed(HashTraceId(trace_id));
    } else {
        std::random_device rd;
        rng.seed((static_cast<uint64_t>(rd()) << 32) ^ rd());
    }
    std::shuffle(pool.begin(), pool.end(), rng);
    pool.resize(spec.n);
    *out = std::move(pool);
}

void ApplySort(const std::vector<SortTerm> &terms,
               const std::vector<std::string> &input,
               const std::function<const NodeMetrics *(const std::string &)> &find_metrics,
               std::vector<std::string> *out) {
    struct Scored {
        std::string id;
        double score;
        std::size_t orig_index;
    };
    std::vector<Scored> scored;
    scored.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        const auto &id = input[i];
        const NodeMetrics *m = find_metrics ? find_metrics(id) : nullptr;
        double score = 0.0;
        if (m != nullptr) {
            for (const auto &t : terms) {
                auto v = MetricCatalog::Extract(t.metric, *m);
                if (v.has_value()) {
                    score += *v * t.weight;
                }
            }
        }
        scored.push_back(Scored{id, score, i});
    }
    std::stable_sort(scored.begin(), scored.end(), [](const Scored &a, const Scored &b) {
        return a.score > b.score; // descending
    });
    out->clear();
    out->reserve(scored.size());
    for (auto &s : scored) {
        out->push_back(std::move(s.id));
    }
}

} // namespace

std::unique_ptr<CandidatePipeline> CandidatePipeline::Parse(const rapidjson::Value &value, std::string *err) {
    if (!value.IsObject()) {
        SetError(err, "strategy must be a JSON object");
        return nullptr;
    }
    static constexpr const char *kKnownKeys[] = {"filter", "prefer_local", "sample", "sort", "limit"};
    for (auto it = value.MemberBegin(); it != value.MemberEnd(); ++it) {
        std::string_view name(it->name.GetString(), it->name.GetStringLength());
        bool ok = false;
        for (const char *k : kKnownKeys) {
            if (name == k) {
                ok = true;
                break;
            }
        }
        if (!ok) {
            SetError(err, std::string("unknown strategy slot: ") + std::string(name));
            return nullptr;
        }
    }

    auto strat = std::make_unique<CandidatePipeline>();

    if (auto it = value.FindMember("filter"); it != value.MemberEnd()) {
        auto cond = FilterCond::Parse(it->value, err);
        if (!cond) {
            return nullptr;
        }
        strat->filter = std::move(cond);
    }

    if (auto it = value.FindMember("prefer_local"); it != value.MemberEnd()) {
        PreferLocalSpec spec;
        if (!ParsePreferLocal(it->value, &spec, err)) {
            return nullptr;
        }
        strat->prefer_local = spec;
    }

    if (auto it = value.FindMember("sample"); it != value.MemberEnd()) {
        SampleSpec spec;
        if (!ParseSample(it->value, &spec, err)) {
            return nullptr;
        }
        strat->sample = std::move(spec);
    }

    if (auto it = value.FindMember("sort"); it != value.MemberEnd()) {
        std::vector<SortTerm> terms;
        if (!ParseSort(it->value, &terms, err)) {
            return nullptr;
        }
        strat->sort = std::move(terms);
    }

    if (auto it = value.FindMember("limit"); it != value.MemberEnd()) {
        if (!it->value.IsInt()) {
            SetError(err, "limit must be an integer");
            return nullptr;
        }
        int n = it->value.GetInt();
        if (n < 1) {
            SetError(err, "limit must be >= 1");
            return nullptr;
        }
        strat->limit = n;
    }

    return strat;
}

namespace {
const rapidjson::Value *UnwrapEnvelope(const rapidjson::Document &doc) {
    if (doc.IsObject()) {
        auto it = doc.FindMember("strategy");
        if (it != doc.MemberEnd()) {
            return &it->value;
        }
    }
    return &doc;
}
} // namespace

std::unique_ptr<CandidatePipeline> CandidatePipeline::ParseJsonString(const std::string &json, std::string *err) {
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    if (doc.HasParseError()) {
        SetError(err, "JSON parse error");
        return nullptr;
    }
    return CandidatePipeline::Parse(*UnwrapEnvelope(doc), err);
}

CandidatePipeline::ApplyResult
CandidatePipeline::Apply(const std::vector<std::string> &candidates,
                         const std::function<const NodeMetrics *(const std::string &)> &find_metrics,
                         const std::string &caller_node_id,
                         const std::string &trace_id) const {
    ApplyResult r;
    r.nodes = candidates;

    if (filter.has_value()) {
        r.nodes = (*filter)->Apply(r.nodes, find_metrics);
    }

    if (prefer_local.has_value()) {
        ApplyPreferLocal(*prefer_local, r.nodes, caller_node_id, &r);
        if (r.status == Status::kAbort) {
            return r;
        }
    }

    if (sample.has_value()) {
        std::vector<std::string> next;
        ApplySample(*sample, r.nodes, find_metrics, trace_id, &next);
        r.nodes = std::move(next);
    }

    if (sort.has_value()) {
        std::vector<std::string> next;
        ApplySort(*sort, r.nodes, find_metrics, &next);
        r.nodes = std::move(next);
    }

    if (limit.has_value() && static_cast<int>(r.nodes.size()) > *limit) {
        r.nodes.resize(*limit);
    }

    return r;
}

} // namespace kv_cache_manager
