#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/affinity/cache_affinity_manager.h"
#include "kv_cache_manager/affinity/node_metrics.h"
#include "kv_cache_manager/affinity/pipeline/candidate_pipeline.h"
#include "kv_cache_manager/affinity/pipeline/filter_cond.h"
#include "kv_cache_manager/common/affinity_types.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

namespace {

NodeMetrics MakeNode(const std::string &id,
                     const std::string &name,
                     int64_t free_bytes = 1LL << 40,
                     double load = 0.0,
                     double rx = 0.0,
                     double tx = 0.0) {
    NodeMetrics m;
    m.node_id = id;
    m.node_name = name;
    m.free_bytes = free_bytes;
    m.load_ratio = load;
    m.rx_mbps = rx;
    m.tx_mbps = tx;
    return m;
}

class MetricsBag {
public:
    void Put(NodeMetrics m) { map_[m.node_id] = std::move(m); }
    const NodeMetrics *Find(const std::string &id) const {
        auto it = map_.find(id);
        return it == map_.end() ? nullptr : &it->second;
    }
    std::function<const NodeMetrics *(const std::string &)> AsFinder() const {
        return [this](const std::string &id) { return Find(id); };
    }

private:
    std::unordered_map<std::string, NodeMetrics> map_;
};

std::unique_ptr<FilterCond> ParseCond(const std::string &json, std::string *err = nullptr) {
    rapidjson::Document doc;
    doc.Parse(json.c_str());
    EXPECT_FALSE(doc.HasParseError()) << "test JSON failed to parse: " << json;
    return FilterCond::Parse(doc, err);
}

std::unique_ptr<CandidatePipeline> ParseStrategy(const std::string &json, std::string *err = nullptr) {
    return CandidatePipeline::ParseJsonString(json, err);
}

} // namespace

// ============================ FilterCond =================================

class FilterCondTest : public TESTBASE {};

TEST_F(FilterCondTest, MetricMinMaxBoundsApplied) {
    auto c = ParseCond(R"({"metric": "free_bytes", "min": 100, "max": 1000})");
    ASSERT_NE(c, nullptr);
    auto small = MakeNode("a", "n", /*free_bytes=*/50);
    auto mid = MakeNode("b", "n", /*free_bytes=*/500);
    auto big = MakeNode("c", "n", /*free_bytes=*/2000);
    EXPECT_FALSE(c->Eval(&small));
    EXPECT_TRUE(c->Eval(&mid));
    EXPECT_FALSE(c->Eval(&big));
}

TEST_F(FilterCondTest, MetricMissingMetricsTreatedAsTrue) {
    auto c = ParseCond(R"({"metric": "free_bytes", "min": 100})");
    ASSERT_NE(c, nullptr);
    EXPECT_TRUE(c->Eval(nullptr)) << "missing metrics must be permissive";
}

TEST_F(FilterCondTest, MetricRequiresMinOrMax) {
    std::string err;
    auto c = ParseCond(R"({"metric": "free_bytes"})", &err);
    EXPECT_EQ(c, nullptr);
    EXPECT_FALSE(err.empty());
}

TEST_F(FilterCondTest, MetricRejectsUnknownName) {
    std::string err;
    auto c = ParseCond(R"({"metric": "bogus_metric", "min": 0})", &err);
    EXPECT_EQ(c, nullptr);
    EXPECT_NE(err.find("not a registered metric"), std::string::npos);
}

TEST_F(FilterCondTest, NodeNameInclude) {
    auto c = ParseCond(R"({"node_name": {"include": ["^gpu-.*$"]}})");
    ASSERT_NE(c, nullptr);
    auto in = MakeNode("a", "gpu-1");
    auto out = MakeNode("b", "cpu-1");
    EXPECT_TRUE(c->Eval(&in));
    EXPECT_FALSE(c->Eval(&out));
}

TEST_F(FilterCondTest, NodeNameExclude) {
    auto c = ParseCond(R"({"node_name": {"exclude": ["^cpu-.*$"]}})");
    ASSERT_NE(c, nullptr);
    auto a = MakeNode("a", "gpu-1");
    auto b = MakeNode("b", "cpu-1");
    EXPECT_TRUE(c->Eval(&a));
    EXPECT_FALSE(c->Eval(&b));
}

TEST_F(FilterCondTest, NodeNameIncludeAndExcludeBothApplied) {
    auto c = ParseCond(R"({"node_name": {"include": ["^gpu-.*$"], "exclude": [".*-bad$"]}})");
    ASSERT_NE(c, nullptr);
    auto good = MakeNode("a", "gpu-good");
    auto bad = MakeNode("b", "gpu-bad");
    auto cpu = MakeNode("c", "cpu-1");
    EXPECT_TRUE(c->Eval(&good));
    EXPECT_FALSE(c->Eval(&bad));
    EXPECT_FALSE(c->Eval(&cpu));
}

TEST_F(FilterCondTest, NodeNameRequiresIncludeOrExclude) {
    std::string err;
    auto c = ParseCond(R"({"node_name": {}})", &err);
    EXPECT_EQ(c, nullptr);
    EXPECT_FALSE(err.empty());
}

TEST_F(FilterCondTest, AndAllChildrenMustPass) {
    auto c = ParseCond(R"({
        "and": [
            {"metric": "free_bytes", "min": 100},
            {"metric": "load_ratio", "max": 0.5}
        ]
    })");
    ASSERT_NE(c, nullptr);
    auto pass = MakeNode("a", "n", /*free=*/200, /*load=*/0.2);
    auto fail_load = MakeNode("b", "n", /*free=*/200, /*load=*/0.9);
    auto fail_free = MakeNode("c", "n", /*free=*/50, /*load=*/0.2);
    EXPECT_TRUE(c->Eval(&pass));
    EXPECT_FALSE(c->Eval(&fail_load));
    EXPECT_FALSE(c->Eval(&fail_free));
}

TEST_F(FilterCondTest, OrAnyChildPasses) {
    auto c = ParseCond(R"({
        "or": [
            {"metric": "free_bytes", "min": 1000},
            {"node_name": {"include": ["^special-.*$"]}}
        ]
    })");
    ASSERT_NE(c, nullptr);
    auto big = MakeNode("a", "ordinary", /*free=*/2000);
    auto special = MakeNode("b", "special-x", /*free=*/0);
    auto neither = MakeNode("c", "ordinary", /*free=*/0);
    EXPECT_TRUE(c->Eval(&big));
    EXPECT_TRUE(c->Eval(&special));
    EXPECT_FALSE(c->Eval(&neither));
}

TEST_F(FilterCondTest, NestedAndInsideOr) {
    auto c = ParseCond(R"({
        "or": [
            {"and": [
                {"metric": "free_bytes", "min": 1000},
                {"metric": "load_ratio", "max": 0.5}
            ]},
            {"node_name": {"include": ["^pinned-.*$"]}}
        ]
    })");
    ASSERT_NE(c, nullptr);
    auto big_idle = MakeNode("a", "ordinary", /*free=*/2000, /*load=*/0.1);
    auto big_busy = MakeNode("b", "ordinary", /*free=*/2000, /*load=*/0.9);
    auto pinned = MakeNode("c", "pinned-1", /*free=*/0, /*load=*/0.99);
    auto small_busy = MakeNode("d", "ordinary", /*free=*/100, /*load=*/0.9);
    EXPECT_TRUE(c->Eval(&big_idle));
    EXPECT_FALSE(c->Eval(&big_busy));
    EXPECT_TRUE(c->Eval(&pinned));
    EXPECT_FALSE(c->Eval(&small_busy));
}

TEST_F(FilterCondTest, AndEmptyArrayRejected) {
    std::string err;
    auto c = ParseCond(R"({"and": []})", &err);
    EXPECT_EQ(c, nullptr);
    EXPECT_FALSE(err.empty());
}

TEST_F(FilterCondTest, OrEmptyArrayRejected) {
    std::string err;
    auto c = ParseCond(R"({"or": []})", &err);
    EXPECT_EQ(c, nullptr);
    EXPECT_FALSE(err.empty());
}

TEST_F(FilterCondTest, AndSingleChildAccepted) {
    auto c = ParseCond(R"({"and": [{"metric": "free_bytes", "min": 0}]})");
    ASSERT_NE(c, nullptr);
}

TEST_F(FilterCondTest, MultipleDispatchKeysRejected) {
    std::string err;
    auto c = ParseCond(R"({"and": [{"metric": "free_bytes", "min": 0}], "or": []})", &err);
    EXPECT_EQ(c, nullptr);
    EXPECT_FALSE(err.empty());
}

TEST_F(FilterCondTest, NoDispatchKeyRejected) {
    std::string err;
    auto c = ParseCond(R"({"foo": "bar"})", &err);
    EXPECT_EQ(c, nullptr);
    EXPECT_FALSE(err.empty());
}

TEST_F(FilterCondTest, ApplyKeepsMatchingDropsOthers) {
    auto c = ParseCond(R"({"metric": "free_bytes", "min": 1000})");
    ASSERT_NE(c, nullptr);
    MetricsBag bag;
    bag.Put(MakeNode("small", "n", /*free=*/100));
    bag.Put(MakeNode("big", "n", /*free=*/2000));
    auto kept = c->Apply({"small", "big", "unknown"}, bag.AsFinder());
    // unknown has no metrics -> permissive -> kept.
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_EQ(kept[0], "big");
    EXPECT_EQ(kept[1], "unknown");
}

// ============================ CandidatePipeline ===================================

class StrategyParseTest : public TESTBASE {};

TEST_F(StrategyParseTest, AcceptsBareAndEnvelope) {
    EXPECT_NE(ParseStrategy(R"({"limit": 1})"), nullptr);
    EXPECT_NE(ParseStrategy(R"({"strategy": {"limit": 1}})"), nullptr);
}

// The strategy snippets pasted into docs/cache-affinity-{zh_CN,en_US}.md
// must round-trip cleanly through ParseJsonString. If you change the schema
// and these stop parsing, update the docs alongside the parser change.
TEST_F(StrategyParseTest, DocsExample1BasicThreeStage) {
    std::string err;
    auto s = ParseStrategy(R"({
        "strategy": {
            "filter": {
                "and": [
                    { "metric": "free_bytes", "min": 1073741824 },
                    { "metric": "load_ratio", "max": 0.8 }
                ]
            },
            "sort":  [ { "metric": "load_ratio", "weight": -1 } ],
            "limit": 3
        }
    })",
                           &err);
    ASSERT_NE(s, nullptr) << err;
}

TEST_F(StrategyParseTest, DocsExample2WithPreferLocal) {
    std::string err;
    auto s = ParseStrategy(R"({
        "strategy": {
            "filter":       { "metric": "free_bytes", "min": 1073741824 },
            "prefer_local": { "on_miss": "passthrough" },
            "sort":         [ { "metric": "load_ratio", "weight": -1 } ],
            "limit":        3
        }
    })",
                           &err);
    ASSERT_NE(s, nullptr) << err;
}

TEST_F(StrategyParseTest, DocsExample3WithSample) {
    std::string err;
    auto s = ParseStrategy(R"({
        "strategy": {
            "filter": { "metric": "load_ratio", "max": 0.8 },
            "sample": {
                "n": 5,
                "node_pattern": "^gpu-.*$",
                "seed": "trace_id"
            },
            "sort":  [ { "metric": "load_ratio", "weight": -1 } ],
            "limit": 2
        }
    })",
                           &err);
    ASSERT_NE(s, nullptr) << err;
}

TEST_F(StrategyParseTest, DocsSortNegativeWeightSnippet) {
    // The "sort: negative weight for ascending" snippet from the docs.
    std::string err;
    auto s = ParseStrategy(R"({
        "sort": [
            { "metric": "load_ratio", "weight": -1 },
            { "metric": "rx_mbps",    "weight": -0.5 }
        ]
    })",
                           &err);
    ASSERT_NE(s, nullptr) << err;
}

TEST_F(StrategyParseTest, RejectsUnknownTopLevelKey) {
    std::string err;
    auto s = ParseStrategy(R"({"bogus": {}})", &err);
    EXPECT_EQ(s, nullptr);
    EXPECT_NE(err.find("unknown strategy slot"), std::string::npos);
}

TEST_F(StrategyParseTest, EmptyStrategyParsesAndIsIdentity) {
    auto s = ParseStrategy(R"({})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    auto r = s->Apply({"a", "b", "c"}, bag.AsFinder(), CallerNode{}, "");
    EXPECT_EQ(r.status, CandidatePipeline::Status::kOk);
    ASSERT_EQ(r.nodes.size(), 3u);
    EXPECT_EQ(r.nodes[0], "a");
    EXPECT_EQ(r.nodes[1], "b");
    EXPECT_EQ(r.nodes[2], "c");
}

TEST_F(StrategyParseTest, LimitMustBePositive) {
    EXPECT_EQ(ParseStrategy(R"({"limit": 0})"), nullptr);
    EXPECT_EQ(ParseStrategy(R"({"limit": -1})"), nullptr);
}

TEST_F(StrategyParseTest, SortRejectsUnknownMetric) {
    std::string err;
    auto s = ParseStrategy(R"({"sort": [{"metric": "foo", "weight": 1}]})", &err);
    EXPECT_EQ(s, nullptr);
    EXPECT_NE(err.find("not a registered metric"), std::string::npos);
}

TEST_F(StrategyParseTest, SortRequiresNonEmptyArray) { EXPECT_EQ(ParseStrategy(R"({"sort": []})"), nullptr); }

TEST_F(StrategyParseTest, PreferLocalRejectsBadOnMiss) {
    std::string err;
    EXPECT_EQ(ParseStrategy(R"({"prefer_local": {"on_miss": "explode"}})", &err), nullptr);
    EXPECT_FALSE(err.empty());
}

TEST_F(StrategyParseTest, SampleNRequired) {
    EXPECT_EQ(ParseStrategy(R"({"sample": {}})"), nullptr);
    EXPECT_EQ(ParseStrategy(R"({"sample": {"n": 0}})"), nullptr);
}

TEST_F(StrategyParseTest, SampleSeedRejectsBadValue) {
    EXPECT_EQ(ParseStrategy(R"({"sample": {"n": 1, "seed": "rand"}})"), nullptr);
}

// ----- Slot behavior ------------------------------------------------------

class StrategyApplyTest : public TESTBASE {};

TEST_F(StrategyApplyTest, FilterSlotDropsAndKeeps) {
    auto s = ParseStrategy(R"({"filter": {"metric": "load_ratio", "max": 0.5}})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    bag.Put(MakeNode("idle", "n", 0, /*load=*/0.1));
    bag.Put(MakeNode("busy", "n", 0, /*load=*/0.9));
    auto r = s->Apply({"idle", "busy"}, bag.AsFinder(), CallerNode{}, "");
    ASSERT_EQ(r.nodes.size(), 1u);
    EXPECT_EQ(r.nodes[0], "idle");
}

TEST_F(StrategyApplyTest, PreferLocalKeepsLocal) {
    auto s = ParseStrategy(R"({"prefer_local": {"on_miss": "passthrough"}})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    auto r = s->Apply({"10.0.0.1", "10.0.0.7", "10.0.0.8"}, bag.AsFinder(), /*caller=*/CallerNode{"10.0.0.7", ""}, "");
    ASSERT_EQ(r.nodes.size(), 1u);
    EXPECT_EQ(r.nodes[0], "10.0.0.7");
}

TEST_F(StrategyApplyTest, PreferLocalPassthroughOnMiss) {
    auto s = ParseStrategy(R"({"prefer_local": {"on_miss": "passthrough"}})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    auto r = s->Apply({"10.0.0.1", "10.0.0.2"}, bag.AsFinder(), /*caller=*/CallerNode{"10.0.0.99", ""}, "");
    EXPECT_EQ(r.status, CandidatePipeline::Status::kOk);
    EXPECT_EQ(r.nodes.size(), 2u);
}

TEST_F(StrategyApplyTest, PreferLocalAbortOnMiss) {
    auto s = ParseStrategy(R"({"prefer_local": {"on_miss": "abort"}})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    auto r = s->Apply({"10.0.0.1", "10.0.0.2"}, bag.AsFinder(), /*caller=*/CallerNode{"10.0.0.99", ""}, "");
    EXPECT_EQ(r.status, CandidatePipeline::Status::kAbort);
    EXPECT_TRUE(r.nodes.empty());
}

TEST_F(StrategyApplyTest, PreferLocalDefaultsToPassthrough) {
    // No on_miss key.
    auto s = ParseStrategy(R"({"prefer_local": {}})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    auto r = s->Apply({"a", "b"}, bag.AsFinder(), CallerNode{"missing", ""}, "");
    EXPECT_EQ(r.status, CandidatePipeline::Status::kOk);
    EXPECT_EQ(r.nodes.size(), 2u);
}

TEST_F(StrategyApplyTest, SampleNCapsCount) {
    auto s = ParseStrategy(R"({"sample": {"n": 2, "seed": "trace_id"}})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    auto r = s->Apply({"a", "b", "c", "d"}, bag.AsFinder(), CallerNode{}, /*trace=*/"trace-1");
    EXPECT_EQ(r.nodes.size(), 2u);
}

TEST_F(StrategyApplyTest, SampleReturnsAllWhenInputSmallerThanN) {
    auto s = ParseStrategy(R"({"sample": {"n": 10}})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    auto r = s->Apply({"a", "b"}, bag.AsFinder(), CallerNode{}, "");
    EXPECT_EQ(r.nodes.size(), 2u);
}

TEST_F(StrategyApplyTest, SampleNodePatternFilters) {
    auto s = ParseStrategy(R"({"sample": {"n": 10, "node_pattern": "^gpu-.*$"}})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    bag.Put(MakeNode("a", "gpu-1"));
    bag.Put(MakeNode("b", "cpu-1"));
    bag.Put(MakeNode("c", "gpu-2"));
    auto r = s->Apply({"a", "b", "c"}, bag.AsFinder(), CallerNode{}, "");
    std::unordered_set<std::string> got(r.nodes.begin(), r.nodes.end());
    EXPECT_EQ(got.size(), 2u);
    EXPECT_TRUE(got.count("a"));
    EXPECT_TRUE(got.count("c"));
    EXPECT_FALSE(got.count("b"));
}

TEST_F(StrategyApplyTest, SampleTraceIdSeedDeterministic) {
    auto s = ParseStrategy(R"({"sample": {"n": 2, "seed": "trace_id"}})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    auto r1 = s->Apply({"a", "b", "c", "d", "e"}, bag.AsFinder(), CallerNode{}, /*trace=*/"trace-X");
    auto r2 = s->Apply({"a", "b", "c", "d", "e"}, bag.AsFinder(), CallerNode{}, /*trace=*/"trace-X");
    ASSERT_EQ(r1.nodes.size(), 2u);
    ASSERT_EQ(r2.nodes.size(), 2u);
    EXPECT_EQ(r1.nodes, r2.nodes);
}

TEST_F(StrategyApplyTest, SortDescendingByDefault) {
    auto s = ParseStrategy(R"({"sort": [{"metric": "free_bytes", "weight": 1}]})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    bag.Put(MakeNode("a", "n", /*free=*/100));
    bag.Put(MakeNode("b", "n", /*free=*/300));
    bag.Put(MakeNode("c", "n", /*free=*/200));
    auto r = s->Apply({"a", "b", "c"}, bag.AsFinder(), CallerNode{}, "");
    ASSERT_EQ(r.nodes.size(), 3u);
    EXPECT_EQ(r.nodes[0], "b");
    EXPECT_EQ(r.nodes[1], "c");
    EXPECT_EQ(r.nodes[2], "a");
}

TEST_F(StrategyApplyTest, SortNegativeWeightAscending) {
    auto s = ParseStrategy(R"({"sort": [{"metric": "load_ratio", "weight": -1}]})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    bag.Put(MakeNode("hot", "n", 0, /*load=*/0.9));
    bag.Put(MakeNode("warm", "n", 0, /*load=*/0.5));
    bag.Put(MakeNode("cold", "n", 0, /*load=*/0.1));
    auto r = s->Apply({"hot", "warm", "cold"}, bag.AsFinder(), CallerNode{}, "");
    ASSERT_EQ(r.nodes.size(), 3u);
    EXPECT_EQ(r.nodes[0], "cold");
    EXPECT_EQ(r.nodes[1], "warm");
    EXPECT_EQ(r.nodes[2], "hot");
}

TEST_F(StrategyApplyTest, SortMissingMetricContributesZero) {
    // "known" has rx=10, "unknown" has no metrics at all.
    // weight=1 ascending=false (descending). known scores 10, unknown scores 0.
    auto s = ParseStrategy(R"({"sort": [{"metric": "rx_mbps", "weight": 1}]})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    bag.Put(MakeNode("known", "n", 0, 0, /*rx=*/10));
    auto r = s->Apply({"unknown", "known"}, bag.AsFinder(), CallerNode{}, "");
    ASSERT_EQ(r.nodes.size(), 2u);
    EXPECT_EQ(r.nodes[0], "known");
    EXPECT_EQ(r.nodes[1], "unknown");
}

TEST_F(StrategyApplyTest, SortMultiTermLinearCombination) {
    // score = rx_mbps * 1 + tx_mbps * 1 ; descending.
    auto s = ParseStrategy(R"({"sort": [
        {"metric": "rx_mbps", "weight": 1},
        {"metric": "tx_mbps", "weight": 1}
    ]})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    bag.Put(MakeNode("a", "n", 0, 0, /*rx=*/100, /*tx=*/100)); // 200
    bag.Put(MakeNode("b", "n", 0, 0, /*rx=*/50, /*tx=*/300));  // 350
    bag.Put(MakeNode("c", "n", 0, 0, /*rx=*/10, /*tx=*/10));   // 20
    auto r = s->Apply({"a", "b", "c"}, bag.AsFinder(), CallerNode{}, "");
    ASSERT_EQ(r.nodes.size(), 3u);
    EXPECT_EQ(r.nodes[0], "b");
    EXPECT_EQ(r.nodes[1], "a");
    EXPECT_EQ(r.nodes[2], "c");
}

TEST_F(StrategyApplyTest, LimitTruncatesTail) {
    auto s = ParseStrategy(R"({"limit": 2})");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    auto r = s->Apply({"a", "b", "c", "d"}, bag.AsFinder(), CallerNode{}, "");
    ASSERT_EQ(r.nodes.size(), 2u);
    EXPECT_EQ(r.nodes[0], "a");
    EXPECT_EQ(r.nodes[1], "b");
}

// ----- Pipeline order -----------------------------------------------------

TEST_F(StrategyApplyTest, FixedOrderFilterBeforeSort) {
    // free_bytes filter drops "small"; remainder sorted ascending by load_ratio
    // (negative weight). Only 1 returned via limit.
    auto s = ParseStrategy(R"({
        "filter": {"metric": "free_bytes", "min": 1000},
        "sort":   [{"metric": "load_ratio", "weight": -1}],
        "limit":  1
    })");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    bag.Put(MakeNode("small", "n", /*free=*/100, /*load=*/0.0));
    bag.Put(MakeNode("hot", "n", /*free=*/2000, /*load=*/0.9));
    bag.Put(MakeNode("warm", "n", /*free=*/2000, /*load=*/0.5));
    bag.Put(MakeNode("cold", "n", /*free=*/2000, /*load=*/0.1));
    auto r = s->Apply({"small", "hot", "warm", "cold"}, bag.AsFinder(), CallerNode{}, "");
    ASSERT_EQ(r.nodes.size(), 1u);
    EXPECT_EQ(r.nodes[0], "cold");
}

TEST_F(StrategyApplyTest, PreferLocalRunsBeforeSort) {
    // prefer_local sees the un-sorted candidate set; sort runs after on the
    // (now single-element) survivor list.
    auto s = ParseStrategy(R"({
        "prefer_local": {"on_miss": "passthrough"},
        "sort":         [{"metric": "free_bytes", "weight": 1}]
    })");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    bag.Put(MakeNode("10.0.0.7", "local", /*free=*/100));
    bag.Put(MakeNode("10.0.0.8", "remote", /*free=*/9999));
    auto r = s->Apply({"10.0.0.7", "10.0.0.8"}, bag.AsFinder(), /*caller=*/CallerNode{"10.0.0.7", ""}, "");
    ASSERT_EQ(r.nodes.size(), 1u);
    EXPECT_EQ(r.nodes[0], "10.0.0.7");
}

TEST_F(StrategyApplyTest, AbortShortCircuitsPipeline) {
    // prefer_local{abort} fires before sort would even see the data.
    auto s = ParseStrategy(R"({
        "prefer_local": {"on_miss": "abort"},
        "sort":         [{"metric": "free_bytes", "weight": 1}]
    })");
    ASSERT_NE(s, nullptr);
    MetricsBag bag;
    bag.Put(MakeNode("a", "n", 100));
    auto r = s->Apply({"a"}, bag.AsFinder(), /*caller=*/CallerNode{"missing", ""}, "");
    EXPECT_EQ(r.status, CandidatePipeline::Status::kAbort);
    EXPECT_TRUE(r.nodes.empty());
}

// ============================ CacheAffinityManager ======================

class CacheAffinityManagerTest : public TESTBASE {};

TEST_F(CacheAffinityManagerTest, UpsertAndRemoveNode) {
    CacheAffinityManager mgr;
    mgr.UpsertNodeMetrics(MakeNode("10.0.0.1", "n1"));
    mgr.UpsertNodeMetrics(MakeNode("10.0.0.2", "n2"));
    EXPECT_EQ(mgr.SnapshotNodes().size(), 2u);
    mgr.RemoveNode("10.0.0.1");
    auto snap = mgr.SnapshotNodes();
    ASSERT_EQ(snap.size(), 1u);
    EXPECT_EQ(snap[0].node_id, "10.0.0.2");
}
