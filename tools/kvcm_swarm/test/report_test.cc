// Report tests: schema completeness, phase buckets, status rendering and the
// bounded violation log.
#include <gtest/gtest.h>
#include <set>
#include <string>

#include "rapidjson/document.h"
#include "tools/kvcm_swarm/clients/registry.h"
#include "tools/kvcm_swarm/evidence/report.h"
#include "tools/kvcm_swarm/scenario/loader.h"

namespace kvcm_swarm {
namespace {

const char *kConfig = R"JSON({
  "name": "report-unit",
  "seed": 3,
  "runtime": {"warmup": "1s", "steady": "2s", "drain_timeout": "3s", "workers": 1},
  "target": {
    "endpoints": {
      "meta_http": "http://127.0.0.1:1",
      "meta_grpc": "127.0.0.1:2",
      "admin_http": "http://127.0.0.1:3"
    },
    "instance_groups": {"grp": {"quota_bytes": 4096}}
  },
  "behaviors": [
    {"id": "health-a", "type": "health_probe", "transport": "http", "config": {"interval": "1s"}}
  ],
  "evidence": {"output_json": "r.json", "violations_jsonl": "v.jsonl"}
})JSON";

class ReportTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = MakeDefaultRegistry();
        LoadResult load = LoadScenarioFromJson(kConfig, registry_);
        ASSERT_TRUE(load.ok) << (load.errors.empty() ? "" : load.errors.front());
        config_ = std::move(load.config);
        executor_ = std::make_unique<SwarmExecutor>(1);
        admission_ = std::make_unique<AdmissionController>(*executor_, config_.runtime.limits);
        transports_ = std::make_unique<TransportProvider>(
            *executor_, *admission_, evidence_, phase_, config_.target.endpoints, config_.runtime.transport, 1, 1);
        RuntimeServices services{*executor_, *admission_, *transports_, evidence_, seeds_, stop_.Token(), phase_};
        behavior_ = registry_.Find("health_probe")->Create(config_.behaviors[0], services);
        ASSERT_NE(behavior_, nullptr);
        behaviors_.push_back(behavior_.get());
        for (const Phase phase :
             {Phase::kPreflight, Phase::kInitialize, Phase::kWarmup, Phase::kSteady, Phase::kDrain, Phase::kReport}) {
            PhaseRecord record;
            record.phase = phase;
            record.start = Now();
            record.end = Now() + std::chrono::milliseconds(1);
            record.entered = true;
            phases_.push_back(record);
        }
        preflight_.executed = true;
        preflight_.passed = true;
        preflight_.temporary_instance_id = "tmp";
        preflight_.steps.emplace_back("admin_endpoint_check_health", true);
    }

    void TearDown() override {
        transports_->Shutdown();
        executor_->Shutdown();
    }

    RunReportInput MakeInput() {
        RunReportInput input;
        input.config = &config_;
        input.evidence = &evidence_;
        input.admission = admission_.get();
        input.executor = executor_.get();
        input.transports = transports_.get();
        input.behaviors = &behaviors_;
        input.phases = &phases_;
        input.preflight = &preflight_;
        input.started_wall_ms = WallClockMs();
        input.ended_wall_ms = WallClockMs();
        input.total_duration = std::chrono::seconds(6);
        input.exit_reason = "completed";
        input.initialize_ok = true;
        input.drain_complete = true;
        input.quiesced = true;
        input.resources = CollectResourceUsage();
        return input;
    }

    BehaviorRegistry registry_;
    ScenarioConfig config_;
    EvidenceSink evidence_;
    PhaseSource phase_;
    StopSource stop_;
    SeedDeriver seeds_{3};
    std::unique_ptr<SwarmExecutor> executor_;
    std::unique_ptr<AdmissionController> admission_;
    std::unique_ptr<TransportProvider> transports_;
    std::unique_ptr<ClientBehavior> behavior_;
    std::vector<ClientBehavior *> behaviors_;
    std::vector<PhaseRecord> phases_;
    PreflightReport preflight_;
};

TEST_F(ReportTest, StableSchemaContainsEveryRequiredSection) {
    const RunReportInput input = MakeInput();
    const std::string json = BuildRunReportJson(input);
    ASSERT_FALSE(json.empty()) << "an empty report is an execution failure";
    rapidjson::Document document;
    ASSERT_FALSE(document.Parse(json.c_str()).HasParseError());
    for (const char *section : {"run",
                                "run_config",
                                "phases",
                                "runtime",
                                "behaviors",
                                "rpc",
                                "transport",
                                "cache",
                                "invariants",
                                "workload_shape",
                                "usage_observations",
                                "limitations",
                                "cleanup"}) {
        EXPECT_TRUE(document.HasMember(section)) << "missing section: " << section;
    }
    ASSERT_TRUE(document["runtime"].HasMember("generator_lag"));
    ASSERT_TRUE(document["runtime"].HasMember("admission"));
    ASSERT_TRUE(document["runtime"].HasMember("resource_usage"));
    EXPECT_TRUE(document["run"]["metadata_only"].GetBool());
    EXPECT_FALSE(document["run"]["generator_saturated"].GetBool());
    EXPECT_GT(document["limitations"].Size(), 0u);
    // Every phase that was entered gets its own bucket.
    for (const char *phase : {"preflight", "initialize", "warmup", "steady", "drain", "report"}) {
        EXPECT_TRUE(document["phases"].HasMember(phase)) << phase;
    }
    // The effective configuration is echoed, including advanced defaults.
    EXPECT_TRUE(document["run_config"]["runtime"]["limits"].HasMember("http_connections_per_endpoint"));
    EXPECT_STREQ(document["run_config"]["target"]["endpoints"]["admin_grpc"].GetString(), "127.0.0.1:2");
    EXPECT_STREQ(document["run_config"]["behaviors"]["health-a"]["config"]["probe_deadline"].GetString(), "1000ms");
    EXPECT_EQ(document["run_config"]["behaviors"]["health-a"]["config"]["streams"].GetUint(), 1u);
    EXPECT_TRUE(document["cleanup"]["preflight"]["passed"].GetBool());
}

TEST_F(ReportTest, NotRunAndInconclusiveAreRenderedExplicitly) {
    const RunReportInput input = MakeInput();
    const std::string json = BuildRunReportJson(input);
    rapidjson::Document document;
    ASSERT_FALSE(document.Parse(json.c_str()).HasParseError());
    const auto &checks = document["invariants"]["checks"];
    ASSERT_EQ(checks.Size(), 1u);
    // The probe never ran, so C5 must be NOT_RUN rather than silently passing.
    EXPECT_STREQ(checks[0]["status"].GetString(), "NOT_RUN");
    EXPECT_EQ(checks[0]["checked"].GetUint64(), 0u);
    EXPECT_STREQ(checks[0]["check_name"].GetString(), "C5_health_probe_bounded_response");

    const std::string summary = RenderRunSummary(input);
    EXPECT_NE(summary.find("[NOT_RUN] C5_health_probe_bounded_response"), std::string::npos);
    EXPECT_NE(summary.find("metadata-only"), std::string::npos);
}

TEST_F(ReportTest, GeneratorSaturationIsSurfacedInTheReport) {
    admission_->MarkSaturated("cache_backpressure");
    const RunReportInput input = MakeInput();
    rapidjson::Document document;
    ASSERT_FALSE(document.Parse(BuildRunReportJson(input).c_str()).HasParseError());
    EXPECT_TRUE(document["run"]["generator_saturated"].GetBool());
    ASSERT_EQ(document["run"]["generator_saturation_reasons"].Size(), 1u);
    EXPECT_STREQ(document["run"]["generator_saturation_reasons"][0].GetString(), "cache_backpressure");
}

TEST(ViolationLogTest, PreviewIsBoundedAndCountsAreExact) {
    EvidenceSink sink;
    ASSERT_TRUE(sink.violations().Open(std::string(::testing::TempDir()) + "/swarm_violations.jsonl"));
    for (int i = 0; i < 50; ++i) {
        sink.violations().Record("C1b_remote_availability", "{\"index\":" + std::to_string(i) + "}");
    }
    EXPECT_EQ(sink.violations().Count("C1b_remote_availability"), 50u);
    EXPECT_EQ(sink.violations().total(), 50u);
    EXPECT_LE(sink.violations().Preview("C1b_remote_availability").size(), 8u);
    EXPECT_FALSE(sink.violations().failed());
    sink.violations().Close();
}

TEST(RpcAggregateTest, PhaseAndLaneAreKeptSeparate) {
    EvidenceSink sink;
    RpcObservation observation;
    observation.behavior_type = "v6d_deployment";
    observation.behavior_id = "v6d-a";
    observation.api = "ReportEvent";
    observation.lane = TrafficLane::kControl;
    observation.phase = Phase::kWarmup;
    observation.result.ok = true;
    observation.result.service_status = 1;
    observation.result.rpc_latency = std::chrono::milliseconds(3);
    sink.RecordRpc(observation);
    observation.phase = Phase::kSteady;
    sink.RecordRpc(observation);
    observation.result.ok = false;
    observation.result.transport_error = TransportError::kTimeout;
    sink.RecordRpc(observation);

    const auto snapshot = sink.RpcSnapshot();
    EXPECT_EQ(snapshot.size(), 2u) << "warmup and steady must be reported separately";
    uint64_t uncertain = 0;
    for (const auto &entry : snapshot) {
        uncertain += entry.second.uncertain;
    }
    EXPECT_EQ(uncertain, 1u) << "a timeout is an uncertain outcome, never a success";
}

} // namespace
} // namespace kvcm_swarm
