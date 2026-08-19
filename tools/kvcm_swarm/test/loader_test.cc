// Loader tests: local validation is pure and rejects every configuration
// problem the design requires it to catch.
#include <gtest/gtest.h>
#include <string>

#include "tools/kvcm_swarm/clients/registry.h"
#include "tools/kvcm_swarm/scenario/config_reader.h"
#include "tools/kvcm_swarm/scenario/loader.h"

namespace kvcm_swarm {
namespace {

const char *kBaseConfig = R"JSON({
  "name": "unit",
  "seed": 7,
  "runtime": {
    "warmup": "1s",
    "steady": "2s",
    "drain_timeout": "3s",
    "workers": 2,
    "limits": {"max_in_flight_business_rpcs": 8, "max_in_flight_control_rpcs": 4}
  },
  "target": {
    "endpoints": {
      "meta_http": "http://127.0.0.1:1",
      "meta_grpc": "127.0.0.1:2",
      "admin_http": "http://127.0.0.1:3"
    },
    "instance_groups": {"grp": {"quota_bytes": 1024}}
  },
  "behaviors": [
    {
      "id": "v6d-a",
      "type": "v6d_deployment",
      "transport": "http",
      "config": {
        "process_count": 2,
        "instance_group": "grp",
        "instance_id": "inst-a",
        "local_cache": {"capacity_bytes": 65536},
        "session_arrival": {"rate": 5, "mode": "even"},
        "session_affinity": 0.5,
        "limits": {"max_active_sessions": 16},
        "heartbeat_interval": "5s",
        "shared_prefix_pool": {"root_count": 4, "prefix_tokens": {"min": 16, "max": 32}},
        "groups": [
          {"id": "full-0", "kind": "full_attention", "block_size": 16, "object_size": 4096,
           "lookup_selector": "prefix"},
          {"id": "mamba-0", "kind": "mamba", "block_size": 64, "object_size": 1024,
           "key_presence_rate": 0.25}
        ],
        "session_classes": [
          {"name": "chat", "weight": 1.0, "turns": {"min": 2, "max": 4},
           "turn_interval": {"min": "10ms", "max": "20ms"},
           "initial_tokens": {"min": 64, "max": 128},
           "new_tokens_per_turn": 16, "rewrite_tail_tokens": 0,
           "shared_prefix_probability": 0.5}
        ]
      }
    },
    {"id": "health-a", "type": "health_probe", "transport": "grpc", "config": {"interval": "1s"}}
  ],
  "evidence": {"output_json": "r.json", "violations_jsonl": "v.jsonl"}
})JSON";

std::string Mutate(const std::string &from, const std::string &to) {
    std::string config = kBaseConfig;
    const size_t position = config.find(from);
    if (position == std::string::npos) {
        return "";
    }
    return config.replace(position, from.size(), to);
}

bool HasError(const LoadResult &result, const std::string &needle) {
    for (const auto &error : result.errors) {
        if (error.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

TEST(LoaderTest, BaseConfigIsValidAndFullyNormalised) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    const LoadResult result = LoadScenarioFromJson(kBaseConfig, registry);
    ASSERT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.front());
    EXPECT_EQ(result.config.name, "unit");
    EXPECT_EQ(result.config.seed, 7u);
    EXPECT_EQ(result.config.runtime.workers, 2u);
    EXPECT_EQ(result.config.behaviors.size(), 2u);
    EXPECT_EQ(result.config.behaviors[0].transport, TransportKind::kHttp);
    EXPECT_EQ(result.config.behaviors[1].transport, TransportKind::kGrpc);
    // admin_grpc defaults to meta_grpc and the effective value is recorded.
    EXPECT_EQ(result.config.target.endpoints.admin_grpc, "127.0.0.1:2");
    EXPECT_TRUE(result.config.preflight_enabled);
}

TEST(LoaderTest, UnknownFieldsAreRejected) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    LoadResult top = LoadScenarioFromJson(Mutate("\"seed\": 7,", "\"seed\": 7, \"mystery\": 1,"), registry);
    EXPECT_FALSE(top.ok);
    EXPECT_TRUE(HasError(top, "unknown configuration field: mystery"));

    LoadResult behavior =
        LoadScenarioFromJson(Mutate("\"process_count\": 2,", "\"process_count\": 2, \"spill_rate\": 1.4,"), registry);
    EXPECT_FALSE(behavior.ok);
    EXPECT_TRUE(HasError(behavior, "spill_rate")) << "removed knobs must not be silently accepted";
}

TEST(LoaderTest, TlsTransportsAreRejectedAndNeverDowngraded) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    LoadResult https = LoadScenarioFromJson(
        Mutate("\"meta_http\": \"http://127.0.0.1:1\"", "\"meta_http\": \"https://127.0.0.1:1\""), registry);
    EXPECT_FALSE(https.ok);
    EXPECT_TRUE(HasError(https, "HTTPS/TLS endpoints are not supported"));

    LoadResult mtls = LoadScenarioFromJson(Mutate("\"transport\": \"http\"", "\"transport\": \"mtls\""), registry);
    EXPECT_FALSE(mtls.ok);
    EXPECT_TRUE(HasError(mtls, "TLS transports are not supported"));
}

TEST(LoaderTest, EndpointShapeIsValidated) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    EXPECT_TRUE(
        HasError(LoadScenarioFromJson(Mutate("\"meta_grpc\": \"127.0.0.1:2\"", "\"meta_grpc\": \"http://127.0.0.1:2\""),
                                      registry),
                 "gRPC endpoint must be host:port without a scheme"));
    EXPECT_TRUE(HasError(
        LoadScenarioFromJson(Mutate("\"admin_http\": \"http://127.0.0.1:3\"", "\"admin_http\": \"http://127.0.0.1:1\""),
                             registry),
        "meta_http and admin_http must differ"));
}

TEST(LoaderTest, DurationsRequireExplicitUnits) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    EXPECT_TRUE(
        HasError(LoadScenarioFromJson(Mutate("\"warmup\": \"1s\"", "\"warmup\": \"1\""), registry), "explicit unit"));
    EXPECT_TRUE(HasError(LoadScenarioFromJson(Mutate("\"steady\": \"2s\"", "\"steady\": \"2weeks\""), registry),
                         "unsupported unit"));
    EXPECT_TRUE(HasError(LoadScenarioFromJson(Mutate("\"steady\": \"2s\"", "\"steady\": \"0s\""), registry),
                         "steady: must be positive"));
}

TEST(LoaderTest, DuplicateBehaviorIdsAndIdentitiesAreRejected) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    EXPECT_TRUE(HasError(LoadScenarioFromJson(Mutate("\"id\": \"health-a\"", "\"id\": \"v6d-a\""), registry),
                         "duplicate behavior id"));
    // Two deployments claiming the same instance_id must fail.
    const std::string two_deployments = Mutate("{\"id\": \"health-a\", \"type\": \"health_probe\", \"transport\": "
                                               "\"grpc\", \"config\": {\"interval\": \"1s\"}}",
                                               R"({
      "id": "v6d-b",
      "type": "v6d_deployment",
      "transport": "http",
      "config": {
        "process_count": 1,
        "instance_group": "grp",
        "instance_id": "inst-a",
        "local_cache": {"capacity_bytes": 65536},
        "session_arrival": {"rate": 5, "mode": "even"},
        "session_affinity": 0.5,
        "limits": {"max_active_sessions": 16},
        "heartbeat_interval": "5s",
        "shared_prefix_pool": {"root_count": 0, "prefix_tokens": 0},
        "groups": [{"id": "full-0", "kind": "full_attention", "block_size": 16, "object_size": 4096,
                    "lookup_selector": "coverage"}],
        "session_classes": [
          {"name": "chat", "weight": 1.0, "turns": 2, "turn_interval": "10ms",
           "initial_tokens": 64, "new_tokens_per_turn": 16, "rewrite_tail_tokens": 0,
           "shared_prefix_probability": 0.0}
        ]
      }
    })");
    const LoadResult result = LoadScenarioFromJson(two_deployments, registry);
    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(HasError(result, "instance_id:inst-a"));
    EXPECT_TRUE(HasError(result, "reporter_host_ip_port:10.99.0.1:40000"));
}

TEST(LoaderTest, InstanceGroupMustBeDeclaredInTheTarget) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    const LoadResult result =
        LoadScenarioFromJson(Mutate("\"instance_group\": \"grp\"", "\"instance_group\": \"other\""), registry);
    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(HasError(result, "instance group 'other' is not declared"));
}

TEST(LoaderTest, GroupAndSelectorConstraints) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    // A Full Attention group must declare an explicit selector.
    EXPECT_TRUE(HasError(
        LoadScenarioFromJson(Mutate("\"lookup_selector\": \"prefix\"", "\"lookup_selector\": \"best\""), registry),
        "must set 'prefix' or 'coverage' explicitly"));
    // Mamba must not carry a selector.
    EXPECT_TRUE(HasError(LoadScenarioFromJson(Mutate("\"key_presence_rate\": 0.25",
                                                     "\"key_presence_rate\": 0.25, \"lookup_selector\": \"prefix\""),
                                              registry),
                         "Mamba groups always use COVERAGE"));
    // key_presence_rate is Mamba only and bounded.
    EXPECT_TRUE(HasError(LoadScenarioFromJson(Mutate("\"lookup_selector\": \"prefix\"",
                                                     "\"lookup_selector\": \"prefix\", \"key_presence_rate\": 0.5"),
                                              registry),
                         "only Mamba groups accept key_presence_rate"));
    EXPECT_TRUE(
        HasError(LoadScenarioFromJson(Mutate("\"key_presence_rate\": 0.25", "\"key_presence_rate\": 1.5"), registry),
                 "must be within [0, 1]"));
    // At least one Full Attention group is required.
    EXPECT_TRUE(HasError(LoadScenarioFromJson(Mutate("\"kind\": \"full_attention\"", "\"kind\": \"mamba\""), registry),
                         "at least one Full Attention group is required"));
}

TEST(LoaderTest, CacheCapacityMustHoldTheLargestObject) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    EXPECT_TRUE(
        HasError(LoadScenarioFromJson(Mutate("\"capacity_bytes\": 65536", "\"capacity_bytes\": 1024"), registry),
                 "must be >= the largest groups[].object_size"));
}

TEST(LoaderTest, SharedPrefixRequiresEnoughInitialTokens) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    EXPECT_TRUE(HasError(LoadScenarioFromJson(Mutate("\"initial_tokens\": {\"min\": 64, \"max\": 128}",
                                                     "\"initial_tokens\": {\"min\": 16, \"max\": 128}"),
                                              registry),
                         "initial_tokens.min must be >= shared_prefix_pool.prefix_tokens.max"));
}

TEST(LoaderTest, ArrivalAndAffinityBounds) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    EXPECT_TRUE(HasError(LoadScenarioFromJson(Mutate("\"rate\": 5", "\"rate\": 0"), registry),
                         "session_arrival.rate: must be positive"));
    EXPECT_TRUE(HasError(LoadScenarioFromJson(Mutate("\"mode\": \"even\"", "\"mode\": \"zipf\""), registry),
                         "must be 'even' or 'poisson'"));
    EXPECT_TRUE(
        HasError(LoadScenarioFromJson(Mutate("\"session_affinity\": 0.5", "\"session_affinity\": 1.5"), registry),
                 "session_affinity: must be within [0, 1]"));
    EXPECT_TRUE(HasError(LoadScenarioFromJson(Mutate("\"process_count\": 2", "\"process_count\": 0"), registry),
                         "process_count: must be at least 1"));
    EXPECT_TRUE(
        HasError(LoadScenarioFromJson(Mutate("\"max_active_sessions\": 16", "\"max_active_sessions\": 0"), registry),
                 "max_active_sessions"));
}

TEST(LoaderTest, UnknownBehaviorTypeIsRejected) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    EXPECT_TRUE(
        HasError(LoadScenarioFromJson(Mutate("\"type\": \"health_probe\"", "\"type\": \"event_reporter\""), registry),
                 "unknown behavior type 'event_reporter'"));
}

TEST(LoaderTest, MalformedJsonIsReportedWithAnOffset) {
    const BehaviorRegistry registry = MakeDefaultRegistry();
    const LoadResult result = LoadScenarioFromJson("{\"name\": ", registry);
    EXPECT_FALSE(result.ok);
    EXPECT_TRUE(HasError(result, "JSON parse error at offset"));
}

TEST(ConfigReaderTest, DurationParsingCoversEveryUnit) {
    Duration value{};
    std::string error;
    ASSERT_TRUE(ParseDuration("250ns", &value, &error));
    EXPECT_EQ(value, Duration(std::chrono::nanoseconds(250)));
    ASSERT_TRUE(ParseDuration("1.5ms", &value, &error));
    EXPECT_EQ(value, Duration(std::chrono::microseconds(1500)));
    ASSERT_TRUE(ParseDuration("2m", &value, &error));
    EXPECT_EQ(value, Duration(std::chrono::minutes(2)));
    ASSERT_TRUE(ParseDuration("1h", &value, &error));
    EXPECT_EQ(value, Duration(std::chrono::hours(1)));
    EXPECT_FALSE(ParseDuration("", &value, &error));
    EXPECT_FALSE(ParseDuration("abc", &value, &error));
    EXPECT_FALSE(ParseDuration("10", &value, &error));
}

} // namespace
} // namespace kvcm_swarm
