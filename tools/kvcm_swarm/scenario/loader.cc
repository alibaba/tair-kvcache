#include "tools/kvcm_swarm/scenario/loader.h"

#include <cstdio>
#include <limits>
#include <optional>
#include <set>

#include "tools/kvcm_swarm/scenario/duration.h"
#include "tools/kvcm_swarm/scenario/json_config.h"
#include "tools/kvcm_swarm/transport/transport_provider.h"

namespace kvcm_swarm {
namespace {

bool ReadFile(const std::string &path, std::string *out, std::string *error) {
    std::FILE *file = std::fopen(path.c_str(), "rb");
    if (file == nullptr) {
        *error = "cannot open configuration file: " + path;
        return false;
    }
    char buffer[8192];
    size_t read = 0;
    while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        out->append(buffer, read);
    }
    const bool failed = std::ferror(file) != 0;
    std::fclose(file);
    if (failed) {
        *error = "cannot read configuration file: " + path;
        return false;
    }
    return true;
}

bool DecodeDuration(const rapidjson::Value &value, Duration *out, std::string *error) {
    if (!value.IsString()) {
        *error = "must be a duration string such as \"10ms\"";
        return false;
    }
    return ParseDuration(std::string(value.GetString(), value.GetStringLength()), out, error);
}

class RuntimeLimitsJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value,
                         {"max_in_flight_business_rpcs",
                          "max_in_flight_control_rpcs",
                          "business_permit_wait_threshold",
                          "http_connections_per_endpoint",
                          "http_control_connections_per_endpoint",
                          "connect_timeout",
                          "default_rpc_timeout"})) {
            return false;
        }
        Optional(value, "max_in_flight_business_rpcs", max_in_flight_business_rpcs, uint64_t{4096});
        Optional(value, "max_in_flight_control_rpcs", max_in_flight_control_rpcs, uint64_t{512});
        OptionalCustom(value,
                       "business_permit_wait_threshold",
                       business_permit_wait_threshold,
                       Duration(std::chrono::seconds(1)),
                       DecodeDuration);
        Optional(value, "http_connections_per_endpoint", http_connections_per_endpoint, uint64_t{8});
        Optional(value, "http_control_connections_per_endpoint", http_control_connections_per_endpoint, uint64_t{2});
        OptionalCustom(value, "connect_timeout", connect_timeout, Duration(std::chrono::seconds(3)), DecodeDuration);
        OptionalCustom(
            value, "default_rpc_timeout", default_rpc_timeout, Duration(std::chrono::seconds(10)), DecodeDuration);
        return true;
    }

    uint64_t max_in_flight_business_rpcs = 4096;
    uint64_t max_in_flight_control_rpcs = 512;
    Duration business_permit_wait_threshold = std::chrono::seconds(1);
    uint64_t http_connections_per_endpoint = 8;
    uint64_t http_control_connections_per_endpoint = 2;
    Duration connect_timeout = std::chrono::seconds(3);
    Duration default_rpc_timeout = std::chrono::seconds(10);
};

class RuntimeJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value,
                         {"warmup",
                          "steady",
                          "drain_timeout",
                          "workers",
                          "reactor_threads",
                          "grpc_completion_queues",
                          "limits"})) {
            return false;
        }
        RequiredCustom(value, "warmup", warmup, DecodeDuration);
        RequiredCustom(value, "steady", steady, DecodeDuration);
        RequiredCustom(value, "drain_timeout", drain_timeout, DecodeDuration);
        Required(value, "workers", workers);
        Optional(value, "reactor_threads", reactor_threads, uint64_t{2});
        Optional(value, "grpc_completion_queues", grpc_completion_queues, uint64_t{2});
        Optional(value, "limits", limits);
        return true;
    }

    void AppendAllErrors(std::string_view path, std::vector<std::string> *errors) const {
        AppendJsonErrors(path, errors);
        if (limits.has_value()) {
            limits->AppendJsonErrors(std::string(path) + ".limits", errors);
        }
    }

    const RuntimeLimitsJson &EffectiveLimits() const {
        static const RuntimeLimitsJson defaults;
        return limits.has_value() ? *limits : defaults;
    }

    Duration warmup{};
    Duration steady{};
    Duration drain_timeout{};
    uint64_t workers = 0;
    uint64_t reactor_threads = 2;
    uint64_t grpc_completion_queues = 2;
    std::optional<RuntimeLimitsJson> limits;
};

class EndpointsJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value, {"meta_http", "meta_grpc", "admin_http", "admin_grpc"})) {
            return false;
        }
        Required(value, "meta_http", meta_http);
        Required(value, "meta_grpc", meta_grpc);
        Required(value, "admin_http", admin_http);
        Optional(value, "admin_grpc", admin_grpc);
        return true;
    }

    std::string meta_http;
    std::string meta_grpc;
    std::string admin_http;
    std::optional<std::string> admin_grpc;
};

class InstanceGroupJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value, {"quota_bytes"})) {
            return false;
        }
        Required(value, "quota_bytes", quota_bytes);
        return true;
    }
    uint64_t quota_bytes = 0;
};

class TargetJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value, {"endpoints", "instance_groups"})) {
            return false;
        }
        Required(value, "endpoints", endpoints);
        Required(value, "instance_groups", instance_groups);
        return true;
    }

    void AppendAllErrors(std::vector<std::string> *errors) const {
        AppendJsonErrors("target", errors);
        endpoints.AppendJsonErrors("target.endpoints", errors);
        for (const auto &entry : instance_groups) {
            entry.second.AppendJsonErrors("target.instance_groups." + entry.first, errors);
        }
    }

    EndpointsJson endpoints;
    std::map<std::string, InstanceGroupJson> instance_groups;
};

class EvidenceJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value, {"output_json", "violations_jsonl", "markdown_summary"})) {
            return false;
        }
        Required(value, "output_json", output_json);
        Required(value, "violations_jsonl", violations_jsonl);
        Optional(value, "markdown_summary", markdown_summary, std::string());
        return true;
    }

    std::string output_json;
    std::string violations_jsonl;
    std::string markdown_summary;
};

class BehaviorJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value, {"id", "type", "transport", "config"})) {
            return false;
        }
        Required(value, "id", id);
        Required(value, "type", type);
        Required(value, "transport", transport);
        Required(value, "config", config);
        return true;
    }

    std::string id;
    std::string type;
    std::string transport;
    RawJsonObject config;
};

class ScenarioJson final : public JsonConfig {
public:
    bool FromRapidValue(const rapidjson::Value &value) override {
        if (!BeginObject(value, {"name", "seed", "runtime", "target", "behaviors", "evidence", "preflight"})) {
            return false;
        }
        Required(value, "name", name);
        Required(value, "seed", seed);
        Required(value, "runtime", runtime);
        Required(value, "target", target);
        Required(value, "behaviors", behaviors);
        Required(value, "evidence", evidence);
        Optional(value, "preflight", preflight, true);
        return true;
    }

    void AppendAllErrors(std::vector<std::string> *errors) const {
        AppendJsonErrors("", errors);
        runtime.AppendAllErrors("runtime", errors);
        target.AppendAllErrors(errors);
        evidence.AppendJsonErrors("evidence", errors);
        for (size_t i = 0; i < behaviors.size(); ++i) {
            behaviors[i].AppendJsonErrors("behaviors[" + std::to_string(i) + "]", errors);
        }
    }

    std::string name;
    uint64_t seed = 0;
    RuntimeJson runtime;
    TargetJson target;
    std::vector<BehaviorJson> behaviors;
    EvidenceJson evidence;
    bool preflight = true;
};

bool ToUint32(uint64_t value, std::string_view path, uint32_t *out, std::vector<std::string> *errors) {
    if (value > std::numeric_limits<uint32_t>::max()) {
        errors->push_back(std::string(path) + ": must fit in uint32");
        return false;
    }
    *out = static_cast<uint32_t>(value);
    return true;
}

} // namespace

LoadResult LoadScenarioFromJson(const std::string &json, const BehaviorRegistry &registry) {
    LoadResult result;
    ScenarioJson input;
    std::string parse_error;
    if (!input.FromJsonString(json, &parse_error)) {
        if (!parse_error.empty()) {
            result.errors.push_back(parse_error);
        }
        input.AppendAllErrors(&result.errors);
        return result;
    }
    input.AppendAllErrors(&result.errors);
    if (!result.errors.empty()) {
        return result;
    }

    ScenarioConfig config;
    config.name = input.name;
    config.seed = input.seed;
    config.runtime.warmup = input.runtime.warmup;
    config.runtime.steady = input.runtime.steady;
    config.runtime.drain_timeout = input.runtime.drain_timeout;
    ToUint32(input.runtime.workers, "runtime.workers", &config.runtime.workers, &result.errors);
    ToUint32(input.runtime.reactor_threads, "runtime.reactor_threads", &config.runtime.reactor_threads, &result.errors);
    ToUint32(input.runtime.grpc_completion_queues,
             "runtime.grpc_completion_queues",
             &config.runtime.grpc_completion_queues,
             &result.errors);
    const RuntimeLimitsJson &limits = input.runtime.EffectiveLimits();
    ToUint32(limits.max_in_flight_business_rpcs,
             "runtime.limits.max_in_flight_business_rpcs",
             &config.runtime.limits.max_in_flight_business_rpcs,
             &result.errors);
    ToUint32(limits.max_in_flight_control_rpcs,
             "runtime.limits.max_in_flight_control_rpcs",
             &config.runtime.limits.max_in_flight_control_rpcs,
             &result.errors);
    config.runtime.limits.business_permit_wait_threshold = limits.business_permit_wait_threshold;
    ToUint32(limits.http_connections_per_endpoint,
             "runtime.limits.http_connections_per_endpoint",
             &config.runtime.transport.http_connections_per_endpoint,
             &result.errors);
    ToUint32(limits.http_control_connections_per_endpoint,
             "runtime.limits.http_control_connections_per_endpoint",
             &config.runtime.transport.http_control_connections_per_endpoint,
             &result.errors);
    config.runtime.transport.connect_timeout = limits.connect_timeout;
    config.runtime.transport.default_rpc_timeout = limits.default_rpc_timeout;

    if (config.runtime.warmup < Duration::zero()) {
        result.errors.push_back("runtime.warmup: must not be negative");
    }
    if (config.runtime.steady <= Duration::zero()) {
        result.errors.push_back("runtime.steady: must be positive");
    }
    if (config.runtime.drain_timeout <= Duration::zero()) {
        result.errors.push_back("runtime.drain_timeout: must be positive");
    }
    if (config.runtime.workers == 0) {
        result.errors.push_back("runtime.workers: must be at least 1");
    }
    if (config.runtime.reactor_threads == 0) {
        result.errors.push_back("runtime.reactor_threads: must be at least 1");
    }
    if (config.runtime.grpc_completion_queues == 0) {
        result.errors.push_back("runtime.grpc_completion_queues: must be at least 1");
    }
    if (config.runtime.limits.max_in_flight_business_rpcs == 0) {
        result.errors.push_back("runtime.limits.max_in_flight_business_rpcs: must be positive");
    }
    if (config.runtime.limits.max_in_flight_control_rpcs == 0) {
        result.errors.push_back("runtime.limits.max_in_flight_control_rpcs: must be positive");
    }
    if (config.runtime.transport.http_connections_per_endpoint == 0) {
        result.errors.push_back("runtime.limits.http_connections_per_endpoint: must be positive");
    }
    if (config.runtime.transport.http_control_connections_per_endpoint == 0) {
        result.errors.push_back("runtime.limits.http_control_connections_per_endpoint: must be positive");
    }

    config.target.endpoints.meta_http = input.target.endpoints.meta_http;
    config.target.endpoints.meta_grpc = input.target.endpoints.meta_grpc;
    config.target.endpoints.admin_http = input.target.endpoints.admin_http;
    config.target.endpoints.admin_grpc = input.target.endpoints.admin_grpc.value_or(input.target.endpoints.meta_grpc);
    for (const auto &endpoint_field : {std::make_pair(config.target.endpoints.meta_http, true),
                                       std::make_pair(config.target.endpoints.admin_http, true),
                                       std::make_pair(config.target.endpoints.meta_grpc, false),
                                       std::make_pair(config.target.endpoints.admin_grpc, false)}) {
        std::string endpoint_error;
        if (!ValidateInsecureEndpoint(endpoint_field.first, endpoint_field.second, &endpoint_error)) {
            result.errors.push_back("target.endpoints: " + endpoint_error);
        }
    }
    if (config.target.endpoints.meta_http == config.target.endpoints.admin_http) {
        result.errors.push_back(
            "target.endpoints: meta_http and admin_http must differ; meta and admin must not share a socket");
    }
    for (const auto &entry : input.target.instance_groups) {
        InstanceGroupTarget target;
        target.name = entry.first;
        target.quota_bytes = entry.second.quota_bytes;
        config.target.instance_groups.emplace(entry.first, std::move(target));
    }

    config.evidence.output_json = input.evidence.output_json;
    config.evidence.violations_jsonl = input.evidence.violations_jsonl;
    config.evidence.markdown_summary = input.evidence.markdown_summary;
    if (config.evidence.output_json.empty()) {
        result.errors.push_back("evidence.output_json: must not be empty");
    }
    if (config.evidence.violations_jsonl.empty()) {
        result.errors.push_back("evidence.violations_jsonl: must not be empty");
    }
    config.preflight_enabled = input.preflight;

    std::set<std::string> behavior_ids;
    for (const BehaviorJson &behavior_input : input.behaviors) {
        BehaviorSpec spec;
        spec.id = behavior_input.id;
        spec.type = behavior_input.type;
        if (behavior_input.transport == "http") {
            spec.transport = TransportKind::kHttp;
        } else if (behavior_input.transport == "grpc") {
            spec.transport = TransportKind::kGrpc;
        } else if (behavior_input.transport == "https" || behavior_input.transport == "grpcs" ||
                   behavior_input.transport == "tls" || behavior_input.transport == "mtls") {
            result.errors.push_back("behaviors[" + spec.id +
                                    "].transport: TLS transports are not supported and must not be silently "
                                    "downgraded");
        } else {
            result.errors.push_back("behaviors[" + spec.id + "].transport: must be 'http' or 'grpc'");
        }
        spec.config_json = behavior_input.config.json();
        if (spec.id.empty()) {
            result.errors.push_back("behaviors.id: must not be empty");
        } else if (!behavior_ids.insert(spec.id).second) {
            result.errors.push_back("behaviors: duplicate behavior id '" + spec.id + "'");
        }
        const BehaviorFactory *factory = registry.Find(spec.type);
        if (factory == nullptr) {
            result.errors.push_back("behaviors[" + spec.id + "].type: unknown behavior type '" + spec.type + "'");
        } else {
            const ValidationResult validation = factory->Validate(spec);
            result.errors.insert(result.errors.end(), validation.errors.begin(), validation.errors.end());
        }
        config.behaviors.push_back(std::move(spec));
    }
    if (config.behaviors.empty()) {
        result.errors.push_back("behaviors: at least one behavior is required");
    }

    std::map<std::string, std::string> claim_owner;
    for (const auto &spec : config.behaviors) {
        const BehaviorFactory *factory = registry.Find(spec.type);
        if (factory == nullptr) {
            continue;
        }
        const BehaviorIdentityClaims claims = factory->Claims(spec);
        for (const auto &name : claims.exclusive_names) {
            const auto existing = claim_owner.find(name);
            if (existing != claim_owner.end()) {
                result.errors.push_back("behaviors: exclusive identity '" + name + "' is claimed by both '" +
                                        existing->second + "' and '" + spec.id + "'");
            } else {
                claim_owner.emplace(name, spec.id);
            }
        }
        for (const auto &group : claims.required_instance_groups) {
            if (config.target.instance_groups.find(group) == config.target.instance_groups.end()) {
                result.errors.push_back("behaviors[" + spec.id + "]: instance group '" + group +
                                        "' is not declared in target.instance_groups");
            }
        }
    }

    result.ok = result.errors.empty();
    result.config = std::move(config);
    return result;
}

LoadResult LoadScenarioFromFile(const std::string &path, const BehaviorRegistry &registry) {
    LoadResult result;
    std::string json;
    std::string error;
    if (!ReadFile(path, &json, &error)) {
        result.errors.push_back(error);
        return result;
    }
    return LoadScenarioFromJson(json, registry);
}

} // namespace kvcm_swarm
