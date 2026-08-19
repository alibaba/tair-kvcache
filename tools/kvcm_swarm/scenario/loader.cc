#include "tools/kvcm_swarm/scenario/loader.h"

#include <cstdio>
#include <set>

#include "tools/kvcm_swarm/scenario/config_reader.h"
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

} // namespace

LoadResult LoadScenarioFromJson(const std::string &json, const BehaviorRegistry &registry) {
    LoadResult result;
    std::string parse_error;
    ConfigNode root = ConfigNode::Parse(json, &parse_error);
    if (!root.valid()) {
        result.errors.push_back(parse_error);
        return result;
    }
    if (!root.IsObject()) {
        result.errors.push_back("run configuration must be a JSON object");
        return result;
    }

    ScenarioConfig config;
    config.document = root;
    ConfigReader reader(root, &result.errors);
    config.name = reader.RequiredString("name");
    config.seed = reader.RequiredUint("seed");

    ConfigReader runtime = reader.Child("runtime");
    config.runtime.warmup = runtime.RequiredDuration("warmup");
    config.runtime.steady = runtime.RequiredDuration("steady");
    config.runtime.drain_timeout = runtime.RequiredDuration("drain_timeout");
    config.runtime.workers = static_cast<uint32_t>(runtime.RequiredUint("workers"));
    config.runtime.reactor_threads = static_cast<uint32_t>(runtime.OptionalUint("reactor_threads", 2));
    config.runtime.grpc_completion_queues = static_cast<uint32_t>(runtime.OptionalUint("grpc_completion_queues", 2));
    ConfigReader limits = runtime.Child("limits");
    config.runtime.limits.max_in_flight_business_rpcs =
        static_cast<uint32_t>(limits.OptionalUint("max_in_flight_business_rpcs", 4096));
    config.runtime.limits.max_in_flight_control_rpcs =
        static_cast<uint32_t>(limits.OptionalUint("max_in_flight_control_rpcs", 512));
    config.runtime.limits.business_permit_wait_threshold =
        limits.OptionalDuration("business_permit_wait_threshold", std::chrono::seconds(1));
    config.runtime.transport.http_connections_per_endpoint =
        static_cast<uint32_t>(limits.OptionalUint("http_connections_per_endpoint", 8));
    config.runtime.transport.http_control_connections_per_endpoint =
        static_cast<uint32_t>(limits.OptionalUint("http_control_connections_per_endpoint", 2));
    config.runtime.transport.connect_timeout = limits.OptionalDuration("connect_timeout", std::chrono::seconds(3));
    config.runtime.transport.default_rpc_timeout =
        limits.OptionalDuration("default_rpc_timeout", std::chrono::seconds(10));

    if (config.runtime.warmup < Duration::zero()) {
        runtime.ErrorAt("warmup", "must not be negative");
    }
    if (config.runtime.steady <= Duration::zero()) {
        runtime.ErrorAt("steady", "must be positive");
    }
    if (config.runtime.drain_timeout <= Duration::zero()) {
        runtime.ErrorAt("drain_timeout", "must be positive");
    }
    if (config.runtime.workers == 0) {
        runtime.ErrorAt("workers", "must be at least 1");
    }
    if (config.runtime.reactor_threads == 0) {
        runtime.ErrorAt("reactor_threads", "must be at least 1");
    }
    if (config.runtime.grpc_completion_queues == 0) {
        runtime.ErrorAt("grpc_completion_queues", "must be at least 1");
    }
    if (config.runtime.limits.max_in_flight_business_rpcs == 0) {
        limits.ErrorAt("max_in_flight_business_rpcs", "must be positive");
    }
    if (config.runtime.limits.max_in_flight_control_rpcs == 0) {
        limits.ErrorAt("max_in_flight_control_rpcs", "must be positive");
    }
    if (config.runtime.transport.http_connections_per_endpoint == 0) {
        limits.ErrorAt("http_connections_per_endpoint", "must be positive");
    }
    if (config.runtime.transport.http_control_connections_per_endpoint == 0) {
        limits.ErrorAt("http_control_connections_per_endpoint", "must be positive");
    }

    ConfigReader target = reader.Child("target");
    ConfigReader endpoints = target.Child("endpoints");
    config.target.endpoints.meta_http = endpoints.RequiredString("meta_http");
    config.target.endpoints.meta_grpc = endpoints.RequiredString("meta_grpc");
    config.target.endpoints.admin_http = endpoints.RequiredString("admin_http");
    // The test topology serves AdminService on the meta gRPC port when the
    // admin RPC port is not separated; the effective value is reported.
    config.target.endpoints.admin_grpc = endpoints.OptionalString("admin_grpc", config.target.endpoints.meta_grpc);

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

    ConfigNode groups_node = target.node().Get("instance_groups");
    if (groups_node.IsObject()) {
        for (const std::string &group_name : groups_node.Keys()) {
            ConfigReader group(groups_node.Get(group_name), &result.errors);
            InstanceGroupTarget entry;
            entry.name = group_name;
            entry.quota_bytes = group.RequiredUint("quota_bytes");
            config.target.instance_groups.emplace(group_name, entry);
        }
    } else {
        result.errors.push_back("target.instance_groups: required object mapping group name to {quota_bytes}");
    }

    ConfigReader evidence = reader.Child("evidence");
    config.evidence.output_json = evidence.RequiredString("output_json");
    config.evidence.violations_jsonl = evidence.RequiredString("violations_jsonl");
    config.evidence.markdown_summary = evidence.OptionalString("markdown_summary", "");
    if (config.evidence.output_json.empty()) {
        evidence.ErrorAt("output_json", "must not be empty");
    }
    if (config.evidence.violations_jsonl.empty()) {
        evidence.ErrorAt("violations_jsonl", "must not be empty");
    }
    config.preflight_enabled = reader.OptionalBool("preflight", true);

    std::set<std::string> behavior_ids;
    for (ConfigNode behavior_node : reader.RequiredArray("behaviors")) {
        ConfigReader behavior(behavior_node, &result.errors);
        BehaviorSpec spec;
        spec.id = behavior.RequiredString("id");
        spec.type = behavior.RequiredString("type");
        const std::string transport = behavior.RequiredString("transport");
        if (transport == "http") {
            spec.transport = TransportKind::kHttp;
        } else if (transport == "grpc") {
            spec.transport = TransportKind::kGrpc;
        } else if (transport == "https" || transport == "grpcs" || transport == "tls" || transport == "mtls") {
            behavior.ErrorAt("transport", "TLS transports are not supported and must not be silently downgraded");
        } else {
            behavior.ErrorAt("transport", "must be 'http' or 'grpc'");
        }
        spec.config = behavior_node.Get("config");
        if (spec.id.empty()) {
            behavior.ErrorAt("id", "must not be empty");
        } else if (!behavior_ids.insert(spec.id).second) {
            result.errors.push_back("behaviors: duplicate behavior id '" + spec.id + "'");
        }
        const BehaviorFactory *factory = registry.Find(spec.type);
        if (factory == nullptr) {
            behavior.ErrorAt("type", "unknown behavior type '" + spec.type + "'");
        } else {
            const ValidationResult validation = factory->Validate(spec);
            for (const auto &error : validation.errors) {
                result.errors.push_back(error);
            }
        }
        config.behaviors.push_back(std::move(spec));
    }
    if (config.behaviors.empty()) {
        result.errors.push_back("behaviors: at least one behavior is required");
    }

    // Cross-behavior checks, expressed only through behavior-declared claims so
    // the common loader stays free of any domain type.
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

    std::vector<std::string> unknown;
    root.CollectUnknown(&unknown);
    for (const auto &key : unknown) {
        result.errors.push_back("unknown configuration field: " + key);
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
