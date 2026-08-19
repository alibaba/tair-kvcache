#include "tools/kvcm_swarm/scenario/config.h"

#include "tools/kvcm_swarm/scenario/duration.h"

namespace kvcm_swarm {
namespace {

void PutDuration(rapidjson::Writer<rapidjson::StringBuffer> &writer, const char *key, Duration value) {
    const std::string text = FormatDuration(value);
    writer.Key(key);
    writer.String(text.data(), static_cast<rapidjson::SizeType>(text.size()), false);
}

} // namespace

void RuntimeConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    PutDuration(writer, "warmup", warmup);
    PutDuration(writer, "steady", steady);
    PutDuration(writer, "drain_timeout", drain_timeout);
    Put(writer, "workers", workers);
    Put(writer, "reactor_threads", reactor_threads);
    Put(writer, "grpc_completion_queues", grpc_completion_queues);
    writer.Key("limits");
    writer.StartObject();
    Put(writer, "max_in_flight_business_rpcs", limits.max_in_flight_business_rpcs);
    Put(writer, "max_in_flight_control_rpcs", limits.max_in_flight_control_rpcs);
    PutDuration(writer, "business_permit_wait_threshold", limits.business_permit_wait_threshold);
    Put(writer, "http_connections_per_endpoint", transport.http_connections_per_endpoint);
    Put(writer, "http_control_connections_per_endpoint", transport.http_control_connections_per_endpoint);
    PutDuration(writer, "connect_timeout", transport.connect_timeout);
    PutDuration(writer, "default_rpc_timeout", transport.default_rpc_timeout);
    writer.EndObject();
}

void InstanceGroupTarget::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "quota_bytes", quota_bytes);
}

void TargetConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    writer.Key("endpoints");
    writer.StartObject();
    Put(writer, "meta_http", endpoints.meta_http);
    Put(writer, "meta_grpc", endpoints.meta_grpc);
    Put(writer, "admin_http", endpoints.admin_http);
    Put(writer, "admin_grpc", endpoints.admin_grpc);
    writer.EndObject();
    Put(writer, "instance_groups", instance_groups);
}

void EvidenceConfig::ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept {
    Put(writer, "output_json", output_json);
    Put(writer, "violations_jsonl", violations_jsonl);
    Put(writer, "markdown_summary", markdown_summary);
}

} // namespace kvcm_swarm
