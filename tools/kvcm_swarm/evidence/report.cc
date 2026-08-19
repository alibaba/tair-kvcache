#include "tools/kvcm_swarm/evidence/report.h"

#include <cstdio>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>

#include "tools/kvcm_swarm/evidence/json_writer.h"
#include "tools/kvcm_swarm/scenario/config_reader.h"

namespace kvcm_swarm {
namespace {

const std::vector<std::string> &Limitations() {
    static const std::vector<std::string> limitations = {
        "metadata-only: no KV bytes are transferred, so backend bandwidth, byte content and SDK write failures are "
        "not exercised",
        "transport is plaintext HTTP and insecure gRPC only; HTTPS/TLS/mTLS are rejected, never downgraded",
        "only the healthy client path is simulated: no circuit breaker, fault injection, client abort, orphan "
        "WRITING, partial write failure or retry storm",
        "no snapshot handling: neither startup nor periodic full BLOCK_SNAPSHOT is sent, and snapshot_required is "
        "not acted upon",
        "single hot tier (ST_EVENT_REPORT_L2); L1P5 + L2 dual reporters are out of scope",
        "server-side usage accounting is not queried: KVCM exposes no API with the same accounting basis, so usage "
        "convergence is an observation and C4 stays non-gating",
        "results apply only to the actual scale, topology and configuration recorded in this report and must not be "
        "extrapolated to the data plane, to failure recovery or to unverified production capacity",
    };
    return limitations;
}

void WriteHistogram(JsonWriter &writer, const Histogram &histogram) {
    writer.BeginObject();
    writer.KeyUint("count", histogram.count());
    writer.KeyDouble("mean_ms", histogram.mean_ms());
    writer.KeyDouble("min_ms", histogram.min_ms());
    writer.KeyDouble("p50_ms", histogram.Quantile(0.5));
    writer.KeyDouble("p90_ms", histogram.Quantile(0.9));
    writer.KeyDouble("p99_ms", histogram.Quantile(0.99));
    writer.KeyDouble("max_ms", histogram.max_ms());
    writer.EndObject();
}

void WriteLane(JsonWriter &writer, const LaneStats &stats) {
    writer.BeginObject();
    writer.KeyUint("acquired", stats.acquired);
    writer.KeyUint("immediate", stats.immediate);
    writer.KeyUint("waited", stats.waited);
    writer.KeyUint("rejected", stats.rejected);
    writer.KeyUint("in_flight_current", stats.in_flight);
    writer.KeyUint("in_flight_peak", stats.peak_in_flight);
    writer.KeyUint("peak_wait_queue", stats.peak_wait_queue);
    writer.KeyDouble("wait_ms_total", static_cast<double>(stats.wait_ns_total) / 1e6);
    writer.KeyDouble("wait_ms_max", static_cast<double>(stats.wait_ns_max) / 1e6);
    writer.EndObject();
}

} // namespace

ResourceUsage CollectResourceUsage() {
    ResourceUsage usage;
    std::FILE *status = std::fopen("/proc/self/status", "r");
    if (status != nullptr) {
        char line[256];
        while (std::fgets(line, sizeof(line), status) != nullptr) {
            uint64_t value = 0;
            if (std::sscanf(line, "Threads: %lu", &value) == 1) {
                usage.threads = value;
            } else if (std::sscanf(line, "VmRSS: %lu kB", &value) == 1) {
                usage.rss_bytes = value * 1024;
            } else if (std::sscanf(line, "VmHWM: %lu kB", &value) == 1) {
                usage.peak_rss_bytes = value * 1024;
            }
        }
        std::fclose(status);
    }
    rusage resources{};
    if (::getrusage(RUSAGE_SELF, &resources) == 0) {
        usage.user_cpu_seconds =
            static_cast<double>(resources.ru_utime.tv_sec) + static_cast<double>(resources.ru_utime.tv_usec) / 1e6;
        usage.system_cpu_seconds =
            static_cast<double>(resources.ru_stime.tv_sec) + static_cast<double>(resources.ru_stime.tv_usec) / 1e6;
    }
    return usage;
}

std::string BuildRunReportJson(const RunReportInput &input) {
    JsonWriter writer(true);
    writer.BeginObject();

    // ---- run ----
    writer.Key("run");
    writer.BeginObject();
    writer.KeyString("name", input.config->name);
    writer.KeyUint("seed", input.config->seed);
    writer.KeyString("started_at", FormatWallClock(input.started_wall_ms));
    writer.KeyString("ended_at", FormatWallClock(input.ended_wall_ms));
    writer.KeyDouble("duration_ms", ToMillis(input.total_duration));
    writer.KeyString("exit_reason", input.exit_reason);
    writer.KeyBool("initialize_ok", input.initialize_ok);
    writer.KeyBool("drain_complete", input.drain_complete);
    writer.KeyBool("quiesced", input.quiesced);
    writer.KeyBool("generator_saturated", input.admission->saturated());
    writer.Key("generator_saturation_reasons");
    writer.BeginArray();
    for (const auto &reason : input.admission->saturation_reasons()) {
        writer.String(reason);
    }
    writer.EndArray();
    writer.KeyBool("metadata_only", true);
    writer.EndObject();

    // ---- run_config (effective) ----
    writer.Key("run_config");
    writer.BeginObject();
    writer.KeyString("name", input.config->name);
    writer.KeyUint("seed", input.config->seed);
    writer.Key("runtime");
    writer.BeginObject();
    writer.KeyString("warmup", FormatDuration(input.config->runtime.warmup));
    writer.KeyString("steady", FormatDuration(input.config->runtime.steady));
    writer.KeyString("drain_timeout", FormatDuration(input.config->runtime.drain_timeout));
    writer.KeyUint("workers", input.config->runtime.workers);
    writer.KeyUint("reactor_threads", input.config->runtime.reactor_threads);
    writer.KeyUint("grpc_completion_queues", input.config->runtime.grpc_completion_queues);
    writer.Key("limits");
    writer.BeginObject();
    writer.KeyUint("max_in_flight_business_rpcs", input.config->runtime.limits.max_in_flight_business_rpcs);
    writer.KeyUint("max_in_flight_control_rpcs", input.config->runtime.limits.max_in_flight_control_rpcs);
    writer.KeyString("business_permit_wait_threshold",
                     FormatDuration(input.config->runtime.limits.business_permit_wait_threshold));
    writer.KeyUint("http_connections_per_endpoint", input.config->runtime.transport.http_connections_per_endpoint);
    writer.KeyUint("http_control_connections_per_endpoint",
                   input.config->runtime.transport.http_control_connections_per_endpoint);
    writer.KeyString("connect_timeout", FormatDuration(input.config->runtime.transport.connect_timeout));
    writer.KeyString("default_rpc_timeout", FormatDuration(input.config->runtime.transport.default_rpc_timeout));
    writer.EndObject();
    writer.EndObject();
    writer.Key("target");
    writer.BeginObject();
    writer.Key("endpoints");
    writer.BeginObject();
    writer.KeyString("meta_http", input.config->target.endpoints.meta_http);
    writer.KeyString("meta_grpc", input.config->target.endpoints.meta_grpc);
    writer.KeyString("admin_http", input.config->target.endpoints.admin_http);
    writer.KeyString("admin_grpc", input.config->target.endpoints.admin_grpc);
    writer.EndObject();
    writer.Key("instance_groups");
    writer.BeginObject();
    for (const auto &entry : input.config->target.instance_groups) {
        writer.Key(entry.first);
        writer.BeginObject();
        writer.KeyUint("quota_bytes", entry.second.quota_bytes);
        writer.EndObject();
    }
    writer.EndObject();
    writer.EndObject();
    writer.KeyBool("preflight", input.config->preflight_enabled);
    writer.Key("behaviors");
    writer.BeginObject();
    for (ClientBehavior *behavior : *input.behaviors) {
        writer.Key(behavior->Id());
        writer.BeginObject();
        writer.KeyString("type", std::string(behavior->TypeName()));
        writer.Key("config");
        behavior->WriteEffectiveConfig(writer);
        writer.EndObject();
    }
    writer.EndObject();
    writer.Key("evidence");
    writer.BeginObject();
    writer.KeyString("output_json", input.config->evidence.output_json);
    writer.KeyString("violations_jsonl", input.config->evidence.violations_jsonl);
    writer.KeyString("markdown_summary", input.config->evidence.markdown_summary);
    writer.EndObject();
    writer.EndObject();

    // ---- phases ----
    writer.Key("phases");
    writer.BeginObject();
    for (const auto &record : *input.phases) {
        if (!record.entered) {
            continue;
        }
        writer.Key(PhaseName(record.phase));
        writer.BeginObject();
        writer.KeyBool("entered", true);
        writer.KeyDouble("duration_ms", ToMillis(record.end - record.start));
        writer.EndObject();
    }
    writer.EndObject();

    // ---- runtime ----
    writer.Key("runtime");
    writer.BeginObject();
    writer.Key("generator_lag");
    writer.BeginObject();
    writer.KeyUint("executor_workers", input.executor->worker_count());
    writer.KeyUint("executor_tasks_scheduled", input.executor->scheduled_total());
    writer.KeyUint("executor_peak_queue_depth", input.executor->peak_queue_depth());
    writer.KeyDouble("executor_queue_delay_ms_total", input.executor->queue_delay_sum_ms());
    writer.KeyUint("executor_queue_delay_samples", input.executor->queue_delay_samples());
    writer.KeyDouble("executor_queue_delay_ms_mean",
                     input.executor->queue_delay_samples() == 0
                         ? 0.0
                         : input.executor->queue_delay_sum_ms() /
                               static_cast<double>(input.executor->queue_delay_samples()));
    writer.KeyUint("timers_scheduled", input.executor->timer_count());
    writer.EndObject();
    writer.Key("admission");
    writer.BeginObject();
    writer.Key("business");
    WriteLane(writer, input.admission->Snapshot(TrafficLane::kBusiness));
    writer.Key("control");
    WriteLane(writer, input.admission->Snapshot(TrafficLane::kControl));
    writer.KeyUint("saturation_events", input.admission->saturation_events());
    writer.EndObject();
    writer.Key("resource_usage");
    writer.BeginObject();
    writer.KeyUint("threads", input.resources.threads);
    writer.KeyUint("io_threads", input.transports->io_thread_count());
    writer.KeyUint("rss_bytes", input.resources.rss_bytes);
    writer.KeyUint("peak_rss_bytes", input.resources.peak_rss_bytes);
    writer.KeyDouble("user_cpu_seconds", input.resources.user_cpu_seconds);
    writer.KeyDouble("system_cpu_seconds", input.resources.system_cpu_seconds);
    writer.EndObject();
    writer.EndObject();

    // ---- behaviors ----
    writer.Key("behaviors");
    writer.BeginObject();
    for (ClientBehavior *behavior : *input.behaviors) {
        writer.Key(behavior->Id());
        behavior->WriteReport(writer);
    }
    writer.EndObject();

    // ---- rpc ----
    const auto aggregates = input.evidence->RpcSnapshot();
    writer.Key("rpc");
    writer.BeginObject();
    writer.Key("by_api_phase");
    writer.BeginArray();
    uint64_t total_rpcs = 0;
    uint64_t total_success = 0;
    for (const auto &entry : aggregates) {
        total_rpcs += entry.second.total;
        total_success += entry.second.success;
        writer.BeginObject();
        writer.KeyString("behavior_type", entry.first.behavior_type);
        writer.KeyString("behavior_id", entry.first.behavior_id);
        writer.KeyString("api", entry.first.api);
        writer.KeyString("phase", PhaseName(entry.first.phase));
        writer.KeyString("lane", TrafficLaneName(entry.first.lane));
        writer.KeyUint("total", entry.second.total);
        writer.KeyUint("success", entry.second.success);
        writer.KeyUint("transport_failures", entry.second.transport_failures);
        writer.KeyUint("service_failures", entry.second.service_failures);
        writer.KeyUint("uncertain", entry.second.uncertain);
        writer.KeyDouble("success_rate",
                         entry.second.total == 0
                             ? 0.0
                             : static_cast<double>(entry.second.success) / static_cast<double>(entry.second.total));
        writer.Key("transport_errors");
        writer.BeginObject();
        for (const auto &error : entry.second.transport_errors) {
            writer.KeyUint(error.first, error.second);
        }
        writer.EndObject();
        writer.Key("service_statuses");
        writer.BeginObject();
        for (const auto &status : entry.second.service_statuses) {
            writer.KeyUint(std::to_string(status.first), status.second);
        }
        writer.EndObject();
        writer.Key("latency");
        WriteHistogram(writer, entry.second.latency);
        writer.Key("permit_wait");
        WriteHistogram(writer, entry.second.permit_wait);
        writer.Key("queue_delay");
        WriteHistogram(writer, entry.second.queue_delay);
        writer.EndObject();
    }
    writer.EndArray();
    writer.KeyUint("total", total_rpcs);
    writer.KeyUint("success", total_success);
    writer.KeyDouble("success_rate",
                     total_rpcs == 0 ? 0.0 : static_cast<double>(total_success) / static_cast<double>(total_rpcs));
    writer.EndObject();

    // ---- transport ----
    writer.Key("transport");
    writer.BeginObject();
    writer.KeyUint("io_threads", input.transports->io_thread_count());
    writer.Key("contexts");
    writer.BeginArray();
    for (const auto &context : input.transports->CollectStats()) {
        writer.BeginObject();
        writer.KeyString("behavior_type", context.identity.behavior_type);
        writer.KeyString("behavior_id", context.identity.behavior_id);
        writer.KeyString("process_id", context.identity.process_id);
        writer.KeyString("kind", TransportKindName(context.kind));
        writer.Key("endpoints");
        writer.BeginArray();
        for (const auto &endpoint : context.endpoints) {
            writer.BeginObject();
            writer.KeyString("endpoint", endpoint.endpoint);
            writer.KeyString("role", endpoint.role);
            writer.KeyUint("channels", endpoint.channels);
            writer.KeyUint("connections_current", endpoint.connections_current);
            writer.KeyUint("connections_peak", endpoint.connections_peak);
            writer.KeyUint("connections_created", endpoint.connections_created);
            writer.KeyUint("connections_reused", endpoint.connections_reused);
            writer.KeyUint("in_flight_peak", endpoint.in_flight_peak);
            writer.Key("establish_latency");
            WriteHistogram(writer, endpoint.establish_latency_ms);
            writer.EndObject();
        }
        writer.EndArray();
        writer.EndObject();
    }
    writer.EndArray();
    writer.EndObject();

    // ---- cache ----
    writer.Key("cache");
    writer.BeginObject();
    for (ClientBehavior *behavior : *input.behaviors) {
        JsonWriter probe(false);
        if (!behavior->WriteCacheReport(probe)) {
            continue;
        }
        writer.Key(behavior->Id());
        writer.RawValue(probe.Take());
    }
    writer.EndObject();

    // ---- invariants ----
    writer.Key("invariants");
    writer.BeginObject();
    writer.Key("checks");
    writer.BeginArray();
    for (ClientBehavior *behavior : *input.behaviors) {
        for (const auto &observation : behavior->Invariants()) {
            writer.BeginObject();
            writer.KeyString("behavior_type", observation.behavior_type);
            writer.KeyString("behavior_id", behavior->Id());
            writer.KeyString("check_name", observation.check_name);
            writer.KeyString("status", CheckStatusName(observation.status));
            writer.KeyUint("checked", observation.checked);
            writer.KeyUint("violations", observation.violations);
            writer.KeyString("reason", observation.reason);
            writer.Key("counters");
            writer.BeginObject();
            for (const auto &counter : observation.counters) {
                writer.KeyInt(counter.first, counter.second);
            }
            writer.EndObject();
            writer.Key("detail_preview");
            writer.BeginArray();
            for (const auto &detail : observation.detail_preview) {
                writer.RawValue(detail);
            }
            writer.EndArray();
            writer.EndObject();
        }
    }
    writer.EndArray();
    writer.KeyString("violations_jsonl", input.evidence->violations().path());
    writer.KeyUint("violations_total", input.evidence->violations().total());
    writer.KeyBool("violations_log_failed", input.evidence->violations().failed());
    writer.EndObject();

    // ---- workload_shape ----
    writer.Key("workload_shape");
    writer.BeginObject();
    for (ClientBehavior *behavior : *input.behaviors) {
        JsonWriter probe(false);
        if (!behavior->WriteWorkloadShape(probe)) {
            continue;
        }
        writer.Key(behavior->Id());
        writer.RawValue(probe.Take());
    }
    writer.EndObject();

    // ---- usage_observations ----
    writer.Key("usage_observations");
    writer.BeginObject();
    writer.KeyString("note",
                     "KVCM exposes no server API with the same accounting basis in this version, so the generator "
                     "reports only what the client can observe: confirmed cold allocations, hot reporter locations "
                     "and local residue. `client_observed` repeats the per-behavior `cache` section so this section "
                     "is self-contained. Usage convergence is an observation, never a gate.");
    writer.Key("client_observed");
    writer.BeginObject();
    for (ClientBehavior *behavior : *input.behaviors) {
        JsonWriter probe(false);
        if (!behavior->WriteCacheReport(probe)) {
            continue;
        }
        writer.Key(behavior->Id());
        writer.RawValue(probe.Take());
    }
    writer.EndObject();
    writer.EndObject();

    // ---- limitations ----
    writer.Key("limitations");
    writer.BeginArray();
    for (const auto &limitation : Limitations()) {
        writer.String(limitation);
    }
    writer.EndArray();

    // ---- cleanup ----
    writer.Key("cleanup");
    writer.BeginObject();
    writer.Key("preflight");
    writer.BeginObject();
    writer.KeyBool("executed", input.preflight->executed);
    writer.KeyBool("passed", input.preflight->passed);
    writer.KeyString("failure_stage", input.preflight->failure_stage);
    writer.KeyString("failure_detail", input.preflight->failure_detail);
    writer.KeyString("temporary_instance_id", input.preflight->temporary_instance_id);
    writer.KeyString("temporary_host_ip_port", input.preflight->temporary_host_ip_port);
    writer.KeyUint("remove_cache_calls", input.preflight->remove_cache_calls);
    writer.Key("steps");
    writer.BeginArray();
    for (const auto &step : input.preflight->steps) {
        writer.BeginObject();
        writer.KeyString("step", step.first);
        writer.KeyBool("ok", step.second);
        writer.EndObject();
    }
    writer.EndArray();
    writer.Key("notes");
    writer.BeginArray();
    for (const auto &note : input.preflight->cleanup_notes) {
        writer.String(note);
    }
    writer.EndArray();
    writer.EndObject();
    writer.Key("behaviors");
    writer.BeginObject();
    for (ClientBehavior *behavior : *input.behaviors) {
        JsonWriter probe(false);
        if (!behavior->WriteCleanupReport(probe)) {
            continue;
        }
        writer.Key(behavior->Id());
        writer.RawValue(probe.Take());
    }
    writer.EndObject();
    writer.KeyBool("drain_complete", input.drain_complete);
    writer.KeyBool("quiesced", input.quiesced);
    writer.EndObject();

    writer.EndObject();
    return writer.Take();
}

std::string RenderRunSummary(const RunReportInput &input) {
    std::ostringstream out;
    const auto aggregates = input.evidence->RpcSnapshot();
    uint64_t total = 0;
    uint64_t success = 0;
    for (const auto &entry : aggregates) {
        total += entry.second.total;
        success += entry.second.success;
    }
    out << "# KVCM Swarm run: " << input.config->name << "\n";
    out << "- seed: " << input.config->seed << "\n";
    out << "- duration: " << ToMillis(input.total_duration) << " ms\n";
    out << "- exit reason: " << input.exit_reason << "\n";
    out << "- metadata-only: yes (no KV bytes moved)\n";
    out << "- transport endpoints: meta_http=" << input.config->target.endpoints.meta_http
        << " meta_grpc=" << input.config->target.endpoints.meta_grpc
        << " admin_http=" << input.config->target.endpoints.admin_http << "\n";
    out << "- generator saturated: " << (input.admission->saturated() ? "yes" : "no") << "\n";
    out << "- RPCs: " << total << " total, " << success << " successful";
    if (total > 0) {
        out << " (" << (100.0 * static_cast<double>(success) / static_cast<double>(total)) << "%)";
    }
    out << "\n";
    out << "- executor workers: " << input.executor->worker_count()
        << ", network I/O threads: " << input.transports->io_thread_count()
        << ", process threads: " << input.resources.threads << "\n";
    out << "- peak RSS: " << input.resources.peak_rss_bytes << " bytes\n";
    out << "\n## Contracts\n";
    for (ClientBehavior *behavior : *input.behaviors) {
        for (const auto &observation : behavior->Invariants()) {
            out << "- [" << CheckStatusName(observation.status) << "] " << observation.check_name << " ("
                << behavior->Id() << "): checked=" << observation.checked << " violations=" << observation.violations
                << " - " << observation.reason << "\n";
        }
    }
    out << "\n## Behaviors\n";
    for (ClientBehavior *behavior : *input.behaviors) {
        out << "- " << behavior->Id() << " (" << behavior->TypeName() << ")\n";
    }
    out << "\n## Known limitations\n";
    for (const auto &limitation : Limitations()) {
        out << "- " << limitation << "\n";
    }
    return out.str();
}

} // namespace kvcm_swarm
