#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace grpc {
class ServerBuilder;
}

namespace kv_cache_manager {

class MetricsRegistry;
class MetricsReporter;

// Resolve 0 to hardware concurrency, with at least a poller and a handler.
// -1 keeps the framework default; positive values are explicit limits.
int ResolveGrpcMaxThreads(int configured_max_threads, unsigned int hardware_threads);

// Apply per-server quotas and count the final status prepared for each RPC,
// including thread quota rejections. A zero memory quota keeps the framework default.
// Returns the resolved thread limit, or -1 for the framework default.
int ConfigureGrpcServer(grpc::ServerBuilder &builder,
                        int configured_max_threads,
                        std::size_t memory_quota_bytes,
                        const std::string &server_name,
                        const std::shared_ptr<MetricsRegistry> &metrics_registry,
                        const std::shared_ptr<MetricsReporter> &metrics_reporter);

} // namespace kv_cache_manager
