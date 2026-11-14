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

// Resolve 0 to at least 32 threads for pollers and synchronous IO handlers,
// or hardware concurrency if larger, plus one per allowed long-lived stream.
// -1 keeps the framework default; positive values are explicit limits.
int ResolveGrpcMaxThreads(int configured_max_threads,
                          unsigned int hardware_threads,
                          std::size_t reserved_stream_threads = 0);

// Apply per-server quotas and count the final status prepared for each RPC,
// including thread quota rejections. A zero memory quota keeps the framework default.
// Returns the resolved thread limit, or -1 for the framework default.
int ConfigureGrpcServer(grpc::ServerBuilder &builder,
                        int configured_max_threads,
                        std::size_t memory_quota_bytes,
                        const std::string &server_name,
                        const std::shared_ptr<MetricsRegistry> &metrics_registry,
                        const std::shared_ptr<MetricsReporter> &metrics_reporter,
                        std::size_t reserved_stream_threads = 0);

} // namespace kv_cache_manager
