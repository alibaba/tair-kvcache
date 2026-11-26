#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace kv_cache_manager {

struct ProcessExtensionContext {
    std::string config_file;
    std::unordered_map<std::string, std::string> overrides;
};

// Process-scoped services, independent of storage and shard leadership.
// The host calls Start after daemonization/logger setup and before Server::Init.
// Stop is called even when Start fails or throws, after the Server is destroyed
// and before logger teardown. It must drain work, be idempotent and not throw.
// Lifecycle calls are serialized. Implementations must not install signal
// handlers or take over the host's process lifecycle.
class ProcessExtension {
public:
    virtual ~ProcessExtension() = default;
    virtual bool Start(const ProcessExtensionContext &context) = 0;
    virtual void Stop() noexcept = 0;
};

// Selected by the existing stub_source build boundary. The open-source
// implementation is a no-op and has no deployment-specific dependencies.
std::unique_ptr<ProcessExtension> CreateProcessExtension();

} // namespace kv_cache_manager
