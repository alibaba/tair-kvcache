#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace kv_cache_manager {

class DataStorageBackend;
class DataStorageManager;
class StorageConfig;
enum class DataStorageType : uint8_t;

struct ReclaimProviderCount {
    uint32_t count{0};
    std::chrono::steady_clock::time_point observed_at{};
};

bool SupportsAdaptiveDelete(DataStorageType type);
std::string GetReclaimClusterEndpoint(const StorageConfig &config);
ReclaimProviderCount GetReclaimProviderCount(const std::weak_ptr<DataStorageBackend> &source);

struct ReclaimDeleteUri {
    const std::string &uri;
    bool skip;
};

struct ReclaimStorageResult {
    // Complete allows the usual conditional metadata cleanup, including ordinary errors listed in failed_specs.
    enum class Action {
        Complete,
        Retry,
        Retain
    };
    struct Location {
        Action action{Action::Complete};
        const char *reason{nullptr}; // Stable diagnostic label for Retain.
    };
    std::vector<Location> locations; // In AddLocation order.
    // Ordinary delete errors follow existing metadata cleanup and identify the affected specs for logging.
    std::vector<std::pair<size_t, size_t>> failed_specs;
    size_t requests{0};
    size_t busy_replies{0};
    uint64_t retry_after_ms{0};
    bool recovery_confirmed{false};
};

// One registered storage and one logical batch, across attempts. The caller owns metadata and scheduling.
class ReclaimStorageBatch {
public:
    virtual ~ReclaimStorageBatch() = default;
    // Consume the URI views during the call. Validate skipped specs too. An invalid Location adds nothing.
    virtual bool AddLocation(const std::vector<ReclaimDeleteUri> &specs) = 0;
    virtual bool HasRequests() const = 0;
    // Consume this attempt's payload. Later attempts resupply revalidated Locations. A non-null
    // stop reason prevents dispatch and supplies the stable reason to record for unsent targets.
    virtual ReclaimStorageResult Send(const DataStorageManager &storage,
                                      const std::string &trace_id,
                                      const std::function<const char *()> &stop_reason) = 0;
};

// Unsupported storage types use the ordinary Delete path.
std::unique_ptr<ReclaimStorageBatch> CreateReclaimStorageBatch(DataStorageType type, const std::string &name);

} // namespace kv_cache_manager
