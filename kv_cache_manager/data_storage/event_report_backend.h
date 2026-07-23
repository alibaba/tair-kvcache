#pragma once

#include <memory>
#include <utility>

#include "kv_cache_manager/data_storage/vineyard_backend.h"

namespace kv_cache_manager {

// Metadata-only backend for cache state owned by an inference engine. It
// deliberately reuses the proven node liveness and generation fencing from
// VineyardBackend, but exposes a distinct storage type and URI protocol so
// clients never mistake engine-local blocks for Vineyard objects.
class EventReportBackend final : public VineyardBackend {
public:
    explicit EventReportBackend(std::shared_ptr<MetricsRegistry> metrics_registry)
        : VineyardBackend(std::move(metrics_registry),
                          DataStorageType::DATA_STORAGE_TYPE_EVENT_REPORT,
                          "event-report",
                          "kvs#event#",
                          "event_report.") {}
};

} // namespace kv_cache_manager
