#pragma once

#include <string>

namespace kv_cache_manager {

class CacheLocation;
class EventReportBackend;

bool IsSnapshotLocationStaleForCleanup(const EventReportBackend *event_backend,
                                       const std::string &instance_id,
                                       const CacheLocation &location,
                                       bool preserve_in_flight);

} // namespace kv_cache_manager
