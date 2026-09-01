#include "kv_cache_manager/manager/event_report_cleanup_util.h"

#include "kv_cache_manager/data_storage/event_report_backend.h"
#include "kv_cache_manager/data_storage/snapshot_uri_utils.h"
#include "kv_cache_manager/meta/cache_location.h"

namespace kv_cache_manager {

bool IsSnapshotLocationStaleForCleanup(const EventReportBackend *event_backend,
                                       const std::string &instance_id,
                                       const CacheLocation &location,
                                       bool preserve_in_flight) {
    if (!event_backend) {
        return false;
    }
    std::string medium;
    std::string reporter_host;
    if (!event_backend->ParseLocationId(location.id(), medium, reporter_host)) {
        return false;
    }

    const ReporterSnapshotKey reporter_key{instance_id, reporter_host};
    std::string committed_version;
    std::string in_flight_version;
    event_backend->GetSnapshotVersionTokens(reporter_key, committed_version, in_flight_version);
    if (location.location_specs().empty()) {
        return true;
    }
    bool contains_committed = false;
    bool contains_in_flight = false;
    for (const auto &spec : location.location_specs()) {
        const size_t version_param_count =
            SnapshotUriUtils::CountUriParam(spec.uri(), SnapshotUriUtils::kSnapshotVersionParam);
        if (version_param_count == 0) {
            continue;
        }
        SnapshotUriInfo info;
        if (version_param_count != 1 || !SnapshotUriUtils::ParseSnapshotUriInfo(spec.uri(), info)) {
            // Legacy cleanup has only a boolean keep/delete contract. Treat
            // malformed metadata fail-closed: a malformed sibling must never
            // authorize deleting a Location that may also contain a current
            // committed or in-flight spec.
            return false;
        }
        contains_committed = contains_committed || (!committed_version.empty() && info.version == committed_version);
        contains_in_flight = contains_in_flight ||
                             (preserve_in_flight && !in_flight_version.empty() && info.version == in_flight_version);
    }
    return !contains_committed && !contains_in_flight;
}

} // namespace kv_cache_manager
