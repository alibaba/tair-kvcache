#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/meta/meta_storage_backend.h"
#include "kv_cache_manager/meta/raft/raft_coordinator.h"

namespace kv_cache_manager {
namespace raft_meta {

// MetaRaftBackend — replicated MetaStorageBackend driven by the *process-wide*
// raft cluster owned by RaftCoordinator. One kvcm process is one raft node;
// every Instance hosted in the process shares the same raft group via
// RaftCoordinator and is isolated by op.instance_id at the state machine
// (Instance isolation, see CLAUDE.md / phase plan).
//
// MetaRaftBackend is a thin per-Instance proxy:
//   * Init pulls the singleton RaftCoordinator (set up by Server::Init in
//     the raft branch). It does *not* own a raft_server, log store, state
//     machine, or per-instance MetaLocalBackend — those all live in the
//     coordinator and the state machine respectively.
//   * Writes encode a LogOp tagged with this backend's instance_id, then
//     hand it to coordinator->AppendAndWait. The state machine routes by
//     instance_id when the entry commits.
//   * Reads delegate straight to the per-instance backend that the state
//     machine materialises on demand via the coordinator's factory closure.
class MetaRaftBackend : public MetaStorageBackend {
public:
    MetaRaftBackend();
    ~MetaRaftBackend() override;

    std::string GetStorageType() noexcept override;

    ErrorCode Init(const std::string &instance_id,
                   const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept override;
    ErrorCode Open() noexcept override;
    ErrorCode Close() noexcept override;

    // Writes — replicated through raft.
    std::vector<ErrorCode> Put(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               const CacheLocationMapVector &locations,
                               const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Upsert(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  const CacheLocationMapVector &locations,
                                  const PropertyMapVector &properties) noexcept override;
    std::vector<ErrorCode> Delete(RequestContext *request_context, const KeyTypeVec &keys) noexcept override;
    std::vector<ErrorCode> DeleteLocations(RequestContext *request_context,
                                           const KeyTypeVec &keys,
                                           const LocationIdsPerKey &location_ids) noexcept override;

    // Reads — served from the per-instance backend in the coordinator.
    std::vector<ErrorCode> Get(RequestContext *request_context,
                               const KeyTypeVec &keys,
                               CacheLocationMapVector &out_locations,
                               PropertyMapVector &out_properties) noexcept override;
    std::vector<ErrorCode> GetLocations(RequestContext *request_context,
                                        const KeyTypeVec &keys,
                                        CacheLocationMapVector &out_locations) noexcept override;
    std::vector<std::vector<ErrorCode>> GetLocations(RequestContext *request_context,
                                                     const KeyTypeVec &keys,
                                                     const LocationIdsPerKey &location_ids,
                                                     LocationsPerKey &out_locations) noexcept override;
    std::vector<ErrorCode> GetLocationIds(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          LocationIdsPerKey &out_location_ids) noexcept override;
    std::vector<ErrorCode> GetProperties(RequestContext *request_context,
                                         const KeyTypeVec &keys,
                                         const std::vector<std::string> &field_names,
                                         PropertyMapVector &out_properties) noexcept override;
    std::vector<ErrorCode> Exists(RequestContext *request_context,
                                  const KeyTypeVec &keys,
                                  std::vector<bool> &out_is_exist_vec) noexcept override;
    std::vector<ErrorCode> ExistsLocation(RequestContext *request_context,
                                          const KeyTypeVec &keys,
                                          std::vector<bool> &out_exists) noexcept override;
    ErrorCode ListKeys(RequestContext *request_context,
                       const std::string &cursor,
                       const int64_t limit,
                       std::string &out_next_cursor,
                       std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode RandomSample(RequestContext *request_context,
                           const int64_t count,
                           std::vector<KeyType> &out_keys) noexcept override;
    ErrorCode SampleReclaimKeys(RequestContext *request_context,
                                const int64_t count,
                                std::vector<KeyType> &out_keys) noexcept override;

    // Metadata — also goes through raft so all replicas agree on it.
    ErrorCode PutMetaData(const FieldMap &field_maps) noexcept override;
    ErrorCode GetMetaData(FieldMap &field_maps) noexcept override;

    // Raft introspection — proxies to the coordinator. Useful for tests and
    // diagnostics; production callers should query the coordinator directly.
    bool IsLeader() const;
    int32_t LeaderId() const;

    // Test hook: lets unit tests inject a stack-allocated coordinator instead
    // of relying on the process singleton. Must be called *before* Init.
    void SetCoordinatorForTest(RaftCoordinator *coordinator) { coordinator_override_ = coordinator; }

private:
    // Resolve the coordinator we should talk to. Returns the test override if
    // one is set, else the process singleton. Logs at error level when both
    // are missing.
    RaftCoordinator *Coordinator() const;

    // Look up the per-instance backend for read-side delegation. Returns
    // nullptr if the coordinator isn't installed; reads fall back to a
    // vector of EC_ERROR / empty results in that case.
    std::shared_ptr<MetaCacheBaseBackend> InstanceBackend() const;

    std::string instance_id_;
    RaftCoordinator *coordinator_override_ = nullptr;
    std::atomic<bool> opened_{false};
};

} // namespace raft_meta
} // namespace kv_cache_manager
