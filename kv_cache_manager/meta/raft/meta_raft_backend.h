#pragma once

#include <libnuraft/launcher.hxx>
#include <libnuraft/raft_server.hxx>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/meta/meta_local_backend.h"
#include "kv_cache_manager/meta/meta_storage_backend.h"
#include "kv_cache_manager/meta/raft/meta_log_store.h"
#include "kv_cache_manager/meta/raft/meta_state_machine.h"
#include "kv_cache_manager/meta/raft/meta_state_mgr.h"

namespace kv_cache_manager {
namespace raft_meta {

// MetaRaftBackend — replicated MetaStorageBackend driven by NuRaft.
//
// The state-of-the-world for the meta layer lives in a single MetaLocalBackend
// owned by the inner state machine. Every external write turns into one log
// entry, gets append_entries() to NuRaft, and is applied to the backend via
// state_machine::commit() once committed by the cluster. Reads bypass the
// raft path and go directly to the local backend so the leader's serving
// path stays cheap (Phase 1 only the leader serves traffic; followers are
// HA backups — see CLAUDE.md / phase plan).
//
// URI format (passed via MetaStorageBackendConfig::storage_uri):
//   raft://0.0.0.0:9201?server_id=1
//                      &peers=1@10.0.0.1:9201,2@10.0.0.2:9201,3@10.0.0.3:9201
//                      &data_dir=/var/lib/kvcm/raft
//                      &snapshot_distance=100000
//                      &election_timeout_lower=300
//                      &election_timeout_upper=600
//                      &heart_beat_interval=100
//                      &local_capacity=32768
//                      &local_num_shard_bits=10
//                      &local_sample_times=10
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

    // Reads — served from the inner backend directly.
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

    // Raft introspection — needed by callers (RaftLeaderElector in PR 4).
    bool IsLeader() const;
    int32_t LeaderId() const;
    int32_t ServerId() const { return server_id_; }

    // Test hook: drives the state machine without an asio service. Used by
    // unit tests that want to verify commit() dispatch without spinning up
    // the full RPC stack.
    nuraft::ptr<MetaStateMachine> StateMachineForTest() const { return state_machine_; }

private:
    struct ParsedRaftConfig {
        int32_t server_id = 0;
        std::string self_endpoint;
        std::vector<PeerEntry> peers;
        std::string data_dir;
        int port = 0;
        int snapshot_distance = 100000;
        int election_timeout_lower = 300;
        int election_timeout_upper = 600;
        int heart_beat_interval = 100;
        // Inner MetaLocalBackend tunables.
        size_t local_capacity = 32 * 1024;
        int32_t local_num_shard_bits = 10;
        int32_t local_sample_times = 10;
    };

    static bool ParseRaftConfig(const std::string &uri, ParsedRaftConfig &out);

    // Append one already-encoded log entry and wait for the result.
    ErrorCode AppendOneAndWait(nuraft::ptr<nuraft::buffer> data);

    std::string instance_id_;
    int32_t server_id_ = 0;
    ParsedRaftConfig parsed_;

    std::shared_ptr<MetaLocalBackend> local_backend_;
    nuraft::ptr<MetaLogStore> log_store_;
    nuraft::ptr<MetaStateMachine> state_machine_;
    nuraft::ptr<MetaStateMgr> state_mgr_;
    std::unique_ptr<nuraft::raft_launcher> launcher_;
    nuraft::ptr<nuraft::raft_server> raft_server_;
    std::atomic<bool> opened_{false};
};

} // namespace raft_meta
} // namespace kv_cache_manager
