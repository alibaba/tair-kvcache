#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "kv_meta_client.h"

namespace kv_cache_manager {

// Increment when the Python-visible object API changes incompatibly.  The
// packaged Python wrapper checks this value before constructing native state so
// a stale extension cannot silently reinterpret exact-size object requests.
inline constexpr std::uint32_t kKvMetaObjectClientApiVersion = 2;

// Returns the capability version compiled into kv_cache_manager_client.so.
// Consumers should compare this with kKvMetaObjectClientApiVersion before
// constructing native state, so a newer header cannot be paired silently with
// an incompatible shared library.
std::uint32_t GetKvMetaObjectClientApiVersion() noexcept;

// An independently versioned extension; binding against an older library must
// fail at load time rather than claiming that unknown DMA is safe to reuse.
std::uint32_t GetKvMetaObjectTransferStateVersion() noexcept;

struct KvMetaObjectResult {
    ClientErrorCode ec;
    bool data_transfer_may_be_inflight{true};
};

// Configuration for the isolated exact-key object path. The regular
// ManagerClient/TransferClient path does not read or enable this policy.
struct KvMetaObjectClientConfig {
    KvMetaClientConfig metadata;
    std::string instance_group;
    std::string user_data;
    std::string transfer_client_config;
    // storage_configs is replaced with the authoritative value returned by
    // RegisterInstance. The transfer config must use block_size=1, the sole
    // location spec marker {"value": 1}, and the same instance/group identity.
    InitParams transfer_init_params;
    std::uint64_t max_object_bytes{1024ULL * 1024ULL * 1024ULL};
    // Server-side lease for one PutStart/data-write/PutFinish transaction.
    std::int32_t write_timeout_seconds{30};
};

// Composes the KVMeta metadata transaction with its dedicated variable-size
// data plane. Each key is one opaque value and every buffer must cover the
// complete value; ignored or zero-length IOVs are rejected before StartWrite.
// GPU callers must make producer work visible before SaveObjects; this API
// accepts raw pointers and does not inherit framework-specific stream order.
// Calls on one object client may run concurrently. Close rejects new calls and
// safely waits for already-admitted synchronous operations.
class KvMetaObjectClient {
public:
    virtual ~KvMetaObjectClient() = default;

    static std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
    Create(const std::string &trace_id, const KvMetaObjectClientConfig &config);
    static std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
    Create(const std::string &trace_id,
           const KvMetaObjectClientConfig &config,
           const SharedMemoryRegistration &shared_memory_registration);

    // Committed keys of the same size are treated as cache hits and are not
    // overwritten. An active writer returns ER_SERVICE_WRITE_IN_PROGRESS
    // instead of being reported as a hit.
    // Missing keys use one all-or-nothing-failure write session; callers must
    // not assume that all keys become visible simultaneously.
    virtual ClientErrorCode SaveObjects(const std::string &trace_id,
                                        const std::vector<std::string> &keys,
                                        const std::vector<std::uint64_t> &value_sizes,
                                        const BlockBuffers &object_buffers) = 0;

    // Register immutable GPU arenas before PACE initialization. Retain every
    // allocation through Close and, if registrations.gpu_release_state is
    // unsafe afterward, until process exit. Registration alone does not prove
    // that the selected transport performed GPUDirect RDMA.
    static std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
    CreateWithMemoryRegistrations(const std::string &trace_id,
                                  const KvMetaObjectClientConfig &config,
                                  const ClientMemoryRegistrations &registrations);

    // All keys must exist and match expected_value_sizes before any data I/O.
    virtual ClientErrorCode LoadObjects(const std::string &trace_id,
                                        const std::vector<std::string> &keys,
                                        const std::vector<std::uint64_t> &expected_value_sizes,
                                        const BlockBuffers &object_buffers) = 0;

    virtual ClientErrorCode Remove(const std::string &trace_id, const std::vector<std::string> &keys) = 0;

    // Waits for operations already admitted by this client and releases its
    // metadata/data-plane resources. Close is idempotent; subsequent valid
    // operations return ER_CLIENT_NOT_EXISTS.
    virtual void Close() noexcept = 0;

    // False proves that this call either did not dispatch data I/O or completed
    // it successfully. Metadata failure/mutation uncertainty is independent of
    // buffer safety. A data-plane error remains conservative even if rollback
    // succeeds. Append these slots to retain the existing interface layout.
    virtual KvMetaObjectResult SaveObjectsWithTransferState(const std::string &trace_id,
                                                            const std::vector<std::string> &keys,
                                                            const std::vector<std::uint64_t> &value_sizes,
                                                            const BlockBuffers &object_buffers) {
        const auto ec = SaveObjects(trace_id, keys, value_sizes, object_buffers);
        return {ec, ec != ER_OK};
    }
    virtual KvMetaObjectResult LoadObjectsWithTransferState(const std::string &trace_id,
                                                            const std::vector<std::string> &keys,
                                                            const std::vector<std::uint64_t> &expected_value_sizes,
                                                            const BlockBuffers &object_buffers) {
        const auto ec = LoadObjects(trace_id, keys, expected_value_sizes, object_buffers);
        return {ec, ec != ER_OK};
    }

protected:
    KvMetaObjectClient() = default;
};

} // namespace kv_cache_manager
