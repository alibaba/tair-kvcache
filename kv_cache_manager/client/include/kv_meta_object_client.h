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
inline constexpr std::uint32_t kKvMetaObjectClientApiVersion = 1;

// Returns the capability version compiled into kv_cache_manager_client.so.
// Consumers should compare this with kKvMetaObjectClientApiVersion before
// constructing native state, so a newer header cannot be paired silently with
// an incompatible shared library.
std::uint32_t GetKvMetaObjectClientApiVersion() noexcept;

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
    // Must be strictly longer than the configured data-plane put timeout plus
    // three metadata call-timeout windows (PutStart hand-off, masked-hit Get,
    // and PutFinish).
    std::int32_t write_timeout_seconds{30};
};

// Composes the KVMeta metadata transaction with its dedicated variable-size
// data plane. Each key is one opaque value and every buffer must cover the
// complete value; ignored or zero-length IOVs are rejected before StartWrite.
// GPU callers must make producer work visible before SaveObjects; this API
// accepts raw pointers and does not inherit framework-specific stream order.
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
    // instead of being reported as a hit; masked hits are checked with Get so
    // this remains true while talking to an older server during an upgrade.
    // Missing keys use one all-or-nothing-failure write session; callers must
    // not assume that all keys become visible simultaneously.
    virtual ClientErrorCode SaveObjects(const std::string &trace_id,
                                        const std::vector<std::string> &keys,
                                        const std::vector<std::uint64_t> &value_sizes,
                                        const BlockBuffers &object_buffers) = 0;

    // All keys must exist and match expected_value_sizes before any data I/O.
    virtual ClientErrorCode LoadObjects(const std::string &trace_id,
                                        const std::vector<std::string> &keys,
                                        const std::vector<std::uint64_t> &expected_value_sizes,
                                        const BlockBuffers &object_buffers) = 0;

    virtual ClientErrorCode Remove(const std::string &trace_id, const std::vector<std::string> &keys) = 0;

protected:
    KvMetaObjectClient() = default;
};

} // namespace kv_cache_manager
