#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "kv_meta_client.h"

namespace kv_cache_manager {

// Configuration for the isolated exact-key object path. The regular
// ManagerClient/TransferClient path does not read or enable this policy.
struct KvMetaObjectClientConfig {
    KvMetaClientConfig metadata;
    std::string instance_group;
    std::string user_data;
    std::string transfer_client_config;
    // storage_configs is replaced with the authoritative value returned by
    // RegisterInstance. The location spec must be the KVMeta marker "value".
    InitParams transfer_init_params;
    std::uint64_t max_object_bytes{1024ULL * 1024ULL * 1024ULL};
    std::int32_t write_timeout_seconds{30};
};

// Composes the KVMeta metadata transaction with its dedicated variable-size
// data plane. Each key is one opaque value and every buffer must cover the
// complete value; ignored or zero-length IOVs are rejected before StartWrite.
class KvMetaObjectClient {
public:
    virtual ~KvMetaObjectClient() = default;

    static std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
    Create(const std::string &trace_id, const KvMetaObjectClientConfig &config);
    static std::pair<ClientErrorCode, std::unique_ptr<KvMetaObjectClient>>
    Create(const std::string &trace_id,
           const KvMetaObjectClientConfig &config,
           const SharedMemoryRegistration &shared_memory_registration);

    // Existing keys of the same size are treated as cache hits and are not
    // overwritten. Missing keys are committed atomically as one write session.
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
