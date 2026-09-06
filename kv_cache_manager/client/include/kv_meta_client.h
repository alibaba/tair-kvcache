#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace kv_cache_manager {

// These values intentionally mirror kv_meta_service.proto while keeping the
// public client API independent from generated protobuf headers.
enum class KvMetaStorageType : std::int32_t {
    UNSPECIFIED = 0,
    HF3FS = 1,
    MOONCAKE = 2,
    TAIR_MEMPOOL = 3,
    NFS = 4,
    VCNS_HF3FS = 5,
    DUMMY = 6,
    EVENT_REPORT_L1P5 = 7,
    EVENT_REPORT_L2 = 8,
    TAIR_MEMPOOL_SSD = 9,
};

struct KvMetaValueLocation {
    KvMetaStorageType type{KvMetaStorageType::UNSPECIFIED};
    // Exact logical byte count requested for this value. A dedicated dynamic
    // object I/O adapter must use this length; the existing fixed-block
    // TransferClient is not compatible. Values in one batch may differ.
    std::uint64_t value_size{0};
    Location location_specs;
};

struct KvMetaGetResult {
    // Both vectors are request-aligned. A miss has hit_mask[i] == false and a
    // default-constructed locations[i].
    std::vector<bool> hit_mask;
    std::vector<KvMetaValueLocation> locations;
};

struct KvMetaStartWriteResult {
    std::string write_session_id;
    // Request-aligned. true means the key already exists or has an active
    // writer and therefore has no entry in locations.
    std::vector<bool> key_mask;
    // Contains only key_mask=false entries, in request-relative order.
    std::vector<KvMetaValueLocation> locations;
};

struct KvMetaInstanceInfo {
    std::string quota_group_name;
    std::string instance_group_name;
    std::string instance_id;
};

struct KvMetaClientConfig {
    // Addresses must point to the isolated kvcm.kv_meta.rpc_port, not to the
    // existing MetaService port. Calls try the preferred endpoint first and
    // fail over on transport, not-leader, or not-ready responses.
    std::vector<std::string> addresses;
    std::string instance_id;
    std::uint32_t call_timeout_ms{3000};
};

class KvMetaClient {
public:
    virtual ~KvMetaClient() = default;

    // Creation validates configuration and creates lazy gRPC channels. It
    // does not require the server to be reachable at construction time.
    static std::unique_ptr<KvMetaClient> Create(const KvMetaClientConfig &config);

    virtual std::pair<ClientErrorCode, std::string>
    RegisterInstance(const std::string &trace_id,
                     const std::string &instance_group,
                     const std::string &user_data) = 0;

    virtual std::pair<ClientErrorCode, KvMetaInstanceInfo>
    GetInstanceInfo(const std::string &trace_id) = 0;

    virtual std::pair<ClientErrorCode, KvMetaGetResult>
    Get(const std::string &trace_id, const std::vector<std::string> &keys) = 0;

    virtual std::pair<ClientErrorCode, KvMetaStartWriteResult>
    StartWrite(const std::string &trace_id,
               const std::vector<std::string> &keys,
               const std::vector<std::uint64_t> &value_sizes,
               std::int32_t write_timeout_seconds) = 0;

    // success_keys is aligned with StartWriteResult.locations, not with the
    // original request. A single false value aborts the complete session.
    virtual ClientErrorCode FinishWrite(const std::string &trace_id,
                                        const std::string &write_session_id,
                                        const std::vector<bool> &success_keys) = 0;

    virtual ClientErrorCode Remove(const std::string &trace_id, const std::vector<std::string> &keys) = 0;
    virtual ClientErrorCode TrimAll(const std::string &trace_id, bool metadata_only = false) = 0;
};

} // namespace kv_cache_manager
