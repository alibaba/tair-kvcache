#pragma once

#include <memory>
#include <string>
#include <vector>

#include "kv_cache_manager/client/include/kv_meta_object_client.h"
#include "kv_cache_manager/client/include/kv_meta_transfer_client.h"

namespace kv_cache_manager {

class KvMetaObjectClientImpl final : public KvMetaObjectClient {
public:
    KvMetaObjectClientImpl(std::unique_ptr<KvMetaClient> metadata_client,
                           std::unique_ptr<KvMetaTransferClient> transfer_client,
                           std::uint64_t max_object_bytes,
                           std::int32_t write_timeout_seconds);
    ~KvMetaObjectClientImpl() override = default;

    ClientErrorCode SaveObjects(const std::string &trace_id,
                                const std::vector<std::string> &keys,
                                const std::vector<std::uint64_t> &value_sizes,
                                const BlockBuffers &object_buffers) override;
    ClientErrorCode LoadObjects(const std::string &trace_id,
                                const std::vector<std::string> &keys,
                                const std::vector<std::uint64_t> &expected_value_sizes,
                                const BlockBuffers &object_buffers) override;
    ClientErrorCode Remove(const std::string &trace_id, const std::vector<std::string> &keys) override;

private:
    static ClientErrorCode ValidateRequest(const std::vector<std::string> &keys,
                                           const std::vector<std::uint64_t> &value_sizes,
                                           const BlockBuffers &object_buffers,
                                           std::uint64_t max_object_bytes);
    static ClientErrorCode ExtractUris(const std::vector<KvMetaValueLocation> &locations,
                                       const std::vector<std::uint64_t> &value_sizes,
                                       UriStrVec &uris);
    ClientErrorCode AbortWrite(const std::string &trace_id,
                               const std::string &write_session_id,
                               std::size_t location_count,
                               ClientErrorCode original_error);

    std::unique_ptr<KvMetaClient> metadata_client_;
    std::unique_ptr<KvMetaTransferClient> transfer_client_;
    std::uint64_t max_object_bytes_{0};
    std::int32_t write_timeout_seconds_{0};
};

} // namespace kv_cache_manager
