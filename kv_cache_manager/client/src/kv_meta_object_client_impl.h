#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
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
                           std::int32_t write_timeout_seconds,
                           std::string instance_id = {});
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
    void Close() noexcept override;

private:
    class OperationGuard {
    public:
        OperationGuard(KvMetaObjectClientImpl *owner, bool require_transfer);
        ~OperationGuard() noexcept;

        OperationGuard(const OperationGuard &) = delete;
        OperationGuard &operator=(const OperationGuard &) = delete;

        [[nodiscard]] bool admitted() const noexcept { return admitted_; }

    private:
        KvMetaObjectClientImpl *owner_{nullptr};
        bool admitted_{false};
    };

    static ClientErrorCode ValidateRequest(const std::vector<std::string> &keys,
                                           const std::vector<std::uint64_t> &value_sizes,
                                           const BlockBuffers &object_buffers,
                                           std::uint64_t max_object_bytes);
    ClientErrorCode ExtractUris(const std::vector<KvMetaValueLocation> &locations,
                                const std::vector<std::string> &keys,
                                const std::vector<std::uint64_t> &value_sizes,
                                UriStrVec &uris) const;
    ClientErrorCode AbortWrite(const std::string &trace_id,
                               const std::string &write_session_id,
                               std::size_t location_count,
                               ClientErrorCode original_error);
    [[nodiscard]] bool TryBeginOperation(bool require_transfer);
    void EndOperation() noexcept;

    std::unique_ptr<KvMetaClient> metadata_client_;
    std::unique_ptr<KvMetaTransferClient> transfer_client_;
    std::uint64_t max_object_bytes_{0};
    std::int32_t write_timeout_seconds_{0};
    // Non-empty for every public factory-created client. The empty value is
    // retained only for isolated implementation tests with synthetic URIs.
    std::string instance_id_;
    std::mutex mutex_;
    std::condition_variable lifecycle_condition_;
    std::size_t active_operations_{0};
    bool closing_{false};
    bool closed_{false};
};

} // namespace kv_cache_manager
