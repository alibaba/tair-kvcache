#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace kv_cache_manager {

class TransferClient {
public:
    virtual ~TransferClient() = default;
    static std::unique_ptr<TransferClient> Create(const std::string &client_config, const InitParams &init_params);

    // Optional expected_checksums enables read-side verification. Callers (typically
    // ManagerClient) feed the CacheLocation.checksum values returned by meta service.
    //   - nullptr or empty vector: verification skipped (matches legacy behavior).
    //   - size must equal block_buffers.size() (1:1).
    //   - An element == 0 is treated as "no checksum for this block" and skipped
    //     individually (compat with legacy data).
    //   - Any mismatch -> ER_CHECKSUM_MISMATCH; buffer contents may be partially
    //     written and should be discarded by the caller.
    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                         const BlockBuffers &block_buffers,
                                         std::shared_ptr<TransferTraceInfo> trace_info = nullptr,
                                         const std::vector<int64_t> *expected_checksums = nullptr) = 0;

    // Optional out_checksums collects the per-block checksum the SDK computed during
    // write, for the caller to forward to FinishWrite.
    //   - nullptr: no checksum computed (matches legacy behavior).
    //   - Non-null but built without CUDA/MUSA: vector cleared + warn log (caller
    //     degrades to no checksum upstream).
    //   - On success, vector size matches block_buffers.size(); if the upstream needs
    //     to pad zero entries for failed blocks within a mask, that is the caller's job.
    virtual std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                               const BlockBuffers &block_buffers,
                                                               std::shared_ptr<TransferTraceInfo> trace_info = nullptr,
                                                               std::vector<int64_t> *out_checksums = nullptr) = 0;

protected:
    TransferClient() = default;
    virtual ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) = 0;
};
} // namespace kv_cache_manager