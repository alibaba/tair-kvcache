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
    static std::unique_ptr<TransferClient>
    Create(const std::string &client_config,
           const InitParams &init_params,
           const SharedMemoryRegistration &shared_memory_registration);

    // deadline_ms: 绝对时间点（steady_clock 毫秒），到点后本次调用不再触碰 block_buffers；
    // 0 = 调用方不施加 deadline，退回 client 配置的静态超时预算。
    // 置于 trace_info 之后以保持既有位置参数调用兼容。
    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                         const BlockBuffers &block_buffers,
                                         std::shared_ptr<TransferTraceInfo> trace_info = nullptr,
                                         int64_t deadline_ms = 0) = 0;
    virtual std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                               const BlockBuffers &block_buffers,
                                                               std::shared_ptr<TransferTraceInfo> trace_info = nullptr,
                                                               int64_t deadline_ms = 0) = 0;

protected:
    TransferClient() = default;
    virtual ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) = 0;
};
} // namespace kv_cache_manager
