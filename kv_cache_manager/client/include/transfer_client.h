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

    // 任务 82620492：可选 expected_hashes 用于读端校验。当 spec 配置了
    // enable_meta_checksum 后，上层 (ManagerClient) 应该把 meta service 返回的
    // CacheLocation.block_hash 收集成 vector 传进来。
    //   - nullptr 或空 vector  → 跳过校验，行为同老版本。
    //   - expected_hashes->size() 必须与 block_buffers.size() 一致 (1:1)。
    //   - 数组元素 == 0 视为"该 block 无 hash"，单独跳过 (兼容老数据)。
    //   - 任一 block hash 不匹配 → 返回 ER_CHECKSUM_MISMATCH，buffer 中的数据
    //     可能已被部分写入；调用方应丢弃。
    virtual ClientErrorCode LoadKvCaches(const UriStrVec &uri_str_vec,
                                         const BlockBuffers &block_buffers,
                                         std::shared_ptr<TransferTraceInfo> trace_info = nullptr,
                                         const std::vector<int64_t> *expected_hashes = nullptr) = 0;

    // 任务 82620492：可选 out_block_hashes 用于写端算 hash 上报。当 spec 配置了
    // enable_meta_checksum 后，上层 (ManagerClient) 应该传入一个 vector，
    // SaveKvCaches 成功后会填入与 block_buffers 对齐的 hash 列表，上层再透传
    // 到 FinishWrite。
    //   - nullptr → 不算 hash，行为同老版本。
    //   - 非 nullptr 但环境没有 CUDA/MUSA → 留空 vector + warn 日志 (上层退化)。
    //   - 返回的 vector size 与 block_buffers.size() 一致；
    //     若上层在 mask 阶段需要为"失败 block"填占位，按需自己 zero-fill。
    virtual std::pair<ClientErrorCode, UriStrVec> SaveKvCaches(const UriStrVec &uri_str_vec,
                                                               const BlockBuffers &block_buffers,
                                                               std::shared_ptr<TransferTraceInfo> trace_info = nullptr,
                                                               std::vector<int64_t> *out_block_hashes = nullptr) = 0;

protected:
    TransferClient() = default;
    virtual ClientErrorCode Init(const std::string &client_config, const InitParams &init_params) = 0;
};
} // namespace kv_cache_manager