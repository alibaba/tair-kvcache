#pragma once

#include <vector>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/client/src/internal/sdk/cuda_util.h"

namespace kv_cache_manager {

struct IovDevice {
    const void *base;
    size_t size;
};

class SdkBufferCheckUtil {
public:
    static std::vector<int64_t> GetBlocksHash(const BlockBuffers &block_buffers);
    static std::vector<int64_t> GetBlocksHash(const BlockBuffers &block_buffers,
                                              IovDevice *iovs_d,
                                              uint32_t *crcs_d,
                                              size_t max_iov_num,
                                              cudaStream_t stream);
    static std::vector<int64_t> GetBlocksHash(const BlockBuffers &block_buffers,
                                              IovDevice *iovs_d,
                                              uint32_t *crcs_d,
                                              IovDevice *iovs_h_to_save,
                                              size_t max_iov_num,
                                              cudaStream_t stream);

    static std::vector<uint32_t> GetIovsCrc(const std::vector<IovDevice> &iovs_h);
    static std::vector<uint32_t>
    GetIovsCrc(const std::vector<IovDevice> &iovs_h, IovDevice *iovs_d, uint32_t *crcs_d, cudaStream_t stream);
    static std::vector<uint32_t>
    GetIovsCrc(const IovDevice *iovs_h_ptr, size_t iovs_size, IovDevice *iovs_d, uint32_t *crcs_d, cudaStream_t stream);

private:
    static size_t min_cal_byte_size_;
};

}; // namespace kv_cache_manager