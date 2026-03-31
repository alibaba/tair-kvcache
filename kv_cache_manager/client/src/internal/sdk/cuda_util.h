#pragma once

#include <cuda_runtime.h>

#include "kv_cache_manager/common/logger.h"

#define CHECK_CUDA_ERROR(cuda_call, format, args...)                                                                   \
    do {                                                                                                               \
        cudaError_t err = (cuda_call);                                                                                 \
        if (err != cudaSuccess) {                                                                                      \
            KVCM_LOG_WARN("cuda error [%d] [%s] | " format, err, cudaGetErrorString(err), ##args);                     \
        }                                                                                                              \
    } while (0)

#define CHECK_CUDA_ERROR_RETURN(cuda_call, return_value, format, args...)                                              \
    do {                                                                                                               \
        cudaError_t err = (cuda_call);                                                                                 \
        if (err != cudaSuccess) {                                                                                      \
            KVCM_LOG_WARN("cuda error [%d] [%s] | " format, err, cudaGetErrorString(err), ##args);                     \
            return return_value;                                                                                       \
        }                                                                                                              \
    } while (0)

namespace kv_cache_manager {

class CudaBufferGuard {
public:
    CudaBufferGuard() = default;
    ~CudaBufferGuard() {
        if (ptr_ != nullptr) {
            auto err = cudaFree(ptr_);
            if (err != cudaSuccess) {
                KVCM_LOG_ERROR("cudaFree [%p] failed in destructor: %s", ptr_, cudaGetErrorString(err));
            }
        }
    }

    bool Alloc(size_t size) {
        auto err = cudaMalloc(&ptr_, size);
        if (err != cudaSuccess) {
            ptr_ = nullptr;
            KVCM_LOG_ERROR("cudaMalloc [%lu] bytes failed: %s", size, cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    void *Get() const { return ptr_; }

    bool Free() {
        if (ptr_ == nullptr) {
            return true;
        }
        auto err = cudaFree(ptr_);
        if (err != cudaSuccess) {
            KVCM_LOG_ERROR("cudaFree [%p] failed: %s", ptr_, cudaGetErrorString(err));
            return false;
        }
        ptr_ = nullptr;
        return true;
    }

private:
    void *ptr_ = nullptr;
};

}; // namespace kv_cache_manager