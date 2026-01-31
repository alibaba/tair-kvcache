#pragma once

#include "kv_cache_manager/common/logger.h"

#ifdef USING_CUDA
#include <cuda_runtime.h>
#define CHECK_CUDA_ERROR(cuda_call)                                                                                    \
    do {                                                                                                               \
        cudaError_t err = (cuda_call);                                                                                 \
        if (err != cudaSuccess) {                                                                                      \
            KVCM_LOG_ERROR("%s failed, err: %s ", #cuda_call, cudaGetErrorString(err));                                \
            throw std::runtime_error("cuda runtime error: " + std::string(cudaGetErrorString(err)));                   \
        }                                                                                                              \
    } while (0)
#endif

namespace kv_cache_manager {

#ifdef USING_CUDA

class Hf3fsCudaUtil final {
public:
    Hf3fsCudaUtil() = default;
    ~Hf3fsCudaUtil() { Release(); }

public:
    bool Init() {
        auto err = cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            KVCM_LOG_WARN("init failed, create cuda stream failed, err: %s", cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    void Release() {
        if (cuda_stream_) {
            CHECK_CUDA_ERROR(cudaStreamDestroy(cuda_stream_));
            cuda_stream_ = nullptr;
        }
    }

    void CopyAsyncHostToDevice(void *dst, const void *src, size_t size) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cuda_stream_));
    }

    void CopyAsyncDeviceToHost(void *dst, const void *src, size_t size) {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, cuda_stream_));
    }

    void Sync() { CHECK_CUDA_ERROR(cudaStreamSynchronize(cuda_stream_)); }

    bool RegisterHost(void *ptr, size_t size) const {
        if (ptr == nullptr || size == 0) {
            return false;
        }
        auto err = cudaHostRegister(ptr, size, cudaHostRegisterDefault);
        if (err != cudaSuccess) {
            KVCM_LOG_WARN("cuda host register failed, err: %s, ptr: %p, size: %zu", cudaGetErrorString(err), ptr, size);
            return false;
        }
        return true;
    }

    void UnregisterHost(void *ptr) const {
        if (ptr != nullptr) {
            CHECK_CUDA_ERROR(cudaHostUnregister(ptr));
        }
    }

private:
    cudaStream_t cuda_stream_{nullptr};
};

#else

// for without cuda
class Hf3fsCudaUtil final {
public:
    Hf3fsCudaUtil() = default;
    ~Hf3fsCudaUtil() { Release(); }

public:
    bool Init() { return true; }
    void Release() {}
    void CopyAsyncHostToDevice(void *dst, const void *src, size_t size) {}
    void CopyAsyncDeviceToHost(void *dst, const void *src, size_t size) {}
    void Sync() {}
    bool RegisterHost(void *ptr, size_t size) const { return true; }
    void UnregisterHost(void *ptr) const {}
};

#endif

} // namespace kv_cache_manager