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
