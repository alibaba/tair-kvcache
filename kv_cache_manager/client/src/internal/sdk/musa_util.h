#pragma once

#include <musa_runtime.h>

#include "kv_cache_manager/common/logger.h"

#define CHECK_MUSA_ERROR(musa_call, format, args...)                                                                   \
    do {                                                                                                               \
        musaError_t err = (musa_call);                                                                                 \
        if (err != musaSuccess) {                                                                                      \
            KVCM_LOG_WARN("musa error [%d] [%s] | " format, err, musaGetErrorString(err), ##args);                     \
        }                                                                                                              \
    } while (0)

#define CHECK_MUSA_ERROR_RETURN(musa_call, return_value, format, args...)                                              \
    do {                                                                                                               \
        musaError_t err = (musa_call);                                                                                 \
        if (err != musaSuccess) {                                                                                      \
            KVCM_LOG_WARN("musa error [%d] [%s] | " format, err, musaGetErrorString(err), ##args);                     \
            return return_value;                                                                                       \
        }                                                                                                              \
    } while (0)
