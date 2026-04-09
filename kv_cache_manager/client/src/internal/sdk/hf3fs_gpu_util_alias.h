#pragma once

#include "kv_cache_manager/client/src/internal/sdk/hf3fs_cuda_util.h"
#include "kv_cache_manager/client/src/internal/sdk/hf3fs_musa_util.h"

namespace kv_cache_manager {

// If both were defined, CUDA takes precedence (builds should enable at most one).
#if defined(USING_CUDA)
using Hf3fsGpuUtil = Hf3fsCudaUtil;
#elif defined(USING_MUSA)
using Hf3fsGpuUtil = Hf3fsMusaUtil;
#else
using Hf3fsGpuUtil = Hf3fsGpuUtilStub;
#endif

} // namespace kv_cache_manager
