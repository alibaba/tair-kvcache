#pragma once

#ifdef USING_CUDA
#include <cuda_runtime.h>
#endif
#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"

namespace kv_cache_manager {
class LocalFileSdk : public SdkInterface {
public:
    ~LocalFileSdk() override;
    ClientErrorCode Init(const std::shared_ptr<SdkBackendConfig> &sdk_backend_config,
                         const std::shared_ptr<StorageConfig> &storage_config) override;
    SdkType Type() override;
    ClientErrorCode Get(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers) override;
    ClientErrorCode Put(const std::vector<DataStorageUri> &remote_uris,
                        const BlockBuffers &local_buffers,
                        std::shared_ptr<std::vector<DataStorageUri>> actual_remote_uris) override;

private:
    ClientErrorCode Alloc(const std::vector<DataStorageUri> &remote_uris,
                          std::vector<DataStorageUri> &alloc_uris) override;
    ClientErrorCode DoGet(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers);
    ClientErrorCode DoPut(const std::vector<DataStorageUri> &remote_uris, const BlockBuffers &local_buffers);

private:
    int64_t byte_size_per_block_;
#ifdef USING_CUDA
    cudaStream_t cuda_stream_ = nullptr;
#endif
};
} // namespace kv_cache_manager