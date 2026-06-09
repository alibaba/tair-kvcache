#include "kv_cache_manager/client/src/internal/sdk/sdk_interface.h"

namespace kv_cache_manager {

SdkInterface::GroupMap SdkInterface::SplitByPath(const std::vector<DataStorageUri> &remote_uris,
                                                 const BlockBuffers &local_buffers) {
    GroupMap groups;
    for (size_t i = 0; i < remote_uris.size(); ++i) {
        const auto &uri = remote_uris[i];
        auto &group = groups[uri.GetPath()];
        group.remote_uris.push_back(uri);
        group.local_buffers.push_back(local_buffers[i]);
        group.buffer_indices.push_back(i);
    }
    return groups;
}
} // namespace kv_cache_manager