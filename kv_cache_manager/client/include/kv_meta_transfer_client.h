#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.h"

namespace kv_cache_manager {

// Dedicated data plane for exact-size KVMeta objects. Unlike TransferClient,
// it validates each URI and buffer against the request's value_sizes and does
// not require every value size to be registered as a fixed location spec.
class KvMetaTransferClient {
public:
    virtual ~KvMetaTransferClient() = default;

    static std::unique_ptr<KvMetaTransferClient> Create(const std::string &client_config,
                                                        const InitParams &init_params,
                                                        std::uint64_t max_object_bytes);
    static std::unique_ptr<KvMetaTransferClient>
    Create(const std::string &client_config,
           const InitParams &init_params,
           std::uint64_t max_object_bytes,
           const SharedMemoryRegistration &shared_memory_registration);

    virtual ClientErrorCode LoadObjects(const UriStrVec &uri_str_vec,
                                        const std::vector<std::uint64_t> &value_sizes,
                                        const BlockBuffers &object_buffers) = 0;
    virtual std::pair<ClientErrorCode, UriStrVec>
    SaveObjects(const UriStrVec &uri_str_vec,
                const std::vector<std::uint64_t> &value_sizes,
                const BlockBuffers &object_buffers) = 0;

protected:
    KvMetaTransferClient() = default;
};

} // namespace kv_cache_manager
