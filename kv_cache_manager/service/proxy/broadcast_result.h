#pragma once

#include <vector>

#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"

namespace kv_cache_manager {
// New broadcast operations must define how to combine their successful responses.
template <typename Response>
bool MergeReplies(std::vector<Response> &, Response &) {
    return false;
}
bool MergeReplies(std::vector<proto::meta::CommonResponse> &replies, proto::meta::CommonResponse &out);
bool MergeReplies(std::vector<proto::meta::RegisterInstanceResponse> &replies,
                  proto::meta::RegisterInstanceResponse &out);
bool MergeReplies(std::vector<proto::meta::GetHostCacheStateResponse> &replies,
                  proto::meta::GetHostCacheStateResponse &out);
} // namespace kv_cache_manager
