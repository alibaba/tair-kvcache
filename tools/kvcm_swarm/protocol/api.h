#pragma once

#include "kv_cache_manager/client/src/internal/async_rpc/api.h"

namespace kvcm_swarm {

using kv_cache_manager::async_rpc::AllApis;
using kv_cache_manager::async_rpc::Api;
using kv_cache_manager::async_rpc::ApiInfo;
using kv_cache_manager::async_rpc::ApiName;
using kv_cache_manager::async_rpc::ExtractServiceMessage;
using kv_cache_manager::async_rpc::ExtractServiceStatus;
using kv_cache_manager::async_rpc::GetApiInfo;
using kv_cache_manager::async_rpc::kStatusOk;
using kv_cache_manager::async_rpc::kStatusServerNotLeader;
using kv_cache_manager::async_rpc::ServiceEndpoint;
using kv_cache_manager::async_rpc::ServiceEndpointName;

} // namespace kvcm_swarm
