#pragma once

#include "kv_cache_manager/client/src/internal/async_rpc/cancellation.h"

namespace kvcm_swarm {

using StopCallbackGuard = kv_cache_manager::async_rpc::CancellationCallbackGuard;
using StopSource = kv_cache_manager::async_rpc::CancellationSource;
using StopState = kv_cache_manager::async_rpc::CancellationState;
using StopToken = kv_cache_manager::async_rpc::CancellationToken;

} // namespace kvcm_swarm
