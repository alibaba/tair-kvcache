// Short aliases for the KVCM protobuf namespaces.
//
// The Swarm depends on the repository's protobuf definitions as the
// authoritative protocol shape; it never reaches into server internals.
#pragma once

#include "service/proto/admin_service.pb.h"
#include "service/proto/meta_service.pb.h"

namespace kvcm_swarm {

namespace meta = ::kv_cache_manager::proto::meta;
namespace admin = ::kv_cache_manager::proto::admin;

} // namespace kvcm_swarm
