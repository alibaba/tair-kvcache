#pragma once

#include "kv_cache_manager/protocol/protobuf/meta_service.pb.h"
#include "kv_cache_manager/routing/routing_table.h"

namespace kv_cache_manager {
template <typename Request>
RouteKey MetaRoute(const Request &request) {
    return request.routing_key_case() == Request::kEmbKey ? RouteKey::Embedding(request.emb_key())
                                                          : RouteKey::Session(request.session_id());
}

inline RouteKey MetaRoute(const proto::meta::RegisterInstanceRequest &) { return {Placement::All, {}}; }
inline RouteKey MetaRoute(const proto::meta::GetHostCacheStateRequest &) { return {Placement::All, {}}; }
inline RouteKey MetaRoute(const proto::meta::TrimCacheRequest &) { return {Placement::All, {}}; }
inline RouteKey MetaRoute(const proto::meta::GetInstanceInfoRequest &) { return {Placement::RoleZero, {}}; }
inline RouteKey MetaRoute(const proto::meta::GetClusterInfoRequest &) { return {Placement::All, {}}; }

bool IsGlobalReporterEvent(const proto::meta::EventItem &event);
RouteKey MetaRoute(const proto::meta::ReportEventRequest &request);
} // namespace kv_cache_manager
