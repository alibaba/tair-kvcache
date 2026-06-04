#include "kv_cache_manager/meta/raft/meta_raft_backend.h"

#include <libnuraft/asio_service.hxx>
#include <libnuraft/asio_service_options.hxx>
#include <libnuraft/async.hxx>
#include <libnuraft/raft_params.hxx>

#include <sstream>
#include <utility>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/common/standard_uri.h"
#include "kv_cache_manager/common/string_util.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/raft/raft_log_codec.h"

namespace kv_cache_manager {
namespace raft_meta {

using nuraft::asio_service;
using nuraft::buffer;
using nuraft::cmd_result;
using nuraft::cmd_result_code;
using nuraft::cs_new;
using nuraft::ptr;
using nuraft::raft_launcher;
using nuraft::raft_params;
using nuraft::raft_server;

namespace {

// Helper: split "id@host:port,id@host:port" into PeerEntry list. Endpoints
// missing the leading id default to a learner-style entry with id 0 (the
// caller treats id 0 as malformed and returns false).
bool ParsePeers(const std::string &raw, std::vector<PeerEntry> &out) {
    std::stringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        size_t at = item.find('@');
        if (at == std::string::npos) {
            return false;
        }
        std::string id_str = item.substr(0, at);
        std::string ep = item.substr(at + 1);
        int64_t id_i64 = 0;
        if (!StringUtil::StrToInt64(id_str.c_str(), id_i64) || id_i64 <= 0 || ep.empty()) {
            return false;
        }
        PeerEntry pe;
        pe.server_id = static_cast<int32_t>(id_i64);
        pe.endpoint = ep;
        out.push_back(std::move(pe));
    }
    return true;
}

// Translate a NuRaft cmd_result_code to KVCM ErrorCode. NOT_LEADER maps to
// EC_BADARGS so callers can detect "wrong server, route to leader" without
// confusing it with a generic IO failure.
ErrorCode CmdCodeToEc(cmd_result_code code) {
    switch (code) {
    case cmd_result_code::OK:
        return EC_OK;
    case cmd_result_code::NOT_LEADER:
        return EC_BADARGS;
    case cmd_result_code::TIMEOUT:
    case cmd_result_code::CANCELLED:
    default:
        return EC_ERROR;
    }
}

} // namespace

MetaRaftBackend::MetaRaftBackend() = default;

MetaRaftBackend::~MetaRaftBackend() {
    if (opened_.load()) {
        Close();
    }
}

std::string MetaRaftBackend::GetStorageType() noexcept { return META_RAFT_BACKEND_TYPE_STR; }

bool MetaRaftBackend::ParseRaftConfig(const std::string &uri, ParsedRaftConfig &out) {
    StandardUri u = StandardUri::FromUri(uri);
    if (!u.Valid()) {
        KVCM_LOG_ERROR("MetaRaftBackend: invalid raft uri[%s]", uri.c_str());
        return false;
    }
    out.port = static_cast<int>(u.GetPort());
    out.self_endpoint = u.GetHostName().empty() ? std::string() : (u.GetHostName() + ":" + std::to_string(out.port));

    int64_t sid = 0;
    StringUtil::StrToInt64(u.GetParam("server_id").c_str(), sid);
    out.server_id = static_cast<int32_t>(sid);
    if (out.server_id <= 0) {
        KVCM_LOG_ERROR("MetaRaftBackend: server_id missing or non-positive in uri[%s]", uri.c_str());
        return false;
    }

    std::string peers_raw = u.GetParam("peers");
    if (!peers_raw.empty() && !ParsePeers(peers_raw, out.peers)) {
        KVCM_LOG_ERROR("MetaRaftBackend: failed to parse peers[%s]", peers_raw.c_str());
        return false;
    }

    out.data_dir = u.GetParam("data_dir");
    if (out.data_dir.empty()) {
        KVCM_LOG_ERROR("MetaRaftBackend: data_dir is required");
        return false;
    }

    u.GetParamAs("snapshot_distance", out.snapshot_distance);
    u.GetParamAs("election_timeout_lower", out.election_timeout_lower);
    u.GetParamAs("election_timeout_upper", out.election_timeout_upper);
    u.GetParamAs("heart_beat_interval", out.heart_beat_interval);
    u.GetParamAs("local_capacity", out.local_capacity);
    u.GetParamAs("local_num_shard_bits", out.local_num_shard_bits);
    u.GetParamAs("local_sample_times", out.local_sample_times);
    return true;
}

ErrorCode MetaRaftBackend::Init(const std::string &instance_id,
                                const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (instance_id.empty() || !config) {
        return EC_BADARGS;
    }
    instance_id_ = instance_id;

    if (!ParseRaftConfig(config->GetStorageUri(), parsed_)) {
        return EC_BADARGS;
    }
    server_id_ = parsed_.server_id;

    // Build inner MetaLocalBackend with a synthetic config so it gets the
    // tuning values from our outer URI.
    auto inner_cfg = std::make_shared<MetaStorageBackendConfig>(META_LOCAL_BACKEND_TYPE_STR);
    {
        std::ostringstream local_uri;
        local_uri << "local://?capacity=" << parsed_.local_capacity << "&num_shard_bits=" << parsed_.local_num_shard_bits
                  << "&sample_times=" << parsed_.local_sample_times;
        inner_cfg->SetStorageUri(local_uri.str());
    }
    local_backend_ = std::make_shared<MetaLocalBackend>();
    if (local_backend_->Init(instance_id_, inner_cfg) != EC_OK) {
        KVCM_LOG_ERROR("MetaRaftBackend: inner MetaLocalBackend Init failed");
        return EC_ERROR;
    }
    if (local_backend_->Open() != EC_OK) {
        KVCM_LOG_ERROR("MetaRaftBackend: inner MetaLocalBackend Open failed");
        return EC_ERROR;
    }

    log_store_ = cs_new<MetaLogStore>();
    state_machine_ = cs_new<MetaStateMachine>(local_backend_, parsed_.data_dir + "/snapshot");
    state_mgr_ = cs_new<MetaStateMgr>(server_id_, parsed_.self_endpoint, parsed_.peers, parsed_.data_dir + "/state",
                                      log_store_);
    return EC_OK;
}

ErrorCode MetaRaftBackend::Open() noexcept {
    if (opened_.load()) {
        return EC_OK;
    }
    raft_params params;
    params.with_election_timeout_lower(parsed_.election_timeout_lower)
        .with_election_timeout_upper(parsed_.election_timeout_upper)
        .with_hb_interval(parsed_.heart_beat_interval)
        .with_snapshot_enabled(parsed_.snapshot_distance);

    asio_service::options asio_opts;
    raft_server::init_options init_opts;
    init_opts.skip_initial_election_timeout_ = (parsed_.peers.size() <= 1);

    launcher_ = std::make_unique<raft_launcher>();
    raft_server_ =
        launcher_->init(state_machine_, state_mgr_, /*logger=*/nullptr, parsed_.port, asio_opts, params, init_opts);
    if (!raft_server_) {
        KVCM_LOG_ERROR("MetaRaftBackend: raft_launcher init failed for server[%d] port[%d]", server_id_, parsed_.port);
        return EC_ERROR;
    }
    opened_.store(true);
    KVCM_LOG_INFO("MetaRaftBackend: opened, server_id[%d] endpoint[%s] peers[%zu] data_dir[%s]",
                  server_id_,
                  parsed_.self_endpoint.c_str(),
                  parsed_.peers.size(),
                  parsed_.data_dir.c_str());
    return EC_OK;
}

ErrorCode MetaRaftBackend::Close() noexcept {
    if (!opened_.exchange(false)) {
        return EC_OK;
    }
    if (launcher_) {
        launcher_->shutdown();
        launcher_.reset();
    }
    raft_server_.reset();
    if (local_backend_) {
        local_backend_->Close();
    }
    return EC_OK;
}

bool MetaRaftBackend::IsLeader() const { return raft_server_ && raft_server_->is_leader(); }

int32_t MetaRaftBackend::LeaderId() const { return raft_server_ ? raft_server_->get_leader() : -1; }

ErrorCode MetaRaftBackend::AppendOneAndWait(ptr<buffer> data) {
    if (!raft_server_) {
        return EC_ERROR;
    }
    if (!raft_server_->is_leader()) {
        return EC_BADARGS;
    }
    std::vector<ptr<buffer>> logs{data};
    auto result = raft_server_->append_entries(logs);
    if (!result) {
        return EC_ERROR;
    }
    if (!result->get_accepted()) {
        return CmdCodeToEc(result->get_result_code());
    }
    // Block until commit; cmd_result.get() returns the value, but for app_log
    // we only care that the commit happened — use get_result_code() afterwards.
    result->get();
    return CmdCodeToEc(result->get_result_code());
}

// ---------- Writes (replicated) ----------

std::vector<ErrorCode> MetaRaftBackend::Put(RequestContext * /*request_context*/,
                                            const KeyTypeVec &keys,
                                            const CacheLocationMapVector &locations,
                                            const PropertyMapVector &properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        LogOp op;
        op.type = OpType::kPut;
        op.key = keys[i];
        op.locations = locations[i];
        op.properties = properties[i];
        results[i] = AppendOneAndWait(Encode(op));
    }
    return results;
}

std::vector<ErrorCode> MetaRaftBackend::Upsert(RequestContext * /*request_context*/,
                                               const KeyTypeVec &keys,
                                               const CacheLocationMapVector &locations,
                                               const PropertyMapVector &properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        LogOp op;
        op.type = OpType::kUpsert;
        op.key = keys[i];
        op.locations = locations[i];
        op.properties = properties[i];
        results[i] = AppendOneAndWait(Encode(op));
    }
    return results;
}

std::vector<ErrorCode> MetaRaftBackend::Delete(RequestContext * /*request_context*/,
                                               const KeyTypeVec &keys) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        LogOp op;
        op.type = OpType::kDelete;
        op.key = keys[i];
        results[i] = AppendOneAndWait(Encode(op));
    }
    return results;
}

std::vector<ErrorCode> MetaRaftBackend::DeleteLocations(RequestContext * /*request_context*/,
                                                        const KeyTypeVec &keys,
                                                        const LocationIdsPerKey &location_ids) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    for (size_t i = 0; i < keys.size(); ++i) {
        if (location_ids[i].empty()) {
            continue;
        }
        LogOp op;
        op.type = OpType::kDeleteLocations;
        op.key = keys[i];
        op.location_ids = location_ids[i];
        results[i] = AppendOneAndWait(Encode(op));
    }
    return results;
}

// ---------- Reads (delegated) ----------

std::vector<ErrorCode> MetaRaftBackend::Get(RequestContext *ctx,
                                            const KeyTypeVec &keys,
                                            CacheLocationMapVector &out_locations,
                                            PropertyMapVector &out_properties) noexcept {
    return local_backend_->Get(ctx, keys, out_locations, out_properties);
}

std::vector<ErrorCode> MetaRaftBackend::GetLocations(RequestContext *ctx,
                                                     const KeyTypeVec &keys,
                                                     CacheLocationMapVector &out_locations) noexcept {
    return local_backend_->GetLocations(ctx, keys, out_locations);
}

std::vector<std::vector<ErrorCode>> MetaRaftBackend::GetLocations(RequestContext *ctx,
                                                                  const KeyTypeVec &keys,
                                                                  const LocationIdsPerKey &location_ids,
                                                                  LocationsPerKey &out_locations) noexcept {
    return local_backend_->GetLocations(ctx, keys, location_ids, out_locations);
}

std::vector<ErrorCode> MetaRaftBackend::GetLocationIds(RequestContext *ctx,
                                                       const KeyTypeVec &keys,
                                                       LocationIdsPerKey &out_location_ids) noexcept {
    return local_backend_->GetLocationIds(ctx, keys, out_location_ids);
}

std::vector<ErrorCode> MetaRaftBackend::GetProperties(RequestContext *ctx,
                                                      const KeyTypeVec &keys,
                                                      const std::vector<std::string> &field_names,
                                                      PropertyMapVector &out_properties) noexcept {
    return local_backend_->GetProperties(ctx, keys, field_names, out_properties);
}

std::vector<ErrorCode> MetaRaftBackend::Exists(RequestContext *ctx,
                                               const KeyTypeVec &keys,
                                               std::vector<bool> &out_is_exist_vec) noexcept {
    return local_backend_->Exists(ctx, keys, out_is_exist_vec);
}

std::vector<ErrorCode> MetaRaftBackend::ExistsLocation(RequestContext *ctx,
                                                       const KeyTypeVec &keys,
                                                       std::vector<bool> &out_exists) noexcept {
    return local_backend_->ExistsLocation(ctx, keys, out_exists);
}

ErrorCode MetaRaftBackend::ListKeys(RequestContext *ctx,
                                    const std::string &cursor,
                                    const int64_t limit,
                                    std::string &out_next_cursor,
                                    std::vector<KeyType> &out_keys) noexcept {
    return local_backend_->ListKeys(ctx, cursor, limit, out_next_cursor, out_keys);
}

ErrorCode MetaRaftBackend::RandomSample(RequestContext *ctx,
                                        const int64_t count,
                                        std::vector<KeyType> &out_keys) noexcept {
    return local_backend_->RandomSample(ctx, count, out_keys);
}

ErrorCode MetaRaftBackend::SampleReclaimKeys(RequestContext *ctx,
                                             const int64_t count,
                                             std::vector<KeyType> &out_keys) noexcept {
    return local_backend_->SampleReclaimKeys(ctx, count, out_keys);
}

ErrorCode MetaRaftBackend::PutMetaData(const FieldMap &field_maps) noexcept {
    LogOp op;
    op.type = OpType::kPutMetaData;
    op.meta_fields = field_maps;
    return AppendOneAndWait(Encode(op));
}

ErrorCode MetaRaftBackend::GetMetaData(FieldMap &field_maps) noexcept {
    return local_backend_->GetMetaData(field_maps);
}

} // namespace raft_meta
} // namespace kv_cache_manager
