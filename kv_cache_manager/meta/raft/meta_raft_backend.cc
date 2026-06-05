#include "kv_cache_manager/meta/raft/meta_raft_backend.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/config/meta_storage_backend_config.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/raft/raft_log_codec.h"

namespace kv_cache_manager {
namespace raft_meta {

MetaRaftBackend::MetaRaftBackend() = default;

MetaRaftBackend::~MetaRaftBackend() {
    if (opened_.load()) {
        Close();
    }
}

std::string MetaRaftBackend::GetStorageType() noexcept { return META_RAFT_BACKEND_TYPE_STR; }

RaftCoordinator *MetaRaftBackend::Coordinator() const {
    if (coordinator_override_) {
        return coordinator_override_;
    }
    auto *coord = RaftCoordinator::GetInstance();
    if (!coord) {
        KVCM_LOG_ERROR("MetaRaftBackend: no RaftCoordinator installed in process singleton; "
                       "Server::Init must bring it up before any raft-backed Instance is created");
    }
    return coord;
}

std::shared_ptr<MetaCacheBaseBackend> MetaRaftBackend::InstanceBackend() const {
    auto *coord = Coordinator();
    if (!coord) {
        return nullptr;
    }
    return coord->GetOrCreateBackend(instance_id_);
}

ErrorCode MetaRaftBackend::Init(const std::string &instance_id,
                                const std::shared_ptr<MetaStorageBackendConfig> &config) noexcept {
    if (instance_id.empty() || !config) {
        return EC_BADARGS;
    }
    instance_id_ = instance_id;
    // Touch the coordinator now so we fail early if it isn't installed; the
    // returned pointer is not stored — every call re-reads the singleton so
    // tests can swap it out via SetCoordinatorForTest.
    if (!Coordinator()) {
        return EC_ERROR;
    }
    return EC_OK;
}

ErrorCode MetaRaftBackend::Open() noexcept {
    if (opened_.exchange(true)) {
        return EC_OK;
    }
    // Eagerly materialise the per-instance backend so subsequent reads on
    // followers (or before the first commit) see a non-null backend. The
    // factory closure is idempotent.
    if (!InstanceBackend()) {
        opened_.store(false);
        return EC_ERROR;
    }
    KVCM_LOG_INFO("MetaRaftBackend: opened proxy for instance[%s]", instance_id_.c_str());
    return EC_OK;
}

ErrorCode MetaRaftBackend::Close() noexcept {
    opened_.store(false);
    return EC_OK;
}

bool MetaRaftBackend::IsLeader() const {
    auto *coord = Coordinator();
    return coord && coord->IsLeader();
}

int32_t MetaRaftBackend::LeaderId() const {
    auto *coord = Coordinator();
    return coord ? coord->LeaderId() : -1;
}

namespace {

ErrorCode AppendOne(RaftCoordinator *coord, LogOp &&op) {
    return coord->AppendAndWait(Encode(op));
}

} // namespace

// ---------- Writes (replicated) ----------

std::vector<ErrorCode> MetaRaftBackend::Put(RequestContext * /*request_context*/,
                                            const KeyTypeVec &keys,
                                            const CacheLocationMapVector &locations,
                                            const PropertyMapVector &properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_ERROR);
    auto *coord = Coordinator();
    if (!coord) {
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        LogOp op;
        op.type = OpType::kPut;
        op.instance_id = instance_id_;
        op.key = keys[i];
        op.locations = locations[i];
        op.properties = properties[i];
        results[i] = AppendOne(coord, std::move(op));
    }
    return results;
}

std::vector<ErrorCode> MetaRaftBackend::Upsert(RequestContext * /*request_context*/,
                                               const KeyTypeVec &keys,
                                               const CacheLocationMapVector &locations,
                                               const PropertyMapVector &properties) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_ERROR);
    auto *coord = Coordinator();
    if (!coord) {
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        LogOp op;
        op.type = OpType::kUpsert;
        op.instance_id = instance_id_;
        op.key = keys[i];
        op.locations = locations[i];
        op.properties = properties[i];
        results[i] = AppendOne(coord, std::move(op));
    }
    return results;
}

std::vector<ErrorCode> MetaRaftBackend::Delete(RequestContext * /*request_context*/,
                                               const KeyTypeVec &keys) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_ERROR);
    auto *coord = Coordinator();
    if (!coord) {
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        LogOp op;
        op.type = OpType::kDelete;
        op.instance_id = instance_id_;
        op.key = keys[i];
        results[i] = AppendOne(coord, std::move(op));
    }
    return results;
}

std::vector<ErrorCode> MetaRaftBackend::DeleteLocations(RequestContext * /*request_context*/,
                                                        const KeyTypeVec &keys,
                                                        const LocationIdsPerKey &location_ids) noexcept {
    std::vector<ErrorCode> results(keys.size(), EC_OK);
    auto *coord = Coordinator();
    if (!coord) {
        std::fill(results.begin(), results.end(), EC_ERROR);
        return results;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        if (location_ids[i].empty()) {
            continue;
        }
        LogOp op;
        op.type = OpType::kDeleteLocations;
        op.instance_id = instance_id_;
        op.key = keys[i];
        op.location_ids = location_ids[i];
        results[i] = AppendOne(coord, std::move(op));
    }
    return results;
}

// ---------- Reads (delegated to the per-instance backend) ----------

std::vector<ErrorCode> MetaRaftBackend::Get(RequestContext *ctx,
                                            const KeyTypeVec &keys,
                                            CacheLocationMapVector &out_locations,
                                            PropertyMapVector &out_properties) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    return backend->Get(ctx, keys, out_locations, out_properties);
}

std::vector<ErrorCode> MetaRaftBackend::GetLocations(RequestContext *ctx,
                                                     const KeyTypeVec &keys,
                                                     CacheLocationMapVector &out_locations) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    return backend->GetLocations(ctx, keys, out_locations);
}

std::vector<std::vector<ErrorCode>> MetaRaftBackend::GetLocations(RequestContext *ctx,
                                                                  const KeyTypeVec &keys,
                                                                  const LocationIdsPerKey &location_ids,
                                                                  LocationsPerKey &out_locations) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return std::vector<std::vector<ErrorCode>>(keys.size());
    }
    return backend->GetLocations(ctx, keys, location_ids, out_locations);
}

std::vector<ErrorCode> MetaRaftBackend::GetLocationIds(RequestContext *ctx,
                                                       const KeyTypeVec &keys,
                                                       LocationIdsPerKey &out_location_ids) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    return backend->GetLocationIds(ctx, keys, out_location_ids);
}

std::vector<ErrorCode> MetaRaftBackend::GetProperties(RequestContext *ctx,
                                                      const KeyTypeVec &keys,
                                                      const std::vector<std::string> &field_names,
                                                      PropertyMapVector &out_properties) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    return backend->GetProperties(ctx, keys, field_names, out_properties);
}

std::vector<ErrorCode> MetaRaftBackend::Exists(RequestContext *ctx,
                                               const KeyTypeVec &keys,
                                               std::vector<bool> &out_is_exist_vec) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    return backend->Exists(ctx, keys, out_is_exist_vec);
}

std::vector<ErrorCode> MetaRaftBackend::ExistsLocation(RequestContext *ctx,
                                                       const KeyTypeVec &keys,
                                                       std::vector<bool> &out_exists) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return std::vector<ErrorCode>(keys.size(), EC_ERROR);
    }
    return backend->ExistsLocation(ctx, keys, out_exists);
}

ErrorCode MetaRaftBackend::ListKeys(RequestContext *ctx,
                                    const std::string &cursor,
                                    const int64_t limit,
                                    std::string &out_next_cursor,
                                    std::vector<KeyType> &out_keys) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return EC_ERROR;
    }
    return backend->ListKeys(ctx, cursor, limit, out_next_cursor, out_keys);
}

ErrorCode MetaRaftBackend::RandomSample(RequestContext *ctx,
                                        const int64_t count,
                                        std::vector<KeyType> &out_keys) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return EC_ERROR;
    }
    return backend->RandomSample(ctx, count, out_keys);
}

ErrorCode MetaRaftBackend::SampleReclaimKeys(RequestContext *ctx,
                                             const int64_t count,
                                             std::vector<KeyType> &out_keys) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return EC_ERROR;
    }
    return backend->SampleReclaimKeys(ctx, count, out_keys);
}

ErrorCode MetaRaftBackend::PutMetaData(const FieldMap &field_maps) noexcept {
    auto *coord = Coordinator();
    if (!coord) {
        return EC_ERROR;
    }
    LogOp op;
    op.type = OpType::kPutMetaData;
    op.instance_id = instance_id_;
    op.meta_fields = field_maps;
    return AppendOne(coord, std::move(op));
}

ErrorCode MetaRaftBackend::GetMetaData(FieldMap &field_maps) noexcept {
    auto backend = InstanceBackend();
    if (!backend) {
        return EC_ERROR;
    }
    return backend->GetMetaData(field_maps);
}

} // namespace raft_meta
} // namespace kv_cache_manager
