#include "kv_cache_manager/meta/raft/meta_state_machine.h"

#include <libnuraft/buffer.hxx>
#include <libnuraft/buffer_serializer.hxx>
#include <libnuraft/cluster_config.hxx>
#include <libnuraft/snapshot.hxx>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>
#include <vector>

#include "kv_cache_manager/common/logger.h"
#include "kv_cache_manager/meta/cache_location.h"
#include "kv_cache_manager/meta/common.h"
#include "kv_cache_manager/meta/raft/raft_log_codec.h"

namespace kv_cache_manager {
namespace raft_meta {

using nuraft::async_result;
using nuraft::buffer;
using nuraft::buffer_serializer;
using nuraft::cluster_config;
using nuraft::cs_new;
using nuraft::ptr;
using nuraft::snapshot;
using nuraft::ulong;

namespace {

constexpr uint8_t kSnapshotVersionV1 = 1;
constexpr uint8_t kSnapshotVersionV2 = 2;
constexpr uint8_t kSnapshotVersionCurrent = kSnapshotVersionV2;

}

MetaStateMachine::MetaStateMachine(BackendFactory factory, std::string snapshot_dir)
    : backend_factory_(std::move(factory)), snapshot_dir_(std::move(snapshot_dir)) {
    if (!EnsureDir(snapshot_dir_)) {
        KVCM_LOG_ERROR("MetaStateMachine: failed to create snapshot dir[%s]", snapshot_dir_.c_str());
    }
}

MetaStateMachine::~MetaStateMachine() = default;

bool MetaStateMachine::EnsureDir(const std::string &dir) {
    if (dir.empty()) {
        return false;
    }
    std::string acc;
    for (size_t i = 0; i < dir.size(); ++i) {
        if (dir[i] == '/' && i > 0 && !acc.empty()) {
            if (mkdir(acc.c_str(), 0755) != 0 && errno != EEXIST) {
                return false;
            }
        }
        acc.push_back(dir[i]);
    }
    if (mkdir(acc.c_str(), 0755) != 0 && errno != EEXIST) {
        return false;
    }
    return true;
}

std::string MetaStateMachine::SnapshotPath(ulong last_log_idx, ulong term) const {
    return snapshot_dir_ + "/snapshot_" + std::to_string(last_log_idx) + "_" + std::to_string(term);
}

std::shared_ptr<MetaCacheBaseBackend> MetaStateMachine::GetBackend(const std::string &instance_id) const {
    std::shared_lock<std::shared_mutex> g(backends_mu_);
    auto it = backends_.find(instance_id);
    return it == backends_.end() ? nullptr : it->second;
}

std::shared_ptr<MetaCacheBaseBackend> MetaStateMachine::GetOrCreateBackend(const std::string &instance_id) {
    {
        std::shared_lock<std::shared_mutex> g(backends_mu_);
        auto it = backends_.find(instance_id);
        if (it != backends_.end()) {
            return it->second;
        }
    }
    std::unique_lock<std::shared_mutex> g(backends_mu_);
    auto it = backends_.find(instance_id);
    if (it != backends_.end()) {
        return it->second;
    }
    if (!backend_factory_) {
        KVCM_LOG_ERROR("MetaStateMachine: backend factory not set, cannot materialise instance[%s]",
                       instance_id.c_str());
        return nullptr;
    }
    auto backend = backend_factory_(instance_id);
    if (!backend) {
        KVCM_LOG_ERROR("MetaStateMachine: backend factory returned null for instance[%s]", instance_id.c_str());
        return nullptr;
    }
    backends_.emplace(instance_id, backend);
    return backend;
}

ptr<buffer> MetaStateMachine::commit(const ulong log_idx, buffer &data) {
    LogOp op;
    if (Decode(data, op) != EC_OK) {
        KVCM_LOG_ERROR("MetaStateMachine: corrupt log entry at idx[%lu]", log_idx);
        last_commit_index_.store(log_idx);
        return nullptr;
    }

    if (op.type == OpType::kNoOp) {
        last_commit_index_.store(log_idx);
        return nullptr;
    }

    if (op.type == OpType::kRegistryFieldSave || op.type == OpType::kRegistryFieldDelete) {
        bool is_field_save = (op.type == OpType::kRegistryFieldSave);
        std::map<std::string, std::string> fields_copy;
        {
            std::unique_lock<std::shared_mutex> g(registry_mu_);
            if (is_field_save) {
                registry_store_[op.registry_key][op.registry_field_id] = op.registry_field_value;
                fields_copy = registry_store_[op.registry_key];
            } else {
                auto it = registry_store_.find(op.registry_key);
                if (it != registry_store_.end()) {
                    it->second.erase(op.registry_field_id);
                    if (it->second.empty()) {
                        registry_store_.erase(it);
                    } else {
                        fields_copy = it->second;
                    }
                }
            }
        }
        {
            std::lock_guard<std::mutex> g(registry_cb_mu_);
            if (registry_commit_cb_) {
                registry_commit_cb_(true, op.registry_key, fields_copy);
            }
        }
        last_commit_index_.store(log_idx);
        return nullptr;
    }

    if (op.type == OpType::kRegistrySave || op.type == OpType::kRegistryDelete) {
        bool is_save = (op.type == OpType::kRegistrySave);
        std::map<std::string, std::string> fields_copy;
        {
            std::unique_lock<std::shared_mutex> g(registry_mu_);
            if (is_save) {
                registry_store_[op.registry_key] = op.registry_fields;
                fields_copy = std::move(op.registry_fields);
            } else {
                registry_store_.erase(op.registry_key);
            }
        }
        {
            std::lock_guard<std::mutex> g(registry_cb_mu_);
            if (registry_commit_cb_) {
                registry_commit_cb_(is_save, op.registry_key, fields_copy);
            }
        }
        last_commit_index_.store(log_idx);
        return nullptr;
    }

    auto backend = GetOrCreateBackend(op.instance_id);
    if (!backend) {
        KVCM_LOG_ERROR("MetaStateMachine: no backend for instance[%s] at idx[%lu]",
                       op.instance_id.c_str(), log_idx);
        last_commit_index_.store(log_idx);
        return nullptr;
    }

    KeyVector keys;
    CacheLocationMapVector locations;
    PropertyMapVector properties;
    LocationIdsPerKey location_ids;

    switch (op.type) {
    case OpType::kPut:
        keys.push_back(op.key);
        locations.push_back(std::move(op.locations));
        properties.push_back(std::move(op.properties));
        backend->Put(nullptr, keys, locations, properties);
        break;
    case OpType::kPutIfAbsent:
        keys.push_back(op.key);
        locations.push_back(std::move(op.locations));
        properties.push_back(std::move(op.properties));
        backend->PutIfAbsent(nullptr, keys, locations, properties);
        break;
    case OpType::kUpsert:
        keys.push_back(op.key);
        locations.push_back(std::move(op.locations));
        properties.push_back(std::move(op.properties));
        backend->Upsert(nullptr, keys, locations, properties);
        break;
    case OpType::kDelete:
        keys.push_back(op.key);
        backend->Delete(nullptr, keys);
        break;
    case OpType::kDeleteLocations:
        keys.push_back(op.key);
        location_ids.push_back(std::move(op.location_ids));
        backend->DeleteLocations(nullptr, keys, location_ids);
        break;
    case OpType::kPutMetaData:
        backend->PutMetaData(op.meta_fields);
        break;
    default:
        break;
    }

    last_commit_index_.store(log_idx);
    return nullptr;
}

namespace {

// Encode one MetaLocalBackend's state in the same layout as the previous
// single-instance snapshot. Caller has already written the instance_id.
void WriteOneInstance(buffer_serializer &bs, MetaCacheBaseBackend *backend) {
    KeyVector all_keys;
    std::string cursor = SCAN_BASE_CURSOR;
    do {
        std::string next_cursor;
        KeyVector page;
        ErrorCode ec = backend->ListKeys(nullptr, cursor, /*limit=*/4096, next_cursor, page);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("MetaStateMachine: ListKeys failed in snapshot, ec[%d]", ec);
            break;
        }
        all_keys.insert(all_keys.end(), page.begin(), page.end());
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);

    CacheLocationMapVector locs(all_keys.size());
    PropertyMapVector props(all_keys.size());
    backend->Get(nullptr, all_keys, locs, props);

    bs.put_u32(static_cast<uint32_t>(all_keys.size()));
    for (size_t i = 0; i < all_keys.size(); ++i) {
        bs.put_i64(all_keys[i]);
        bs.put_u32(static_cast<uint32_t>(locs[i].size()));
        for (const auto &[id, loc] : locs[i]) {
            bs.put_str(id);
            bs.put_str(loc ? loc->ToJsonString() : std::string());
        }
        size_t real_props = 0;
        for (const auto &[k, v] : props[i]) {
            if (k != PROPERTY_LRU_TIME) {
                ++real_props;
            }
        }
        bs.put_u32(static_cast<uint32_t>(real_props));
        for (const auto &[k, v] : props[i]) {
            if (k == PROPERTY_LRU_TIME) {
                continue;
            }
            bs.put_str(k);
            bs.put_str(v);
        }
    }
}

size_t EstimateOneInstanceSize(MetaCacheBaseBackend *backend) {
    // Walk the same path as WriteOneInstance, but only sum sizes. Doubles the
    // ListKeys cost for snapshot creation; acceptable for Phase 1 sizes.
    KeyVector all_keys;
    std::string cursor = SCAN_BASE_CURSOR;
    do {
        std::string next_cursor;
        KeyVector page;
        if (backend->ListKeys(nullptr, cursor, /*limit=*/4096, next_cursor, page) != EC_OK) {
            break;
        }
        all_keys.insert(all_keys.end(), page.begin(), page.end());
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);

    CacheLocationMapVector locs(all_keys.size());
    PropertyMapVector props(all_keys.size());
    backend->Get(nullptr, all_keys, locs, props);

    size_t total = sizeof(uint32_t);
    for (size_t i = 0; i < all_keys.size(); ++i) {
        total += sizeof(int64_t) + sizeof(uint32_t);
        for (const auto &[id, loc] : locs[i]) {
            total += sizeof(uint32_t) + id.size();
            total += sizeof(uint32_t) + (loc ? loc->ToJsonString().size() : 0);
        }
        total += sizeof(uint32_t);
        for (const auto &[k, v] : props[i]) {
            if (k == PROPERTY_LRU_TIME) {
                continue;
            }
            total += sizeof(uint32_t) + k.size();
            total += sizeof(uint32_t) + v.size();
        }
    }
    return total;
}

bool ReadOneInstance(buffer_serializer &bs, MetaCacheBaseBackend *backend) {
    // Drain whatever was there before — apply_snapshot semantics replace
    // the whole state.
    KeyVector to_delete;
    std::string cursor = SCAN_BASE_CURSOR;
    do {
        std::string next_cursor;
        KeyVector page;
        if (backend->ListKeys(nullptr, cursor, 4096, next_cursor, page) != EC_OK) {
            return false;
        }
        to_delete.insert(to_delete.end(), page.begin(), page.end());
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);
    if (!to_delete.empty()) {
        backend->Delete(nullptr, to_delete);
    }

    uint32_t key_count = bs.get_u32();
    for (uint32_t i = 0; i < key_count; ++i) {
        KeyType key = bs.get_i64();
        CacheLocationMap loc_map;
        uint32_t loc_n = bs.get_u32();
        for (uint32_t j = 0; j < loc_n; ++j) {
            std::string id = bs.get_str();
            std::string json = bs.get_str();
            if (json.empty()) {
                loc_map.emplace(std::move(id), CacheLocationConstPtr{});
            } else {
                auto loc = std::make_shared<CacheLocation>();
                if (!loc->FromJsonString(json)) {
                    KVCM_LOG_ERROR("MetaStateMachine: corrupt CacheLocation json in snapshot");
                    return false;
                }
                loc_map.emplace(std::move(id), std::const_pointer_cast<const CacheLocation>(loc));
            }
        }
        PropertyMap prop_map;
        uint32_t prop_n = bs.get_u32();
        for (uint32_t j = 0; j < prop_n; ++j) {
            std::string k = bs.get_str();
            std::string v = bs.get_str();
            prop_map.emplace(std::move(k), std::move(v));
        }
        KeyVector keys{key};
        CacheLocationMapVector locs{std::move(loc_map)};
        PropertyMapVector props{std::move(prop_map)};
        backend->Put(nullptr, keys, locs, props);
    }
    return true;
}

} // namespace

ptr<buffer> MetaStateMachine::SerializeAll() const {
    std::vector<std::pair<std::string, std::shared_ptr<MetaCacheBaseBackend>>> entries;
    {
        std::shared_lock<std::shared_mutex> g(backends_mu_);
        entries.reserve(backends_.size());
        for (const auto &[iid, b] : backends_) {
            entries.emplace_back(iid, b);
        }
    }

    std::unordered_map<std::string, std::map<std::string, std::string>> registry_snapshot;
    {
        std::shared_lock<std::shared_mutex> g(registry_mu_);
        registry_snapshot = registry_store_;
    }

    size_t total = 1 /*ver*/ + sizeof(uint32_t) /*instance_count*/;
    for (const auto &[iid, b] : entries) {
        total += sizeof(uint32_t) + iid.size();
        total += EstimateOneInstanceSize(b.get());
    }
    // Registry section size.
    total += sizeof(uint32_t); // registry_entry_count
    for (const auto &[key, fields] : registry_snapshot) {
        total += sizeof(uint32_t) + key.size();
        total += sizeof(uint32_t); // field_count
        for (const auto &[fk, fv] : fields) {
            total += sizeof(uint32_t) + fk.size();
            total += sizeof(uint32_t) + fv.size();
        }
    }

    ptr<buffer> out = buffer::alloc(total);
    buffer_serializer bs(out);
    bs.put_u8(kSnapshotVersionCurrent);
    bs.put_u32(static_cast<uint32_t>(entries.size()));
    for (const auto &[iid, b] : entries) {
        bs.put_str(iid);
        WriteOneInstance(bs, b.get());
    }

    // Registry section.
    bs.put_u32(static_cast<uint32_t>(registry_snapshot.size()));
    for (const auto &[key, fields] : registry_snapshot) {
        bs.put_str(key);
        bs.put_u32(static_cast<uint32_t>(fields.size()));
        for (const auto &[fk, fv] : fields) {
            bs.put_str(fk);
            bs.put_str(fv);
        }
    }
    return out;
}

bool MetaStateMachine::DeserializeAll(buffer &buf) {
    buffer_serializer bs(buf);
    uint8_t ver = bs.get_u8();
    if (ver != kSnapshotVersionV1 && ver != kSnapshotVersionV2) {
        KVCM_LOG_ERROR("MetaStateMachine: unknown snapshot version[%u]", ver);
        return false;
    }
    uint32_t instance_count = bs.get_u32();
    for (uint32_t i = 0; i < instance_count; ++i) {
        std::string instance_id = bs.get_str();
        auto backend = GetOrCreateBackend(instance_id);
        if (!backend) {
            return false;
        }
        if (!ReadOneInstance(bs, backend.get())) {
            return false;
        }
    }

    // V2: registry section follows per-instance data.
    {
        std::unique_lock<std::shared_mutex> g(registry_mu_);
        registry_store_.clear();
        if (ver >= kSnapshotVersionV2) {
            uint32_t reg_count = bs.get_u32();
            for (uint32_t i = 0; i < reg_count; ++i) {
                std::string key = bs.get_str();
                uint32_t field_count = bs.get_u32();
                std::map<std::string, std::string> fields;
                for (uint32_t j = 0; j < field_count; ++j) {
                    std::string fk = bs.get_str();
                    std::string fv = bs.get_str();
                    fields.emplace(std::move(fk), std::move(fv));
                }
                registry_store_.emplace(std::move(key), std::move(fields));
            }
        }
    }
    return true;
}

void MetaStateMachine::create_snapshot(snapshot &s, async_result<bool>::handler_type &when_done) {
    ptr<buffer> body = SerializeAll();
    std::string path = SnapshotPath(s.get_last_log_idx(), s.get_last_log_term());
    {
        std::ofstream out(path + ".tmp", std::ios::binary | std::ios::trunc);
        if (out) {
            body->pos(0);
            out.write(reinterpret_cast<const char *>(body->data_begin()),
                      static_cast<std::streamsize>(body->size()));
        }
    }
    bool ok = std::rename((path + ".tmp").c_str(), path.c_str()) == 0;

    if (ok) {
        std::lock_guard<std::mutex> g(snapshot_mutex_);
        last_snapshot_ = cs_new<snapshot>(s.get_last_log_idx(),
                                          s.get_last_log_term(),
                                          s.get_last_config(),
                                          body->size(),
                                          s.get_type());
    }

    ptr<std::exception> exp;
    when_done(ok, exp);
}

bool MetaStateMachine::apply_snapshot(snapshot &s) {
    std::string path = SnapshotPath(s.get_last_log_idx(), s.get_last_log_term());
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        KVCM_LOG_ERROR("MetaStateMachine: missing snapshot file[%s]", path.c_str());
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string raw = ss.str();
    ptr<buffer> buf = buffer::alloc(raw.size());
    std::memcpy(buf->data_begin(), raw.data(), raw.size());
    if (!DeserializeAll(*buf)) {
        return false;
    }

    {
        std::lock_guard<std::mutex> g(snapshot_mutex_);
        last_snapshot_ = cs_new<snapshot>(s.get_last_log_idx(),
                                          s.get_last_log_term(),
                                          s.get_last_config(),
                                          raw.size(),
                                          s.get_type());
    }
    last_commit_index_.store(s.get_last_log_idx());
    return true;
}

int MetaStateMachine::read_logical_snp_obj(snapshot &s,
                                           void *&user_snp_ctx,
                                           ulong obj_id,
                                           ptr<buffer> &data_out,
                                           bool &is_last_obj) {
    (void)user_snp_ctx;
    if (obj_id == 0) {
        // Sentinel chunk so NuRaft sees a non-zero first read; the actual
        // bytes go in obj_id == 1.
        data_out = buffer::alloc(0);
        is_last_obj = false;
        return 0;
    }
    std::string path = SnapshotPath(s.get_last_log_idx(), s.get_last_log_term());
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        KVCM_LOG_ERROR("MetaStateMachine: read_logical_snp_obj missing file[%s]", path.c_str());
        data_out = buffer::alloc(0);
        is_last_obj = true;
        return -1;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string raw = ss.str();
    data_out = buffer::alloc(raw.size());
    std::memcpy(data_out->data_begin(), raw.data(), raw.size());
    is_last_obj = true;
    return 0;
}

void MetaStateMachine::save_logical_snp_obj(snapshot &s,
                                            ulong &obj_id,
                                            buffer &data,
                                            bool /*is_first_obj*/,
                                            bool is_last_obj) {
    if (obj_id == 0) {
        // First sentinel object — nothing to save yet.
        obj_id = 1;
        return;
    }
    std::string path = SnapshotPath(s.get_last_log_idx(), s.get_last_log_term());
    {
        std::ofstream out(path + ".tmp", std::ios::binary | std::ios::trunc);
        if (out) {
            data.pos(0);
            out.write(reinterpret_cast<const char *>(data.data_begin()),
                      static_cast<std::streamsize>(data.size()));
        }
    }
    if (std::rename((path + ".tmp").c_str(), path.c_str()) != 0) {
        KVCM_LOG_ERROR("MetaStateMachine: save_logical_snp_obj rename failed for[%s]", path.c_str());
    }
    if (is_last_obj) {
        std::lock_guard<std::mutex> g(snapshot_mutex_);
        last_snapshot_ = cs_new<snapshot>(s.get_last_log_idx(),
                                          s.get_last_log_term(),
                                          s.get_last_config(),
                                          data.size(),
                                          s.get_type());
    }
}

void MetaStateMachine::free_user_snp_ctx(void *&user_snp_ctx) { user_snp_ctx = nullptr; }

ptr<snapshot> MetaStateMachine::last_snapshot() {
    std::lock_guard<std::mutex> g(snapshot_mutex_);
    return last_snapshot_;
}

ulong MetaStateMachine::last_commit_index() { return last_commit_index_.load(); }

ErrorCode MetaStateMachine::RegistryLoad(const std::string &key,
                                         std::map<std::string, std::string> &out) const {
    std::shared_lock<std::shared_mutex> g(registry_mu_);
    auto it = registry_store_.find(key);
    if (it == registry_store_.end()) {
        return EC_NOENT;
    }
    out = it->second;
    return EC_OK;
}

void MetaStateMachine::RegistryClear() {
    std::unique_lock<std::shared_mutex> g(registry_mu_);
    registry_store_.clear();
}

void MetaStateMachine::SetRegistryCommitCallback(RegistryCommitCallback cb) {
    std::lock_guard<std::mutex> g(registry_cb_mu_);
    registry_commit_cb_ = std::move(cb);
}

} // namespace raft_meta
} // namespace kv_cache_manager
