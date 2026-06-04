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

constexpr uint8_t kSnapshotVersion = 1;

}

MetaStateMachine::MetaStateMachine(std::shared_ptr<MetaCacheBaseBackend> backend, std::string snapshot_dir)
    : backend_(std::move(backend)), snapshot_dir_(std::move(snapshot_dir)) {
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

ptr<buffer> MetaStateMachine::commit(const ulong log_idx, buffer &data) {
    LogOp op;
    if (Decode(data, op) != EC_OK) {
        KVCM_LOG_ERROR("MetaStateMachine: corrupt log entry at idx[%lu]", log_idx);
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
        backend_->Put(nullptr, keys, locations, properties);
        break;
    case OpType::kPutIfAbsent:
        keys.push_back(op.key);
        locations.push_back(std::move(op.locations));
        properties.push_back(std::move(op.properties));
        backend_->PutIfAbsent(nullptr, keys, locations, properties);
        break;
    case OpType::kUpsert:
        keys.push_back(op.key);
        locations.push_back(std::move(op.locations));
        properties.push_back(std::move(op.properties));
        backend_->Upsert(nullptr, keys, locations, properties);
        break;
    case OpType::kDelete:
        keys.push_back(op.key);
        backend_->Delete(nullptr, keys);
        break;
    case OpType::kDeleteLocations:
        keys.push_back(op.key);
        location_ids.push_back(std::move(op.location_ids));
        backend_->DeleteLocations(nullptr, keys, location_ids);
        break;
    case OpType::kPutMetaData:
        backend_->PutMetaData(op.meta_fields);
        break;
    }

    last_commit_index_.store(log_idx);
    return nullptr;
}

ptr<buffer> MetaStateMachine::SerializeBackend() const {
    // Layout:
    //   u8 version
    //   u32 key_count
    //   for each key:
    //     i64 key
    //     u32 location_count
    //     for each location: str id, str json
    //     u32 property_count
    //     for each property: str name, str value
    KeyVector all_keys;
    std::string cursor = SCAN_BASE_CURSOR;
    do {
        std::string next_cursor;
        KeyVector page;
        ErrorCode ec = backend_->ListKeys(nullptr, cursor, /*limit=*/4096, next_cursor, page);
        if (ec != EC_OK) {
            KVCM_LOG_ERROR("MetaStateMachine: ListKeys failed in snapshot, ec[%d]", ec);
            break;
        }
        all_keys.insert(all_keys.end(), page.begin(), page.end());
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);

    CacheLocationMapVector locs(all_keys.size());
    PropertyMapVector props(all_keys.size());
    backend_->Get(nullptr, all_keys, locs, props);

    // Estimate buffer size.
    size_t total = 1 + sizeof(uint32_t);
    for (size_t i = 0; i < all_keys.size(); ++i) {
        total += sizeof(int64_t) + sizeof(uint32_t);
        for (const auto &[id, loc] : locs[i]) {
            total += sizeof(uint32_t) + id.size();
            total += sizeof(uint32_t) + (loc ? loc->ToJsonString().size() : 0);
        }
        total += sizeof(uint32_t);
        for (const auto &[k, v] : props[i]) {
            total += sizeof(uint32_t) + k.size();
            total += sizeof(uint32_t) + v.size();
        }
    }
    ptr<buffer> out = buffer::alloc(total);
    buffer_serializer bs(out);
    bs.put_u8(kSnapshotVersion);
    bs.put_u32(static_cast<uint32_t>(all_keys.size()));
    for (size_t i = 0; i < all_keys.size(); ++i) {
        bs.put_i64(all_keys[i]);
        bs.put_u32(static_cast<uint32_t>(locs[i].size()));
        for (const auto &[id, loc] : locs[i]) {
            bs.put_str(id);
            bs.put_str(loc ? loc->ToJsonString() : std::string());
        }
        // Strip the synthetic PROPERTY_LRU_TIME (added by Get) so reload
        // doesn't pollute backend properties with derived state.
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
    return out;
}

bool MetaStateMachine::DeserializeIntoBackend(buffer &buf) {
    buffer_serializer bs(buf);
    uint8_t ver = bs.get_u8();
    if (ver != kSnapshotVersion) {
        KVCM_LOG_ERROR("MetaStateMachine: unknown snapshot version[%u]", ver);
        return false;
    }

    // Wipe current state. Cheapest way without an explicit Clear API is to
    // drain via ListKeys + Delete.
    KeyVector to_delete;
    std::string cursor = SCAN_BASE_CURSOR;
    do {
        std::string next_cursor;
        KeyVector page;
        if (backend_->ListKeys(nullptr, cursor, 4096, next_cursor, page) != EC_OK) {
            return false;
        }
        to_delete.insert(to_delete.end(), page.begin(), page.end());
        cursor = next_cursor;
    } while (cursor != SCAN_BASE_CURSOR);
    if (!to_delete.empty()) {
        backend_->Delete(nullptr, to_delete);
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
        backend_->Put(nullptr, keys, locs, props);
    }
    return true;
}

void MetaStateMachine::create_snapshot(snapshot &s, async_result<bool>::handler_type &when_done) {
    ptr<buffer> body = SerializeBackend();
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
        // We hold the cluster_config from the snapshot ref directly; raft
        // owns the lifetime of the passed snapshot during this call but
        // last_log_idx/term/config are plain values we copy.
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
    if (!DeserializeIntoBackend(*buf)) {
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

} // namespace raft_meta
} // namespace kv_cache_manager
