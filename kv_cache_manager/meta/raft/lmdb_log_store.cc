#include "kv_cache_manager/meta/raft/lmdb_log_store.h"

#include <libnuraft/buffer.hxx>
#include <libnuraft/buffer_serializer.hxx>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
namespace raft_meta {

using nuraft::buffer;
using nuraft::buffer_serializer;
using nuraft::log_entry;
using nuraft::log_val_type;
using nuraft::ptr;
using nuraft::ulong;

static constexpr size_t kDefaultMapSize = 1ULL << 30; // 1 GB
static const char *kStartIndexKey = "start_index";

static bool EnsureDir(const std::string &dir) {
    if (dir.empty()) return false;
    std::string acc;
    for (size_t i = 0; i < dir.size(); ++i) {
        if (dir[i] == '/' && i > 0 && !acc.empty()) {
            if (mkdir(acc.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
        acc.push_back(dir[i]);
    }
    return mkdir(acc.c_str(), 0755) == 0 || errno == EEXIST;
}

LmdbLogStore::LmdbLogStore(const std::string &dir) {
    EnsureDir(dir);

    int rc = mdb_env_create(&env_);
    if (rc != 0) {
        KVCM_LOG_ERROR("LmdbLogStore: mdb_env_create failed: %s", mdb_strerror(rc));
        return;
    }
    mdb_env_set_maxdbs(env_, 2);
    mdb_env_set_mapsize(env_, kDefaultMapSize);

    rc = mdb_env_open(env_, dir.c_str(), 0, 0664);
    if (rc != 0) {
        KVCM_LOG_ERROR("LmdbLogStore: mdb_env_open(%s) failed: %s", dir.c_str(), mdb_strerror(rc));
        mdb_env_close(env_);
        env_ = nullptr;
        return;
    }

    MDB_txn *txn = nullptr;
    rc = mdb_txn_begin(env_, nullptr, 0, &txn);
    if (rc != 0) {
        KVCM_LOG_ERROR("LmdbLogStore: mdb_txn_begin failed: %s", mdb_strerror(rc));
        mdb_env_close(env_);
        env_ = nullptr;
        return;
    }
    mdb_dbi_open(txn, "logs", MDB_CREATE, &logs_dbi_);
    mdb_dbi_open(txn, "meta", MDB_CREATE, &meta_dbi_);
    mdb_txn_commit(txn);
}

LmdbLogStore::~LmdbLogStore() {
    if (env_) {
        mdb_env_close(env_);
        env_ = nullptr;
    }
}

ptr<log_entry> LmdbLogStore::MakeDummyEntry() {
    ptr<buffer> empty = buffer::alloc(0);
    return nuraft::cs_new<log_entry>(0ULL, empty, log_val_type::app_log);
}

void LmdbLogStore::EncodeIndex(ulong index, uint8_t out[8]) {
    for (int i = 7; i >= 0; --i) {
        out[i] = static_cast<uint8_t>(index & 0xFF);
        index >>= 8;
    }
}

ulong LmdbLogStore::DecodeIndex(const uint8_t data[8]) {
    ulong v = 0;
    for (int i = 0; i < 8; ++i) {
        v = (v << 8) | data[i];
    }
    return v;
}

void LmdbLogStore::SaveStartIndex(MDB_txn *txn, ulong index) {
    uint8_t val[8];
    EncodeIndex(index, val);
    MDB_val k{strlen(kStartIndexKey), const_cast<char *>(kStartIndexKey)};
    MDB_val v{8, val};
    mdb_put(txn, meta_dbi_, &k, &v, 0);
}

ulong LmdbLogStore::LoadStartIndex(MDB_txn *txn) const {
    MDB_val k{strlen(kStartIndexKey), const_cast<char *>(kStartIndexKey)};
    MDB_val v;
    if (mdb_get(txn, meta_dbi_, &k, &v) == 0 && v.mv_size == 8) {
        return DecodeIndex(static_cast<const uint8_t *>(v.mv_data));
    }
    return 1;
}

ulong LmdbLogStore::next_slot() const {
    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return 1;

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
    MDB_cursor *cur = nullptr;
    mdb_cursor_open(txn, logs_dbi_, &cur);
    MDB_val k, v;
    ulong result;
    if (mdb_cursor_get(cur, &k, &v, MDB_LAST) == 0 && k.mv_size == 8) {
        result = DecodeIndex(static_cast<const uint8_t *>(k.mv_data)) + 1;
    } else {
        result = LoadStartIndex(txn);
    }
    mdb_cursor_close(cur);
    mdb_txn_abort(txn);
    return result;
}

ulong LmdbLogStore::start_index() const {
    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return 1;

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
    ulong result = LoadStartIndex(txn);
    mdb_txn_abort(txn);
    return result;
}

ptr<log_entry> LmdbLogStore::last_entry() const {
    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return MakeDummyEntry();

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
    MDB_cursor *cur = nullptr;
    mdb_cursor_open(txn, logs_dbi_, &cur);
    MDB_val k, v;
    ptr<log_entry> result;
    if (mdb_cursor_get(cur, &k, &v, MDB_LAST) == 0) {
        ptr<buffer> buf = buffer::alloc(v.mv_size);
        std::memcpy(buf->data_begin(), v.mv_data, v.mv_size);
        result = log_entry::deserialize(*buf);
    } else {
        result = MakeDummyEntry();
    }
    mdb_cursor_close(cur);
    mdb_txn_abort(txn);
    return result;
}

ulong LmdbLogStore::append(ptr<log_entry> &entry) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return 0;

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, 0, &txn);

    MDB_cursor *cur = nullptr;
    mdb_cursor_open(txn, logs_dbi_, &cur);
    MDB_val lk, lv;
    ulong idx;
    if (mdb_cursor_get(cur, &lk, &lv, MDB_LAST) == 0 && lk.mv_size == 8) {
        idx = DecodeIndex(static_cast<const uint8_t *>(lk.mv_data)) + 1;
    } else {
        idx = LoadStartIndex(txn);
    }
    mdb_cursor_close(cur);

    uint8_t key_buf[8];
    EncodeIndex(idx, key_buf);
    ptr<buffer> ser = entry->serialize();
    MDB_val k{8, key_buf};
    MDB_val v{ser->size(), ser->data_begin()};
    mdb_put(txn, logs_dbi_, &k, &v, 0);
    mdb_txn_commit(txn);
    return idx;
}

void LmdbLogStore::write_at(ulong index, ptr<log_entry> &entry) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return;

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, 0, &txn);

    // Delete entries >= index.
    uint8_t key_buf[8];
    EncodeIndex(index, key_buf);
    MDB_val k{8, key_buf};
    MDB_cursor *cur = nullptr;
    mdb_cursor_open(txn, logs_dbi_, &cur);
    MDB_val ck, cv;
    ck = k;
    if (mdb_cursor_get(cur, &ck, &cv, MDB_SET_RANGE) == 0) {
        do {
            mdb_cursor_del(cur, 0);
        } while (mdb_cursor_get(cur, &ck, &cv, MDB_NEXT) == 0);
    }
    mdb_cursor_close(cur);

    // Insert the new entry.
    ptr<buffer> ser = entry->serialize();
    MDB_val v{ser->size(), ser->data_begin()};
    k = {8, key_buf};
    mdb_put(txn, logs_dbi_, &k, &v, 0);
    mdb_txn_commit(txn);
}

ptr<std::vector<ptr<log_entry>>> LmdbLogStore::log_entries(ulong start, ulong end) {
    auto out = nuraft::cs_new<std::vector<ptr<log_entry>>>();
    if (start >= end) return out;

    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return nullptr;

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);

    out->reserve(end - start);
    for (ulong i = start; i < end; ++i) {
        uint8_t key_buf[8];
        EncodeIndex(i, key_buf);
        MDB_val k{8, key_buf};
        MDB_val v;
        if (mdb_get(txn, logs_dbi_, &k, &v) != 0) {
            mdb_txn_abort(txn);
            return nullptr;
        }
        ptr<buffer> buf = buffer::alloc(v.mv_size);
        std::memcpy(buf->data_begin(), v.mv_data, v.mv_size);
        out->push_back(log_entry::deserialize(*buf));
    }
    mdb_txn_abort(txn);
    return out;
}

ptr<log_entry> LmdbLogStore::entry_at(ulong index) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return MakeDummyEntry();

    uint8_t key_buf[8];
    EncodeIndex(index, key_buf);
    MDB_val k{8, key_buf};
    MDB_val v;

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);
    int rc = mdb_get(txn, logs_dbi_, &k, &v);
    ptr<log_entry> result;
    if (rc == 0) {
        ptr<buffer> buf = buffer::alloc(v.mv_size);
        std::memcpy(buf->data_begin(), v.mv_data, v.mv_size);
        result = log_entry::deserialize(*buf);
    } else {
        result = MakeDummyEntry();
    }
    mdb_txn_abort(txn);
    return result;
}

ulong LmdbLogStore::term_at(ulong index) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return 0;

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);

    ulong si = LoadStartIndex(txn);
    if (index < si) {
        mdb_txn_abort(txn);
        return 0;
    }

    uint8_t key_buf[8];
    EncodeIndex(index, key_buf);
    MDB_val k{8, key_buf};
    MDB_val v;
    ulong result = 0;
    if (mdb_get(txn, logs_dbi_, &k, &v) == 0) {
        ptr<buffer> buf = buffer::alloc(v.mv_size);
        std::memcpy(buf->data_begin(), v.mv_data, v.mv_size);
        result = log_entry::deserialize(*buf)->get_term();
    }
    mdb_txn_abort(txn);
    return result;
}

ptr<buffer> LmdbLogStore::pack(ulong index, nuraft::int32 cnt) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return buffer::alloc(sizeof(uint32_t));

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, MDB_RDONLY, &txn);

    std::vector<ptr<buffer>> serialized;
    serialized.reserve(cnt);
    size_t total = sizeof(uint32_t);
    for (nuraft::int32 i = 0; i < cnt; ++i) {
        uint8_t key_buf[8];
        EncodeIndex(index + static_cast<ulong>(i), key_buf);
        MDB_val k{8, key_buf};
        MDB_val v;
        if (mdb_get(txn, logs_dbi_, &k, &v) != 0) break;
        ptr<buffer> buf = buffer::alloc(v.mv_size);
        std::memcpy(buf->data_begin(), v.mv_data, v.mv_size);
        ptr<log_entry> entry = log_entry::deserialize(*buf);
        ptr<buffer> ser = entry->serialize();
        total += sizeof(uint32_t) + ser->size();
        serialized.push_back(ser);
    }
    mdb_txn_abort(txn);

    ptr<buffer> out = buffer::alloc(total);
    buffer_serializer bs(out);
    bs.put_u32(static_cast<uint32_t>(serialized.size()));
    for (auto &s : serialized) {
        bs.put_u32(static_cast<uint32_t>(s->size()));
        s->pos(0);
        bs.put_raw(s->data_begin(), s->size());
    }
    return out;
}

void LmdbLogStore::apply_pack(ulong index, buffer &pack) {
    buffer_serializer bs(pack);
    uint32_t count = bs.get_u32();

    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return;

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, 0, &txn);

    // Delete entries >= index.
    uint8_t start_key[8];
    EncodeIndex(index, start_key);
    MDB_val sk{8, start_key};
    MDB_cursor *cur = nullptr;
    mdb_cursor_open(txn, logs_dbi_, &cur);
    MDB_val ck = sk, cv;
    if (mdb_cursor_get(cur, &ck, &cv, MDB_SET_RANGE) == 0) {
        do {
            mdb_cursor_del(cur, 0);
        } while (mdb_cursor_get(cur, &ck, &cv, MDB_NEXT) == 0);
    }
    mdb_cursor_close(cur);

    // Insert unpacked entries.
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t entry_size = bs.get_u32();
        ptr<buffer> entry_buf = buffer::alloc(entry_size);
        void *raw = bs.get_raw(entry_size);
        std::memcpy(entry_buf->data_begin(), raw, entry_size);
        ptr<log_entry> entry = log_entry::deserialize(*entry_buf);
        ptr<buffer> ser = entry->serialize();

        uint8_t key_buf[8];
        EncodeIndex(index + i, key_buf);
        MDB_val k{8, key_buf};
        MDB_val v{ser->size(), ser->data_begin()};
        mdb_put(txn, logs_dbi_, &k, &v, 0);
    }

    ulong si = LoadStartIndex(txn);
    if (si < index) {
        SaveStartIndex(txn, index);
    }
    mdb_txn_commit(txn);
}

bool LmdbLogStore::compact(ulong last_log_index) {
    std::lock_guard<std::mutex> g(mutex_);
    if (!env_) return false;

    MDB_txn *txn = nullptr;
    mdb_txn_begin(env_, nullptr, 0, &txn);

    // Delete entries <= last_log_index.
    MDB_cursor *cur = nullptr;
    mdb_cursor_open(txn, logs_dbi_, &cur);
    MDB_val ck, cv;
    if (mdb_cursor_get(cur, &ck, &cv, MDB_FIRST) == 0) {
        do {
            if (ck.mv_size == 8 && DecodeIndex(static_cast<const uint8_t *>(ck.mv_data)) <= last_log_index) {
                mdb_cursor_del(cur, 0);
            } else {
                break;
            }
        } while (mdb_cursor_get(cur, &ck, &cv, MDB_NEXT) == 0);
    }
    mdb_cursor_close(cur);

    ulong si = LoadStartIndex(txn);
    if (si < last_log_index + 1) {
        SaveStartIndex(txn, last_log_index + 1);
    }
    mdb_txn_commit(txn);
    return true;
}

bool LmdbLogStore::flush() {
    if (!env_) return false;
    return mdb_env_sync(env_, 1) == 0;
}

} // namespace raft_meta
} // namespace kv_cache_manager
