#include "kv_cache_manager/meta/raft/meta_log_store.h"

#include <libnuraft/buffer.hxx>
#include <libnuraft/buffer_serializer.hxx>

#include <cstring>
#include <utility>

namespace kv_cache_manager {
namespace raft_meta {

using nuraft::buffer;
using nuraft::buffer_serializer;
using nuraft::log_entry;
using nuraft::log_val_type;
using nuraft::ptr;
using nuraft::ulong;

MetaLogStore::MetaLogStore() = default;

ptr<log_entry> MetaLogStore::MakeDummyEntry() {
    ptr<buffer> empty = buffer::alloc(0);
    return nuraft::cs_new<log_entry>(0ULL, empty, log_val_type::app_log);
}

ulong MetaLogStore::next_slot() const {
    std::lock_guard<std::mutex> g(mutex_);
    if (entries_.empty()) {
        return start_index_;
    }
    return entries_.rbegin()->first + 1;
}

ulong MetaLogStore::start_index() const {
    std::lock_guard<std::mutex> g(mutex_);
    return start_index_;
}

ptr<log_entry> MetaLogStore::last_entry() const {
    std::lock_guard<std::mutex> g(mutex_);
    if (entries_.empty()) {
        return MakeDummyEntry();
    }
    return entries_.rbegin()->second;
}

ulong MetaLogStore::append(ptr<log_entry> &entry) {
    std::lock_guard<std::mutex> g(mutex_);
    ulong idx = entries_.empty() ? start_index_ : entries_.rbegin()->first + 1;
    entries_.emplace(idx, entry);
    return idx;
}

void MetaLogStore::write_at(ulong index, ptr<log_entry> &entry) {
    std::lock_guard<std::mutex> g(mutex_);
    // Truncate any entries at or after `index`, then insert the new one.
    entries_.erase(entries_.lower_bound(index), entries_.end());
    entries_.emplace(index, entry);
}

ptr<std::vector<ptr<log_entry>>> MetaLogStore::log_entries(ulong start, ulong end) {
    auto out = nuraft::cs_new<std::vector<ptr<log_entry>>>();
    if (start >= end) {
        return out;
    }
    std::lock_guard<std::mutex> g(mutex_);
    out->reserve(end - start);
    for (ulong i = start; i < end; ++i) {
        auto it = entries_.find(i);
        if (it == entries_.end()) {
            return nullptr;
        }
        out->push_back(it->second);
    }
    return out;
}

ptr<log_entry> MetaLogStore::entry_at(ulong index) {
    std::lock_guard<std::mutex> g(mutex_);
    auto it = entries_.find(index);
    if (it == entries_.end()) {
        return MakeDummyEntry();
    }
    return it->second;
}

ulong MetaLogStore::term_at(ulong index) {
    std::lock_guard<std::mutex> g(mutex_);
    if (index < start_index_) {
        return 0;
    }
    auto it = entries_.find(index);
    if (it == entries_.end()) {
        return 0;
    }
    return it->second->get_term();
}

ptr<buffer> MetaLogStore::pack(ulong index, nuraft::int32 cnt) {
    std::lock_guard<std::mutex> g(mutex_);
    // Layout: [count:u32] then for each entry [size:u32][serialized log_entry bytes].
    std::vector<ptr<buffer>> serialized;
    serialized.reserve(cnt);
    size_t total = sizeof(uint32_t);
    for (nuraft::int32 i = 0; i < cnt; ++i) {
        auto it = entries_.find(index + static_cast<ulong>(i));
        if (it == entries_.end()) {
            break;
        }
        ptr<buffer> ser = it->second->serialize();
        total += sizeof(uint32_t) + ser->size();
        serialized.push_back(ser);
    }
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

void MetaLogStore::apply_pack(ulong index, buffer &pack) {
    buffer_serializer bs(pack);
    uint32_t count = bs.get_u32();
    std::lock_guard<std::mutex> g(mutex_);
    // Truncate everything >= index to receive a fresh prefix.
    entries_.erase(entries_.lower_bound(index), entries_.end());
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t entry_size = bs.get_u32();
        ptr<buffer> entry_buf = buffer::alloc(entry_size);
        void *raw = bs.get_raw(entry_size);
        std::memcpy(entry_buf->data_begin(), raw, entry_size);
        ptr<log_entry> entry = log_entry::deserialize(*entry_buf);
        entries_.emplace(index + i, entry);
    }
    // After applying a snapshot pack, start_index advances to the first slot
    // we just wrote so subsequent term_at/entry_at queries before that point
    // return 0 / dummy as expected.
    if (start_index_ < index) {
        start_index_ = index;
    }
}

bool MetaLogStore::compact(ulong last_log_index) {
    std::lock_guard<std::mutex> g(mutex_);
    entries_.erase(entries_.begin(), entries_.upper_bound(last_log_index));
    if (start_index_ < last_log_index + 1) {
        start_index_ = last_log_index + 1;
    }
    return true;
}

bool MetaLogStore::flush() {
    // In-memory store has nothing to durably flush. PR 6 swaps this for LMDB.
    return true;
}

} // namespace raft_meta
} // namespace kv_cache_manager
