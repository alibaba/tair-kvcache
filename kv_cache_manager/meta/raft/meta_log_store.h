#pragma once

#include <libnuraft/log_entry.hxx>
#include <libnuraft/log_store.hxx>

#include <map>
#include <mutex>

namespace kv_cache_manager {
namespace raft_meta {

// In-memory implementation of nuraft::log_store. Holds entries in a
// std::map keyed by log index so that compact (truncate-prefix) and
// write_at (truncate-suffix) are O(log n). PR 6 will swap the underlying
// container for a durable LMDB-backed store; the interface here is the
// stable boundary.
class MetaLogStore : public nuraft::log_store {
public:
    MetaLogStore();
    ~MetaLogStore() override = default;

    nuraft::ulong next_slot() const override;
    nuraft::ulong start_index() const override;
    nuraft::ptr<nuraft::log_entry> last_entry() const override;
    nuraft::ulong append(nuraft::ptr<nuraft::log_entry> &entry) override;
    void write_at(nuraft::ulong index, nuraft::ptr<nuraft::log_entry> &entry) override;
    nuraft::ptr<std::vector<nuraft::ptr<nuraft::log_entry>>>
    log_entries(nuraft::ulong start, nuraft::ulong end) override;
    nuraft::ptr<nuraft::log_entry> entry_at(nuraft::ulong index) override;
    nuraft::ulong term_at(nuraft::ulong index) override;
    nuraft::ptr<nuraft::buffer> pack(nuraft::ulong index, nuraft::int32 cnt) override;
    void apply_pack(nuraft::ulong index, nuraft::buffer &pack) override;
    bool compact(nuraft::ulong last_log_index) override;
    bool flush() override;

private:
    // Returns a default zero-term entry; used when caller asks for
    // last_entry on an empty store or queries an out-of-range index.
    static nuraft::ptr<nuraft::log_entry> MakeDummyEntry();

    mutable std::mutex mutex_;
    std::map<nuraft::ulong, nuraft::ptr<nuraft::log_entry>> entries_;
    // Smallest live index. Bumped by compact() when entries are purged.
    nuraft::ulong start_index_ = 1;
};

} // namespace raft_meta
} // namespace kv_cache_manager
