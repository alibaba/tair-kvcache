#pragma once

#include <lmdb.h>
#include <libnuraft/log_entry.hxx>
#include <libnuraft/log_store.hxx>

#include <mutex>
#include <string>

namespace kv_cache_manager {
namespace raft_meta {

class LmdbLogStore : public nuraft::log_store {
public:
    explicit LmdbLogStore(const std::string &dir);
    ~LmdbLogStore() override;

    LmdbLogStore(const LmdbLogStore &) = delete;
    LmdbLogStore &operator=(const LmdbLogStore &) = delete;

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
    static nuraft::ptr<nuraft::log_entry> MakeDummyEntry();
    static void EncodeIndex(nuraft::ulong index, uint8_t out[8]);
    static nuraft::ulong DecodeIndex(const uint8_t data[8]);

    void SaveStartIndex(MDB_txn *txn, nuraft::ulong index);
    nuraft::ulong LoadStartIndex(MDB_txn *txn) const;

    MDB_env *env_ = nullptr;
    MDB_dbi logs_dbi_ = 0;
    MDB_dbi meta_dbi_ = 0;
    mutable std::mutex mutex_;
};

} // namespace raft_meta
} // namespace kv_cache_manager
