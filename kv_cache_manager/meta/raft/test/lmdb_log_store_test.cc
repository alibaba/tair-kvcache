#include "kv_cache_manager/meta/raft/lmdb_log_store.h"

#include <libnuraft/buffer.hxx>
#include <libnuraft/log_entry.hxx>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {
namespace raft_meta {

using nuraft::buffer;
using nuraft::cs_new;
using nuraft::log_entry;
using nuraft::log_val_type;
using nuraft::ptr;
using nuraft::ulong;

class LmdbLogStoreTest : public TESTBASE {
protected:
    void SetUp() override {
        TESTBASE::SetUp();
        char tpl[] = "/tmp/lmdb_log_store_test_XXXXXX";
        dir_ = mkdtemp(tpl);
    }

    void TearDown() override {
        std::filesystem::remove_all(dir_);
        TESTBASE::TearDown();
    }

    std::string dir_;

    static ptr<log_entry> MakeEntry(ulong term, const std::string &payload) {
        ptr<buffer> buf = buffer::alloc(payload.size());
        buf->put_raw(reinterpret_cast<const nuraft::byte *>(payload.data()), payload.size());
        return cs_new<log_entry>(term, buf, log_val_type::app_log);
    }
};

TEST_F(LmdbLogStoreTest, EmptyStartsAtSlotOne) {
    LmdbLogStore store(dir_);
    EXPECT_EQ(1u, store.next_slot());
    EXPECT_EQ(1u, store.start_index());
    auto last = store.last_entry();
    ASSERT_TRUE(last);
    EXPECT_EQ(0u, last->get_term());
}

TEST_F(LmdbLogStoreTest, AppendIncrementsSlot) {
    LmdbLogStore store(dir_);
    auto e1 = MakeEntry(1, "a");
    auto e2 = MakeEntry(1, "b");
    auto e3 = MakeEntry(2, "c");
    EXPECT_EQ(1u, store.append(e1));
    EXPECT_EQ(2u, store.append(e2));
    EXPECT_EQ(3u, store.append(e3));
    EXPECT_EQ(4u, store.next_slot());
    EXPECT_EQ(2u, store.last_entry()->get_term());
}

TEST_F(LmdbLogStoreTest, EntryAtAndTermAt) {
    LmdbLogStore store(dir_);
    auto e1 = MakeEntry(7, "x");
    store.append(e1);
    EXPECT_EQ(7u, store.term_at(1));
    EXPECT_EQ(0u, store.term_at(2));
    auto fetched = store.entry_at(1);
    EXPECT_EQ(7u, fetched->get_term());
    auto missing = store.entry_at(5);
    ASSERT_TRUE(missing);
    EXPECT_EQ(0u, missing->get_term());
}

TEST_F(LmdbLogStoreTest, LogEntriesRange) {
    LmdbLogStore store(dir_);
    for (int i = 0; i < 5; ++i) {
        auto e = MakeEntry(static_cast<ulong>(i + 1), "p");
        store.append(e);
    }
    auto out = store.log_entries(2, 5);
    ASSERT_TRUE(out);
    ASSERT_EQ(3u, out->size());
    EXPECT_EQ(2u, (*out)[0]->get_term());
    EXPECT_EQ(3u, (*out)[1]->get_term());
    EXPECT_EQ(4u, (*out)[2]->get_term());
    auto empty = store.log_entries(5, 5);
    ASSERT_TRUE(empty);
    EXPECT_TRUE(empty->empty());
    auto missing = store.log_entries(10, 12);
    EXPECT_FALSE(missing);
}

TEST_F(LmdbLogStoreTest, WriteAtTruncatesSuffix) {
    LmdbLogStore store(dir_);
    for (int i = 0; i < 5; ++i) {
        auto e = MakeEntry(1, "p");
        store.append(e);
    }
    auto replacement = MakeEntry(9, "r");
    store.write_at(3, replacement);
    EXPECT_EQ(4u, store.next_slot());
    EXPECT_EQ(9u, store.term_at(3));
    EXPECT_EQ(0u, store.term_at(4));
    EXPECT_EQ(0u, store.term_at(5));
}

TEST_F(LmdbLogStoreTest, CompactAdvancesStartIndex) {
    LmdbLogStore store(dir_);
    for (int i = 0; i < 5; ++i) {
        auto e = MakeEntry(static_cast<ulong>(i + 1), "p");
        store.append(e);
    }
    EXPECT_TRUE(store.compact(3));
    EXPECT_EQ(4u, store.start_index());
    EXPECT_EQ(0u, store.term_at(1));
    EXPECT_EQ(0u, store.term_at(3));
    EXPECT_EQ(4u, store.term_at(4));
    EXPECT_EQ(6u, store.next_slot());
}

TEST_F(LmdbLogStoreTest, PackApplyPackRoundTrip) {
    LmdbLogStore source(dir_);
    for (int i = 0; i < 3; ++i) {
        auto e = MakeEntry(static_cast<ulong>(i + 1), "p");
        source.append(e);
    }
    auto packed = source.pack(1, 3);
    ASSERT_TRUE(packed);

    std::string dest_dir = dir_ + "/dest";
    LmdbLogStore dest(dest_dir);
    dest.apply_pack(1, *packed);
    EXPECT_EQ(4u, dest.next_slot());
    EXPECT_EQ(1u, dest.term_at(1));
    EXPECT_EQ(2u, dest.term_at(2));
    EXPECT_EQ(3u, dest.term_at(3));
}

// --- Persistence-specific tests ---

TEST_F(LmdbLogStoreTest, PersistenceAfterReopen) {
    {
        LmdbLogStore store(dir_);
        auto e1 = MakeEntry(10, "hello");
        auto e2 = MakeEntry(11, "world");
        store.append(e1);
        store.append(e2);
        EXPECT_EQ(3u, store.next_slot());
    }
    {
        LmdbLogStore store(dir_);
        EXPECT_EQ(3u, store.next_slot());
        EXPECT_EQ(1u, store.start_index());
        EXPECT_EQ(10u, store.term_at(1));
        EXPECT_EQ(11u, store.term_at(2));
        EXPECT_EQ(11u, store.last_entry()->get_term());
    }
}

TEST_F(LmdbLogStoreTest, CompactPersistsAcrossReopen) {
    {
        LmdbLogStore store(dir_);
        for (int i = 0; i < 5; ++i) {
            auto e = MakeEntry(static_cast<ulong>(i + 1), "p");
            store.append(e);
        }
        store.compact(3);
        EXPECT_EQ(4u, store.start_index());
    }
    {
        LmdbLogStore store(dir_);
        EXPECT_EQ(4u, store.start_index());
        EXPECT_EQ(6u, store.next_slot());
        EXPECT_EQ(0u, store.term_at(1));
        EXPECT_EQ(4u, store.term_at(4));
        EXPECT_EQ(5u, store.term_at(5));
    }
}

TEST_F(LmdbLogStoreTest, WriteAtPersists) {
    {
        LmdbLogStore store(dir_);
        for (int i = 0; i < 5; ++i) {
            auto e = MakeEntry(1, "p");
            store.append(e);
        }
        auto replacement = MakeEntry(99, "new");
        store.write_at(3, replacement);
        EXPECT_EQ(4u, store.next_slot());
    }
    {
        LmdbLogStore store(dir_);
        EXPECT_EQ(4u, store.next_slot());
        EXPECT_EQ(99u, store.term_at(3));
        EXPECT_EQ(0u, store.term_at(4));
    }
}

TEST_F(LmdbLogStoreTest, FlushSyncsToOS) {
    LmdbLogStore store(dir_);
    auto e = MakeEntry(1, "data");
    store.append(e);
    EXPECT_TRUE(store.flush());
}

} // namespace raft_meta
} // namespace kv_cache_manager
