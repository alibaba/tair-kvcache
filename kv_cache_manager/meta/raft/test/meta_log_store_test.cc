#include "kv_cache_manager/meta/raft/meta_log_store.h"

#include <libnuraft/buffer.hxx>
#include <libnuraft/log_entry.hxx>

#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {
namespace raft_meta {

using nuraft::buffer;
using nuraft::cs_new;
using nuraft::log_entry;
using nuraft::log_val_type;
using nuraft::ptr;
using nuraft::ulong;

class MetaLogStoreTest : public TESTBASE {
protected:
    static ptr<log_entry> MakeEntry(ulong term, const std::string &payload) {
        ptr<buffer> buf = buffer::alloc(payload.size());
        buf->put_raw(reinterpret_cast<const nuraft::byte *>(payload.data()), payload.size());
        return cs_new<log_entry>(term, buf, log_val_type::app_log);
    }
};

TEST_F(MetaLogStoreTest, EmptyStartsAtSlotOne) {
    MetaLogStore store;
    EXPECT_EQ(1u, store.next_slot());
    EXPECT_EQ(1u, store.start_index());
    auto last = store.last_entry();
    ASSERT_TRUE(last);
    EXPECT_EQ(0u, last->get_term());
}

TEST_F(MetaLogStoreTest, AppendIncrementsSlot) {
    MetaLogStore store;
    auto e1 = MakeEntry(1, "a");
    auto e2 = MakeEntry(1, "b");
    auto e3 = MakeEntry(2, "c");
    EXPECT_EQ(1u, store.append(e1));
    EXPECT_EQ(2u, store.append(e2));
    EXPECT_EQ(3u, store.append(e3));
    EXPECT_EQ(4u, store.next_slot());
    EXPECT_EQ(2u, store.last_entry()->get_term());
}

TEST_F(MetaLogStoreTest, EntryAtAndTermAt) {
    MetaLogStore store;
    auto e1 = MakeEntry(7, "x");
    store.append(e1);
    EXPECT_EQ(7u, store.term_at(1));
    EXPECT_EQ(0u, store.term_at(2)); // out of range
    auto fetched = store.entry_at(1);
    EXPECT_EQ(7u, fetched->get_term());
    auto missing = store.entry_at(5);
    ASSERT_TRUE(missing);
    EXPECT_EQ(0u, missing->get_term());
}

TEST_F(MetaLogStoreTest, LogEntriesRange) {
    MetaLogStore store;
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

TEST_F(MetaLogStoreTest, WriteAtTruncatesSuffix) {
    MetaLogStore store;
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

TEST_F(MetaLogStoreTest, CompactAdvancesStartIndex) {
    MetaLogStore store;
    for (int i = 0; i < 5; ++i) {
        auto e = MakeEntry(static_cast<ulong>(i + 1), "p");
        store.append(e);
    }
    EXPECT_TRUE(store.compact(3));
    EXPECT_EQ(4u, store.start_index());
    EXPECT_EQ(0u, store.term_at(1)); // compacted
    EXPECT_EQ(0u, store.term_at(3)); // compacted
    EXPECT_EQ(4u, store.term_at(4));
    EXPECT_EQ(6u, store.next_slot());
}

TEST_F(MetaLogStoreTest, PackApplyPackRoundTrip) {
    MetaLogStore source;
    for (int i = 0; i < 3; ++i) {
        auto e = MakeEntry(static_cast<ulong>(i + 1), "p");
        source.append(e);
    }
    auto packed = source.pack(1, 3);
    ASSERT_TRUE(packed);

    MetaLogStore dest;
    dest.apply_pack(1, *packed);
    EXPECT_EQ(4u, dest.next_slot());
    EXPECT_EQ(1u, dest.term_at(1));
    EXPECT_EQ(2u, dest.term_at(2));
    EXPECT_EQ(3u, dest.term_at(3));
}

} // namespace raft_meta
} // namespace kv_cache_manager
