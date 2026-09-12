#include <cerrno>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "kv_cache_manager/common/scoped_jemalloc_arena_rotation.h"

namespace kv_cache_manager {
namespace {
struct FakeAllocator {
    unsigned count = 4;
    unsigned current = 2;
    const char *percpu = "disabled";
    std::string fail_read;
    int fail_bind = -1;
    bool short_read = false;
    std::vector<unsigned> writes;
    std::vector<std::string> reads;
};
thread_local FakeAllocator allocator;

int Control(const char *name, void *out, size_t *out_size, void *in, size_t in_size) {
    if (in) {
        EXPECT_STREQ(name, "thread.arena");
        EXPECT_EQ(in_size, sizeof(unsigned));
        EXPECT_EQ(out, nullptr);
        auto arena = *static_cast<unsigned *>(in);
        allocator.writes.push_back(arena);
        if (static_cast<int>(arena) == allocator.fail_bind) {
            return EINVAL;
        }
        allocator.current = arena;
        return 0;
    }
    allocator.reads.emplace_back(name);
    if (allocator.fail_read == name) {
        return ENOENT;
    }
    if (std::strcmp(name, "opt.percpu_arena") == 0) {
        EXPECT_EQ(*out_size, sizeof(const char *));
        *static_cast<const char **>(out) = allocator.percpu;
    } else if (std::strcmp(name, "opt.narenas") == 0) {
        EXPECT_EQ(*out_size, sizeof(unsigned));
        *static_cast<unsigned *>(out) = allocator.count;
    } else if (std::strcmp(name, "thread.arena") == 0) {
        EXPECT_EQ(*out_size, sizeof(unsigned));
        *static_cast<unsigned *>(out) = allocator.current;
    } else {
        ADD_FAILURE() << "Unexpected control (including tcache flush): " << name;
        return ENOENT;
    }
    if (allocator.short_read) {
        --*out_size;
    }
    return 0;
}

class ArenaRotationTest : public testing::Test {
    void SetUp() override { allocator = FakeAllocator{}; }
};

TEST_F(ArenaRotationTest, WrapsAutomaticArenasAndRestoresOriginal) {
    {
        ScopedJemallocArenaRotation rotation(Control);
        for (unsigned expected : {2, 3, 0, 1, 2, 3}) {
            rotation.NextBatch();
            EXPECT_EQ(allocator.current, expected);
        }
        EXPECT_EQ(allocator.reads, (std::vector<std::string>{"opt.percpu_arena", "opt.narenas", "thread.arena"}));
    }
    EXPECT_EQ(allocator.current, 2u);
    EXPECT_EQ(allocator.writes, (std::vector<unsigned>{3, 0, 1, 2, 3, 2}));
}

TEST_F(ArenaRotationTest, NoWritesWithoutBatches) {
    { ScopedJemallocArenaRotation rotation(Control); }
    EXPECT_TRUE(allocator.writes.empty());
}

TEST_F(ArenaRotationTest, NullControlAndDisabledAreNoops) {
    ScopedJemallocArenaRotation absent(static_cast<ScopedJemallocArenaRotation::Mallctl>(nullptr));
    ScopedJemallocArenaRotation disabled(false);
    absent.NextBatch();
    disabled.NextBatch();
    EXPECT_TRUE(allocator.reads.empty());
    EXPECT_TRUE(allocator.writes.empty());
}

TEST_F(ArenaRotationTest, SingleOrZeroArenaIsNoop) {
    for (unsigned count : {0, 1}) {
        allocator.count = count;
        ScopedJemallocArenaRotation rotation(Control);
        rotation.NextBatch();
    }
    EXPECT_TRUE(allocator.writes.empty());
}

TEST_F(ArenaRotationTest, PerCpuPoliciesAreNotOverridden) {
    for (const char *mode : {"percpu", "phycpu", "unknown"}) {
        allocator.percpu = mode;
        ScopedJemallocArenaRotation rotation(Control);
        rotation.NextBatch();
    }
    EXPECT_TRUE(allocator.writes.empty());
}

TEST_F(ArenaRotationTest, FailedReadsDoNotChangeBinding) {
    for (const char *name : {"opt.percpu_arena", "opt.narenas", "thread.arena"}) {
        allocator.fail_read = name;
        ScopedJemallocArenaRotation rotation(Control);
        rotation.NextBatch();
    }
    EXPECT_TRUE(allocator.writes.empty());
}

TEST_F(ArenaRotationTest, WrongControlSizeIsRejected) {
    allocator.short_read = true;
    {
        ScopedJemallocArenaRotation rotation(Control);
        rotation.NextBatch();
    }
    EXPECT_TRUE(allocator.writes.empty());
}

TEST_F(ArenaRotationTest, ExplicitOriginalArenaIsRestoredButNotInRotation) {
    allocator.current = 65;
    {
        ScopedJemallocArenaRotation rotation(Control);
        for (unsigned expected : {0, 1, 2, 3, 0}) {
            rotation.NextBatch();
            EXPECT_EQ(allocator.current, expected);
        }
    }
    EXPECT_EQ(allocator.current, 65u);
}

TEST_F(ArenaRotationTest, BindingFailureStopsRotationAndRestoresImmediately) {
    {
        ScopedJemallocArenaRotation rotation(Control);
        rotation.NextBatch();
        rotation.NextBatch(); // 3
        allocator.fail_bind = 0;
        rotation.NextBatch(); // failed 0, restore 2
        EXPECT_EQ(allocator.current, 2u);
        rotation.NextBatch();
    }
    EXPECT_EQ(allocator.writes, (std::vector<unsigned>{3, 0, 2}));
}

TEST_F(ArenaRotationTest, DestructorRetriesFailedRestore) {
    {
        ScopedJemallocArenaRotation rotation(Control);
        rotation.NextBatch();
        rotation.NextBatch(); // 3
        rotation.NextBatch(); // 0
        rotation.NextBatch(); // 1
        allocator.fail_bind = 2;
        rotation.NextBatch(); // cannot bind or restore original
        EXPECT_EQ(allocator.current, 1u);
        allocator.fail_bind = -1;
    }
    EXPECT_EQ(allocator.current, 2u);
}

TEST_F(ArenaRotationTest, ReadsConfigurationForEachScope) {
    for (unsigned count : {2, 8}) {
        allocator.count = count;
        allocator.current = 0;
        ScopedJemallocArenaRotation rotation(Control);
        for (unsigned i = 0; i < count * 2; ++i) {
            rotation.NextBatch();
            EXPECT_EQ(allocator.current, i % count);
        }
    }
}

TEST_F(ArenaRotationTest, EarlyExitRestoresBinding) {
    auto recover = [] {
        ScopedJemallocArenaRotation rotation(Control);
        rotation.NextBatch();
        rotation.NextBatch();
        return;
    };
    recover();
    EXPECT_EQ(allocator.current, 2u);
}
} // namespace
} // namespace kv_cache_manager
