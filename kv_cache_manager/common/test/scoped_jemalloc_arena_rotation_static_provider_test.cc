#include <cerrno>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>

#include "kv_cache_manager/common/scoped_jemalloc_arena_rotation.h"

#if defined(ADDRESS_SANITIZER)
TEST(ArenaRotationStaticProviderTest, KeepsMallocDefinedByExecutable) {
    GTEST_SKIP() << "overriding malloc bypasses the address sanitizer allocator";
}
#else
namespace {
unsigned current_arena = 0;
unsigned arena_writes = 0;
const char *percpu_arena = "disabled";
} // namespace

extern "C" void *__libc_malloc(size_t size) noexcept;

extern "C" void *malloc(size_t size) noexcept { return __libc_malloc(size); }

extern "C" int mallctl(const char *name, void *out, size_t *out_size, void *in, size_t in_size) {
    if (in) {
        if (std::strcmp(name, "thread.arena") != 0 || in_size != sizeof(unsigned)) {
            return EINVAL;
        }
        current_arena = *static_cast<unsigned *>(in);
        ++arena_writes;
        return 0;
    }
    if (std::strcmp(name, "opt.percpu_arena") == 0) {
        *static_cast<const char **>(out) = percpu_arena;
        *out_size = sizeof(const char *);
    } else if (std::strcmp(name, "opt.narenas") == 0) {
        *static_cast<unsigned *>(out) = 2;
        *out_size = sizeof(unsigned);
    } else if (std::strcmp(name, "thread.arena") == 0) {
        *static_cast<unsigned *>(out) = current_arena;
        *out_size = sizeof(unsigned);
    } else {
        return ENOENT;
    }
    return 0;
}

namespace kv_cache_manager {
namespace {

TEST(ArenaRotationStaticProviderTest, KeepsMallocDefinedByExecutable) {
    current_arena = 0;
    arena_writes = 0;
    {
        ScopedJemallocArenaRotation rotation(true);
        rotation.Rotate();
        rotation.Rotate();
        EXPECT_EQ(current_arena, 1u);
        EXPECT_EQ(arena_writes, 1u);
    }
    EXPECT_EQ(current_arena, 0u);
    EXPECT_EQ(arena_writes, 2u);
}

} // namespace
} // namespace kv_cache_manager
#endif
