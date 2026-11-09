#include <cstddef>

#include "kv_cache_manager/common/cache/allocation_size.h"
#include "kv_cache_manager/common/unittest.h"

namespace {
size_t foreign_calls = 0;
}

extern "C" size_t nallocx(size_t size, int) {
    ++foreign_calls;
    return size + 4096;
}

namespace kv_cache_manager {

TEST(AllocationSizeTest, IgnoresAllocatorThatDoesNotProvideMalloc) {
    EXPECT_EQ(0u, EstimateAllocationSize(0));
    EXPECT_EQ(24u, EstimateAllocationSize(24));
    EXPECT_EQ(79u, EstimateAllocationSize(79));
    EXPECT_EQ(0u, foreign_calls);
}

} // namespace kv_cache_manager
