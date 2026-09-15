#include <cstdint>
#include <functional>
#include <vector>

#include "kv_cache_manager/common/hash_util.h"
#include "kv_cache_manager/common/unittest.h"

namespace kv_cache_manager {

class HashUtilTest : public TESTBASE {};

TEST_F(HashUtilTest, HashIntArrayHasStableModuloAndArithmeticShiftSemantics) {
    std::vector<uint32_t> values = {0, 1, 0x9AE0DAAFu, 0xFFFFFFFFu};
    const std::vector<int64_t> expected_prefixes = {
        -7046029254386353131LL,
        -8366447769517737510LL,
        INT64_C(0x6F5CAD4E83A46702),
        INT64_C(0x0650CCBBB6B32E60),
    };

    int64_t hash = 0;
    std::hash<uint32_t> hasher;
    for (size_t i = 0; i < values.size(); ++i) {
        hash = HashUtil::HashIntFunc(hasher, hash, values[i]);
        EXPECT_EQ(expected_prefixes[i], hash);
    }
    EXPECT_EQ(expected_prefixes.back(), HashUtil::HashIntArray(values.data(), values.data() + values.size(), 0));
}

} // namespace kv_cache_manager
