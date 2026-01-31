#include <gtest/gtest.h>
#include "kv_cache_manager/common/unittest.h"

TEST(Hf3fsUnittestEmpty, SkipTestIfHf3fsDisabled) { GTEST_SKIP() << "ENABLE_HF3FS is not set; skipping 3FS tests."; }
