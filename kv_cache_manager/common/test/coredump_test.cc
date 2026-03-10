#include <gtest/gtest.h>
#include <csignal>

// This test deliberately causes a segfault to verify CI coredump collection.
// It should be tagged "manual" and only run explicitly.
TEST(CoredumpCollectionTest, TriggerSegfault) {
  volatile int *p = nullptr;
  // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
  *p = 42;
}
