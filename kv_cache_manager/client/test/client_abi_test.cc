#include <cstddef>
#include <vector>

#include "kv_cache_manager/client/include/common.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

// client/include/*.h is packaged by //kv_cache_manager/client:include and the
// registration structs cross the kv_cache_manager_client.so boundary by
// reference, so their layout is an ABI: binaries built against an older header
// keep calling the same Create overloads. Any field/shape edit must fail here
// first — add a new overload instead of changing a published struct.
static_assert(sizeof(RegistSpan) == 16, "RegistSpan ABI changed");
static_assert(offsetof(RegistSpan, base) == 0, "RegistSpan ABI changed");
static_assert(offsetof(RegistSpan, size) == 8, "RegistSpan ABI changed");

static_assert(sizeof(SharedMemoryRegistration) == 24, "SharedMemoryRegistration ABI changed");
static_assert(offsetof(SharedMemoryRegistration, base) == 0, "SharedMemoryRegistration ABI changed");
static_assert(offsetof(SharedMemoryRegistration, size) == 8, "SharedMemoryRegistration ABI changed");
static_assert(offsetof(SharedMemoryRegistration, fd) == 16, "SharedMemoryRegistration ABI changed");

static_assert(sizeof(GpuMemorySpan) == 24, "GpuMemorySpan ABI changed");
static_assert(offsetof(GpuMemorySpan, base) == 0, "GpuMemorySpan ABI changed");
static_assert(offsetof(GpuMemorySpan, size) == 8, "GpuMemorySpan ABI changed");
static_assert(offsetof(GpuMemorySpan, device_id) == 16, "GpuMemorySpan ABI changed");

static_assert(sizeof(ClientMemoryRegistrations) == 48, "ClientMemoryRegistrations ABI changed");
static_assert(offsetof(ClientMemoryRegistrations, host) == 0, "ClientMemoryRegistrations ABI changed");
static_assert(offsetof(ClientMemoryRegistrations, gpu) == 24, "ClientMemoryRegistrations ABI changed");

// LP64 invariants behind the numbers above: members are packed back to back.
static_assert(offsetof(ClientMemoryRegistrations, gpu) == sizeof(SharedMemoryRegistration));
static_assert(sizeof(ClientMemoryRegistrations) ==
                  sizeof(SharedMemoryRegistration) + sizeof(std::vector<GpuMemorySpan>),
              "ClientMemoryRegistrations grew a field");

TEST(ClientAbiTest, RegistrationStructLayoutIsFrozen) {
    EXPECT_EQ(sizeof(SharedMemoryRegistration), 24u);
    EXPECT_EQ(offsetof(ClientMemoryRegistrations, host), 0u);
    EXPECT_EQ(offsetof(ClientMemoryRegistrations, gpu), 24u);
    EXPECT_EQ(sizeof(ClientMemoryRegistrations), 48u);
}
