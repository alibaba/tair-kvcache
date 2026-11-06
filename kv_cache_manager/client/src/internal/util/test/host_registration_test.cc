#include "kv_cache_manager/client/src/internal/util/host_registration.h"
#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

namespace {
constexpr size_t kSize = 4096;
} // namespace

// Reference partition: the two predicate sets below are written out literally, on
// purpose independent of host_registration.h, so an edit to the classifier has to
// agree with them over every combination of the three fields.
TEST(HostRegistrationTest, AgreesWithTheReferencePartition) {
    void *const bases[] = {nullptr, reinterpret_cast<void *>(0x1000)};
    const size_t sizes[] = {0, kSize};
    const int fds[] = {-2, -1, 0, 3};

    for (void *base : bases) {
        for (size_t size : sizes) {
            for (int fd : fds) {
                const SharedMemoryRegistration registration{base, size, fd};
                SCOPED_TRACE(testing::Message() << "base=" << base << " size=" << size << " fd=" << fd);

                const bool legacy_disabled =
                    registration.fd == -1 && registration.base == nullptr && registration.size == 0;
                const bool legacy_rejected =
                    !legacy_disabled && (registration.fd < 0 || registration.base == nullptr || registration.size == 0);

                const HostRegistrationState state = ClassifyHostRegistration(registration);
                EXPECT_EQ(legacy_disabled, state == HostRegistrationState::Disabled);
                EXPECT_EQ(legacy_rejected, state == HostRegistrationState::Partial);
                EXPECT_EQ(!legacy_disabled && !legacy_rejected, state == HostRegistrationState::Complete);
            }
        }
    }
}

TEST(HostRegistrationTest, EmptyRegistrationIsDisabled) {
    EXPECT_EQ(HostRegistrationState::Disabled, ClassifyHostRegistration(SharedMemoryRegistration{}));
}

TEST(HostRegistrationTest, FullyFilledRegistrationIsComplete) {
    EXPECT_EQ(HostRegistrationState::Complete,
              ClassifyHostRegistration(SharedMemoryRegistration{reinterpret_cast<void *>(0x1000), kSize, 3}));
    // fd 0 is a legal descriptor, not a missing field.
    EXPECT_EQ(HostRegistrationState::Complete,
              ClassifyHostRegistration(SharedMemoryRegistration{reinterpret_cast<void *>(0x1000), kSize, 0}));
}

TEST(HostRegistrationTest, AnyMissingFieldIsPartial) {
    void *const base = reinterpret_cast<void *>(0x1000);
    EXPECT_EQ(HostRegistrationState::Partial, ClassifyHostRegistration(SharedMemoryRegistration{base, kSize, -1}));
    EXPECT_EQ(HostRegistrationState::Partial, ClassifyHostRegistration(SharedMemoryRegistration{nullptr, kSize, 3}));
    EXPECT_EQ(HostRegistrationState::Partial, ClassifyHostRegistration(SharedMemoryRegistration{base, 0, 3}));
    // base and size never arrived, but an fd did: not the empty sentinel.
    EXPECT_EQ(HostRegistrationState::Partial, ClassifyHostRegistration(SharedMemoryRegistration{nullptr, 0, 3}));
    // Only -1 marks "no fd"; any other negative fd is a partially filled entry.
    EXPECT_EQ(HostRegistrationState::Partial, ClassifyHostRegistration(SharedMemoryRegistration{nullptr, 0, -2}));
}
