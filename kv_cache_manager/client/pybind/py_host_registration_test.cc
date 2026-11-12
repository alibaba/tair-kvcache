#include "kv_cache_manager/client/pybind/py_host_registration.h"

#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

namespace {
RegistSpan MakeSpan(void *base, size_t size) {
    RegistSpan span;
    span.base = base;
    span.size = size;
    return span;
}

SharedMemoryRegistration MakeRegistration(void *base, size_t size, int fd) {
    return SharedMemoryRegistration{base, size, fd};
}

// Same two steps Create performs: normalize the caller's list, then pick what
// single-region backends receive.
std::optional<RegistSpan> SelectFromList(const std::vector<SharedMemoryRegistration> &entries) {
    return SelectSingularRegistSpan(std::nullopt, NormalizeHostRegistrations(entries));
}
} // namespace

TEST(PyHostRegistrationTest, DropsUnsetEntriesAndKeepsOrder) {
    auto first = MakeRegistration(reinterpret_cast<void *>(0x1000), 100, 3);
    auto second = MakeRegistration(reinterpret_cast<void *>(0x2000), 200, 4);

    const auto registrations =
        NormalizeHostRegistrations({first, SharedMemoryRegistration{}, second, SharedMemoryRegistration{}});
    ASSERT_EQ(2u, registrations.size());
    EXPECT_EQ(first.base, registrations[0].base);
    EXPECT_EQ(second.base, registrations[1].base);
    EXPECT_TRUE(NormalizeHostRegistrations({}).empty());
    EXPECT_TRUE(NormalizeHostRegistrations({SharedMemoryRegistration{}}).empty());
}

TEST(PyHostRegistrationTest, RejectsPartiallyFilledEntries) {
    auto valid = MakeRegistration(reinterpret_cast<void *>(0x9000), 900, 9);
    auto no_fd = MakeRegistration(reinterpret_cast<void *>(0x1000), 100, -1);
    auto no_size = MakeRegistration(reinterpret_cast<void *>(0x1000), 0, 3);
    auto no_base = MakeRegistration(nullptr, 100, 3);

    for (const auto &entry : {no_fd, no_size, no_base}) {
        EXPECT_THROW(NormalizeHostRegistrations({valid, entry}), std::invalid_argument);
    }
    try {
        NormalizeHostRegistrations({valid, no_fd});
        ADD_FAILURE() << "partially filled entry was accepted";
    } catch (const std::invalid_argument &error) {
        const std::string message = error.what();
        EXPECT_NE(std::string::npos, message.find("regist_spans[1]"));
        EXPECT_NE(std::string::npos, message.find("fd"));
    }
}

TEST(PyHostRegistrationTest, ExplicitSpanWins) {
    auto explicit_span = MakeSpan(reinterpret_cast<void *>(0x1000), 100);
    auto first = MakeRegistration(reinterpret_cast<void *>(0x2000), 200, 3);
    auto second = MakeRegistration(reinterpret_cast<void *>(0x3000), 300, 4);

    for (const auto &entries : {std::vector<SharedMemoryRegistration>{},
                                std::vector<SharedMemoryRegistration>{first},
                                std::vector<SharedMemoryRegistration>{first, second}}) {
        auto selected = SelectSingularRegistSpan(explicit_span, NormalizeHostRegistrations(entries));
        ASSERT_TRUE(selected.has_value());
        EXPECT_EQ(explicit_span.base, selected->base);
        EXPECT_EQ(explicit_span.size, selected->size);
    }
}

// The "exactly one segment" criterion counts registrations that actually
// register something, not list elements.
TEST(PyHostRegistrationTest, SingleEffectiveSegmentIsTheSingularRegistration) {
    auto only = MakeRegistration(reinterpret_cast<void *>(0x2000), 200, 3);

    for (const auto &entries : {std::vector<SharedMemoryRegistration>{only},
                                std::vector<SharedMemoryRegistration>{only, SharedMemoryRegistration{}},
                                std::vector<SharedMemoryRegistration>{SharedMemoryRegistration{}, only}}) {
        auto selected = SelectFromList(entries);
        ASSERT_TRUE(selected.has_value());
        EXPECT_EQ(only.base, selected->base);
        EXPECT_EQ(only.size, selected->size);
    }
}

TEST(PyHostRegistrationTest, SeveralSegmentsAreNotCollapsed) {
    auto first = MakeRegistration(reinterpret_cast<void *>(0x2000), 200, 3);
    auto second = MakeRegistration(reinterpret_cast<void *>(0x3000), 300, 4);

    EXPECT_FALSE(SelectFromList({}).has_value());
    EXPECT_FALSE(SelectFromList({SharedMemoryRegistration{}}).has_value());
    EXPECT_FALSE(SelectFromList({first, second}).has_value());
    EXPECT_FALSE(SelectFromList({first, second, SharedMemoryRegistration{}}).has_value());
}

// The result must be a snapshot: the source objects are Python-owned and may be
// retyped by another thread while Create runs without the GIL.
TEST(PyHostRegistrationTest, SelectionIsACopy) {
    auto source = MakeSpan(reinterpret_cast<void *>(0x2000), 200);

    auto selected = SelectSingularRegistSpan(source, {});
    ASSERT_TRUE(selected.has_value());
    source.base = reinterpret_cast<void *>(0x9000);
    source.size = 900;
    EXPECT_EQ(reinterpret_cast<void *>(0x2000), selected->base);
    EXPECT_EQ(200u, selected->size);
}
