#include "kv_cache_manager/client/pybind/py_span_selection.h"

#include "kv_cache_manager/common/unittest.h"

using namespace kv_cache_manager;

namespace {
RegistSpan MakeSpan(void *base, size_t size) {
    RegistSpan span;
    span.base = base;
    span.size = size;
    return span;
}
} // namespace

// Which span single-region backends (mooncake) receive. This is pure logic on
// purpose: the pybind layer owns the Python objects, the decision lives here so
// it stays testable without a Python runtime.
TEST(PySpanSelectionTest, ExplicitSpanWins) {
    auto explicit_span = MakeSpan(reinterpret_cast<void *>(0x1000), 100);
    auto first = MakeSpan(reinterpret_cast<void *>(0x2000), 200);
    auto second = MakeSpan(reinterpret_cast<void *>(0x3000), 300);
    const std::vector<const RegistSpan *> segments{&first, &second};

    for (const auto &host_spans : {std::vector<const RegistSpan *>{}, std::vector<const RegistSpan *>{&first},
                                   segments}) {
        auto selected = SelectSingularRegistSpan(&explicit_span, host_spans);
        ASSERT_TRUE(selected.has_value());
        EXPECT_EQ(explicit_span.base, selected->base);
        EXPECT_EQ(explicit_span.size, selected->size);
    }
}

TEST(PySpanSelectionTest, SingleSegmentListIsTheSingularRegistration) {
    auto only = MakeSpan(reinterpret_cast<void *>(0x2000), 200);

    auto selected = SelectSingularRegistSpan(nullptr, {&only});
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(only.base, selected->base);
    EXPECT_EQ(only.size, selected->size);

    const std::vector<const RegistSpan *> unset_segment{nullptr};
    EXPECT_FALSE(SelectSingularRegistSpan(nullptr, unset_segment).has_value());
}

TEST(PySpanSelectionTest, SeveralSegmentsAreNotCollapsed) {
    auto first = MakeSpan(reinterpret_cast<void *>(0x2000), 200);
    auto second = MakeSpan(reinterpret_cast<void *>(0x3000), 300);

    EXPECT_FALSE(SelectSingularRegistSpan(nullptr, {}).has_value());
    EXPECT_FALSE(SelectSingularRegistSpan(nullptr, {&first, &second}).has_value());
    EXPECT_FALSE(SelectSingularRegistSpan(nullptr, {&first, nullptr}).has_value());
}

// The result must be a snapshot: the source objects are Python-owned and may be
// retyped by another thread while Create runs without the GIL.
TEST(PySpanSelectionTest, SelectionIsACopy) {
    auto source = MakeSpan(reinterpret_cast<void *>(0x2000), 200);

    auto selected = SelectSingularRegistSpan(&source, {});
    ASSERT_TRUE(selected.has_value());
    source.base = reinterpret_cast<void *>(0x9000);
    source.size = 900;
    EXPECT_EQ(reinterpret_cast<void *>(0x2000), selected->base);
    EXPECT_EQ(200u, selected->size);
}
