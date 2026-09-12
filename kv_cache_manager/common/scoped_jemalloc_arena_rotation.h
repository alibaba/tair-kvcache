#pragma once

#include <cstddef>

namespace kv_cache_manager {

// Thread-confined scope: spreads successive allocation batches across jemalloc's
// automatic arenas and restores the caller's binding on every exit path.
// Existing allocations and tcache entries keep their original arena ownership.
class ScopedJemallocArenaRotation {
public:
    using Mallctl = int (*)(const char *, void *, size_t *, void *, size_t);

    explicit ScopedJemallocArenaRotation(bool enabled);
    // Injectable public control API, also used by deterministic allocator tests.
    explicit ScopedJemallocArenaRotation(Mallctl mallctl);
    ~ScopedJemallocArenaRotation();

    ScopedJemallocArenaRotation(const ScopedJemallocArenaRotation &) = delete;
    ScopedJemallocArenaRotation &operator=(const ScopedJemallocArenaRotation &) = delete;

    void NextBatch();

private:
    static Mallctl ResolveMallctl();
    void Initialize();
    void Restore();

    Mallctl mallctl_ = nullptr;
    unsigned arena_count_ = 0;
    unsigned original_arena_ = 0;
    unsigned current_arena_ = 0;
    unsigned next_arena_ = 0;
    bool active_ = false;
    bool changed_ = false;
};

} // namespace kv_cache_manager
