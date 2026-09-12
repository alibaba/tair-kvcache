#pragma once

#include <cstddef>

namespace kv_cache_manager {

// Thread-confined scope: rotates the calling thread's binding across jemalloc's
// automatic arenas and attempts to restore the original binding on scope exit.
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

    // Select the next automatic arena without flushing tcache. The first call
    // keeps the original arena if it is automatic; subsequent calls advance.
    void Rotate();

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
