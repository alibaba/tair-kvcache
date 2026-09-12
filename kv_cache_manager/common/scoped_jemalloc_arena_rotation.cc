#include "kv_cache_manager/common/scoped_jemalloc_arena_rotation.h"

#include <cstring>
#include <dlfcn.h>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {
namespace {
template <typename T>
bool ReadControl(ScopedJemallocArenaRotation::Mallctl mallctl, const char *name, T &value) {
    size_t size = sizeof(value);
    const int error = mallctl(name, &value, &size, nullptr, 0);
    if (error != 0 || size != sizeof(value)) {
        KVCM_LOG_WARN("skip jemalloc arena rotation: read %s failed, error[%d] size[%zu]", name, error, size);
        return false;
    }
    return true;
}
} // namespace

ScopedJemallocArenaRotation::Mallctl ScopedJemallocArenaRotation::ResolveMallctl() {
    // Do not dlopen another allocator. A linked but non-interposing jemalloc
    // must not cause us to change the state of an allocator that malloc bypasses.
    void *control = dlsym(RTLD_DEFAULT, "mallctl");
    void *allocate = dlsym(RTLD_DEFAULT, "malloc");
    Dl_info control_info{}, allocate_info{};
    if (!control || !allocate || dladdr(control, &control_info) == 0 || dladdr(allocate, &allocate_info) == 0 ||
        control_info.dli_fbase != allocate_info.dli_fbase) {
        return nullptr;
    }
    return reinterpret_cast<Mallctl>(control);
}

ScopedJemallocArenaRotation::ScopedJemallocArenaRotation(bool enabled)
    : ScopedJemallocArenaRotation(enabled ? ResolveMallctl() : nullptr) {}

ScopedJemallocArenaRotation::ScopedJemallocArenaRotation(Mallctl mallctl) : mallctl_(mallctl) { Initialize(); }

void ScopedJemallocArenaRotation::Initialize() {
    if (!mallctl_) {
        return;
    }
    const char *percpu = nullptr;
    if (!ReadControl(mallctl_, "opt.percpu_arena", percpu)) {
        return;
    }
    if (!percpu || std::strcmp(percpu, "disabled") != 0) {
        KVCM_LOG_INFO("skip jemalloc arena rotation: percpu_arena[%s]", percpu ? percpu : "unknown");
        return;
    }
    // arenas.narenas also includes manually created and the oversize arena.
    // opt.narenas is the effective, immutable automatic arena count.
    if (!ReadControl(mallctl_, "opt.narenas", arena_count_) || arena_count_ <= 1 ||
        !ReadControl(mallctl_, "thread.arena", original_arena_)) {
        return;
    }
    current_arena_ = original_arena_;
    next_arena_ = original_arena_ < arena_count_ ? original_arena_ : 0;
    active_ = true;
    KVCM_LOG_INFO(
        "jemalloc recovery arena rotation enabled, arenas[%u] original_arena[%u]", arena_count_, original_arena_);
}

void ScopedJemallocArenaRotation::Rotate() {
    if (!active_) {
        return;
    }
    if (next_arena_ != current_arena_) {
        const int error = mallctl_("thread.arena", nullptr, nullptr, &next_arena_, sizeof(next_arena_));
        if (error != 0) {
            KVCM_LOG_WARN("stop jemalloc arena rotation: bind arena[%u] failed, error[%d]", next_arena_, error);
            active_ = false;
            Restore();
            return;
        }
        current_arena_ = next_arena_;
        changed_ = current_arena_ != original_arena_;
    }
    next_arena_ = (next_arena_ + 1) % arena_count_;
}

void ScopedJemallocArenaRotation::Restore() {
    if (!changed_) {
        return;
    }
    const int error = mallctl_("thread.arena", nullptr, nullptr, &original_arena_, sizeof(original_arena_));
    if (error != 0) {
        KVCM_LOG_WARN("restore jemalloc thread arena[%u] failed, error[%d]", original_arena_, error);
        return;
    }
    current_arena_ = original_arena_;
    changed_ = false;
}

ScopedJemallocArenaRotation::~ScopedJemallocArenaRotation() { Restore(); }

} // namespace kv_cache_manager
