#include "kv_cache_manager/common/cache/allocation_size.h"

#include <dlfcn.h>
#include <link.h>

namespace kv_cache_manager {
namespace {

using Nallocx = size_t (*)(size_t, int);

Nallocx ResolveNallocx() noexcept {
    void *size_class = dlsym(RTLD_DEFAULT, "nallocx");
    void *allocate = dlsym(RTLD_DEFAULT, "malloc");
    Dl_info allocate_info{}, size_class_info{};
    void *symbol_entry = nullptr;
    if (allocate && dladdr1(allocate, &allocate_info, &symbol_entry, RTLD_DL_SYMENT) != 0) {
        const ElfW(Sym) *symbol = static_cast<const ElfW(Sym) *>(symbol_entry);
        if (symbol && symbol->st_shndx == SHN_UNDEF && symbol->st_value != 0) {
            allocate = dlsym(RTLD_NEXT, "malloc");
        }
    }
    // Never load or consult an allocator that does not provide the active malloc.
    if (!size_class || !allocate || dladdr(size_class, &size_class_info) == 0 ||
        dladdr(allocate, &allocate_info) == 0 || size_class_info.dli_fbase != allocate_info.dli_fbase) {
        return nullptr;
    }
    return reinterpret_cast<Nallocx>(size_class);
}

} // namespace

size_t EstimateAllocationSize(size_t requested) noexcept {
    static const Nallocx size_class = ResolveNallocx();
    if (requested == 0 || !size_class) {
        return requested;
    }
    const size_t allocated = size_class(requested, 0);
    return allocated >= requested ? allocated : requested;
}

} // namespace kv_cache_manager
