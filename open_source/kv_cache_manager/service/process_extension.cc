#include "kv_cache_manager/service/process_extension.h"

namespace kv_cache_manager {
namespace {
class NoopProcessExtension final : public ProcessExtension {
public:
    bool Start(const ProcessExtensionContext &) override { return true; }
    void Stop() noexcept override {}
};
} // namespace

std::unique_ptr<ProcessExtension> CreateProcessExtension() { return std::make_unique<NoopProcessExtension>(); }
} // namespace kv_cache_manager
