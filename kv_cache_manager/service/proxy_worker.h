#pragma once

#include <memory>

#include "kv_cache_manager/service/worker.h"

namespace kv_cache_manager {
class ProxyWorker final : public Worker {
public:
    ProxyWorker();
    ~ProxyWorker() override;
    bool Init(const ServerConfig &config) override;
    bool Start() override;
    void Stop() override;
    bool Wait() override;

private:
    struct Runtime;
    std::unique_ptr<Runtime> runtime_;
};
} // namespace kv_cache_manager
