#pragma once

namespace kv_cache_manager {
class ServerConfig;

class Worker {
public:
    virtual ~Worker() = default;
    virtual bool Init(const ServerConfig &config) = 0;
    virtual bool Start() = 0;
    virtual void Stop() = 0;
    virtual bool Wait() = 0;
};
} // namespace kv_cache_manager
