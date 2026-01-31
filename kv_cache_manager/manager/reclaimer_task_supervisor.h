#pragma once

#include <atomic>
#include <autil/LockFreeQueue.h>
#include <memory>
#include <string>
#include <thread>

#include "kv_cache_manager/manager/schedule_plan_executor.h"

namespace kv_cache_manager {

class ReclaimerTaskSupervisor {
public:
    explicit ReclaimerTaskSupervisor(std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor);
    ~ReclaimerTaskSupervisor();
    void Start();
    void Submit(const std::string &trace_id, CacheMetaDelRequest &&request);
    void Submit(const std::string &trace_id, CacheLocationDelRequest &&request);

private:
    struct ReclaimerTaskSupervisorCell {
        std::string trace_id;
        std::string instance_id;
        std::future<PlanExecuteResult> result;
    };

    void Stop();
    void WorkLoop();

    autil::LockFreeQueue<std::shared_ptr<ReclaimerTaskSupervisorCell>> cell_queue_;
    std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor_;
    std::thread supervisor_;
    std::atomic_bool stop_ = false;
};

} // namespace kv_cache_manager