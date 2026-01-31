#include "kv_cache_manager/manager/reclaimer_task_supervisor.h"

#include <assert.h>
#include <chrono>

#include "kv_cache_manager/common/logger.h"

namespace kv_cache_manager {

namespace {
constexpr auto kDefaultSupervisorSleepTime = std::chrono::milliseconds(100);
constexpr auto kDefaultFutureWaitTime = std::chrono::microseconds(200);
} // namespace

ReclaimerTaskSupervisor::ReclaimerTaskSupervisor(std::shared_ptr<SchedulePlanExecutor> schedule_plan_executor)
    : schedule_plan_executor_(schedule_plan_executor) {
    assert(schedule_plan_executor_);
}

ReclaimerTaskSupervisor::~ReclaimerTaskSupervisor() { Stop(); }

void ReclaimerTaskSupervisor::Start() {
    supervisor_ = std::thread([this]() { this->WorkLoop(); });
}

void ReclaimerTaskSupervisor::Stop() {
    stop_.store(true, std::memory_order_relaxed);
    if (supervisor_.joinable()) {
        supervisor_.join();
    }
}

void ReclaimerTaskSupervisor::Submit(const std::string &trace_id, CacheMetaDelRequest &&request) {
    auto cell = std::make_shared<ReclaimerTaskSupervisorCell>();
    cell->trace_id = trace_id;
    cell->instance_id = request.instance_id;
    cell->result = schedule_plan_executor_->Submit(request);
    if (cell->result.valid()) {
        cell_queue_.Push(cell);
    }
}

void ReclaimerTaskSupervisor::Submit(const std::string &trace_id, CacheLocationDelRequest &&request) {
    auto cell = std::make_shared<ReclaimerTaskSupervisorCell>();
    cell->trace_id = trace_id;
    cell->instance_id = request.instance_id;
    cell->result = schedule_plan_executor_->Submit(request);
    if (cell->result.valid()) {
        cell_queue_.Push(cell);
    }
}

void ReclaimerTaskSupervisor::WorkLoop() {
    while (!stop_.load(std::memory_order_relaxed)) {
        if (cell_queue_.Empty()) {
            std::this_thread::sleep_for(kDefaultSupervisorSleepTime);
            continue;
        }
        std::shared_ptr<ReclaimerTaskSupervisorCell> cell;
        if (cell_queue_.Pop(&cell)) {
            auto status = cell->result.wait_for(kDefaultFutureWaitTime);
            if (status == std::future_status::ready) {
                auto del_result = cell->result.get();
                KVCM_LOG_INFO("delete task finish : instance_id[%s] trace_id [%s] ec[%d] message[%s]",
                              cell->instance_id.c_str(),
                              cell->trace_id.c_str(),
                              del_result.status,
                              del_result.error_message.c_str());
            } else {
                cell_queue_.Push(cell);
            }
        }
    }
}

} // namespace kv_cache_manager