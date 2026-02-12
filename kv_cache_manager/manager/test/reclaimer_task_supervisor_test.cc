#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <utility>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/data_storage/data_storage_manager.h"
#include "kv_cache_manager/manager/reclaimer_task_supervisor.h"
#include "kv_cache_manager/manager/schedule_plan_executor.h"
#include "kv_cache_manager/meta/meta_indexer_manager.h"
#include "kv_cache_manager/metrics/metrics_registry.h"
#include "stub.h"

using namespace kv_cache_manager;

namespace {

using spe_submit_blk = std::future<PlanExecuteResult> (SchedulePlanExecutor::*)(const CacheMetaDelRequest &);
using spe_submit_loc = std::future<PlanExecuteResult> (SchedulePlanExecutor::*)(const CacheLocationDelRequest &);

PlanExecuteResult blk_del_result;
std::mutex blk_del_reqs_mut;
std::condition_variable blk_del_reqs_cv; // condition being not empty of submitted_blk_del_reqs
std::vector<CacheMetaDelRequest> submitted_blk_del_reqs;

std::future<PlanExecuteResult> SchedulePlanExecutor_Submit_blk_stub(void *obj, const CacheMetaDelRequest &req) {
    const auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    {
        std::lock_guard<std::mutex> lock{blk_del_reqs_mut};
        submitted_blk_del_reqs.emplace_back(req);
        blk_del_reqs_cv.notify_one();
    }
    promise->set_value(blk_del_result);
    return promise->get_future();
}

PlanExecuteResult loc_del_result;
std::mutex loc_del_reqs_mut;
std::condition_variable loc_del_reqs_cv; // condition being not empty of submitted_loc_del_reqs
std::vector<CacheLocationDelRequest> submitted_loc_del_reqs;

std::future<PlanExecuteResult> SchedulePlanExecutor_Submit_loc_stub(void *obj, const CacheLocationDelRequest &req) {
    const auto promise = std::make_shared<std::promise<PlanExecuteResult>>();
    {
        std::lock_guard<std::mutex> lock{loc_del_reqs_mut};
        submitted_loc_del_reqs.emplace_back(req);
        loc_del_reqs_cv.notify_one();
    }
    promise->set_value(loc_del_result);
    return promise->get_future();
}

} // namespace

class ReclaimerTaskSupervisorTest : public TESTBASE {
public:
    void SetUp() override {
        stub_.set(static_cast<spe_submit_blk>(ADDR(SchedulePlanExecutor, Submit)),
                  SchedulePlanExecutor_Submit_blk_stub);
        stub_.set(static_cast<spe_submit_loc>(ADDR(SchedulePlanExecutor, Submit)),
                  SchedulePlanExecutor_Submit_loc_stub);

        blk_del_result = {ErrorCode::EC_OK, ""};
        loc_del_result = {ErrorCode::EC_OK, ""};

        auto mr = std::make_shared<MetricsRegistry>();
        auto mim = std::make_shared<MetaIndexerManager>();
        auto dsm = std::make_shared<DataStorageManager>(mr);
        auto spe = std::make_shared<SchedulePlanExecutor>(0, mim, dsm, mr);
        task_supervisor_ = std::make_unique<ReclaimerTaskSupervisor>(spe);
        task_supervisor_->Start();
    }

    void TearDown() override {
        submitted_blk_del_reqs.clear();
        submitted_loc_del_reqs.clear();
    }

    Stub stub_;
    std::unique_ptr<ReclaimerTaskSupervisor> task_supervisor_;
};

TEST_F(ReclaimerTaskSupervisorTest, TestCacheMetaDelRequest) {
    CacheMetaDelRequest req;
    task_supervisor_->Submit("foo", std::move(req));
    {
        std::unique_lock<std::mutex> lock{blk_del_reqs_mut};
        blk_del_reqs_cv.wait_for(lock, std::chrono::milliseconds(8000));
        ASSERT_FALSE(submitted_blk_del_reqs.empty());
    }
}

TEST_F(ReclaimerTaskSupervisorTest, TestCacheLocDelRequest) {
    CacheLocationDelRequest req;
    task_supervisor_->Submit("foo", std::move(req));
    {
        std::unique_lock<std::mutex> lock{loc_del_reqs_mut};
        loc_del_reqs_cv.wait_for(lock, std::chrono::milliseconds(8000));
        ASSERT_FALSE(submitted_loc_del_reqs.empty());
    }
}
