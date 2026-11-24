#pragma once

#include "kv_cache_manager/manager/schedule_plan_executor.h"

namespace kv_cache_manager {

using ReclaimFreeLocations = std::vector<std::pair<int64_t, CacheLocationConstPtr>>;

void RunPaceReclaimDelete(SchedulePlanExecutor &executor,
                          const std::shared_ptr<DataStorageManager> &storage,
                          const std::shared_ptr<MetaIndexerManager> &meta,
                          const std::shared_ptr<MetaIndexer> &indexer,
                          const CacheLocationDelRequest &request,
                          const ReclaimFreeLocations &locations,
                          PlanExecuteResult result,
                          std::function<void(PlanExecuteResult)> complete);

} // namespace kv_cache_manager
