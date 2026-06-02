#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "kv_cache_manager/optimizer/config/types.h"

namespace kv_cache_manager {

enum class TierFlowEventKind {
    ENTER_TIER = 0,
    LEAVE_TIER = 1,
    READ_TOUCH = 2,
    WRITE_TOUCH = 3,
    FINAL_EVICT = 4,
};

enum class TierFlowEventReason {
    UNKNOWN = 0,
    WRITE = 1,
    WRITE_THROUGH = 2,
    WRITE_PROPAGATION = 3,
    WRITE_THROUGH_SELECTIVE = 4,
    CASCADING_DEMOTE = 5,
    PROMOTE = 6,
    READ = 7,
    CAPACITY_EVICTION = 8,
};

struct TierFlowEvent {
    TierFlowEventKind kind = TierFlowEventKind::ENTER_TIER;
    TierFlowEventReason reason = TierFlowEventReason::UNKNOWN;
    BlockEntry *block = nullptr;
    std::string instance_id;
    std::string from_tier;
    std::string to_tier;
    int64_t timestamp_ns = 0;
};

class TierFlowRecorder {
public:
    void Clear() { events_.clear(); }
    [[nodiscard]] bool empty() const { return events_.empty(); }
    [[nodiscard]] const std::vector<TierFlowEvent> &events() const { return events_; }

    void RecordEnter(const std::string &instance_id,
                     BlockEntry *block,
                     const std::string &from_tier,
                     const std::string &to_tier,
                     TierFlowEventReason reason,
                     int64_t timestamp_ns) {
        Record(TierFlowEventKind::ENTER_TIER, instance_id, block, from_tier, to_tier, reason, timestamp_ns);
    }

    void RecordLeave(const std::string &instance_id,
                     BlockEntry *block,
                     const std::string &from_tier,
                     TierFlowEventReason reason,
                     int64_t timestamp_ns) {
        Record(TierFlowEventKind::LEAVE_TIER, instance_id, block, from_tier, "", reason, timestamp_ns);
    }

    void RecordReadTouch(const std::string &instance_id,
                         BlockEntry *block,
                         const std::string &tier,
                         TierFlowEventReason reason,
                         int64_t timestamp_ns) {
        Record(TierFlowEventKind::READ_TOUCH, instance_id, block, tier, tier, reason, timestamp_ns);
    }

    void RecordWriteTouch(const std::string &instance_id,
                          BlockEntry *block,
                          const std::string &tier,
                          TierFlowEventReason reason,
                          int64_t timestamp_ns) {
        Record(TierFlowEventKind::WRITE_TOUCH, instance_id, block, tier, tier, reason, timestamp_ns);
    }

    void RecordFinalEvict(const std::string &instance_id,
                          BlockEntry *block,
                          TierFlowEventReason reason,
                          int64_t timestamp_ns) {
        Record(TierFlowEventKind::FINAL_EVICT, instance_id, block, "", "", reason, timestamp_ns);
    }

    [[nodiscard]] std::vector<BlockEntry *>
    BlocksForTier(const std::string &instance_id,
                  const std::string &tier,
                  const std::vector<TierFlowEventKind> &kinds,
                  const std::vector<TierFlowEventReason> &excluded_reasons = {}) const {
        std::vector<BlockEntry *> blocks;
        std::unordered_set<BlockEntry *> seen;
        for (const auto &event : events_) {
            if (event.instance_id != instance_id || event.block == nullptr || !MatchesKind(event.kind, kinds) ||
                MatchesReason(event.reason, excluded_reasons)) {
                continue;
            }
            const std::string &event_tier =
                event.kind == TierFlowEventKind::ENTER_TIER ? event.to_tier : event.from_tier;
            if (event_tier == tier && seen.insert(event.block).second) {
                blocks.push_back(event.block);
            }
        }
        return blocks;
    }

    [[nodiscard]] std::vector<BlockEntry *> FinalEvictedBlocks(const std::string &instance_id) const {
        std::vector<BlockEntry *> blocks;
        std::unordered_set<BlockEntry *> seen;
        for (const auto &event : events_) {
            if (event.kind == TierFlowEventKind::FINAL_EVICT && event.instance_id == instance_id &&
                event.block != nullptr && seen.insert(event.block).second) {
                blocks.push_back(event.block);
            }
        }
        return blocks;
    }

    void MergeFrom(const TierFlowRecorder &other) {
        events_.insert(events_.end(), other.events_.begin(), other.events_.end());
    }

private:
    static bool MatchesKind(TierFlowEventKind kind, const std::vector<TierFlowEventKind> &allowed) {
        for (const auto allowed_kind : allowed) {
            if (kind == allowed_kind) {
                return true;
            }
        }
        return false;
    }

    static bool MatchesReason(TierFlowEventReason reason, const std::vector<TierFlowEventReason> &reasons) {
        for (const auto excluded_reason : reasons) {
            if (reason == excluded_reason) {
                return true;
            }
        }
        return false;
    }

    void Record(TierFlowEventKind kind,
                const std::string &instance_id,
                BlockEntry *block,
                const std::string &from_tier,
                const std::string &to_tier,
                TierFlowEventReason reason,
                int64_t timestamp_ns) {
        if (block == nullptr) {
            return;
        }
        events_.push_back(TierFlowEvent{kind, reason, block, instance_id, from_tier, to_tier, timestamp_ns});
    }

    std::vector<TierFlowEvent> events_;
};

} // namespace kv_cache_manager
