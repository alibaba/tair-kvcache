#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kv_cache_manager/event/base_event.h"

namespace kv_cache_manager {

class OptimizerQueryHitEvent : public BaseEvent {
public:
    class CapacityResult : public Jsonizable {
    public:
        CapacityResult(double capacity_gb, int64_t cache_hit_count, double hit_rate, int64_t current_unique_keys)
            : capacity_gb_(capacity_gb)
            , cache_hit_count_(cache_hit_count)
            , hit_rate_(hit_rate)
            , current_unique_keys_(current_unique_keys) {}

        bool FromRapidValue(const rapidjson::Value &) override { return false; }

        void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
            Put(writer, "capacity_gb", capacity_gb_);
            Put(writer, "cache_hit_count", cache_hit_count_);
            Put(writer, "hit_rate", hit_rate_);
            Put(writer, "current_unique_keys", current_unique_keys_);
        }

        double capacity_gb() const { return capacity_gb_; }
        int64_t cache_hit_count() const { return cache_hit_count_; }
        double hit_rate() const { return hit_rate_; }
        int64_t current_unique_keys() const { return current_unique_keys_; }

    private:
        double capacity_gb_{0};
        int64_t cache_hit_count_{0};
        double hit_rate_{0};
        int64_t current_unique_keys_{0};
    };

    class TheoreticalResult : public Jsonizable {
    public:
        bool FromRapidValue(const rapidjson::Value &) override { return false; }

        void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
            Put(writer, "max_hit_count", max_hit_count_);
            Put(writer, "hit_rate", hit_rate_);
            Put(writer, "current_unique_keys", current_unique_keys_);
        }

        void Set(int64_t max_hit_count, double hit_rate, int64_t current_unique_keys) {
            max_hit_count_ = max_hit_count;
            hit_rate_ = hit_rate;
            current_unique_keys_ = current_unique_keys;
        }

        int64_t max_hit_count() const { return max_hit_count_; }
        double hit_rate() const { return hit_rate_; }
        int64_t current_unique_keys() const { return current_unique_keys_; }

    private:
        int64_t max_hit_count_{0};
        double hit_rate_{0};
        int64_t current_unique_keys_{0};
    };

    explicit OptimizerQueryHitEvent(const std::string &source)
        : BaseEvent(source, "optimizer", "OptimizerQueryHitEvent") {}

    void SetAdditionalArgs(const std::string &trace_id,
                           int64_t request_timestamp_ns,
                           int64_t input_token_len,
                           int64_t total_blocks) {
        trace_id_ = trace_id;
        request_timestamp_ns_ = request_timestamp_ns;
        input_token_len_ = input_token_len;
        total_blocks_ = total_blocks;
    }

    void AddCapacityResult(double capacity_gb, int64_t cache_hit_count, double hit_rate, int64_t current_unique_keys) {
        capacity_results_.emplace_back(capacity_gb, cache_hit_count, hit_rate, current_unique_keys);
    }

    void SetTheoreticalResult(int64_t max_hit_count, double hit_rate, int64_t current_unique_keys) {
        theoretical_result_.Set(max_hit_count, hit_rate, current_unique_keys);
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        BaseEvent::ToRapidWriter(writer);
        Put(writer, "trace_id", trace_id_);
        Put(writer, "request_timestamp_ns", request_timestamp_ns_);
        Put(writer, "input_token_len", input_token_len_);
        Put(writer, "total_blocks", total_blocks_);
        Put(writer, "capacity_results", capacity_results_);
        Put(writer, "theoretical_result", theoretical_result_);
    }

    const std::string &trace_id() const { return trace_id_; }
    int64_t request_timestamp_ns() const { return request_timestamp_ns_; }
    int64_t input_token_len() const { return input_token_len_; }
    int64_t total_blocks() const { return total_blocks_; }
    const std::vector<CapacityResult> &capacity_results() const { return capacity_results_; }
    const TheoreticalResult &theoretical_result() const { return theoretical_result_; }

private:
    std::string trace_id_;
    int64_t request_timestamp_ns_{0};
    int64_t input_token_len_{0};
    int64_t total_blocks_{0};
    std::vector<CapacityResult> capacity_results_;
    TheoreticalResult theoretical_result_;
};

} // namespace kv_cache_manager
