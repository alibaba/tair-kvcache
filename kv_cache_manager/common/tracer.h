#pragma once

#include <stdio.h>
#include <string>
#include <string_view>
#include <vector>

#include "jsonizable.h"
#include "timestamp_util.h"

namespace kv_cache_manager {

class QueryTracer {};

class ErrorTracer : public Jsonizable {
public:
    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "error_msg", error_msgs_);
    }
    bool FromRapidValue(const rapidjson::Value &rapid_value) override { return false; }
    void AddErrorMsg(std::string_view error_msg) { error_msgs_.emplace_back(error_msg); }

private:
    std::vector<std::string> error_msgs_;
};

class SpanTracer : public Jsonizable {
public:
    SpanTracer(SpanTracer *parent, std::string_view file, std::string_view func)
        : start_(TimestampUtil::GetSteadyTimeUs()), parent_(parent), file_(file), func_(func) {}
    SpanTracer(int64_t span, std::string_view file, std::string_view func, std::vector<SpanTracer> &&childs)
        : span_(span), file_(file), func_(func), childs_(std::move(childs)) {}
    ~SpanTracer() { End(); }

    std::string EndAndGetTracerStr() {
        End();
        return ToJsonString();
    }

    void ToRapidWriter(rapidjson::Writer<rapidjson::StringBuffer> &writer) const noexcept override {
        Put(writer, "span_time_us", span_);
        std::string key(file_);
        key += ":";
        key += func_;
        Put(writer, key, childs_);
    }
    bool FromRapidValue(const rapidjson::Value &rapid_value) override { return false; }

    void Add(SpanTracer *tracer) {
        childs_.emplace_back(tracer->span_, tracer->file_, tracer->func_, std::move(tracer->childs_));
    }

private:
    void End() {
        span_ = TimestampUtil::GetSteadyTimeUs() - start_;
        if (parent_) {
            parent_->Add(this);
            parent_ = nullptr;
        }
    }

    int64_t start_ = 0;
    int64_t span_ = 0;
    SpanTracer *parent_ = nullptr;
    std::string_view file_;
    std::string_view func_;
    std::vector<SpanTracer> childs_;
};

} // namespace kv_cache_manager