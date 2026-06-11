#include "kv_cache_manager/metrics/metrics_registry.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <variant>

namespace kv_cache_manager {

std::shared_ptr<MetricsValue> MakeCounterValue() {
    return std::make_shared<MetricsValue>(std::in_place_type<CounterValue>, 0);
}

std::shared_ptr<MetricsValue> MakeGaugeValue() {
    return std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, 0.);
}

/* ---------------------------- Wrapper ----------------------------- */

MetricsValueWrapper::MetricsValueWrapper(std::shared_ptr<MetricsValue> v) noexcept : value_(std::move(v)) {}

std::shared_ptr<MetricsValue> MetricsValueWrapper::GetRaw() const noexcept { return value_; }

void MetricsValueWrapper::MarkTouched() noexcept {
    if (value_ != nullptr) {
        value_->touched.store(true, std::memory_order_relaxed);
    }
}

/* ---------------------------- Counter ----------------------------- */

Counter::Counter() noexcept : MetricsValueWrapper(nullptr) {}

Counter::Counter(std::shared_ptr<MetricsValue> v) : MetricsValueWrapper(std::move(v)) {
    if (!std::holds_alternative<CounterValue>(value_->value)) {
        value_.reset();
        throw std::runtime_error{"mismatched metrics value type (expecting CounterValue)"};
    }
}

std::uint64_t Counter::Get() const {
    if (value_ == nullptr) {
        return 0;
    }
    return std::get<CounterValue>(value_->value).load(std::memory_order_acquire);
}

void Counter::Reset() const {
    if (value_ == nullptr) {
        return;
    }
    // Reset is intentionally not marking the value as touched: a
    // counter that was touched then explicitly reset to zero is
    // still a deliberately-observable series and should keep
    // appearing in /metrics.
    std::get<CounterValue>(value_->value).store(0, std::memory_order_release);
}

Counter &Counter::operator++() {
    if (value_ == nullptr) {
        return *this;
    }
    std::get<CounterValue>(value_->value).fetch_add(1, std::memory_order_acq_rel);
    MarkTouched();
    return *this;
}

Counter &Counter::operator--() {
    if (value_ == nullptr) {
        return *this;
    }
    std::get<CounterValue>(value_->value).fetch_sub(1, std::memory_order_acq_rel);
    MarkTouched();
    return *this;
}

Counter Counter::operator++(int) {
    Counter ret = *this;
    ++*this;
    return ret;
}

Counter Counter::operator--(int) {
    Counter ret = *this;
    --*this;
    return ret;
}

Counter &Counter::operator+=(const std::uint64_t v) {
    if (value_ == nullptr) {
        return *this;
    }
    std::get<CounterValue>(value_->value).fetch_add(v, std::memory_order_acq_rel);
    MarkTouched();
    return *this;
}

Counter &Counter::operator-=(const std::uint64_t v) {
    if (value_ == nullptr) {
        return *this;
    }
    std::get<CounterValue>(value_->value).fetch_sub(v, std::memory_order_acq_rel);
    MarkTouched();
    return *this;
}

/* ----------------------------- Gauge ------------------------------ */

Gauge::Gauge() noexcept : MetricsValueWrapper(nullptr) {}

Gauge::Gauge(std::shared_ptr<MetricsValue> v) : MetricsValueWrapper(std::move(v)) {
    if (!std::holds_alternative<GaugeValue>(value_->value)) {
        value_.reset();
        throw std::runtime_error{"mismatched metrics value type (expecting GaugeValue)"};
    }
}

double Gauge::Get() const {
    if (value_ == nullptr) {
        return 0.;
    }
    return std::get<GaugeValue>(value_->value).load();
}

double Gauge::Steal() {
    // Steal is a read-and-invalidate — the gauge was already
    // touched by the original writer.
    if (value_ == nullptr) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double old_v;
    double invalid_v = std::numeric_limits<double>::quiet_NaN();
    do {
        old_v = std::get<GaugeValue>(value_->value).load(std::memory_order_relaxed);
    } while (!std::get<GaugeValue>(value_->value).compare_exchange_weak(old_v, invalid_v, std::memory_order_relaxed));
    return old_v;
}

Gauge &Gauge::operator=(const double v) {
    if (value_ == nullptr) {
        return *this;
    }
    std::get<GaugeValue>(value_->value).store(v);
    MarkTouched();
    return *this;
}

Gauge &Gauge::operator+=(const double v) {
    if (value_ == nullptr) {
        return *this;
    }
    double old_v, new_v;
    do {
        old_v = std::get<GaugeValue>(value_->value).load();
        new_v = old_v + v;
    } while (!std::get<GaugeValue>(value_->value).compare_exchange_weak(old_v, new_v));
    MarkTouched();
    return *this;
}

Gauge &Gauge::operator-=(const double v) {
    if (value_ == nullptr) {
        return *this;
    }
    double old_v, new_v;
    do {
        old_v = std::get<GaugeValue>(value_->value).load();
        new_v = old_v - v;
    } while (!std::get<GaugeValue>(value_->value).compare_exchange_weak(old_v, new_v));
    MarkTouched();
    return *this;
}

/* -------------------------- MetricsData --------------------------- */

std::size_t MetricsData::GetSize() noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    return metrics_data_.size();
}

std::vector<MetricsData::metrics_pair_t> MetricsData::GetMetricsValues() noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<metrics_pair_t> data;
    data.reserve(metrics_data_.size());
    for (const auto &[tags, val] : metrics_data_) {
        data.emplace_back(tags, val);
    }
    return data;
}

void MetricsData::GetMetricsValues(std::vector<metrics_pair_t> &out_metrics_values) noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    out_metrics_values.clear();
    out_metrics_values.reserve(metrics_data_.size());
    for (const auto &[tags, val] : metrics_data_) {
        out_metrics_values.emplace_back(tags, val);
    }
}

Counter MetricsData::GetOrCreateCounter(const MetricsTags &tags) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = metrics_data_.find(tags);
    if (it == metrics_data_.end()) {
        it = metrics_data_.emplace(tags, MakeCounterValue()).first;
    } else if (it->second == nullptr) {
        it->second = MakeCounterValue();
    } else if (!std::holds_alternative<CounterValue>(it->second->value)) {
        throw std::runtime_error{"a metrics value holds a type other than CounterValue already exists"};
    }
    return Counter{it->second};
}

Gauge MetricsData::GetOrCreateGauge(const MetricsTags &tags) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = metrics_data_.find(tags);
    if (it == metrics_data_.end()) {
        it = metrics_data_.emplace(tags, MakeGaugeValue()).first;
    } else if (it->second == nullptr) {
        it->second = MakeGaugeValue();
    } else if (!std::holds_alternative<GaugeValue>(it->second->value)) {
        throw std::runtime_error{"a metrics value holds a type other than GaugeValue already exists"};
    }
    return Gauge{it->second};
}

namespace {

// returns true if tags contains all key-value pairs in filter
bool TagsMatch(const MetricsTags &tags, const MetricsTags &filter) {
    return std::all_of(filter.begin(), filter.end(), [&tags](const auto &kv) {
        auto it = tags.find(kv.first);
        return it != tags.end() && it->second == kv.second;
    });
}

} // anonymous namespace

bool MetricsData::Remove(const MetricsTags &tags) {
    std::lock_guard<std::mutex> guard(mutex_);
    return metrics_data_.erase(tags) > 0;
}

std::size_t MetricsData::RemoveByTagFilter(const MetricsTags &filter) {
    std::lock_guard<std::mutex> guard(mutex_);
    std::size_t removed = 0;
    for (auto it = metrics_data_.begin(); it != metrics_data_.end();) {
        if (TagsMatch(it->first, filter)) {
            it = metrics_data_.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }
    return removed;
}

std::optional<Gauge> MetricsData::GetGauge(const MetricsTags &tags) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = metrics_data_.find(tags);
    if (it == metrics_data_.end() || it->second == nullptr ||
        !std::holds_alternative<GaugeValue>(it->second->value)) {
        return std::nullopt;
    }
    return Gauge{it->second};
}

bool MetricsData::RemoveByTags(const MetricsTags &tags) {
    std::lock_guard<std::mutex> guard(mutex_);
    return metrics_data_.erase(tags) > 0;
}

/* ---------------------------- Registry ---------------------------- */

std::size_t MetricsRegistry::GetSize() noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    std::size_t size = 0;
    for (const auto &[name, data] : metrics_data_map_) {
        size += data->GetSize();
    }
    return size;
}

std::vector<std::string> MetricsRegistry::GetNames() noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    std::vector<std::string> names;
    names.reserve(metrics_data_map_.size());
    for (const auto &[name, _] : metrics_data_map_) {
        names.emplace_back(name);
    }
    return names;
}

void MetricsRegistry::GetNames(std::vector<std::string> &out_names) noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    out_names.clear();
    out_names.reserve(metrics_data_map_.size());
    for (const auto &[name, _] : metrics_data_map_) {
        out_names.emplace_back(name);
    }
}

void MetricsRegistry::GetAllMetrics(std::vector<metrics_tuple_t> &out_all_metrics) noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    out_all_metrics.clear();
    for (const auto &[name, data] : metrics_data_map_) {
        if (data != nullptr) {
            std::vector<MetricsData::metrics_pair_t> metrics_values;
            data->GetMetricsValues(metrics_values);
            for (const auto &[tags, val] : metrics_values) {
                out_all_metrics.emplace_back(name, tags, val);
            }
        }
    }
}

std::shared_ptr<MetricsData> MetricsRegistry::GetMetricsData(const std::string &name) noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto it = metrics_data_map_.find(name);
    if (it == metrics_data_map_.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<MetricsData> MetricsRegistry::GetOrCreateMetricsData(const std::string &name) noexcept {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = metrics_data_map_.find(name);
    if (it == metrics_data_map_.end()) {
        it = metrics_data_map_.emplace(name, std::make_shared<MetricsData>()).first;
    }
    return it->second;
}

bool MetricsRegistry::Remove(const std::string &name, const MetricsTags &tags) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto it = metrics_data_map_.find(name);
    if (it == metrics_data_map_.end() || it->second == nullptr) {
        return false;
    }
    return it->second->Remove(tags);
}

std::size_t MetricsRegistry::RemoveByTagFilter(const MetricsTags &filter) {
    std::lock_guard<std::mutex> guard(mutex_);
    std::size_t total = 0;
    for (auto &[name, data] : metrics_data_map_) {
        if (data != nullptr) {
            total += data->RemoveByTagFilter(filter);
        }
    }
    return total;
}

Counter MetricsRegistry::GetCounter(const std::string &name, const MetricsTags &tags) {
    const auto metrics_data = GetOrCreateMetricsData(name);
    assert(metrics_data);
    return metrics_data->GetOrCreateCounter(tags);
}

Gauge MetricsRegistry::GetGauge(const std::string &name, const MetricsTags &tags) {
    const auto metrics_data = GetOrCreateMetricsData(name);
    assert(metrics_data);
    return metrics_data->GetOrCreateGauge(tags);
}

void MetricsRegistry::RegisterHistogramFamily(const std::string &family_name) {
    std::lock_guard<std::mutex> guard(mutex_);
    histogram_families_.insert(family_name);
}

void MetricsRegistry::MapMetricToFamily(const std::string &metric_name, const std::string &family_name) {
    std::lock_guard<std::mutex> guard(mutex_);
    metric_to_family_.emplace(metric_name, family_name);
}

std::string MetricsRegistry::GetMetricFamily(const std::string &metric_name) const {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = metric_to_family_.find(metric_name);
    if (it == metric_to_family_.end()) {
        return {};
    }
    return it->second;
}

std::set<std::string> MetricsRegistry::GetHistogramFamilies() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return histogram_families_;
}

} // namespace kv_cache_manager
