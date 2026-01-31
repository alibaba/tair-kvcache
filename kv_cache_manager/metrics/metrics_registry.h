#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace kv_cache_manager {

#ifndef DEFINE_METRICS_
#define DEFINE_METRICS_

#define METRICS_NAME_(group, name) k_metrics_name_##group##_##name##_

#define METRICS_(group, name) group##_##name##_metrics_

#define SCOPED_METRICS_NAME_(scope, group, name) scope::METRICS_NAME_(group, name)

#define DECLARE_METRICS_NAME_(group, name) static const std::string METRICS_NAME_(group, name)

#define DEFINE_METRICS_NAME_(scope, group, name)                                                                       \
    const std::string SCOPED_METRICS_NAME_(scope, group, name) { #group "." #name }

#define DECLARE_METRICS_COUNTER_(group, name) Counter METRICS_(group, name)

#define DECLARE_METRICS_GAUGE_(group, name) Gauge METRICS_(group, name)

#define REGISTER_METRICS_COUNTER_(metrics_registry, group, name)                                                       \
    do {                                                                                                               \
        METRICS_(group, name) = (metrics_registry)->GetCounter(METRICS_NAME_(group, name));                            \
    } while (0)

#define REGISTER_METRICS_W_TAGS_COUNTER_(metrics_registry, group, name, tags)                                          \
    do {                                                                                                               \
        METRICS_(group, name) = (metrics_registry)->GetCounter(METRICS_NAME_(group, name), tags);                      \
    } while (0)

#define REGISTER_METRICS_GAUGE_(metrics_registry, group, name)                                                         \
    do {                                                                                                               \
        METRICS_(group, name) = (metrics_registry)->GetGauge(METRICS_NAME_(group, name));                              \
    } while (0)

#define REGISTER_METRICS_W_TAGS_GAUGE_(metrics_registry, group, name, tags)                                            \
    do {                                                                                                               \
        METRICS_(group, name) = (metrics_registry)->GetGauge(METRICS_NAME_(group, name), tags);                        \
    } while (0)

#define DEFINE_COPY_METRICS_COUNTER_(group, name)                                                                      \
    void copy_##group##_##name##_metrics(Counter &v) { v = METRICS_(group, name); }

#define DEFINE_COPY_METRICS_GAUGE_(group, name)                                                                        \
    void copy_##group##_##name##_metrics(Gauge &v) { v = METRICS_(group, name); }

#define DEFINE_SET_METRICS_COUNTER_(group, name)                                                                       \
    void set_##group##_##name##_metrics(std::uint64_t) {}

#define DEFINE_SET_METRICS_GAUGE_(group, name)                                                                         \
    void set_##group##_##name##_metrics(double v) { METRICS_(group, name) = v; }

#define DEFINE_GET_METRICS_COUNTER_(group, name)                                                                       \
    std::uint64_t get_##group##_##name##_metrics() const { return METRICS_(group, name).Get(); }

#define DEFINE_GET_METRICS_GAUGE_(group, name)                                                                         \
    double get_##group##_##name##_metrics() const { return METRICS_(group, name).Get(); }

#define DEFINE_STEAL_METRICS_GAUGE_(group, name)                                                                       \
    double steal_##group##_##name##_metrics() { return METRICS_(group, name).Steal(); }

#define COPY_METRICS_(ptr, group, name, value)                                                                         \
    do {                                                                                                               \
        (ptr)->copy_##group##_##name##_metrics(value);                                                                 \
    } while (0)

#define SET_METRICS_(ptr, group, name, value)                                                                          \
    do {                                                                                                               \
        (ptr)->set_##group##_##name##_metrics(value);                                                                  \
    } while (0)

#define GET_METRICS_(ptr, group, name, value)                                                                          \
    do {                                                                                                               \
        (value) = (ptr)->get_##group##_##name##_metrics();                                                             \
    } while (0)

#define STEAL_METRICS_(ptr, group, name, value)                                                                        \
    do {                                                                                                               \
        (value) = (ptr)->steal_##group##_##name##_metrics();                                                           \
    } while (0)

#endif

using CounterValue = std::atomic<std::uint64_t>;
using GaugeValue = std::atomic<double>;
using MetricsValue = std::variant<CounterValue, GaugeValue>;
using MetricsTags = std::map<std::string, std::string>;

class MetricsValueWrapper {
public:
    MetricsValueWrapper() = delete;
    virtual ~MetricsValueWrapper() = default;

    [[nodiscard]] std::shared_ptr<MetricsValue> GetRaw() const noexcept;

protected:
    explicit MetricsValueWrapper(std::shared_ptr<MetricsValue> v) noexcept;
    MetricsValueWrapper(const MetricsValueWrapper &) = default;
    MetricsValueWrapper(MetricsValueWrapper &&) = default;

    MetricsValueWrapper &operator=(const MetricsValueWrapper &) = default;
    MetricsValueWrapper &operator=(MetricsValueWrapper &&) = default;

    std::shared_ptr<MetricsValue> value_;
};

class Counter final : public MetricsValueWrapper {
public:
    Counter() noexcept;
    explicit Counter(std::shared_ptr<MetricsValue> v);
    Counter(const Counter &) = default;
    Counter(Counter &&) = default;
    ~Counter() override = default;

    Counter &operator=(const Counter &) = default;
    Counter &operator=(Counter &&) = default;

    [[nodiscard]] std::uint64_t Get() const;
    void Reset() const;

    // prefix
    Counter &operator++();
    Counter &operator--();

    // postfix
    Counter operator++(int);
    Counter operator--(int);

    Counter &operator+=(std::uint64_t v);
    Counter &operator-=(std::uint64_t v);
};

class Gauge final : public MetricsValueWrapper {
public:
    Gauge() noexcept;
    explicit Gauge(std::shared_ptr<MetricsValue> v);
    Gauge(const Gauge &) = default;
    Gauge(Gauge &&) = default;
    ~Gauge() override = default;

    Gauge &operator=(const Gauge &) = default;
    Gauge &operator=(Gauge &&) = default;

    [[nodiscard]] double Get() const;
    [[nodiscard]] double Steal();

    Gauge &operator=(double v);
    Gauge &operator+=(double v);
    Gauge &operator-=(double v);
};

class MetricsData {
public:
    MetricsData() = default;
    ~MetricsData() = default;
    MetricsData(const MetricsData &) = delete;
    MetricsData(MetricsData &&) = delete;
    MetricsData &operator=(const MetricsData &) = delete;
    MetricsData &operator=(MetricsData &&) = delete;

    std::size_t GetSize() noexcept;

    using metrics_pair_t = std::pair<MetricsTags, std::shared_ptr<MetricsValue>>;
    std::vector<metrics_pair_t> GetMetricsValues() noexcept;
    void GetMetricsValues(std::vector<metrics_pair_t> &out_metrics_values) noexcept;

    Counter GetOrCreateCounter(const MetricsTags &tags);
    Gauge GetOrCreateGauge(const MetricsTags &tags);

private:
    std::mutex mutex_;
    std::map<MetricsTags, std::shared_ptr<MetricsValue>> metrics_data_;
};

class MetricsRegistry {
public:
    MetricsRegistry() = default;
    ~MetricsRegistry() = default;
    MetricsRegistry(const MetricsRegistry &) = delete;
    MetricsRegistry(MetricsRegistry &&) = delete;
    MetricsRegistry &operator=(const MetricsRegistry &) = delete;
    MetricsRegistry &operator=(MetricsRegistry &&) = delete;

    std::size_t GetSize() noexcept;

    std::vector<std::string> GetNames() noexcept;
    void GetNames(std::vector<std::string> &out_names) noexcept;

    using metrics_tuple_t = std::tuple<std::string, MetricsTags, std::shared_ptr<MetricsValue>>;
    void GetAllMetrics(std::vector<metrics_tuple_t> &out_all_metrics) noexcept;

    Counter GetCounter(const std::string &name, const MetricsTags &tags = {});
    Gauge GetGauge(const std::string &name, const MetricsTags &tags = {});

    [[nodiscard]] std::shared_ptr<MetricsData> GetMetricsData(const std::string &name) noexcept;
    [[nodiscard]] std::shared_ptr<MetricsData> GetOrCreateMetricsData(const std::string &name) noexcept;

private:
    std::mutex mutex_;
    std::map<std::string, std::shared_ptr<MetricsData>> metrics_data_map_;
};

} // namespace kv_cache_manager
