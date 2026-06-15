#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>

#include "kv_cache_manager/common/unittest.h"
#include "kv_cache_manager/metrics/metrics_registry.h"

using namespace kv_cache_manager;

bool VecContains(const std::vector<std::string> &vec, const std::string &v) {
    return std::any_of(vec.cbegin(), vec.cend(), [v](const std::string &e) { return e == v; });
}

bool VecContains(const std::vector<MetricsData::metrics_pair_t> &vec, const MetricsData::metrics_pair_t &v) {
    return std::any_of(vec.cbegin(), vec.cend(), [v](const MetricsData::metrics_pair_t &e) { return e == v; });
}

bool AlmostEqual(const double a, const double b, const double epsilon = 1e-9) { return std::abs(a - b) < epsilon; }

bool VecContains(const std::vector<MetricsRegistry::metrics_tuple_t> &vec, const MetricsRegistry::metrics_tuple_t &v) {
    return std::any_of(vec.cbegin(), vec.cend(), [v](const MetricsRegistry::metrics_tuple_t &e) {
        const auto &[name_e, tags_e, val_e] = e;
        const auto &[name_v, tags_v, val_v] = v;

        if (name_e != name_v || tags_e != tags_v || val_e->value.index() != val_v->value.index()) {
            return false;
        }

        if (std::holds_alternative<CounterValue>(val_e->value)) {
            return std::get<CounterValue>(val_e->value).load() == std::get<CounterValue>(val_v->value).load();
        }

        if (std::holds_alternative<GaugeValue>(val_e->value)) {
            return AlmostEqual(std::get<GaugeValue>(val_e->value).load(), std::get<GaugeValue>(val_v->value).load());
        }

        return false;
    });
}

class MetricsRegistryTest : public TESTBASE {
public:
    void SetUp() override { registry_ = std::make_shared<MetricsRegistry>(); }

    void TearDown() override {}

    std::shared_ptr<MetricsRegistry> registry_;
};

TEST_F(MetricsRegistryTest, TestCounterCtor) {
    // default ctor
    Counter a, b;
    ASSERT_EQ(nullptr, a.GetRaw());
    ASSERT_EQ(nullptr, b.GetRaw());

    // explicit ctor
    Counter c{std::make_shared<MetricsValue>(std::in_place_type<CounterValue>, 0)};
    Counter d{c};            // copy ctor
    Counter e{std::move(d)}; // move ctor
    a = c;                   // lvalue assignment
    b = std::move(e);        // rvalue assignment

    ++c;
    ASSERT_EQ(1, a.Get());
    ASSERT_EQ(1, b.Get());
    ASSERT_EQ(1, c.Get());

    ASSERT_THROW(Counter{std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, 0.)}, std::runtime_error);
}

TEST_F(MetricsRegistryTest, TestCounterNullptr) {
    Counter a{};

    ASSERT_EQ(0, a.Get());

    ASSERT_NO_FATAL_FAILURE(a.Reset());

    ASSERT_EQ(0, (++a).Get());
    ASSERT_EQ(0, (a++).Get());
    ASSERT_EQ(0, (a--).Get());
    ASSERT_EQ(0, (--a).Get());
    ASSERT_EQ(0, (a += 1).Get());
    ASSERT_EQ(0, (a -= 1).Get());
}

TEST_F(MetricsRegistryTest, TestCounterOperate) {
    Counter a{std::make_shared<MetricsValue>(std::in_place_type<CounterValue>, 0)};
    const auto orig_use_count = a.GetRaw().use_count();

    {
        // Counter holds pointer, so the value would be updated anyway
        // even under postfix op
        ASSERT_EQ(1, (a++).Get());
        ASSERT_EQ(2, (++a).Get());

        ASSERT_EQ(4, (++(++a)).Get());
        ASSERT_EQ(6, ((++a)++).Get());
        ASSERT_EQ(8, (++(a++)).Get());
        ASSERT_EQ(10, ((a++)++).Get());

        ASSERT_EQ(12, (++ ++a).Get());
        ASSERT_EQ(14, (++a++).Get());
        ASSERT_EQ(16, (a++ ++).Get());
    }

    {
        // b holds a copy of the underlying shared_ptr
        // this is the only subtle difference for the prefix and postfix
        // ops for the Counter type
        auto b = a++;
        ASSERT_EQ(17, b.Get());
        ASSERT_EQ(orig_use_count + 1, a.GetRaw().use_count());
    }

    {
        auto &c = ++a;
        ASSERT_EQ(18, c.Get());
        ASSERT_EQ(orig_use_count, a.GetRaw().use_count());
    }

    {
        ASSERT_EQ(18, (--(++a)).Get());
        ASSERT_EQ(18, ((++a)--).Get());
        ASSERT_EQ(18, (--(a++)).Get());
        ASSERT_EQ(18, ((a++)--).Get());

        ASSERT_EQ(18, (++(--a)).Get());
        ASSERT_EQ(18, ((--a)++).Get());
        ASSERT_EQ(18, (++(a--)).Get());
        ASSERT_EQ(18, ((a--)++).Get());
    }

    {
        ASSERT_EQ(17, (a--).Get());
        ASSERT_EQ(16, (--a).Get());

        ASSERT_EQ(14, (--(--a)).Get());
        ASSERT_EQ(12, ((--a)--).Get());
        ASSERT_EQ(10, (--(a--)).Get());
        ASSERT_EQ(8, ((a--)--).Get());

        ASSERT_EQ(6, (-- --a).Get());
        ASSERT_EQ(4, (--a--).Get());
        ASSERT_EQ(2, (a-- --).Get());
    }

    {
        auto b = a--;
        ASSERT_EQ(1, b.Get());
        ASSERT_EQ(orig_use_count + 1, a.GetRaw().use_count());
    }

    {
        auto &c = --a;
        ASSERT_EQ(0, c.Get());
        ASSERT_EQ(orig_use_count, a.GetRaw().use_count());
    }

    ASSERT_EQ(16, ((a += 64) -= 48).Get());

    a++.Reset();
    ASSERT_EQ(0, a.Get());

    {
        // force to point to an invalid value
        a.value_ = std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, 0.);

        // the ops should throw
        ASSERT_THROW(a.Get(), std::bad_variant_access);
        ASSERT_THROW(a.Reset(), std::bad_variant_access);
        ASSERT_THROW(++a, std::bad_variant_access);
        ASSERT_THROW(--a, std::bad_variant_access);
        ASSERT_THROW(a++, std::bad_variant_access);
        ASSERT_THROW(a--, std::bad_variant_access);
        ASSERT_THROW(a += 1, std::bad_variant_access);
        ASSERT_THROW(a -= 1, std::bad_variant_access);
    }
}

TEST_F(MetricsRegistryTest, TestGaugeCtor) {
    // default ctor
    Gauge a, b;
    ASSERT_EQ(nullptr, a.GetRaw());
    ASSERT_EQ(nullptr, b.GetRaw());

    // explicit ctor
    Gauge c{std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, 0.)};
    Gauge d{c};            // copy ctor
    Gauge e{std::move(d)}; // move ctor
    a = c;                 // lvalue assignment
    b = std::move(e);      // rvalue assignment

    c = 1.;
    ASSERT_EQ(1., a.Get());
    ASSERT_EQ(1., b.Get());
    ASSERT_EQ(1., c.Get());

    ASSERT_THROW(Gauge{std::make_shared<MetricsValue>(std::in_place_type<CounterValue>, 0)}, std::runtime_error);
}

TEST_F(MetricsRegistryTest, TestGaugeNullptr) {
    Gauge a{};

    ASSERT_DOUBLE_EQ(0., a.Get());

    ASSERT_DOUBLE_EQ(0., (a = 8.0).Get());
    ASSERT_DOUBLE_EQ(0., (a += 1.0).Get());
    ASSERT_DOUBLE_EQ(0., (a -= 1.0).Get());
}

TEST_F(MetricsRegistryTest, TestGaugeOperate) {
    Gauge a{std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, 0.)};
    ASSERT_DOUBLE_EQ(8.0, (((a = 6.4) += 3.2) -= 1.6).Get());

    // force to point to an invalid value
    a.value_ = std::make_shared<MetricsValue>(std::in_place_type<CounterValue>, 0);

    // the ops should throw
    ASSERT_THROW(a.Get(), std::bad_variant_access);
    ASSERT_THROW(a = 1., std::bad_variant_access);
    ASSERT_THROW(a += 1., std::bad_variant_access);
    ASSERT_THROW(a -= 1., std::bad_variant_access);
}

TEST_F(MetricsRegistryTest, TestMetricsDataGetSize) {
    {
        // functionality
        auto md = std::make_shared<MetricsData>();
        ASSERT_EQ(0, md->GetSize());
        md->metrics_data_.emplace(MetricsTags{}, nullptr);
        ASSERT_EQ(1, md->GetSize());
        md->metrics_data_.clear();
        ASSERT_EQ(0, md->GetSize());
    }

    {
        // thread-safety and reentrance
        auto md = std::make_shared<MetricsData>();

        std::thread t([md]() -> void {
            while (md->GetSize() == 0) {}
            ASSERT_EQ(1, md->GetSize());
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            {
                std::lock_guard<std::mutex> guard(md->mutex_);
                md->metrics_data_.clear();
            }
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        {
            std::lock_guard<std::mutex> guard(md->mutex_);
            md->metrics_data_.emplace(MetricsTags{}, nullptr);
        }
        ASSERT_EQ(1, md->GetSize());
        while (md->GetSize() == 1) {}
        t.join();
        ASSERT_EQ(0, md->GetSize());
    }
}

TEST_F(MetricsRegistryTest, TestMetricsDataGetMetricsValues) {
    {
        // functionality
        auto md = std::make_shared<MetricsData>();
        std::vector<MetricsData::metrics_pair_t> vals;

        vals = md->GetMetricsValues();
        ASSERT_TRUE(vals.empty());
        md->GetMetricsValues(vals);
        ASSERT_TRUE(vals.empty());

        md->metrics_data_.emplace(MetricsTags{}, nullptr);
        vals = md->GetMetricsValues();
        ASSERT_FALSE(vals.empty());
        ASSERT_EQ(1, vals.size());
        ASSERT_TRUE(VecContains(vals, {MetricsTags{}, nullptr}));
        md->GetMetricsValues(vals);
        ASSERT_FALSE(vals.empty());
        ASSERT_EQ(1, vals.size());
        ASSERT_TRUE(VecContains(vals, {MetricsTags{}, nullptr}));

        md->metrics_data_.emplace(MetricsTags{{"foo", "bar"}}, nullptr);
        vals = md->GetMetricsValues();
        ASSERT_EQ(2, vals.size());
        ASSERT_TRUE(VecContains(vals, {MetricsTags{{"foo", "bar"}}, nullptr}));
        md->GetMetricsValues(vals);
        ASSERT_EQ(2, vals.size());
        ASSERT_TRUE(VecContains(vals, {MetricsTags{{"foo", "bar"}}, nullptr}));

        md->metrics_data_.clear();
        vals = md->GetMetricsValues();
        ASSERT_TRUE(vals.empty());
        md->GetMetricsValues(vals);
        ASSERT_TRUE(vals.empty());
    }

    {
        // thread-safety and reentrance
        auto md = std::make_shared<MetricsData>();

        std::thread t([md]() -> void {
            std::vector<MetricsData::metrics_pair_t> vals;
            while (vals = md->GetMetricsValues(), vals.empty()) {}
            ASSERT_EQ(1, vals.size());
            ASSERT_TRUE(VecContains(vals, {MetricsTags{{"foo", "bar"}}, nullptr}));
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            {
                std::lock_guard<std::mutex> guard(md->mutex_);
                md->metrics_data_.clear();
            }
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        {
            std::lock_guard<std::mutex> guard(md->mutex_);
            md->metrics_data_.emplace(MetricsTags{{"foo", "bar"}}, nullptr);
        }
        std::vector<MetricsData::metrics_pair_t> vals;
        while (md->GetMetricsValues(vals), !vals.empty()) {}
        t.join();
        vals = md->GetMetricsValues();
        ASSERT_TRUE(vals.empty());
        md->GetMetricsValues(vals);
        ASSERT_TRUE(vals.empty());
    }
}

TEST_F(MetricsRegistryTest, TestMetricsDataGetOrCreate) {
    {
        // functionality
        auto md = std::make_shared<MetricsData>();

        Counter c0 = md->GetOrCreateCounter(MetricsTags{{"foo", "bar"}});
        ASSERT_EQ(1, md->GetSize());
        Counter c1 = md->GetOrCreateCounter(MetricsTags{{"foo", "bar"}});
        ASSERT_EQ(1, md->GetSize());
        ASSERT_EQ(c0.GetRaw(), c1.GetRaw());

        Gauge g0 = md->GetOrCreateGauge(MetricsTags{{"foo2", "bar2"}});
        ASSERT_EQ(2, md->GetSize());
        Gauge g1 = md->GetOrCreateGauge(MetricsTags{{"foo2", "bar2"}});
        ASSERT_EQ(2, md->GetSize());
        ASSERT_EQ(g0.GetRaw(), g1.GetRaw());

        md->metrics_data_.emplace(MetricsTags{{"foo3", "bar3"}}, nullptr);
        Counter c2 = md->GetOrCreateCounter(MetricsTags{{"foo3", "bar3"}});
        ASSERT_EQ(3, md->GetSize());
        ASSERT_NE(nullptr, c2.GetRaw());

        md->metrics_data_.emplace(MetricsTags{{"foo4", "bar4"}}, nullptr);
        Gauge g2 = md->GetOrCreateGauge(MetricsTags{{"foo4", "bar4"}});
        ASSERT_EQ(4, md->GetSize());
        ASSERT_NE(nullptr, g2.GetRaw());

        ASSERT_THROW(md->GetOrCreateGauge(MetricsTags{{"foo", "bar"}}), std::runtime_error);
        ASSERT_THROW(md->GetOrCreateCounter(MetricsTags{{"foo2", "bar2"}}), std::runtime_error);
    }

    {
        // thread-safety and reentrance
        auto md = std::make_shared<MetricsData>();

        std::thread t([md]() -> void { Counter c = md->GetOrCreateCounter(MetricsTags{{"foo", "bar"}}); });

        Counter c = md->GetOrCreateCounter(MetricsTags{{"foo", "bar"}});
        t.join();
        ASSERT_EQ(1, md->GetSize());
    }
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryGetCounter) {
    auto c = registry_->GetCounter("foo.bar");
    ASSERT_TRUE(std::holds_alternative<CounterValue>(c.GetRaw()->value));
    ASSERT_EQ(0, c.Get());
    ++c;
    ASSERT_EQ(1, c.Get());
    c += 8;
    ASSERT_EQ(9, c.Get());
    c.Reset();
    ASSERT_EQ(0, c.Get());

    auto d = registry_->GetCounter("foo.bar");
    auto e = registry_->GetCounter("foo.bar2");
    ASSERT_EQ(0, d.Get());
    ASSERT_EQ(0, e.Get());

    ++c;
    ASSERT_EQ(1, c.Get());
    ASSERT_EQ(1, d.Get());
    ASSERT_EQ(0, e.Get());
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryGetCounterWithTags) {
    auto c = registry_->GetCounter("foo.bar", {{"foo", "bar"}});
    ASSERT_TRUE(std::holds_alternative<CounterValue>(c.GetRaw()->value));
    ASSERT_EQ(0, c.Get());
    ++c;
    ASSERT_EQ(1, c.Get());
    c += 8;
    ASSERT_EQ(9, c.Get());
    c.Reset();
    ASSERT_EQ(0, c.Get());

    auto d = registry_->GetCounter("foo.bar", {{"foo", "bar"}});
    auto e = registry_->GetCounter("foo.bar", {{"foo2", "bar2"}});
    auto f = registry_->GetCounter("foo.bar", {{"foo", "bar"}, {"foo2", "bar2"}});
    ASSERT_EQ(0, d.Get());
    ASSERT_EQ(0, e.Get());
    ASSERT_EQ(0, f.Get());

    ++c;
    ASSERT_EQ(1, c.Get());
    ASSERT_EQ(1, d.Get());
    ASSERT_EQ(0, e.Get());
    ASSERT_EQ(0, f.Get());
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryGetGauge) {
    auto c = registry_->GetGauge("foo.bar");
    ASSERT_TRUE(std::holds_alternative<GaugeValue>(c.GetRaw()->value));
    ASSERT_DOUBLE_EQ(0., c.Get());
    c += 8.;
    ASSERT_DOUBLE_EQ(8., c.Get());
    c -= 2.;
    ASSERT_DOUBLE_EQ(6., c.Get());
    c = 10.;
    ASSERT_DOUBLE_EQ(10., c.Get());

    auto d = registry_->GetGauge("foo.bar");
    auto e = registry_->GetGauge("foo.bar2");
    ASSERT_DOUBLE_EQ(10., d.Get());
    ASSERT_DOUBLE_EQ(0., e.Get());

    c += 128.;
    ASSERT_DOUBLE_EQ(138., c.Get());
    ASSERT_DOUBLE_EQ(138., d.Get());
    ASSERT_DOUBLE_EQ(0., e.Get());
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryGetGaugeWithTags) {
    auto c = registry_->GetGauge("foo.bar", {{"foo", "bar"}});
    ASSERT_TRUE(std::holds_alternative<GaugeValue>(c.GetRaw()->value));
    ASSERT_DOUBLE_EQ(0., c.Get());
    c += 8.;
    ASSERT_DOUBLE_EQ(8., c.Get());
    c -= 2.;
    ASSERT_DOUBLE_EQ(6., c.Get());
    c = 10.;
    ASSERT_DOUBLE_EQ(10., c.Get());

    auto d = registry_->GetGauge("foo.bar", {{"foo", "bar"}});
    auto e = registry_->GetGauge("foo.bar", {{"foo2", "bar2"}});
    auto f = registry_->GetGauge("foo.bar", {{"foo", "bar"}, {"foo2", "bar2"}});
    ASSERT_DOUBLE_EQ(10., d.Get());
    ASSERT_DOUBLE_EQ(0., e.Get());
    ASSERT_DOUBLE_EQ(0., f.Get());

    c += 128.;
    ASSERT_DOUBLE_EQ(138., c.Get());
    ASSERT_DOUBLE_EQ(138., d.Get());
    ASSERT_DOUBLE_EQ(0., e.Get());
    ASSERT_DOUBLE_EQ(0., f.Get());
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryAdaptiveGet) {
    // adaptive get should not be supported

    auto counter = registry_->GetCounter("foo.bar");
    ASSERT_TRUE(std::holds_alternative<CounterValue>(counter.GetRaw()->value));
    ASSERT_THROW(registry_->GetGauge("foo.bar"), std::runtime_error);

    auto gauge = registry_->GetGauge("foo2.bar2");
    ASSERT_TRUE(std::holds_alternative<GaugeValue>(gauge.GetRaw()->value));
    ASSERT_THROW(registry_->GetCounter("foo2.bar2"), std::runtime_error);
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryGetSize) {
    ASSERT_EQ(0, registry_->GetSize());

    auto a = registry_->GetCounter("foo.bar.counter", {{"foo", "bar"}});
    auto b = registry_->GetCounter("foo.bar.counter", {{"foo2", "bar2"}});

    ASSERT_EQ(2, registry_->GetSize());

    auto c = registry_->GetGauge("foo.bar.gauge", {{"foo", "bar"}});
    auto e = registry_->GetGauge("foo.bar.gauge", {{"foo2", "bar2"}});
    auto f = registry_->GetGauge("foo.bar.gauge", {{"foo", "bar"}, {"foo2", "bar2"}});

    ASSERT_EQ(5, registry_->GetSize());

    auto g = registry_->GetCounter("foo.bar.counter2", {{"foo", "bar"}});

    ASSERT_EQ(6, registry_->GetSize());
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryGetNames) {
    {
        ASSERT_TRUE(registry_->GetNames().empty());

        std::vector<std::string> names;
        registry_->GetNames(names);
        ASSERT_TRUE(names.empty());

        names.emplace_back("a");
        registry_->GetNames(names);
        ASSERT_TRUE(names.empty());
    }

    std::string name_counter = "foo.bar.counter";
    std::string name_gauge = "foo.bar.gauge";

    MetricsTags tags_a{{"foo", "bar"}};
    auto a = registry_->GetCounter(name_counter, tags_a);

    MetricsTags tags_b{{"foo2", "bar2"}};
    auto b = registry_->GetCounter(name_counter, tags_b);

    MetricsTags tags_c{{"foo", "bar"}};
    auto c = registry_->GetGauge(name_gauge, tags_c);

    MetricsTags tags_d{{"foo2", "bar2"}};
    auto d = registry_->GetGauge(name_gauge, tags_d);

    MetricsTags tags_e{{"foo", "bar"}, {"foo2", "bar2"}};
    auto e = registry_->GetGauge(name_gauge, tags_e);

    {
        std::vector<std::string> names = registry_->GetNames();
        ASSERT_EQ(2, names.size());
        ASSERT_TRUE(VecContains(names, name_counter));
        ASSERT_TRUE(VecContains(names, name_gauge));
    }

    {
        std::vector<std::string> names;
        registry_->GetNames(names);
        ASSERT_EQ(2, names.size());
        ASSERT_TRUE(VecContains(names, name_counter));
        ASSERT_TRUE(VecContains(names, name_gauge));
    }

    {
        std::vector<std::string> names;
        names.emplace_back("a");
        registry_->GetNames(names);
        ASSERT_EQ(2, names.size());
        ASSERT_TRUE(VecContains(names, name_counter));
        ASSERT_TRUE(VecContains(names, name_gauge));
    }
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryGetAllMetrics) {
    std::string name_counter = "foo.bar.counter";
    std::string name_gauge = "foo.bar.gauge";

    MetricsTags tags_a{{"foo", "bar"}};
    auto a = registry_->GetCounter(name_counter, tags_a);

    MetricsTags tags_b{{"foo2", "bar2"}};
    auto b = registry_->GetCounter(name_counter, tags_b);

    MetricsTags tags_c{{"foo", "bar"}};
    auto c = registry_->GetGauge(name_gauge, tags_c);

    MetricsTags tags_d{{"foo2", "bar2"}};
    auto d = registry_->GetGauge(name_gauge, tags_d);

    MetricsTags tags_e{{"foo", "bar"}, {"foo2", "bar2"}};
    auto e = registry_->GetGauge(name_gauge, tags_e);

    {
        std::vector<MetricsRegistry::metrics_tuple_t> all_metrics;
        registry_->GetAllMetrics(all_metrics);

        ASSERT_EQ(5, all_metrics.size());

        ++a, b += 2, c = 3.0, d += 4.0, e -= 5.0;

        std::shared_ptr<MetricsValue> val_a = std::make_shared<MetricsValue>(std::in_place_type<CounterValue>, 1);
        ASSERT_TRUE(VecContains(all_metrics, std::make_tuple(name_counter, tags_a, val_a)));

        std::shared_ptr<MetricsValue> val_b = std::make_shared<MetricsValue>(std::in_place_type<CounterValue>, 2);
        ASSERT_TRUE(VecContains(all_metrics, std::make_tuple(name_counter, tags_b, val_b)));

        std::shared_ptr<MetricsValue> val_c = std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, 3.0);
        ASSERT_TRUE(VecContains(all_metrics, std::make_tuple(name_gauge, tags_c, val_c)));

        std::shared_ptr<MetricsValue> val_d = std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, 4.0);
        ASSERT_TRUE(VecContains(all_metrics, std::make_tuple(name_gauge, tags_d, val_d)));

        std::shared_ptr<MetricsValue> val_e = std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, -5.0);
        ASSERT_TRUE(VecContains(all_metrics, std::make_tuple(name_gauge, tags_e, val_e)));
    }

    {
        std::vector<MetricsRegistry::metrics_tuple_t> all_metrics(8);
        registry_->GetAllMetrics(all_metrics);

        ASSERT_EQ(5, all_metrics.size());

        ++a, b += 2, c = 3.0, d += 4.0, e -= 5.0;

        std::shared_ptr<MetricsValue> val_a = std::make_shared<MetricsValue>(std::in_place_type<CounterValue>, 2);
        ASSERT_TRUE(VecContains(all_metrics, std::make_tuple(name_counter, tags_a, val_a)));

        std::shared_ptr<MetricsValue> val_b = std::make_shared<MetricsValue>(std::in_place_type<CounterValue>, 4);
        ASSERT_TRUE(VecContains(all_metrics, std::make_tuple(name_counter, tags_b, val_b)));

        std::shared_ptr<MetricsValue> val_c = std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, 3.0);
        ASSERT_TRUE(VecContains(all_metrics, std::make_tuple(name_gauge, tags_c, val_c)));

        std::shared_ptr<MetricsValue> val_d = std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, 8.0);
        ASSERT_TRUE(VecContains(all_metrics, std::make_tuple(name_gauge, tags_d, val_d)));

        std::shared_ptr<MetricsValue> val_e = std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, -10.0);
        ASSERT_TRUE(VecContains(all_metrics, std::make_tuple(name_gauge, tags_e, val_e)));
    }
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryGetMetricsData) {
    ASSERT_EQ(nullptr, registry_->GetMetricsData("foo"));
    ASSERT_NE(nullptr, registry_->GetOrCreateMetricsData("foo"));
    ASSERT_NE(nullptr, registry_->GetMetricsData("foo"));
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryMultiThreads) {
    std::thread t([this]() -> void {
        for (int i = 0; i < 1024; ++i) {
            auto c = registry_->GetCounter(std::string{"foo"}.append(std::to_string(i)));
            auto g = registry_->GetGauge(std::string{"bar"}.append(std::to_string(i)));
            ++c;
            g += 1.0;
        }
    });

    std::vector<std::string> names;
    std::vector<MetricsRegistry::metrics_tuple_t> all_metrics;
    while (registry_->GetSize() != 2048) {
        registry_->GetNames(names);
        if (!names.empty()) {
            auto suffix = std::to_string((names.size() - 1) / 2);
            {
                // 1, 3, 5, 7, ...
                auto d = registry_->GetMetricsData(std::string{"foo"}.append(suffix));

                // GetCounter internally calls GetOrCreateMetricsData then GetOrCreateCounter
                // under separate locks, so MetricsData may temporarily be empty; spin until ready.
                auto metrics = d->GetMetricsValues();
                while (metrics.empty()) {
                    metrics = d->GetMetricsValues();
                }

                Counter c;
                ASSERT_NO_THROW(c = Counter{metrics.front().second});

                while (c.Get() == 0) {}
                ASSERT_EQ(1, c.Get());

                registry_->GetAllMetrics(all_metrics);
                ASSERT_TRUE(
                    VecContains(all_metrics,
                                std::make_tuple(std::string{"foo"}.append(suffix),
                                                MetricsTags{},
                                                std::make_shared<MetricsValue>(std::in_place_type<CounterValue>, 1))));
            }
            if (names.size() % 2 == 0) {
                // 2, 4, 6, 8, ...
                auto d = registry_->GetMetricsData(std::string{"bar"}.append(suffix));

                // GetGauge internally calls GetOrCreateMetricsData then GetOrCreateGauge
                // under separate locks, so MetricsData may temporarily be empty; spin until ready.
                auto metrics = d->GetMetricsValues();
                while (metrics.empty()) {
                    metrics = d->GetMetricsValues();
                }

                Gauge g;
                ASSERT_NO_THROW(g = Gauge{metrics.front().second});

                while (AlmostEqual(g.Get(), 0.)) {}
                ASSERT_EQ(1., g.Get());

                registry_->GetAllMetrics(all_metrics);
                ASSERT_TRUE(
                    VecContains(all_metrics,
                                std::make_tuple(std::string{"bar"}.append(suffix),
                                                MetricsTags{},
                                                std::make_shared<MetricsValue>(std::in_place_type<GaugeValue>, 1.))));
            }
        }
    }

    t.join();
}

/* ---------------------- MetricsData::Remove ----------------------- */

TEST_F(MetricsRegistryTest, TestMetricsDataRemoveExact) {
    auto md = std::make_shared<MetricsData>();

    MetricsTags tags_a{{"instance_id", "inst1"}};
    MetricsTags tags_b{{"instance_id", "inst2"}};
    MetricsTags tags_c{{"instance_id", "inst1"}, {"instance_group", "grp1"}};

    md->GetOrCreateCounter(tags_a);
    md->GetOrCreateGauge(tags_b);
    md->GetOrCreateCounter(tags_c);
    ASSERT_EQ(3, md->GetSize());

    // remove non-existent tags returns false
    ASSERT_FALSE(md->Remove(MetricsTags{{"instance_id", "inst999"}}));
    ASSERT_EQ(3, md->GetSize());

    // remove exact match
    ASSERT_TRUE(md->Remove(tags_a));
    ASSERT_EQ(2, md->GetSize());

    // removing again returns false
    ASSERT_FALSE(md->Remove(tags_a));
    ASSERT_EQ(2, md->GetSize());

    // remaining entries unaffected
    ASSERT_TRUE(md->Remove(tags_b));
    ASSERT_EQ(1, md->GetSize());
    ASSERT_TRUE(md->Remove(tags_c));
    ASSERT_EQ(0, md->GetSize());
}

TEST_F(MetricsRegistryTest, TestMetricsDataRemoveByTagFilter) {
    auto md = std::make_shared<MetricsData>();

    MetricsTags tags_a{{"instance_id", "inst1"}, {"instance_group", "grp1"}};
    MetricsTags tags_b{{"instance_id", "inst2"}, {"instance_group", "grp1"}};
    MetricsTags tags_c{{"instance_id", "inst1"}, {"instance_group", "grp2"}};
    MetricsTags tags_d{{"instance_id", "inst3"}, {"instance_group", "grp2"}};

    md->GetOrCreateCounter(tags_a);
    md->GetOrCreateGauge(tags_b);
    md->GetOrCreateCounter(tags_c);
    md->GetOrCreateGauge(tags_d);
    ASSERT_EQ(4, md->GetSize());

    // filter by instance_id=inst1 should remove tags_a and tags_c
    ASSERT_EQ(2, md->RemoveByTagFilter({{"instance_id", "inst1"}}));
    ASSERT_EQ(2, md->GetSize());

    // filter by instance_group=grp1 should remove tags_b only
    ASSERT_EQ(1, md->RemoveByTagFilter({{"instance_group", "grp1"}}));
    ASSERT_EQ(1, md->GetSize());

    // filter by non-existent tag should remove nothing
    ASSERT_EQ(0, md->RemoveByTagFilter({{"instance_id", "inst999"}}));
    ASSERT_EQ(1, md->GetSize());

    // empty filter matches everything
    ASSERT_EQ(1, md->RemoveByTagFilter({}));
    ASSERT_EQ(0, md->GetSize());
}

TEST_F(MetricsRegistryTest, TestMetricsDataRemoveByTagFilterMultiKey) {
    auto md = std::make_shared<MetricsData>();

    MetricsTags tags_a{{"instance_id", "inst1"}, {"instance_group", "grp1"}};
    MetricsTags tags_b{{"instance_id", "inst1"}, {"instance_group", "grp2"}};
    MetricsTags tags_c{{"instance_id", "inst2"}, {"instance_group", "grp1"}};

    md->GetOrCreateCounter(tags_a);
    md->GetOrCreateCounter(tags_b);
    md->GetOrCreateCounter(tags_c);
    ASSERT_EQ(3, md->GetSize());

    // filter with two keys: only tags_a matches both
    ASSERT_EQ(1, md->RemoveByTagFilter({{"instance_id", "inst1"}, {"instance_group", "grp1"}}));
    ASSERT_EQ(2, md->GetSize());
}

/* -------------------- MetricsRegistry::Remove -------------------- */

TEST_F(MetricsRegistryTest, TestMetricsRegistryRemoveExact) {
    MetricsTags tags_a{{"instance_id", "inst1"}};
    MetricsTags tags_b{{"instance_id", "inst2"}};

    auto c1 = registry_->GetCounter("metric1", tags_a);
    auto c2 = registry_->GetCounter("metric1", tags_b);
    auto g1 = registry_->GetGauge("metric2", tags_a);
    ASSERT_EQ(3, registry_->GetSize());

    // remove non-existent name
    ASSERT_FALSE(registry_->Remove("no_such_metric", tags_a));
    ASSERT_EQ(3, registry_->GetSize());

    // remove non-existent tags under existing name
    ASSERT_FALSE(registry_->Remove("metric1", {{"instance_id", "inst999"}}));
    ASSERT_EQ(3, registry_->GetSize());

    // remove existing entry
    ASSERT_TRUE(registry_->Remove("metric1", tags_a));
    ASSERT_EQ(2, registry_->GetSize());

    // the counter handle is still usable (detached)
    ++c1;
    ASSERT_EQ(1, c1.Get());

    // metric1 still exists with tags_b
    ASSERT_NE(nullptr, registry_->GetMetricsData("metric1"));

    // remove last entry under metric1 -> MetricsData remains (empty shell)
    ASSERT_TRUE(registry_->Remove("metric1", tags_b));
    ASSERT_EQ(1, registry_->GetSize());
    // the MetricsData shell persists to avoid orphaning concurrent readers
    auto md = registry_->GetMetricsData("metric1");
    ASSERT_NE(nullptr, md);
    ASSERT_EQ(0, md->GetSize());

    // metric2 still intact
    ASSERT_TRUE(registry_->Remove("metric2", tags_a));
    ASSERT_EQ(0, registry_->GetSize());
}

/* --------------- MetricsRegistry::RemoveByTagFilter -------------- */

TEST_F(MetricsRegistryTest, TestMetricsRegistryRemoveByTagFilter) {
    MetricsTags tags_inst1_grp1{{"instance_id", "inst1"}, {"instance_group", "grp1"}};
    MetricsTags tags_inst2_grp1{{"instance_id", "inst2"}, {"instance_group", "grp1"}};
    MetricsTags tags_inst1_grp2{{"instance_id", "inst1"}, {"instance_group", "grp2"}};
    MetricsTags tags_inst3_grp2{{"instance_id", "inst3"}, {"instance_group", "grp2"}};

    // spread metrics across multiple names
    registry_->GetCounter("service.query_counter", tags_inst1_grp1);
    registry_->GetGauge("service.query_rt_us", tags_inst1_grp1);
    registry_->GetCounter("service.query_counter", tags_inst2_grp1);
    registry_->GetGauge("cache_manager_instance.key_count", tags_inst1_grp2);
    registry_->GetGauge("cache_manager_instance.byte_size", tags_inst3_grp2);
    ASSERT_EQ(5, registry_->GetSize());

    // remove all metrics for instance_id=inst1 (should remove 3 entries)
    ASSERT_EQ(3, registry_->RemoveByTagFilter({{"instance_id", "inst1"}}));
    ASSERT_EQ(2, registry_->GetSize());

    // verify remaining
    std::vector<MetricsRegistry::metrics_tuple_t> all;
    registry_->GetAllMetrics(all);
    ASSERT_EQ(2, all.size());

    // remove by instance_group=grp2 (should remove 1)
    ASSERT_EQ(1, registry_->RemoveByTagFilter({{"instance_group", "grp2"}}));
    ASSERT_EQ(1, registry_->GetSize());

    // remove remaining by instance_group=grp1
    ASSERT_EQ(1, registry_->RemoveByTagFilter({{"instance_group", "grp1"}}));
    ASSERT_EQ(0, registry_->GetSize());

    // no-op on empty registry
    ASSERT_EQ(0, registry_->RemoveByTagFilter({{"instance_id", "inst1"}}));
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryRemoveByTagFilterKeepsEmptyShell) {
    MetricsTags tags{{"instance_id", "inst1"}};
    registry_->GetCounter("metric1", tags);
    ASSERT_EQ(1, registry_->GetSize());
    ASSERT_NE(nullptr, registry_->GetMetricsData("metric1"));

    // removing the only entry leaves an empty MetricsData shell
    // (avoids orphaning concurrent GetCounter/GetGauge callers)
    registry_->RemoveByTagFilter({{"instance_id", "inst1"}});
    ASSERT_EQ(0, registry_->GetSize());
    auto md = registry_->GetMetricsData("metric1");
    ASSERT_NE(nullptr, md);
    ASSERT_EQ(0, md->GetSize());

    // GetNames still includes the name (shell is present)
    auto names = registry_->GetNames();
    ASSERT_EQ(1, names.size());
    ASSERT_EQ("metric1", names[0]);

    // re-creating a metric under the same name reuses the shell
    auto counter = registry_->GetCounter("metric1", tags);
    ASSERT_EQ(1, registry_->GetSize());
    ASSERT_EQ(md, registry_->GetMetricsData("metric1"));
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryRemoveDetachedHandleStillWorks) {
    MetricsTags tags{{"instance_id", "inst1"}};
    auto counter = registry_->GetCounter("metric1", tags);
    ++counter;
    ASSERT_EQ(1, counter.Get());

    // remove from registry
    registry_->Remove("metric1", tags);

    // the handle still works (shared_ptr keeps it alive)
    ++counter;
    ASSERT_EQ(2, counter.Get());

    // but it's no longer visible in GetAllMetrics
    std::vector<MetricsRegistry::metrics_tuple_t> all;
    registry_->GetAllMetrics(all);
    ASSERT_TRUE(all.empty());
}

TEST_F(MetricsRegistryTest, TestMetricsRegistryRemoveByTagFilterMultiThread) {
    // populate registry with metrics across two instances
    for (int i = 0; i < 100; ++i) {
        registry_->GetCounter("counter." + std::to_string(i), {{"instance_id", "inst1"}});
        registry_->GetGauge("gauge." + std::to_string(i), {{"instance_id", "inst2"}});
    }
    ASSERT_EQ(200, registry_->GetSize());

    // concurrently remove inst1 metrics while reading
    std::thread remover([this]() { registry_->RemoveByTagFilter({{"instance_id", "inst1"}}); });

    std::thread reader([this]() {
        std::vector<MetricsRegistry::metrics_tuple_t> all;
        for (int i = 0; i < 50; ++i) {
            registry_->GetAllMetrics(all);
            // size should be somewhere between 100 and 200
            ASSERT_GE(all.size(), 100);
            ASSERT_LE(all.size(), 200);
        }
    });

    remover.join();
    reader.join();

    // after removal, only inst2 metrics remain
    ASSERT_EQ(100, registry_->GetSize());

    // clean up
    ASSERT_EQ(100, registry_->RemoveByTagFilter({{"instance_id", "inst2"}}));
    ASSERT_EQ(0, registry_->GetSize());
}
