package(default_visibility = ["//visibility:private"])

cc_test(
    name = "MetricsCollectorTest",
    srcs = [
        "metrics_collector_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_collector",
        "//kv_cache_manager/metrics:metrics_registry",
    ],
)

cc_test(
    name = "KmonitorMetricsReporterTest",
    srcs = [
        "kmonitor_metrics_reporter_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_reporter",
        "@havenask//aios/kmonitor:kmonitor_client_cpp",
    ],
)

cc_test(
    name = "MetricsReporterFactoryTest",
    srcs = [
        "metrics_reporter_factory_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_reporter",
        "//kv_cache_manager/metrics:metrics_reporter_factory",
    ],
)

cc_test(
    name = "LocalMetricsReporterTest",
    srcs = [
        "local_metrics_reporter_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    linkstatic = True,
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_reporter",
        "@cpp_stub",
    ],
)

cc_test(
    name = "LoggingMetricsReporterTest",
    srcs = [
        "logging_metrics_reporter_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_reporter",
    ],
)

cc_test(
    name = "KmonParamTest",
    srcs = [
        "kmon_param_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_reporter",
    ],
)

cc_test(
    name = "DummyMetricsReporterTest",
    srcs = [
        "dummy_metrics_reporter_test.cc",
    ],
    copts = ["-fno-access-control"],
    data = [],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_reporter",
    ],
)

cc_test(
    name = "MetricsRegistryTest",
    srcs = [
        "metrics_registry_test.cc",
    ],
    copts = [
        "-fno-access-control",
    ],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/metrics:metrics_registry",
    ],
)
