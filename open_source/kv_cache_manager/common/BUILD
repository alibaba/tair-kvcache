package(default_visibility = ["//visibility:public"])

cc_library(
    name = "vipserver_subscriber",
    srcs = [
        "vipserver_subscriber.cc",
    ],
    hdrs = [
        "vipserver_subscriber.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:service_discovery",
    ],
)

cc_library(
    name = "spectrum_service_discovery",
    srcs = [
        "spectrum_service_discovery.cc",
    ],
    hdrs = [
        "spectrum_service_discovery.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:service_discovery",
    ],
)

cc_library(
    name = "service_discovery_factory",
    srcs = [
        "service_discovery_factory.cc",
    ],
    hdrs = [
        "service_discovery_factory.h",
    ],
    deps = [
        "//kv_cache_manager/common:logger",
        "//kv_cache_manager/common:service_discovery",
        "//kv_cache_manager/common:service_discovery_url",
        "//kv_cache_manager/common:static_service_discovery",
    ],
)
