package(default_visibility = ["//visibility:public"])

cc_library(
    name = "routing",
    srcs = ["routing_table.cc"],
    hdrs = ["routing_table.h"],
    deps = ["//kv_cache_manager/common/hash"],
)

cc_test(
    name = "RoutingTableTest",
    srcs = ["routing_table_test.cc"],
    deps = [
        ":routing",
        "//kv_cache_manager/common:unittest",
    ],
)
