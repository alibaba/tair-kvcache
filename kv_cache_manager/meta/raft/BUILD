package(default_visibility = ["//visibility:public"])

cc_library(
    name = "raft_mode",
    srcs = ["raft_mode.cc"],
    hdrs = ["raft_mode.h"],
)

cc_library(
    name = "raft_log_codec",
    srcs = [
        "raft_log_codec.cc",
    ],
    hdrs = [
        "raft_log_codec.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/meta:types",
        "@nuraft//:nuraft",
    ],
)

cc_library(
    name = "meta_log_store",
    srcs = [
        "meta_log_store.cc",
    ],
    hdrs = [
        "meta_log_store.h",
    ],
    deps = [
        "@nuraft//:nuraft",
    ],
)

cc_library(
    name = "lmdb_log_store",
    srcs = [
        "lmdb_log_store.cc",
    ],
    hdrs = [
        "lmdb_log_store.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "@lmdb//:lmdb",
        "@nuraft//:nuraft",
    ],
)

cc_library(
    name = "meta_state_mgr",
    srcs = [
        "meta_state_mgr.cc",
    ],
    hdrs = [
        "meta_state_mgr.h",
    ],
    deps = [
        "//kv_cache_manager/common",
        "@nuraft//:nuraft",
    ],
)

cc_library(
    name = "meta_state_machine",
    srcs = [
        "meta_state_machine.cc",
    ],
    hdrs = [
        "meta_state_machine.h",
    ],
    deps = [
        ":raft_log_codec",
        "//kv_cache_manager/common",
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/meta:common",
        "//kv_cache_manager/meta:meta_cache_base_backend",
        "@nuraft//:nuraft",
    ],
)

cc_library(
    name = "raft_coordinator",
    srcs = [
        "raft_coordinator.cc",
    ],
    hdrs = [
        "raft_coordinator.h",
    ],
    deps = [
        ":lmdb_log_store",
        ":meta_state_machine",
        ":meta_state_mgr",
        ":raft_mode",
        "//kv_cache_manager/common",
        "//kv_cache_manager/config",
        "//kv_cache_manager/meta:common",
        "//kv_cache_manager/meta:meta_cache_base_backend",
        "//kv_cache_manager/meta:meta_local_backend",
        "@nuraft//:nuraft",
    ],
)

cc_library(
    name = "meta_raft_backend",
    srcs = [
        "meta_raft_backend.cc",
    ],
    hdrs = [
        "meta_raft_backend.h",
    ],
    deps = [
        ":raft_coordinator",
        ":raft_log_codec",
        "//kv_cache_manager/common",
        "//kv_cache_manager/common:standard_uri",
        "//kv_cache_manager/common:string_util",
        "//kv_cache_manager/config",
        "//kv_cache_manager/meta:common",
        "//kv_cache_manager/meta:meta_storage_backend",
        "@nuraft//:nuraft",
    ],
)
