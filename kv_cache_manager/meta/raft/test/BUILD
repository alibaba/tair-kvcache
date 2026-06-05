package(default_visibility = ["//visibility:private"])

cc_test(
    name = "nuraft_smoke_test",
    srcs = ["nuraft_smoke_test.cc"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "@nuraft//:nuraft",
    ],
)

cc_test(
    name = "raft_log_codec_test",
    srcs = ["raft_log_codec_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/meta/raft:raft_log_codec",
        "@nuraft//:nuraft",
    ],
)

cc_test(
    name = "meta_log_store_test",
    srcs = ["meta_log_store_test.cc"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/meta/raft:meta_log_store",
        "@nuraft//:nuraft",
    ],
)

cc_test(
    name = "lmdb_log_store_test",
    srcs = ["lmdb_log_store_test.cc"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/meta/raft:lmdb_log_store",
        "@nuraft//:nuraft",
    ],
)

cc_test(
    name = "meta_state_machine_test",
    srcs = ["meta_state_machine_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/meta:common",
        "//kv_cache_manager/meta:meta_local_backend",
        "//kv_cache_manager/meta/raft:meta_state_machine",
        "//kv_cache_manager/meta/raft:raft_log_codec",
        "@nuraft//:nuraft",
    ],
)

cc_test(
    name = "raft_coordinator_test",
    srcs = ["raft_coordinator_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/meta/raft:raft_coordinator",
        "//kv_cache_manager/meta/raft:raft_log_codec",
        "@nuraft//:nuraft",
    ],
)

cc_test(
    name = "registry_raft_backend_test",
    srcs = ["registry_raft_backend_test.cc"],
    copts = ["-fno-access-control"],
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/config",
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/meta:common",
        "//kv_cache_manager/meta:meta_local_backend",
        "//kv_cache_manager/meta/raft:meta_state_machine",
        "//kv_cache_manager/meta/raft:raft_log_codec",
        "@nuraft//:nuraft",
    ],
)

cc_test(
    name = "multi_node_raft_test",
    srcs = ["multi_node_raft_test.cc"],
    data = ["lsan_suppressions.txt"],
    env = {"LSAN_OPTIONS": "suppressions=kv_cache_manager/meta/raft/test/lsan_suppressions.txt"},
    deps = [
        "//kv_cache_manager/common:unittest",
        "//kv_cache_manager/meta:cache_location",
        "//kv_cache_manager/meta/raft:raft_coordinator",
        "//kv_cache_manager/meta/raft:raft_log_codec",
        "@nuraft//:nuraft",
    ],
)
