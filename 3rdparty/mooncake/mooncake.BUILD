# mooncake-common/common.cmake 定义的编译配置选项
config_setting(
    name = "enable_asan",
    values = {"define": "MOONCAKE_ENABLE_ASAN=true"},
)

config_setting(
    name = "enable_cuda",
    values = {"define": "MOONCAKE_USE_CUDA=true"},
)

config_setting(
    name = "enable_nvmeof",
    values = {"define": "MOONCAKE_USE_NVMEOF=true"},
)

config_setting(
    name = "enable_tcp",
    values = {"define": "MOONCAKE_USE_TCP=true"},
)

config_setting(
    name = "enable_ascend",
    values = {"define": "MOONCAKE_USE_ASCEND=true"},
)

config_setting(
    name = "enable_mnnvl",
    values = {"define": "MOONCAKE_USE_MNNVL=true"},
)

config_setting(
    name = "enable_cxl",
    values = {"define": "MOONCAKE_USE_CXL=true"},
)

config_setting(
    name = "enable_etcd",
    values = {"define": "MOONCAKE_USE_ETCD=true"},
)

config_setting(
    name = "enable_etcd_legacy",
    values = {"define": "MOONCAKE_USE_ETCD_LEGACY=true"},
)

config_setting(
    name = "enable_redis",
    values = {"define": "MOONCAKE_USE_REDIS=true"},
)

config_setting(
    name = "enable_http",
    values = {"define": "MOONCAKE_USE_HTTP=true"},
)

config_setting(
    name = "with_rust_example",
    values = {"define": "MOONCAKE_WITH_RUST_EXAMPLE=true"},
)

config_setting(
    name = "with_metrics",
    values = {"define": "MOONCAKE_WITH_METRICS=true"},
)

# lru
config_setting(
    name = "use_lru_master",
    values = {"define": "MOONCAKE_USE_LRU_MASTER=true"},
)

config_setting(
    name = "store_use_etcd",
    values = {"define": "MOONCAKE_STORE_USE_ETCD=false"},
)

mooncake_copts = [
    "-std=c++20",  # C++ 标准
    "-g",
    "-Wall",
    "-Wextra",
    "-Wno-unused-parameter",
    "-fcoroutines",
    "-fPIC",
    "-DCONFIG_ERDMA",
] + select({
    ":enable_asan": ["-fsanitize=address", "-fsanitize=leak"],
    "//conditions:default": [],
}) + select({
    ":enable_cuda": ["-DUSE_CUDA"],
    "//conditions:default": [],
}) + select({
    # metadata service type
    ":enable_http": ["-DUSE_HTTP"],
    "//conditions:default": [],
}) + select({
    ":enable_etcd": ["-DUSE_ETCD"],
    "//conditions:default": [],
}) + select({
    ":enable_etcd_legacy": ["-DUSE_ETCD_LEGACY"],
    "//conditions:default": [],
}) + select({
    ":enable_redis": ["-DUSE_REDIS"],
    "//conditions:default": [],
}) + select({
    # transport type
    ":enable_nvmeof": ["-DUSE_NVMEOF", "-DUSE_CUDA"],  # NVMe-oF support
    "//conditions:default": [],
}) + select({
    ":enable_mnnvl": ["-DUSE_MNNVL", "-DUSE_CUDA"],  # Multi-Node nvlink support
    "//conditions:default": [],
}) + select({
    ":enable_tcp": ["-DUSE_TCP"],
    "//conditions:default": [],
}) + select({
    ":enable_cxl": ["-DUSE_CXL"],
    "//conditions:default": [],
}) + select({
    ":enable_ascend": ["-DUSE_ASCEND"],
    "//conditions:default": [],
}) + select({
    ":with_metrics": ["-DWITH_METRICS"],  # enable metrics and metrics reporting thread
    "//conditions:default": [],
}) + select({
    ":use_lru_master": ["-DUSE_LRU_MASTER", "-DLRU_MAX_CAPACITY=100"],  # option for using LRU in master service
    "//conditions:default": [],
})

cc_library(
    name = "mooncake_transfer_engine",
    srcs = glob([
        "mooncake-transfer-engine/src/*.cpp",
        "mooncake-transfer-engine/src/common/**/*.cpp",
        "mooncake-transfer-engine/src/transport/*.cpp",
        "mooncake-transfer-engine/src/transport/rdma_transport/**/*.cpp",
    ]) + select({
        ":enable_ascend": glob(["mooncake-transfer-engine/src/transport/ascend_transport/**/*.cpp"]),
        "//conditions:default": [],
    }) + select({
        ":enable_cxl": glob(["mooncake-transfer-engine/src/transport/cxl_transport/**/*.cpp"]),
        "//conditions:default": [],
    }) + select({
        ":enable_mnnvl": glob([
            "mooncake-transfer-engine/src/transport/nvlink_transport/**/*.cpp",
            "mooncake-transfer-engine/nvlink-allocator/**/*.cpp",
        ]),
        "//conditions:default": [],
    }) + select({
        ":enable_tcp": glob(["mooncake-transfer-engine/src/transport/tcp_transport/**/*.cpp"]),
        "//conditions:default": [],
    }) + select({
        ":enable_nvmeof": glob(["mooncake-transfer-engine/src/transport/nvmeof_transport/**/*.cpp"]),
        "//conditions:default": [],
    }),
    hdrs = glob(["mooncake-transfer-engine/include/**/*.h"]),
    defines = ["YLT_ENABLE_IBV"],
    includes = [
        "mooncake-transfer-engine/include",
        # "/usr/include",
        # "/usr/include/curl"
    ],
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@jsoncpp_git//:jsoncpp",
        "@yalantinglibs//:ylt",
        "@glog//:glog",
        "@//3rdparty/libnuma",
    ] + select({
        ":enable_cuda": ["@local_config_cuda//cuda:cuda"],
        "//conditions:default": [],
    }) + select({
        ":enable_http": ["@curl//:curl"],
        "//conditions:default": [],
    }),
    linkopts = [
        "-L/usr/local/lib64/",
        "-libverbs",
        "-lrdmacm",
    ] + select({
        ":enable_cuda": ["-lcuda"],
        "//conditions:default": [],
    }),
    copts = mooncake_copts,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mooncake_store",
    srcs = glob(
        [
            "mooncake-store/src/**/*.cpp",
            "mooncake-common/src/default_config.cpp",
        ],
        exclude = ["mooncake-store/src/hf3fs/*.cpp"],
    ),  # exclude hf3fs
    hdrs = glob(
        [
            "mooncake-store/include/**/*.h",
            "mooncake-common/include/default_config.h",
        ],
        exclude = ["mooncake-store/src/hf3fs/*.h"],
    ),
    includes = [
        "mooncake-common/include",
        "mooncake-store/include",
        "mooncake-store/include/cachelib_memory_allocator",
        "mooncake-store/include/utils",
        "mooncake-store/include/cachelib_memory_allocator/include",
        "mooncake-store/include/cachelib_memory_allocator/fake_include",
    ],
    deps = [
        ":mooncake_transfer_engine",
        "@boost//:headers-base",
        "@boost//:headers-all",
        "@boost//:interprocess",
        "@boost//:date_time",
        "@boost//:property_tree",
        "@yaml-cpp//:yaml_cpp",
    ],
    copts = mooncake_copts,
    visibility = ["//visibility:public"],
)
