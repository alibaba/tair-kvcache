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
    name = "enable_peermem",
    values = {"define": "MOONCAKE_USE_PEERMEM=true"},
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
    name = "enable_efa",
    values = {"define": "MOONCAKE_USE_EFA=true"},
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
    "-Wno-narrowing",            # Suppress narrowing conversions warning -> error
    "-Wno-parentheses",          # Mooncake src: && within || without explicit parentheses
    "-Wno-array-parameter",      # Not recognized on GCC < 11, suppress gracefully
    "-Wno-error=parentheses",    # Don't treat this as error even with global -Werror
    "-Wno-reorder",              # Warning for member initialization out of order
    "-Wno-error=reorder",        # Don't treat reorder warning as error
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
 ":enable_peermem": ["-DWITH_NVIDIA_PEERMEM"],
    "//conditions:default": [],
})+ select({
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

genrule(
    name = "mooncake_version_h",
    outs = ["mooncake-store/include/version.h"],
    cmd = """cat > $@ << 'EOF'
#pragma once
#include <string>
namespace mooncake {
constexpr const char* MOONCAKE_STORE_VERSION = "2.0.0";
inline const std::string& GetMooncakeStoreVersion() {
    static const std::string version(MOONCAKE_STORE_VERSION);
    return version;
}
constexpr const char* MOONCAKE_DISPLAY_VERSION = "0.3.13.post1";
}  // namespace mooncake
EOF
""",
)

# asio stays header-only (no asio_impl.cpp / ASIO_SEPARATE_COMPILATION) so it
# matches the other yalantinglibs users linked into the same binaries.
cc_library(
    name = "mooncake_common",
    srcs = glob(
        ["mooncake-common/src/*.cpp"],
        exclude = ["mooncake-common/src/asio_impl.cpp"],
    ),
    hdrs = glob(["mooncake-common/include/**/*.h"]),
    includes = ["mooncake-common/include"],
    deps = [
        "@glog//:glog",
        "@jsoncpp_git//:jsoncpp",
        "@yalantinglibs//:ylt",
        "@yaml-cpp//:yaml_cpp",
    ],
    copts = mooncake_copts,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mooncake_transfer_engine",
    srcs = glob([
        "mooncake-transfer-engine/src/*.cpp",
        "mooncake-transfer-engine/src/shared_segment/*.cpp",
        "mooncake-transfer-engine/src/shared_segment/*.h",
        "mooncake-transfer-engine/src/common/**/*.cpp",
        "mooncake-transfer-engine/src/transport/*.cpp",
        "mooncake-transfer-engine/src/transport/rdma_transport/**/*.cpp",
        "mooncake-transfer-engine/src/transport/rdma_twosided/**/*.cpp",
        # rpc_communicator is the pybind11-only Python RPC helper; not needed here.
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
        ":enable_tcp": glob([
            "mooncake-transfer-engine/src/transport/tcp_transport/**/*.cpp",
            "mooncake-transfer-engine/src/transport/tcp_transport/**/*.h",
        ]),
        "//conditions:default": [],
    }) + select({
        ":enable_nvmeof": glob(["mooncake-transfer-engine/src/transport/nvmeof_transport/**/*.cpp"]),
        "//conditions:default": [],
    }) + select({
        ":enable_efa": glob(["mooncake-transfer-engine/src/transport/efa_transport/**/*.cpp"]),
        "//conditions:default": [],
    }),
    hdrs = glob(["mooncake-transfer-engine/include/**/*.h"]),
    # USE_EFA changes TransferMetadata::HandShakeDesc layout, so it must reach
    # every dependent (defines), not just the files compiled here (copts).
    defines = ["YLT_ENABLE_IBV"] + select({
        ":enable_efa": ["USE_EFA"],
        "//conditions:default": [],
    }),
    includes = [
        "mooncake-transfer-engine/include",
        "mooncake-transfer-engine/src/shared_segment",
    ],
    deps = [
        ":mooncake_common",
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
    }) + select({
        ":enable_efa": ["@libfabric//:libfabric"],
        "//conditions:default": [],
    }),
    linkopts = [
        "-L/usr/local/lib64/",
        "-L/usr/lib64/",
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
        ["mooncake-store/src/**/*.cpp"],
        exclude = [
            # executables
            "mooncake-store/src/master.cpp",
            "mooncake-store/src/real_client_main.cpp",
            # optional backends not built here (USE_3FS / USE_NOF / AWS SDK /
            # ENABLE_KV_EVENTS / STORE_USE_K8S_LEASE / liburing)
            "mooncake-store/src/hf3fs/**",
            "mooncake-store/src/storage/distributed/hf3fs_adapter.cpp",
            "mooncake-store/src/spdk/**",
            "mooncake-store/src/utils/s3_helper.cpp",
            "mooncake-store/src/kv_event/**",
            "mooncake-store/src/ha/leadership/backends/k8s/**",
            "mooncake-store/src/uring_file.cpp",
            # unit-test stub and the separately linked libmooncake_store_c.so
            "mooncake-store/src/nvme_kv_executor_stub.cpp",
            "mooncake-store/src/store_c_shared.cpp",
        ],
    ) + glob(["mooncake-store/src/**/*.h"]) + [":mooncake_version_h"],
    hdrs = glob(
        [
            "mooncake-store/include/**/*.h",
            "mooncake-store/include/**/*.hpp",
        ],
        exclude = [
            "mooncake-store/include/hf3fs/**",
            "mooncake-store/include/utils/s3_helper.h",
        ],
    ),
    includes = [
        "mooncake-store/include",
        "mooncake-store/include/cachelib_memory_allocator",
        "mooncake-store/include/utils",
        "mooncake-store/include/cachelib_memory_allocator/include",
        "mooncake-store/include/cachelib_memory_allocator/fake_include",
        "mooncake-store/include/offset_allocator",
        "mooncake-store/include/serialize",
    ],
    deps = [
        ":mooncake_common",
        ":mooncake_transfer_engine",
        "@boost//:headers-base",
        "@boost//:headers-all",
        "@boost//:interprocess",
        "@boost//:date_time",
        "@boost//:property_tree",
        "@yaml-cpp//:yaml_cpp",
        "@msgpack_cxx//:msgpack",
        "@xxhash//:xxhash",
        "@zstd_lib//:zstd",
        "@com_github_gflags_gflags//:gflags",
        "@hiredis//:hiredis",
    ],
    copts = mooncake_copts + [
        # Upstream real_client.cpp uses DEFINE_* without including gflags.h.
        "-include",
        "gflags/gflags.h",
        "-Wno-error=invalid-memory-model",
        "-DSTORE_USE_REDIS",
    ],
    visibility = ["//visibility:public"],
)
