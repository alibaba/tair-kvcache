package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache-2.0

# NuRaft (eBay) — Raft consensus library.
# Build configuration choices for KVCM phase-1 HA:
#   * Use Boost.Asio (already vendored at @boost) instead of standalone Asio.
#   * Skip OpenSSL by defining SSL_LIBRARY_NOT_FOUND=1; NuRaft falls back to its
#     mock_ssl shim. Inter-Manager replication runs on a private network, so
#     plaintext is acceptable for phase-1 and avoids dragging OpenSSL into the
#     dep graph.
#   * NuRaft's own .cxx files include public headers as `"raft_server.hxx"`
#     (no `libnuraft/` prefix), so we add `include/libnuraft` and `src` to the
#     include search path.

NURAFT_DEFINES = [
    "USE_BOOST_ASIO",
    "SSL_LIBRARY_NOT_FOUND=1",
    "ENABLE_RAFT_STATS=1",
    "LOGGER_NO_COLOR=1",
]

NURAFT_COPTS = [
    "-std=c++17",
    "-Wno-unused-parameter",
    "-Wno-unused-variable",
    "-Wno-deprecated-declarations",
]

cc_library(
    name = "nuraft",
    srcs = glob([
        "src/*.cxx",
        "src/*.hxx",
        "src/*.h",
    ]),
    hdrs = glob(["include/libnuraft/*.hxx"]),
    copts = NURAFT_COPTS,
    defines = NURAFT_DEFINES,
    includes = [
        "include",
        "include/libnuraft",
        "src",
    ],
    deps = [
        "@boost//:asio",
        "@boost//:system",
    ],
)
