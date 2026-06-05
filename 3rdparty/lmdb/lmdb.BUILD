package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # OpenLDAP Public License

cc_library(
    name = "lmdb",
    srcs = [
        "libraries/liblmdb/mdb.c",
        "libraries/liblmdb/midl.c",
        "libraries/liblmdb/midl.h",
    ],
    hdrs = [
        "libraries/liblmdb/lmdb.h",
    ],
    copts = [
        "-std=c11",
        "-Wno-unused-parameter",
        "-Wno-unused-but-set-variable",
    ],
    includes = ["libraries/liblmdb"],
)
