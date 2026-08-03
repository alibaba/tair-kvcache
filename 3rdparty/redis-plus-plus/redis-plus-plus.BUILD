genrule(
    name = "hiredis_features",
    outs = ["generated/sw/redis++/hiredis_features.h"],
    cmd = "mkdir -p $$(dirname $@) && touch $@",
)

cc_library(
    name = "redis_plus_plus",
    srcs = [
        "src/sw/redis++/command.cpp",
        "src/sw/redis++/command_options.cpp",
        "src/sw/redis++/connection.cpp",
        "src/sw/redis++/connection_pool.cpp",
        "src/sw/redis++/crc16.cpp",
        "src/sw/redis++/errors.cpp",
        "src/sw/redis++/patterns/redlock.cpp",
        "src/sw/redis++/pipeline.cpp",
        "src/sw/redis++/redis.cpp",
        "src/sw/redis++/redis_cluster.cpp",
        "src/sw/redis++/redis_uri.cpp",
        "src/sw/redis++/reply.cpp",
        "src/sw/redis++/sentinel.cpp",
        "src/sw/redis++/shards.cpp",
        "src/sw/redis++/shards_pool.cpp",
        "src/sw/redis++/subscriber.cpp",
        "src/sw/redis++/tls/sw/redis++/tls.cpp",
        "src/sw/redis++/transaction.cpp",
    ],
    hdrs = glob([
        "src/sw/redis++/*.h",
        "src/sw/redis++/*.hpp",
        "src/sw/redis++/cxx17/sw/redis++/*.h",
        "src/sw/redis++/patterns/*.h",
        "src/sw/redis++/tls/sw/redis++/*.h",
    ]) + [":hiredis_features"],
    includes = [
        "generated",
        "src",
        "src/sw/redis++/cxx17",
        "src/sw/redis++/tls",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@hiredis//:hiredis_ssl",
    ],
)
