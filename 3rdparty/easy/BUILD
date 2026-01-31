load("//bazel:defs.bzl", "rpm_library")

hdrs = [
    "easy/easy_aes.h",
    "easy/easy_aio.h",
    "easy/easy_array.h",
    "easy/easy_atomic.h",
    "easy/easy_baseth_pool.h",
    "easy/easy_buf.h",
    "easy/easy_client.h",
    "easy/easy_connection.h",
    "easy/easy_crc.h",
    "easy/easy_define.h",
    "easy/easy_hash.h",
    "easy/easy_hashx.h",
    "easy/easy_http_handler.h",
    "easy/easy_inet.h",
    "easy/easy_io.h",
    "easy/easy_io_struct.h",
    "easy/easy_list.h",
    "easy/easy_log.h",
    "easy/easy_message.h",
    "easy/easy_pool.h",
    "easy/easy_request.h",
    "easy/easy_simple_handler.h",
    "easy/easy_socket.h",
    "easy/easy_ssl.h",
    "easy/easy_ssl_init.h",
    "easy/easy_string.h",
    "easy/easy_test.h",
    "easy/easy_time.h",
    "easy/easy_timer.h",
    "easy/easy_trace.h",
    "easy/easy_uthread.h",
    "easy/ev.h",
    "easy/http_parser.h",
]

rpm_library(
    name = "easy_headers",
    hdrs = hdrs,
    include_path = "usr/local/include/",
    include_prefix = ".",
    lib_path = "",
    rpms = [
        "@easy//file:file",
    ],
)

rpm_library(
    name = "easy",
    hdrs = [],
    include_path = "usr/local/include",
    lib_path = "usr/local/lib",
    rpms = [
        "@easy//file:file",
    ],
    shared_lib = "libeasy.so",
    static_lib = "libeasy.a",
    deps = [":easy_headers"],
)
