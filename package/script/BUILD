package(default_visibility = ["//visibility:public"])

py_test(
    name = "start_server_spectrum_test",
    srcs = ["start_server_spectrum_test.py"],
    data = ["start_server_spectrum.sh"],
)

filegroup(
    name = "start_server_script",
    srcs = [
        "start_server.sh",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "start_server_spectrum_script",
    srcs = [
        "start_server_spectrum.sh",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "decode_stack_trace_script",
    srcs = [
        "decode_stack_trace.sh",
    ],
    visibility = ["//visibility:public"],
)
