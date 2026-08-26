package(default_visibility = ["//visibility:public"])

py_binary(
    name = "report_event_load",
    srcs = ["report_event_load.py"],
    deps = ["@pip_cpu//requests"],
)

py_test(
    name = "report_event_load_test",
    srcs = [
        "report_event_load.py",
        "report_event_load_test.py",
    ],
    main = "report_event_load_test.py",
    deps = ["@pip_cpu//requests"],
)
