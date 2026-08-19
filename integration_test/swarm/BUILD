package(default_visibility = ["//integration_test:__subpackages__"])

filegroup(
    name = "scenario_data",
    srcs = glob([
        "scenarios/*.json",
        "expectations/*.json",
    ]),
)

py_library(
    name = "swarm_testlib",
    srcs = [
        "__init__.py",
        "evaluator.py",
        "fixture.py",
        "runner.py",
        "swarm_test_base.py",
    ],
    data = [
        ":scenario_data",
        "//tools/kvcm_swarm",
    ],
    imports = [
        "..",
        "../testlib",
    ],
    deps = [
        "//integration_test/testlib:test_base",
        "@pip_cpu//requests",
    ],
)

py_test(
    name = "test_swarm_http",
    size = "enormous",
    srcs = ["test_swarm_http.py"],
    main = "test_swarm_http.py",
    tags = ["no-remote-exec"],
    deps = [":swarm_testlib"],
)

py_test(
    name = "test_swarm_grpc",
    size = "enormous",
    srcs = ["test_swarm_grpc.py"],
    main = "test_swarm_grpc.py",
    tags = ["no-remote-exec"],
    deps = [":swarm_testlib"],
)

py_test(
    name = "test_swarm_capacity",
    size = "enormous",
    srcs = ["test_swarm_capacity.py"],
    main = "test_swarm_capacity.py",
    tags = ["no-remote-exec"],
    deps = [":swarm_testlib"],
)

py_test(
    name = "test_swarm_health",
    size = "enormous",
    srcs = ["test_swarm_health.py"],
    main = "test_swarm_health.py",
    tags = ["no-remote-exec"],
    deps = [":swarm_testlib"],
)

py_test(
    name = "test_swarm_drain",
    size = "enormous",
    srcs = ["test_swarm_drain.py"],
    main = "test_swarm_drain.py",
    tags = ["no-remote-exec"],
    deps = [":swarm_testlib"],
)

py_test(
    name = "test_swarm_direct_json",
    size = "enormous",
    srcs = ["test_swarm_direct_json.py"],
    main = "test_swarm_direct_json.py",
    tags = ["no-remote-exec"],
    deps = [":swarm_testlib"],
)

py_test(
    name = "test_swarm_transport_contract",
    size = "enormous",
    srcs = ["test_swarm_transport_contract.py"],
    main = "test_swarm_transport_contract.py",
    tags = ["no-remote-exec"],
    deps = [":swarm_testlib"],
)

py_test(
    name = "test_swarm_evaluator",
    size = "small",
    srcs = ["test_swarm_evaluator.py"],
    main = "test_swarm_evaluator.py",
    tags = ["no-remote-exec"],
    deps = [":swarm_testlib"],
)
