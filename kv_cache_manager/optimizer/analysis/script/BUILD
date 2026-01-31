package(default_visibility = ["//visibility:public"])

py_binary(
    name = "optimizer_run",
    srcs = [
        "optimizer_run.py",
        "plot_hit_rate_with_storage.py",
    ],
    deps = [
        "//kv_cache_manager/optimizer/pybind:kvcm_py_optimizer_lib",
    ],
    main = "optimizer_run.py",
)

py_binary(
    name = "tradeoff_analysis_run_by_instances",
    srcs = [
        "tradeoff_curve_by_instances.py",
        "optimizer_analysis_utils.py",
    ],
    deps = [
        "//kv_cache_manager/optimizer/pybind:kvcm_py_optimizer_lib",
    ],
    main = "tradeoff_curve_by_instances.py",
)

py_binary(
    name = "tradeoff_analysis_run_by_policies",
    srcs = [
        "tradeoff_curve_by_policies.py",
        "optimizer_analysis_utils.py",
    ],
    deps = [
        "//kv_cache_manager/optimizer/pybind:kvcm_py_optimizer_lib",
    ],
    main = "tradeoff_curve_by_policies.py",
)

py_binary(
    name = "visualize_tree",
    srcs = [
        "export_and_visualize_tree.py",
        "plot_radix_tree.py",
    ],
    deps = [
        "//kv_cache_manager/optimizer/pybind:kvcm_py_optimizer_lib",
    ],
    main = "export_and_visualize_tree.py",
)