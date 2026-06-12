workspace(name = "kv_cache_manager")

# TODO for open_source
load("//stub_source:workspace.bzl", "kv_cache_manager_workspace")

kv_cache_manager_workspace()

# NOTE: bazel_skylib_workspace() not needed in bzlmod mode.

load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")

load("//3rdparty/gpus:musa_configure.bzl", "musa_configure")
musa_configure(name = "local_config_musa")

load("//3rdparty/py:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

# NOTE: pip_cpu is now set up via MODULE.bazel pip extension (bzlmod).
# The legacy WORKSPACE-based pip_parse is removed to avoid conflicts.

load("//3rdparty/py:python_configure.bzl", "declare_python_abi", "declare_python_platform")
declare_python_abi(name = "python_abi", python_version = "3")
declare_python_platform(name = "python_platform", python_version = "3")

# NOTE: rules_pkg is managed by MODULE.bazel (bzlmod)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()

load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")

hedron_compile_commands_setup_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive_transitive()
# NOTE: rules_cuda dependencies now handled by MODULE.bazel (bzlmod)