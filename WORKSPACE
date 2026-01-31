workspace(name = "kv_cache_manager")

# TODO for open_source
load("//stub_source:workspace.bzl", "kv_cache_manager_workspace")

kv_cache_manager_workspace()

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies", "rules_proto_toolchains")

rules_proto_dependencies()

rules_proto_toolchains()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")

py_repositories()

load("//3rdparty/py:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")

load("//stub_source/deps:pip.bzl", "pip_deps")
pip_deps()
load("@pip_cpu//:requirements.bzl", pip_cpu_install_deps = "install_deps")
pip_cpu_install_deps()
load("//3rdparty/py:python_configure.bzl", "declare_python_abi", "declare_python_platform")
declare_python_abi(name = "python_abi", python_version = "3")
declare_python_platform(name = "python_platform", python_version = "3")

load("@rules_pkg//toolchains:rpmbuild_configure.bzl", "find_system_rpmbuild")
find_system_rpmbuild(name="rules_pkg_rpmbuild")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()

load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")

hedron_compile_commands_setup_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive_transitive()
load("@rules_cuda//cuda:repositories.bzl", "register_detected_cuda_toolchains", "rules_cuda_dependencies")
rules_cuda_dependencies()
register_detected_cuda_toolchains()