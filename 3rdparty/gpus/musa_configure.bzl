"""Repository rule for MUSA (Moore Threads) autoconfiguration.

`musa_configure` depends on the following environment variables:

  * `MUSA_TOOLKIT_PATH`: The path to the MUSA toolkit. Default is
    `/usr/local/musa`.
  * `MUSA_ARCH`: The MUSA compute architecture targets. Default is
    `mp_21,mp_22`.
"""

_MUSA_TOOLKIT_PATH = "MUSA_TOOLKIT_PATH"
_MUSA_ARCH = "MUSA_ARCH"

_DEFAULT_MUSA_TOOLKIT_PATH = "/usr/local/musa"
_DEFAULT_MUSA_ARCH = ["mp_21", "mp_22"]

def _get_musa_toolkit_path(repository_ctx):
    musa_toolkit_path = _DEFAULT_MUSA_TOOLKIT_PATH
    if _MUSA_TOOLKIT_PATH in repository_ctx.os.environ:
        musa_toolkit_path = repository_ctx.os.environ[_MUSA_TOOLKIT_PATH].strip()
    return musa_toolkit_path

def _musa_autoconf_impl(repository_ctx):
    musa_toolkit_path = _get_musa_toolkit_path(repository_ctx)

    # Create symlinks so Bazel can access toolkit files inside the sandbox
    repository_ctx.symlink(
        musa_toolkit_path + "/include",
        "musa/include",
    )
    repository_ctx.symlink(
        musa_toolkit_path + "/lib",
        "musa/lib",
    )

    repository_ctx.template(
        "BUILD",
        Label("//3rdparty/gpus/musa:BUILD.tpl"),
        {
            "%{musa_toolkit_path}": musa_toolkit_path,
        },
    )

    repository_ctx.template(
        "build_defs.bzl",
        Label("//3rdparty/gpus/musa:build_defs.bzl.tpl"),
        {},
    )

    repository_ctx.template(
        "musa/musa_config.h",
        Label("//3rdparty/gpus/musa:musa_config.h.tpl"),
        {
            "%{musa_toolkit_path}": musa_toolkit_path,
        },
    )

musa_configure = repository_rule(
    implementation = _musa_autoconf_impl,
    environ = [
        _MUSA_TOOLKIT_PATH,
        _MUSA_ARCH,
    ],
)
