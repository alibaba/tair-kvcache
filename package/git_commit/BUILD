load("//bazel:bundle.bzl", "bundle_files")

genrule(
    name = "git_commit_version",
    srcs = [],
    outs = ["git_commit"],
    cmd = "cat bazel-out/stable-status.txt bazel-out/volatile-status.txt > $(OUTS)",
    stamp = 1,
)

genrule(
    name = "git_commit_version_log_latest",
    srcs = [
        ":git_log.py",
        ":git_commit_version",
    ],
    outs = ["git_commit_log"],
    cmd = "python3 $(location git_log.py) -o $(OUTS)",
)

# rules_pkg's pkg_files consumes DefaultInfo files.  The custom
# commit_version_file target below intentionally exposes BundleFilesInfo for
# bundle_tar, so RPM rules must use this filegroup instead of treating that
# bundle provider as ordinary files.
filegroup(
    name = "commit_version_files",
    srcs = [
        ":git_commit_version",
        ":git_commit_version_log_latest",
    ],
    visibility = ["//visibility:public"],
)

bundle_files(
    name = "commit_version_file",
    srcs = [
        ":git_commit_version",
        ":git_commit_version_log_latest",
    ],
    prefix = "etc",
    visibility = ["//visibility:public"],
)
