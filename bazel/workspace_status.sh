#!/bin/bash
# Bazel workspace status command.
# Outputs key-value pairs consumed by genrules with stamp = 1.
#
# STABLE_ prefixed keys: changes trigger rebuild of stamped targets.
# Non-prefixed keys: volatile, written to volatile-status.txt,
#   do NOT trigger rebuild (suitable for timestamps).

emit_git_status() {
    local prefix=$1
    local directory=$2
    local branch=${3:-}
    if [[ -z "$directory" ]]; then
        echo "${prefix}_COMMIT unknown"
        echo "${prefix}_COMMIT_FULL unknown"
        echo "${prefix}_REPO unknown"
        echo "${prefix}_BRANCH unknown"
        return
    fi
    echo "${prefix}_COMMIT $(git -C "$directory" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
    echo "${prefix}_COMMIT_FULL $(git -C "$directory" rev-parse HEAD 2>/dev/null || echo unknown)"
    local repository_url
    repository_url=$(git -C "$directory" remote get-url origin 2>/dev/null || echo unknown)
    # Checkout URLs may contain credentials; never stamp URL userinfo.
    repository_url=$(printf '%s' "$repository_url" | sed -E 's|^([[:alpha:]][[:alnum:]+.-]*://)[^/?#]*@|\1|')
    echo "${prefix}_REPO $repository_url"
    if [[ -z "$branch" ]]; then
        branch=$(git -C "$directory" symbolic-ref --quiet --short HEAD 2>/dev/null)
        if [[ -z "$branch" ]]; then
            if git -C "$directory" rev-parse --verify HEAD >/dev/null 2>&1; then
                branch="detached HEAD"
            else
                branch=unknown
            fi
        fi
    fi
    echo "${prefix}_BRANCH $branch"
}

emit_git_status STABLE_GIT . "${KVCM_GIT_BRANCH:-}"
stub_directory=$(cd ./stub_source 2>/dev/null && pwd -P)
emit_git_status STABLE_STUB_GIT "$stub_directory" "${KVCM_STUB_GIT_BRANCH:-}"
echo "STABLE_KVCM_VERSION 0.0.1"
# Used by py_wheel version templates; keep stable so repeat packaging for the
# same commit invalidates the cached wheel action and gets a fresh version.
echo "STABLE_BUILD_TIMESTAMP $(date +%Y%m%d%H%M%S)"
echo "BUILD_DATE $(date +%Y%m%d)"
echo "BUILD_TIME $(date '+%Y-%m-%d %H:%M:%S')"
