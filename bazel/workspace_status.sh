#!/bin/bash
# Bazel workspace status command.
# Outputs key-value pairs consumed by genrules with stamp = 1.
#
# STABLE_ prefixed keys: changes trigger rebuild of stamped targets.
# Non-prefixed keys: volatile, written to volatile-status.txt,
#   do NOT trigger rebuild (suitable for timestamps).

echo "STABLE_GIT_COMMIT $(git rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
echo "STABLE_GIT_COMMIT_FULL $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "STABLE_GIT_REPO $(git remote get-url origin 2>/dev/null || echo unknown)"
echo "STABLE_KVCM_VERSION 0.0.1"
echo "BUILD_DATE $(date +%Y%m%d)"
echo "BUILD_TIME $(date '+%Y-%m-%d %H:%M:%S')"

# Internal repo commit (only populated when built as a submodule inside the
# internal repo, so that open-source standalone builds stay clean).
# Detection: submodule has a .git *file* (gitlink), not a .git directory.
_internal_commit=""
_internal_commit_full=""
if [ -n "$INTERNAL_GIT_COMMIT" ]; then
    _internal_commit="$INTERNAL_GIT_COMMIT"
    _internal_commit_full="${INTERNAL_GIT_COMMIT_FULL:-unknown}"
elif [ -f .git ] && [ -d ../.git ]; then
    _internal_commit="$(git --git-dir=../.git rev-parse --short=8 HEAD 2>/dev/null)"
    _internal_commit_full="$(git --git-dir=../.git rev-parse HEAD 2>/dev/null)"
fi
if [ -n "$_internal_commit" ]; then
    echo "STABLE_INTERNAL_GIT_COMMIT $_internal_commit"
    echo "STABLE_INTERNAL_GIT_COMMIT_FULL $_internal_commit_full"
    echo "STABLE_INTERNAL_VERSION_SUFFIX .i-$_internal_commit"
else
    echo "STABLE_INTERNAL_VERSION_SUFFIX"
fi
