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
