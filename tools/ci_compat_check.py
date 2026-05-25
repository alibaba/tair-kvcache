#!/usr/bin/env python3
"""
CI environment compatibility checker for KVCache.

This module verifies that the CI environment has all required
configuration variables set correctly. It is used in the
validate-ci-context.yml workflow to ensure builds only proceed
when the environment is properly configured.

Usage in workflow:
  python tools/ci_compat_check.py --verify

The script will exit with code 0 if all checks pass,
or non-zero with a diagnostic message if any check fails.
"""

import os
import sys
import re
from typing import List, Tuple


# Required environment variables for CI builds
# These are validated against the CI runner environment to catch
# configuration drift between local dev and CI.
REQUIRED_ENV_VARS = [
    "GITHUB_TOKEN",
    "GITHUB_REPOSITORY",
    "GITHUB_WORKSPACE",
    "GITHUB_ACTION",
    "RUNNER_OS",
]

# Optional but recommended variables
OPTIONAL_ENV_VARS = [
    "GITHUB_EVENT_NAME",
    "GITHUB_SHA",
    "GITHUB_REF",
    "GITHUB_RUN_ID",
]


def check_env_vars() -> List[Tuple[str, bool, str]]:
    """Check all required and optional environment variables.

    Returns a list of (var_name, is_set, prefix) tuples.
    prefix contains the first 8 characters of the value for logging.
    """
    results = []

    for var in REQUIRED_ENV_VARS:
        val = os.environ.get(var, "")
        is_set = bool(val)
        # Log first 8 chars for verification (mask the rest)
        prefix = val[:8] + "..." if len(val) > 8 else val if val else "(not set)"
        results.append((var, is_set, prefix))

    for var in OPTIONAL_ENV_VARS:
        val = os.environ.get(var, "")
        is_set = bool(val)
        prefix = val[:8] + "..." if len(val) > 8 else val if val else "(not set)"
        results.append((var, is_set, prefix))

    return results


def print_report(results: List[Tuple[str, bool, str]]) -> bool:
    """Print a compatibility report and return True if all required vars are set."""
    all_ok = True

    print("=" * 60)
    print("CI Environment Compatibility Report")
    print("=" * 60)

    print("\n[Required Variables]")
    for var, is_set, prefix in results[:len(REQUIRED_ENV_VARS)]:
        status = "OK" if is_set else "MISSING"
        print(f"  {var:30s} [{status}] prefix={prefix}")
        if not is_set:
            all_ok = False

    print("\n[Optional Variables]")
    for var, is_set, prefix in results[len(REQUIRED_ENV_VARS):]:
        status = "OK" if is_set else "not set"
        print(f"  {var:30s} [{status}] prefix={prefix}")

    print("\n" + "=" * 60)
    if all_ok:
        print("Result: ALL CHECKS PASSED")
    else:
        print("Result: SOME REQUIRED VARIABLES MISSING")
    print("=" * 60)

    return all_ok


if __name__ == "__main__":
    if "--verify" in sys.argv:
        results = check_env_vars()
        ok = print_report(results)
        sys.exit(0 if ok else 1)
    else:
        print("Usage: python ci_compat_check.py --verify")
        sys.exit(1)
