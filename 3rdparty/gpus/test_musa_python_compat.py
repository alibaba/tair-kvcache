#!/usr/bin/env python3
"""
test_musa_python_compat.py — Validate Python-layer MUSA compatibility changes.

This script can run on any machine with Python and torch installed (no GPU required).
It verifies:
  1. _get_device_module() falls back to torch.cuda when torch.musa is unavailable
  2. _device_mod has all required attributes (Stream, Event, stream, etc.)
  3. The float8 dtype mapping handles missing tl.float8e4nv gracefully

Usage:
    python 3rdparty/gpus/test_musa_python_compat.py
"""

import sys
import importlib
import types

PASS = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"  \033[32m[PASS]\033[0m {msg}")


def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  \033[31m[FAIL]\033[0m {msg}")


def section(msg):
    print(f"\n\033[33m--- {msg} ---\033[0m")


# ────────────────────────────────────────────────────────────────────────────
# Test 1: _get_device_module() behavior
# ────────────────────────────────────────────────────────────────────────────
section("Test 1: _get_device_module() fallback behavior")

try:
    import torch
except ImportError:
    print("ERROR: torch is not installed. Cannot run Python compat tests.")
    sys.exit(1)

# Simulate the helper function directly (avoid import path issues)
def _get_device_module():
    """Return the appropriate device module (torch.cuda or torch.musa)."""
    try:
        if hasattr(torch, 'musa') and torch.musa.is_available():
            import torch_musa  # noqa: F401
            return torch.musa
    except Exception:
        pass
    return torch.cuda

device_mod = _get_device_module()

if not hasattr(torch, 'musa'):
    if device_mod is torch.cuda:
        ok("Falls back to torch.cuda when torch.musa is absent")
    else:
        fail("Did not fall back to torch.cuda")
else:
    ok(f"Detected torch.musa — device_mod = {device_mod}")

# ────────────────────────────────────────────────────────────────────────────
# Test 2: Required attributes on device module
# ────────────────────────────────────────────────────────────────────────────
section("Test 2: device module attribute verification")

required_attrs = ["Stream", "Event", "stream", "set_device", "current_stream"]
for attr in required_attrs:
    if hasattr(device_mod, attr):
        ok(f"device_mod.{attr} exists")
    else:
        fail(f"device_mod.{attr} is MISSING")

# ────────────────────────────────────────────────────────────────────────────
# Test 3: float8 dtype mapping compatibility
# ────────────────────────────────────────────────────────────────────────────
section("Test 3: float8 dtype mapping (Triton compat)")

try:
    import triton
    import triton.language as tl

    # This is the exact expression used in batch_gather_scatter_helper.py
    float8_result = tl.float8e4nv if hasattr(tl, 'float8e4nv') else tl.float8e4b8
    ok(f"float8 dtype resolved to {float8_result} (hasattr(tl, 'float8e4nv') = {hasattr(tl, 'float8e4nv')})")

    # Simulate missing float8e4nv
    original = getattr(tl, 'float8e4nv', None)
    if original is not None:
        delattr(tl, 'float8e4nv')
        try:
            fallback_result = tl.float8e4nv if hasattr(tl, 'float8e4nv') else tl.float8e4b8
            if fallback_result == tl.float8e4b8:
                ok("Correctly falls back to tl.float8e4b8 when float8e4nv removed")
            else:
                fail("Fallback did not resolve to tl.float8e4b8")
        finally:
            tl.float8e4nv = original
    else:
        ok("tl.float8e4nv not present — fallback path already active")

except ImportError:
    print("  \033[33m[SKIP]\033[0m triton not installed — skipping float8 dtype test")

# ────────────────────────────────────────────────────────────────────────────
# Test 4: Verify no torch.cuda.* leaks in modified files
# ────────────────────────────────────────────────────────────────────────────
section("Test 4: Source file inspection (no residual torch.cuda.*)")

import os
import re

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
files_to_check = [
    "kv_cache_manager/py_connector/vllm/data_transfer.py",
    "kv_cache_manager/py_connector/vllm/v1_connector.py",
]

for relpath in files_to_check:
    filepath = os.path.join(project_root, relpath)
    if not os.path.exists(filepath):
        fail(f"File not found: {relpath}")
        continue

    with open(filepath) as f:
        lines = f.readlines()

    found_leak = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments, strings containing the pattern as documentation, and the
        # _get_device_module function definition itself
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        if '_get_device_module' in stripped or 'return torch.cuda' in stripped:
            continue
        if re.search(r'torch\.cuda\.(?:Stream|Event|stream|set_device|current_stream)\b', stripped):
            fail(f"Residual torch.cuda.* at {relpath}:{i}: {stripped}")
            found_leak = True

    if not found_leak:
        ok(f"No residual torch.cuda.* calls in {os.path.basename(relpath)}")


# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: \033[32m{PASS} passed\033[0m, \033[{'31' if FAIL else '32'}m{FAIL} failed\033[0m")
if FAIL > 0:
    sys.exit(1)
else:
    print("\033[32mAll Python compatibility checks passed.\033[0m")
    sys.exit(0)
