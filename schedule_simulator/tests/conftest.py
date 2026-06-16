"""
Auto-skip tests that require kunlun_commons/deepestim when not installed.
"""
import pytest
from schedule_simulator._compat import HAS_KUNLUN_COMMONS, HAS_DEEPESTIM


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests requiring kunlun_commons or deepestim when unavailable."""
    if HAS_KUNLUN_COMMONS and HAS_DEEPESTIM:
        return  # All deps available, nothing to skip

    skip_no_kunlun = pytest.mark.skip(reason="kunlun_commons not installed")
    skip_no_deepestim = pytest.mark.skip(reason="deepestim not installed")

    # Test files that only require request-level mode (no kunlun_commons needed)
    request_level_only_files = {
        "test_load_counter.py",
        "test_direct_cache_aware_page_size.py",
    }

    for item in items:
        test_file = item.fspath.basename
        if test_file in request_level_only_files:
            continue

        # Check if test uses request_level marker
        markers = [m.name for m in item.iter_markers()]
        if "request_level" in markers:
            continue

        if not HAS_KUNLUN_COMMONS:
            item.add_marker(skip_no_kunlun)
        elif not HAS_DEEPESTIM:
            item.add_marker(skip_no_deepestim)
