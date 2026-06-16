"""
Auto-skip tests that directly import kunlun_commons when not installed.
"""
import pytest
from schedule_simulator._compat import HAS_KUNLUN_COMMONS, HAS_DEEPESTIM


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that directly import kunlun_commons in test code."""
    if HAS_KUNLUN_COMMONS and HAS_DEEPESTIM:
        return

    skip_no_kunlun = pytest.mark.skip(reason="kunlun_commons not installed")

    kunlun_internal_test_files = {
        "test_time_predictor.py",
    }

    for item in items:
        test_file = item.fspath.basename
        if test_file in kunlun_internal_test_files:
            if not HAS_KUNLUN_COMMONS:
                item.add_marker(skip_no_kunlun)
