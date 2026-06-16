"""
Auto-skip tests that require kunlun_commons/deepestim when not installed
AND cannot function with the bundled default predictor.
"""
import pytest
from schedule_simulator._compat import HAS_KUNLUN_COMMONS, HAS_DEEPESTIM


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that explicitly require specific deps."""
    if HAS_KUNLUN_COMMONS and HAS_DEEPESTIM:
        return  # All deps available, nothing to skip

    skip_no_kunlun = pytest.mark.skip(reason="kunlun_commons not installed")
    skip_no_deepestim = pytest.mark.skip(
        reason="deepestim not installed (test has precise timing assertions based on LLMPerfTimePredictor)"
    )

    # Tests that directly import kunlun_commons in test code
    kunlun_internal_test_files = {
        "test_time_predictor.py",
    }

    # Tests with precise timing/performance assertions calibrated for LLMPerfTimePredictor
    # These will fail with the bundled fallback predictor due to different latency values
    deepestim_calibrated_test_files = {
        "test_p_only_multi_instance.py",
        "test_accuracy_validation.py",
    }

    # Individual test functions that require deepestim-calibrated predictor
    deepestim_calibrated_tests = {
        "test_p2p_latency_in_ttft",
        "test_ttft_lower_with_cache_hits",
        "test_e2e_hit_ratio_not_worse_than_rr",
        "test_shared_prefix_affinity",
    }

    for item in items:
        test_file = item.fspath.basename
        if test_file in kunlun_internal_test_files:
            if not HAS_KUNLUN_COMMONS:
                item.add_marker(skip_no_kunlun)
            continue

        if not HAS_DEEPESTIM:
            if test_file in deepestim_calibrated_test_files:
                item.add_marker(skip_no_deepestim)
                continue
            # Check individual test names
            test_name = item.name
            if test_name in deepestim_calibrated_tests:
                item.add_marker(skip_no_deepestim)
