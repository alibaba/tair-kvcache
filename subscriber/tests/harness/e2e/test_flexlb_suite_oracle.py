from __future__ import annotations

from harness.e2e.flexlb_suite.model import CacheComparison
from harness.e2e.flexlb_suite.oracle import evaluate_cache_comparison


def test_cold_kvcm_match_with_no_hit_passes() -> None:
    verdict = evaluate_cache_comparison(
        CacheComparison(
            case_id="cold-32000",
            cache_relation="cold",
            input_tokens=32_000,
            expected_prefix_tokens=0,
            predicted_hit_tokens=0,
            actual_hit_tokens=0,
            cache_unit_tokens=128,
            cache_match_source="KVCM",
            settled=True,
        )
    )

    assert verdict.status == "pass"
    assert verdict.delta_hit_tokens == 0
    assert verdict.reasons == ()


def test_exact_and_partial_matches_allow_one_cache_unit_of_alignment() -> None:
    exact = evaluate_cache_comparison(
        CacheComparison(
            case_id="exact-32000",
            cache_relation="exact",
            input_tokens=32_000,
            expected_prefix_tokens=32_000,
            predicted_hit_tokens=31_872,
            actual_hit_tokens=31_872,
            cache_unit_tokens=128,
            cache_match_source="KVCM",
            settled=True,
        )
    )
    partial = evaluate_cache_comparison(
        CacheComparison(
            case_id="partial-32000",
            cache_relation="partial",
            input_tokens=32_000,
            expected_prefix_tokens=24_000,
            predicted_hit_tokens=23_936,
            actual_hit_tokens=24_064,
            cache_unit_tokens=128,
            cache_match_source="KVCM",
            settled=True,
        )
    )

    assert exact.status == "pass"
    assert partial.status == "pass"


def test_partial_reuse_allows_eviction_but_rejects_impossible_extra_hit() -> None:
    evicted = evaluate_cache_comparison(
        CacheComparison(
            case_id="partial-32000",
            cache_relation="partial",
            input_tokens=32_000,
            expected_prefix_tokens=24_000,
            predicted_hit_tokens=12_000,
            actual_hit_tokens=12_000,
            cache_unit_tokens=128,
            cache_match_source="KVCM",
            settled=True,
        )
    )
    impossible = evaluate_cache_comparison(
        CacheComparison(
            case_id="partial-32000",
            cache_relation="partial",
            input_tokens=32_000,
            expected_prefix_tokens=24_000,
            predicted_hit_tokens=30_000,
            actual_hit_tokens=30_000,
            cache_unit_tokens=128,
            cache_match_source="KVCM",
            settled=True,
        )
    )

    assert evicted.status == "pass"
    assert impossible.status == "fail"
    assert "actual_hit_exceeds_shared_prefix" in impossible.reasons


def test_stress_prediction_parity_does_not_require_cache_residency() -> None:
    verdict = evaluate_cache_comparison(
        CacheComparison(
            case_id="stress-partial-32000-p1",
            cache_relation="partial",
            input_tokens=32_000,
            expected_prefix_tokens=24_000,
            predicted_hit_tokens=0,
            actual_hit_tokens=0,
            cache_unit_tokens=128,
            cache_match_source="KVCM",
            settled=True,
            enforce_expected_prefix=False,
        )
    )

    assert verdict.status == "pass"
    assert verdict.reasons == ()


def test_steady_kvcm_overprediction_fails_even_when_request_succeeds() -> None:
    verdict = evaluate_cache_comparison(
        CacheComparison(
            case_id="partial-64000",
            cache_relation="partial",
            input_tokens=64_000,
            expected_prefix_tokens=32_000,
            predicted_hit_tokens=48_000,
            actual_hit_tokens=32_000,
            cache_unit_tokens=128,
            cache_match_source="KVCM",
            settled=True,
        )
    )

    assert verdict.status == "fail"
    assert "kvcm_overprediction" in verdict.reasons


def test_non_kvcm_source_fails_a_measured_cache_case() -> None:
    verdict = evaluate_cache_comparison(
        CacheComparison(
            case_id="exact-4096",
            cache_relation="exact",
            input_tokens=4_096,
            expected_prefix_tokens=4_096,
            predicted_hit_tokens=4_096,
            actual_hit_tokens=4_096,
            cache_unit_tokens=128,
            cache_match_source="LOCAL_SYNC",
            settled=True,
        )
    )

    assert verdict.status == "fail"
    assert "cache_source_not_kvcm" in verdict.reasons


def test_unsettled_and_observational_cases_are_inconclusive() -> None:
    unsettled = evaluate_cache_comparison(
        CacheComparison(
            case_id="incremental-convergence",
            cache_relation="exact",
            input_tokens=32_000,
            expected_prefix_tokens=32_000,
            predicted_hit_tokens=0,
            actual_hit_tokens=31_872,
            cache_unit_tokens=128,
            cache_match_source="KVCM",
            settled=False,
        )
    )
    observational = evaluate_cache_comparison(
        CacheComparison(
            case_id="boundary-127",
            cache_relation="observational",
            input_tokens=127,
            expected_prefix_tokens=0,
            predicted_hit_tokens=0,
            actual_hit_tokens=0,
            cache_unit_tokens=128,
            cache_match_source="KVCM",
            settled=True,
        )
    )

    assert unsettled.status == "inconclusive"
    assert observational.status == "inconclusive"
