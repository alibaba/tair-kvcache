from __future__ import annotations

from harness.e2e.flexlb_suite.manifest import build_default_manifest


def test_default_manifest_covers_boundaries_lengths_and_cache_relations() -> None:
    manifest = build_default_manifest(
        engine_kind="vllm",
        cache_unit_tokens=128,
        max_context_tokens=65_536,
    )

    boundary_targets = {
        case.target_tokens for case in manifest.cases if case.family == "boundary"
    }
    assert boundary_targets == {127, 128, 129, 511, 512, 513}

    standard_targets = {
        case.target_tokens
        for case in manifest.cases
        if case.family == "standard_length"
    }
    assert standard_targets == {1_024, 4_096, 16_384, 32_000, 64_000}

    relations = {case.cache_relation for case in manifest.cases}
    assert relations == {"cold", "exact", "partial", "negative", "observational"}
    assert manifest.stress.duration_s == 1_200
    assert manifest.stress.concurrency_steps == (1, 4, 8, 4)


def test_default_manifest_adds_sglang_bigram_cases_only_for_sglang() -> None:
    sglang = build_default_manifest(
        engine_kind="sglang",
        cache_unit_tokens=128,
        max_context_tokens=131_072,
    )
    vllm = build_default_manifest(
        engine_kind="vllm",
        cache_unit_tokens=1_152,
        max_context_tokens=131_072,
    )

    assert {case.case_id for case in sglang.cases if case.family == "bigram"} == {
        "bigram-overlap",
        "bigram-reordered-negative",
    }
    assert not [case for case in vllm.cases if case.family == "bigram"]
    assert max(case.target_tokens for case in vllm.cases) == 64_000
    assert max(length for length, _ in vllm.stress.length_weights) == 64_000


def test_default_manifest_omits_lengths_that_do_not_fit_context_window() -> None:
    manifest = build_default_manifest(
        engine_kind="vllm",
        cache_unit_tokens=1_152,
        max_context_tokens=32_768,
    )

    assert all(case.target_tokens <= 32_768 for case in manifest.cases)
    assert 64_000 not in {case.target_tokens for case in manifest.cases}
