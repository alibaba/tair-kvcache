from __future__ import annotations

from harness.e2e.flexlb_suite.model import ScenarioManifest, StressSpec
from harness.e2e.flexlb_suite.stress import build_stress_schedule


def test_exact_stress_schedule_reuses_a_bounded_prefix_pool() -> None:
    manifest = ScenarioManifest(
        schema_version=1,
        engine_kind="vllm",
        cache_unit_tokens=128,
        max_context_tokens=64_000,
        cases=(),
        stress=StressSpec(
            duration_s=1_200,
            concurrency_steps=(1, 4, 8, 4),
            length_weights=((32_000, 100),),
            relation_weights=(("exact", 100),),
        ),
    )

    schedule = build_stress_schedule(
        manifest=manifest,
        seed=7,
        request_count=6,
        prefix_pool_size=2,
    )

    assert [item.case.case_id for item in schedule] == [
        "stress-exact-32000-p0",
        "stress-exact-32000-p1",
        "stress-exact-32000-p0",
        "stress-exact-32000-p1",
        "stress-exact-32000-p0",
        "stress-exact-32000-p1",
    ]
    assert [item.sample_index for item in schedule] == [0, 0, 1, 1, 2, 2]


def test_cold_stress_schedule_never_reuses_a_case_identity() -> None:
    manifest = ScenarioManifest(
        schema_version=1,
        engine_kind="sglang",
        cache_unit_tokens=128,
        max_context_tokens=64_000,
        cases=(),
        stress=StressSpec(
            duration_s=1_200,
            concurrency_steps=(1,),
            length_weights=((4_096, 100),),
            relation_weights=(("cold", 100),),
        ),
    )

    schedule = build_stress_schedule(
        manifest=manifest,
        seed=9,
        request_count=4,
        prefix_pool_size=2,
    )

    assert len({item.case.case_id for item in schedule}) == 4
    assert all(item.sample_index == 0 for item in schedule)
