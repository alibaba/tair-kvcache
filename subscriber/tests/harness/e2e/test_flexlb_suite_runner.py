from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from harness.e2e.flexlb_suite.model import (
    ScenarioCase,
    ScenarioManifest,
    StressSpec,
)
from harness.e2e.flexlb_suite.runner import (
    run_functional_suite,
    run_stress_suite,
    summarize_requests,
)


@pytest.mark.asyncio
async def test_functional_runner_preserves_each_case_prompt_relationship(
    tmp_path: Path,
) -> None:
    observed_messages: dict[str, list[list[dict[str, str]]]] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        marker = request.headers["x-kv-test-case"]
        observed_messages.setdefault(marker, []).append(body["messages"])
        return httpx.Response(
            200,
            headers={"x-request-id": f"gateway-{marker}"},
            json={
                "id": f"response-{marker}",
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1_024,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
        )

    manifest = ScenarioManifest(
        schema_version=1,
        engine_kind="vllm",
        cache_unit_tokens=128,
        max_context_tokens=4_096,
        cases=(
            ScenarioCase(
                case_id="cold",
                family="standard_length",
                cache_relation="cold",
                target_tokens=1_024,
                repetitions=2,
            ),
            ScenarioCase(
                case_id="exact",
                family="standard_length",
                cache_relation="exact",
                target_tokens=1_024,
                prefix_tokens=1_024,
                repetitions=2,
            ),
            ScenarioCase(
                case_id="partial",
                family="standard_length",
                cache_relation="partial",
                target_tokens=1_024,
                prefix_tokens=768,
                tail_tokens=256,
                repetitions=2,
            ),
            ScenarioCase(
                case_id="branch-a-b",
                family="branch",
                cache_relation="partial",
                target_tokens=1_024,
                prefix_tokens=768,
                tail_tokens=256,
                repetitions=4,
            ),
            ScenarioCase(
                case_id="reordered-negative",
                family="negative",
                cache_relation="negative",
                target_tokens=1_024,
                repetitions=2,
            ),
            ScenarioCase(
                case_id="bigram-overlap",
                family="bigram",
                cache_relation="observational",
                target_tokens=1_024,
                repetitions=2,
            ),
        ),
        stress=StressSpec(0, (), (), ()),
    )
    output_path = tmp_path / "requests.jsonl"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="https://example.test/v1"
    ) as client:
        await run_functional_suite(
            manifest=manifest,
            client=client,
            model="test-model",
            output_path=output_path,
            run_id="run-123",
            chars_per_token=4.8,
            max_tokens=1,
        )

    cold = observed_messages["cold"]
    exact = observed_messages["exact"]
    partial = observed_messages["partial"]
    branch = observed_messages["branch-a-b"]
    reordered = observed_messages["reordered-negative"]
    bigram = observed_messages["bigram-overlap"]
    assert cold[0] != cold[1]
    assert exact[0] == exact[1]
    assert partial[0][0] == partial[1][0]
    assert partial[0][1] != partial[1][1]
    assert branch[0][0] == branch[2][0]
    assert branch[1][0] == branch[3][0]
    assert branch[0][0] != branch[1][0]
    reordered_words = [messages[0]["content"].split() for messages in reordered]
    assert sorted(reordered_words[0]) == sorted(reordered_words[1])
    assert reordered_words[0] != reordered_words[1]
    bigram_words = [messages[0]["content"].split() for messages in bigram]
    first_bigrams = set(zip(bigram_words[0], bigram_words[0][1:], strict=False))
    second_bigrams = set(zip(bigram_words[1], bigram_words[1][1:], strict=False))
    assert bigram_words[0][0] != bigram_words[1][0]
    assert len(first_bigrams & second_bigrams) >= len(first_bigrams) * 0.8

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 14
    assert {row["gateway_request_id"] for row in rows} == {
        "gateway-bigram-overlap",
        "gateway-branch-a-b",
        "gateway-cold",
        "gateway-exact",
        "gateway-partial",
        "gateway-reordered-negative",
    }
    assert all(row["body_ok"] for row in rows)
    assert all("prompt_sha256" in row for row in rows)
    assert all(row["cache_unit_tokens"] == 128 for row in rows)
    assert [row["settled"] for row in rows if row["case_id"] == "exact"] == [
        False,
        True,
    ]


@pytest.mark.asyncio
async def test_stress_runner_executes_all_concurrency_phases(tmp_path: Path) -> None:
    active_requests = 0
    max_active_requests = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active_requests, max_active_requests
        active_requests += 1
        max_active_requests = max(max_active_requests, active_requests)
        await asyncio.sleep(0)
        active_requests -= 1
        return httpx.Response(
            200,
            headers={"x-request-id": request.headers["x-kv-test-marker"]},
            json={
                "id": "response",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1_024, "completion_tokens": 1},
            },
        )

    manifest = ScenarioManifest(
        schema_version=1,
        engine_kind="vllm",
        cache_unit_tokens=128,
        max_context_tokens=4_096,
        cases=(),
        stress=StressSpec(
            duration_s=60,
            concurrency_steps=(1, 2),
            length_weights=((1_024, 100),),
            relation_weights=(("exact", 100),),
        ),
    )
    output_path = tmp_path / "stress.jsonl"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="https://example.test/v1"
    ) as client:
        records = await run_stress_suite(
            manifest=manifest,
            client=client,
            model="test-model",
            output_path=output_path,
            run_id="stress-run",
            chars_per_token=4.8,
            max_tokens=1,
            duration_s=60,
            max_requests=6,
            seed=11,
            prefix_pool_size=1,
        )

    assert len(records) == 6
    assert max_active_requests == 2
    assert {row["concurrency"] for row in records} == {1, 2}


@pytest.mark.asyncio
async def test_stress_runner_only_settles_after_a_prior_batch_completed(
    tmp_path: Path,
) -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            headers={"x-request-id": request.headers["x-kv-test-marker"]},
            json={
                "id": "response",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1_024, "completion_tokens": 1},
            },
        )

    manifest = ScenarioManifest(
        schema_version=1,
        engine_kind="vllm",
        cache_unit_tokens=128,
        max_context_tokens=4_096,
        cases=(),
        stress=StressSpec(
            duration_s=60,
            concurrency_steps=(2,),
            length_weights=((1_024, 100),),
            relation_weights=(("exact", 100),),
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="https://example.test/v1/"
    ) as client:
        records = await run_stress_suite(
            manifest=manifest,
            client=client,
            model="test-model",
            output_path=tmp_path / "stress-warmup.jsonl",
            run_id="warmup-run",
            chars_per_token=4.8,
            max_tokens=1,
            duration_s=60,
            max_requests=4,
            seed=11,
            prefix_pool_size=1,
        )

    assert [row["settled"] for row in records] == [False, False, True, True]


@pytest.mark.asyncio
async def test_stress_runner_aborts_after_consecutive_response_failures(
    tmp_path: Path,
) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "unavailable"})

    manifest = ScenarioManifest(
        schema_version=1,
        engine_kind="vllm",
        cache_unit_tokens=128,
        max_context_tokens=4_096,
        cases=(),
        stress=StressSpec(
            duration_s=60,
            concurrency_steps=(1,),
            length_weights=((1_024, 100),),
            relation_weights=(("cold", 100),),
        ),
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), base_url="https://example.test/v1/"
    ) as client:
        records = await run_stress_suite(
            manifest=manifest,
            client=client,
            model="test-model",
            output_path=tmp_path / "stress.jsonl",
            run_id="failing-run",
            chars_per_token=4.8,
            max_tokens=1,
            duration_s=60,
            max_requests=100,
            seed=11,
            prefix_pool_size=1,
            failure_backoff_s=0,
            max_consecutive_failures=2,
        )

    assert len(records) == 2


def test_request_summary_keeps_failures_separate_from_cache_expectations(
    tmp_path: Path,
) -> None:
    requests_path = tmp_path / "requests.jsonl"
    requests_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "case_id": "cold",
                        "http_code": 200,
                        "body_ok": True,
                        "cache_relation": "cold",
                    }
                ),
                json.dumps(
                    {
                        "case_id": "exact",
                        "http_code": 500,
                        "body_ok": False,
                        "cache_relation": "exact",
                    }
                ),
            )
        )
        + "\n"
    )

    summary = summarize_requests(requests_path)

    assert summary == {
        "total_requests": 2,
        "valid_responses": 1,
        "failed_responses": 1,
        "by_case": {
            "cold": {"total": 1, "valid": 1},
            "exact": {"total": 1, "valid": 0},
        },
    }
