from __future__ import annotations

import json
from pathlib import Path

from harness.e2e.flexlb_suite.feedback import analyze_feedback, extract_cache_feedback


def test_extract_cache_feedback_ignores_other_pv_events(tmp_path: Path) -> None:
    pv_path = tmp_path / "pv.log"
    output_path = tmp_path / "feedback.jsonl"
    pv_path.write_text(
        "\n".join(
            (
                '2026 INFO pvLogger - {"requestId":"route","success":true}',
                "not-json",
                '2026 INFO pvLogger - {"event":"cache_hit_comparison",'
                '"requestId":"feedback","source":"KVCM"}',
            )
        )
        + "\n"
    )

    extracted = extract_cache_feedback(pv_path=pv_path, output_path=output_path)

    assert extracted == 1
    assert json.loads(output_path.read_text())["requestId"] == "feedback"


def test_feedback_analysis_correlates_request_ids_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    requests_path = tmp_path / "requests.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    verdicts_path = tmp_path / "verdicts.jsonl"
    requests_path.write_text(
        json.dumps(
            {
                "gateway_request_id": "request-1",
                "response_id": "response-1",
                "case_id": "partial-32000",
                "cache_relation": "partial",
                "expected_prefix_tokens": 24_000,
                "prompt_tokens": 32_000,
                "cache_unit_tokens": 128,
                "settled": True,
            }
        )
        + "\n"
    )
    feedback_path.write_text(
        json.dumps(
            {
                "requestId": "request-1",
                "event": "cache_hit_comparison",
                "source": "KVCM",
                "worker": "10.0.0.7:8000",
                "inputTokens": 32_000,
                "kvcm": {"hit": 23_936, "delta": 64},
                "actual": {"hit": 24_000},
            }
        )
        + "\n"
    )

    summary = analyze_feedback(
        requests_path=requests_path,
        feedback_path=feedback_path,
        output_path=verdicts_path,
    )

    assert summary == {
        "total_requests": 1,
        "matched_feedback": 1,
        "unmatched_requests": 0,
        "pass": 1,
        "fail": 0,
        "inconclusive": 0,
        "prediction_match": 1,
        "prediction_underprediction": 0,
        "prediction_overprediction": 0,
    }
    verdict = json.loads(verdicts_path.read_text())
    assert verdict["status"] == "pass"
    assert verdict["selected_worker"] == "10.0.0.7:8000"
    assert verdict["predicted_hit_tokens"] == 23_936
    assert verdict["actual_hit_tokens"] == 24_000
    assert verdict["prediction_status"] == "match"


def test_feedback_analysis_marks_missing_or_invalid_feedback_inconclusive(
    tmp_path: Path,
) -> None:
    requests_path = tmp_path / "requests.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    verdicts_path = tmp_path / "verdicts.jsonl"
    requests_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "gateway_request_id": "missing",
                        "case_id": "cold",
                        "cache_relation": "cold",
                        "expected_prefix_tokens": 0,
                        "prompt_tokens": 1_024,
                        "cache_unit_tokens": 128,
                        "settled": True,
                    }
                ),
                json.dumps(
                    {
                        "gateway_request_id": "invalid",
                        "case_id": "exact",
                        "cache_relation": "exact",
                        "expected_prefix_tokens": 1_024,
                        "prompt_tokens": 1_024,
                        "cache_unit_tokens": 128,
                        "settled": True,
                    }
                ),
            )
        )
        + "\n"
    )
    feedback_path.write_text(
        json.dumps(
            {
                "requestId": "invalid",
                "cacheMatchSource": "KVCM",
                "predictedHitTokens": "not-an-int",
                "actualHitTokens": 1_024,
            }
        )
        + "\n"
    )

    summary = analyze_feedback(
        requests_path=requests_path,
        feedback_path=feedback_path,
        output_path=verdicts_path,
    )

    assert summary["matched_feedback"] == 1
    assert summary["unmatched_requests"] == 1
    assert summary["inconclusive"] == 2
    verdicts = [json.loads(line) for line in verdicts_path.read_text().splitlines()]
    assert [row["reasons"] for row in verdicts] == [
        ["feedback_not_found"],
        ["feedback_invalid"],
    ]


def test_feedback_analysis_scales_planned_prefix_to_actual_tokenization(
    tmp_path: Path,
) -> None:
    requests_path = tmp_path / "requests.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    verdicts_path = tmp_path / "verdicts.jsonl"
    requests_path.write_text(
        json.dumps(
            {
                "gateway_request_id": "request-1",
                "case_id": "exact-32000",
                "cache_relation": "exact",
                "target_tokens": 32_000,
                "expected_prefix_tokens": 32_000,
                "prompt_tokens": 29_788,
                "cache_unit_tokens": 1_152,
                "settled": True,
            }
        )
        + "\n"
    )
    feedback_path.write_text(
        json.dumps(
            {
                "requestId": "request-1",
                "source": "KVCM",
                "inputTokens": 29_799,
                "kvcm": {"hit": 28_800},
                "actual": {"hit": 28_800},
            }
        )
        + "\n"
    )

    summary = analyze_feedback(
        requests_path=requests_path,
        feedback_path=feedback_path,
        output_path=verdicts_path,
    )

    assert summary["pass"] == 1
    assert summary["fail"] == 0


def test_feedback_analysis_separates_stress_parity_from_prefix_residency(
    tmp_path: Path,
) -> None:
    requests_path = tmp_path / "requests.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"
    verdicts_path = tmp_path / "verdicts.jsonl"
    requests_path.write_text(
        json.dumps(
            {
                "gateway_request_id": "request-1",
                "case_id": "stress-partial-32000-p1",
                "family": "stress",
                "cache_relation": "partial",
                "target_tokens": 32_000,
                "expected_prefix_tokens": 24_000,
                "prompt_tokens": 29_800,
                "cache_unit_tokens": 1_152,
                "settled": True,
            }
        )
        + "\n"
    )
    feedback_path.write_text(
        json.dumps(
            {
                "requestId": "request-1",
                "source": "KVCM",
                "inputTokens": 29_800,
                "kvcm": {"hit": 0},
                "actual": {"hit": 17_280},
            }
        )
        + "\n"
    )

    summary = analyze_feedback(
        requests_path=requests_path,
        feedback_path=feedback_path,
        output_path=verdicts_path,
    )

    assert summary["fail"] == 1
    assert summary["prediction_underprediction"] == 1
    verdict = json.loads(verdicts_path.read_text())
    assert verdict["reasons"] == ["kvcm_underprediction"]
    assert verdict["prediction_status"] == "underprediction"
