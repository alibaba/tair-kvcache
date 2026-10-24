from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_RUNNER_PATH = Path(__file__).parents[2] / "benchmark" / "cold_prefill_latency.py"
_SPEC = importlib.util.spec_from_file_location("cold_prefill_latency", _RUNNER_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_build_request_body_uses_unique_prompt_and_single_token_output() -> None:
    first = _MODULE.build_request_body(
        model="pre-flexlb-hongyi-test-glm-5.2",
        target_tokens=256_000,
        seed=101,
        chars_per_token=4.8,
    )
    second = _MODULE.build_request_body(
        model="pre-flexlb-hongyi-test-glm-5.2",
        target_tokens=256_000,
        seed=102,
        chars_per_token=4.8,
    )

    assert first.api_body["max_tokens"] == 1
    assert first.api_body["stream"] is False
    assert (
        first.api_body["messages"][0]["content"]
        != second.api_body["messages"][0]["content"]
    )
    assert first.prompt_sha256 != second.prompt_sha256


def test_parse_response_extracts_usage_fields() -> None:
    response = {
        "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 256_123,
            "completion_tokens": 1,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
    }

    parsed = _MODULE.parse_response(json.dumps(response))

    assert parsed.body_ok is True
    assert parsed.prompt_tokens == 256_123
    assert parsed.completion_tokens == 1
    assert parsed.cached_tokens == 0
    assert parsed.reason == "-"


def test_parse_response_rejects_malformed_or_empty_content() -> None:
    parsed = _MODULE.parse_response('{"choices": []}')

    assert parsed.body_ok is False
    assert "no_choices" in parsed.reason


def test_parse_response_accepts_one_token_reasoning_output() -> None:
    response = {
        "choices": [
            {
                "message": {"content": "", "reasoning_content": "The"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 256, "completion_tokens": 1},
    }

    parsed = _MODULE.parse_response(json.dumps(response))

    assert parsed.body_ok is True
    assert parsed.completion_tokens == 1


def test_cache_hit_response_cannot_be_accepted() -> None:
    accepted, reason = _MODULE._is_accepted(
        http_code=200,
        parsed=_MODULE.ParsedResponse(
            body_ok=True,
            reason="-",
            prompt_tokens=256_000,
            cached_tokens=128,
            completion_tokens=1,
        ),
        target_tokens=256_000,
        token_tolerance=0.02,
    )

    assert accepted is False
    assert reason == "cache_hit"


def test_curl_config_keeps_api_key_out_of_process_arguments() -> None:
    config = _MODULE._build_curl_config(
        base_url="https://example.test/v1",
        api_key="test-secret",
        request_path=Path("/tmp/request.json"),
        response_path=Path("/tmp/response.json"),
        timeout_s=60,
    )

    assert "Authorization: Bearer test-secret" in config
    assert _MODULE._CURL_COMMAND == ("curl", "--config", "-")


def test_chart_svg_includes_samples_means_and_axis_units() -> None:
    chart = _MODULE._chart_svg(
        [
            (256_000, 256_000, 12_000.0),
            (256_000, 256_500, 13_000.0),
            (400_000, 400_000, 21_000.0),
        ]
    )

    assert "Actual prompt tokens" in chart
    assert "End-to-end latency (s)" in chart
    assert '<rect width="100%" height="100%" fill="#ffffff"/>' in chart
    assert 'aria-label="individual cold-cache samples"' in chart
    assert 'aria-label="mean latency by target length"' in chart
    assert 'aria-label="individual sample coordinates"' in chart
    assert 'aria-label="quadratic least-squares fit"' in chart
    assert "1M fit:" in chart


def test_quadratic_fit_predicts_known_curve() -> None:
    coefficients = _MODULE._fit_quadratic(
        [
            (0, 0, 1.0),
            (1_000_000, 1_000_000, 6.0),
            (2_000_000, 2_000_000, 17.0),
        ]
    )

    assert _MODULE._quadratic_value(coefficients, 1.0) == pytest.approx(6.0)


@pytest.mark.skipif(
    sys.platform != "darwin"
    or shutil.which("qlmanage") is None
    or shutil.which("sips") is None,
    reason="PNG conversion uses local macOS image tools",
)
def test_render_chart_writes_png(tmp_path: Path) -> None:
    result_path = tmp_path / "results.csv"
    result_path.write_text(
        "target_tokens,prompt_tokens,elapsed_ms,accepted\n"
        "256000,256120,12000.0,True\n"
        "256000,255990,13000.0,True\n",
        encoding="utf-8",
    )
    chart_path = _MODULE.render_chart(result_path, tmp_path / "chart.png")

    assert chart_path.exists()
    assert chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    dimensions = subprocess.run(
        ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(chart_path)],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    assert "pixelWidth: 1280" in dimensions
    assert "pixelHeight: 1280" in dimensions
