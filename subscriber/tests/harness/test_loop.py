from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parents[2]
_LOOP = _REPO_ROOT / "harness" / "loop.py"


def _write_manifest(
    path: Path, repository: Path, gates: list[dict[str, object]]
) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "workspace_root": ".",
                "policy": {"output_tail_lines": 20, "slow_gate_threshold_s": 10},
                "repositories": {
                    "sample": {
                        "path": str(repository),
                        "profiles": {"quality": gates},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _run_loop(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_LOOP), *args],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_profile_records_passed_gate(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    records = tmp_path / "runs"
    _write_manifest(
        manifest,
        tmp_path,
        [
            {
                "id": "sample.pass",
                "category": "unit",
                "timeout_s": 10,
                "command": [sys.executable, "-c", "print('gate-ok')"],
            }
        ],
    )

    completed = _run_loop(
        "run",
        "quality",
        "all",
        "--manifest",
        str(manifest),
        "--record-dir",
        str(records),
    )

    assert completed.returncode == 0, completed.stderr
    record_paths = list(records.glob("*.json"))
    assert len(record_paths) == 1
    record = json.loads(record_paths[0].read_text(encoding="utf-8"))
    assert record["profile"] == "quality"
    assert record["results"][0]["gate_id"] == "sample.pass"
    assert record["results"][0]["status"] == "passed"
    assert record["results"][0]["output_tail"] == "gate-ok"


def test_run_profile_fails_fast_and_records_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    records = tmp_path / "runs"
    marker = tmp_path / "should-not-exist"
    _write_manifest(
        manifest,
        tmp_path,
        [
            {
                "id": "sample.fail",
                "category": "unit",
                "timeout_s": 10,
                "command": [sys.executable, "-c", "raise SystemExit(7)"],
            },
            {
                "id": "sample.after",
                "category": "unit",
                "timeout_s": 10,
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
            },
        ],
    )

    completed = _run_loop(
        "run",
        "quality",
        "sample",
        "--manifest",
        str(manifest),
        "--record-dir",
        str(records),
    )

    assert completed.returncode == 1
    assert not marker.exists()
    record_path = next(records.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["selected_gate_ids"] == ["sample.fail", "sample.after"]
    assert [result["status"] for result in record["results"]] == ["failed"]
    assert record["results"][0]["exit_code"] == 7

    review = _run_loop(
        "review",
        "--feedback-file",
        str(tmp_path / "missing-feedback.jsonl"),
        "--run-dir",
        str(records),
    )
    assert review.returncode == 0, review.stderr
    assert "`sample.fail`: gate_failed x1" in review.stdout


def test_keep_going_records_blocked_gate_and_runs_next_gate(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    records = tmp_path / "runs"
    marker = tmp_path / "continued"
    _write_manifest(
        manifest,
        tmp_path,
        [
            {
                "id": "sample.blocked",
                "category": "static",
                "timeout_s": 10,
                "command": ["definitely-missing-harness-executable"],
            },
            {
                "id": "sample.continued",
                "category": "unit",
                "timeout_s": 10,
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
            },
        ],
    )

    completed = _run_loop(
        "run",
        "quality",
        "sample",
        "--manifest",
        str(manifest),
        "--record-dir",
        str(records),
        "--keep-going",
    )

    assert completed.returncode == 1
    assert marker.exists()
    record_path = next(records.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert [result["status"] for result in record["results"]] == [
        "blocked",
        "passed",
    ]
    assert record["results"][0]["signals"] == ["prerequisite_blocked"]


def test_missing_repository_is_recorded_as_blocked(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    records = tmp_path / "runs"
    _write_manifest(
        manifest,
        tmp_path / "missing-repository",
        [
            {
                "id": "sample.missing-repository",
                "category": "workspace",
                "timeout_s": 10,
                "command": [sys.executable, "-c", "print('unreachable')"],
            }
        ],
    )

    completed = _run_loop(
        "run",
        "quality",
        "sample",
        "--manifest",
        str(manifest),
        "--record-dir",
        str(records),
    )

    assert completed.returncode == 1
    record_path = next(records.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["repositories"]["sample"]["error"] == (
        "repository path does not exist"
    )
    assert record["results"][0]["status"] == "blocked"


def test_dry_run_is_planned_and_does_not_execute_gate(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    records = tmp_path / "runs"
    marker = tmp_path / "not-executed"
    _write_manifest(
        manifest,
        tmp_path,
        [
            {
                "id": "sample.plan",
                "category": "unit",
                "timeout_s": 10,
                "command": [
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(marker)!r}).touch()",
                ],
            }
        ],
    )

    completed = _run_loop(
        "run",
        "quality",
        "sample",
        "--manifest",
        str(manifest),
        "--record-dir",
        str(records),
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    record_path = next(records.glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "planned"
    assert record["results"][0]["status"] == "planned"


def test_feedback_is_append_only_and_resolution_leaves_no_open_item(
    tmp_path: Path,
) -> None:
    feedback_file = tmp_path / "feedback.jsonl"
    created = _run_loop(
        "feedback",
        "--feedback-file",
        str(feedback_file),
        "--gate-id",
        "sample.fail",
        "--signal",
        "failure",
        "--observation",
        "dependency missing",
        "--proposal",
        "document the prerequisite",
    )
    assert created.returncode == 0, created.stderr
    feedback_id = json.loads(created.stdout)["id"]

    resolved = _run_loop(
        "feedback",
        "--feedback-file",
        str(feedback_file),
        "--gate-id",
        "sample.fail",
        "--signal",
        "resolved",
        "--observation",
        "prerequisite documented",
        "--resolves",
        feedback_id,
    )
    assert resolved.returncode == 0, resolved.stderr
    assert len(feedback_file.read_text(encoding="utf-8").splitlines()) == 2

    review = _run_loop(
        "review",
        "--feedback-file",
        str(feedback_file),
        "--run-dir",
        str(tmp_path / "missing-runs"),
    )
    assert review.returncode == 0, review.stderr
    assert "No open feedback." in review.stdout
