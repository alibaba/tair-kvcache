#!/usr/bin/env python3
"""Generate overall and diff line coverage summaries from an LCOV report."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


DIFF_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def normalize_source_path(source_path, workspace):
    path = Path(source_path)
    workspace = workspace.resolve()

    if path.is_absolute():
        resolved = path.resolve(strict=False)
        try:
            return resolved.relative_to(workspace).as_posix()
        except ValueError:
            text = resolved.as_posix()
            for prefix in ("kv_cache_manager/", "integration_test/", "tools/"):
                marker = "/" + prefix
                index = text.rfind(marker)
                if index >= 0:
                    return text[index + 1 :]
            return text

    return path.as_posix().lstrip("./")


def should_include(path, include_prefixes):
    if not include_prefixes:
        return True
    return any(path.startswith(prefix) for prefix in include_prefixes)


def parse_lcov(lcov_path, workspace, include_prefixes):
    coverage = {}
    current_file = None
    current_lines = {}

    def flush_record():
        if current_file and should_include(current_file, include_prefixes):
            file_coverage = coverage.setdefault(current_file, {})
            for line_number, hits in current_lines.items():
                file_coverage[line_number] = file_coverage.get(line_number, 0) + hits

    with lcov_path.open("r", encoding="utf-8") as lcov:
        for raw_line in lcov:
            line = raw_line.strip()
            if line.startswith("SF:"):
                flush_record()
                current_file = normalize_source_path(line[3:], workspace)
                current_lines = {}
            elif line.startswith("DA:") and current_file:
                fields = line[3:].split(",", 2)
                if len(fields) >= 2:
                    current_lines[int(fields[0])] = max(int(fields[1]), 0)
            elif line == "end_of_record":
                flush_record()
                current_file = None
                current_lines = {}
    flush_record()
    return coverage


def parse_unified_diff(diff_text, include_prefixes):
    changed_lines = {}
    current_file = None

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target == "/dev/null":
                current_file = None
            elif target.startswith("b/"):
                current_file = target[2:]
            else:
                current_file = target

            if current_file and not should_include(current_file, include_prefixes):
                current_file = None
            continue

        if not current_file:
            continue

        match = DIFF_HUNK_RE.match(line)
        if not match:
            continue

        start = int(match.group(1))
        count = int(match.group(2) or "1")
        if count == 0:
            continue
        changed_lines.setdefault(current_file, set()).update(range(start, start + count))

    return changed_lines


def git_diff(base_ref, head_ref):
    if not base_ref:
        return ""

    ranges = [f"{base_ref}...{head_ref}", f"{base_ref}..{head_ref}"]
    last_error = None
    for rev_range in ranges:
        result = subprocess.run(
            ["git", "diff", "--unified=0", "--no-ext-diff", rev_range, "--"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout
        last_error = result.stderr.strip()

    raise RuntimeError(f"failed to diff {base_ref} against {head_ref}: {last_error}")


def coverage_rate(covered, coverable):
    if coverable == 0:
        return None
    return covered / coverable


def format_rate(rate):
    if rate is None:
        return "N/A"
    return f"{rate * 100:.2f}%"


def format_line_ranges(lines):
    if not lines:
        return ""

    ranges = []
    start = previous = None
    for line in sorted(lines):
        if start is None:
            start = previous = line
        elif line == previous + 1:
            previous = line
        else:
            ranges.append((start, previous))
            start = previous = line
    ranges.append((start, previous))

    return ", ".join(str(s) if s == e else f"{s}-{e}" for s, e in ranges)


def summarize_coverage(coverage):
    coverable = 0
    covered = 0
    for lines in coverage.values():
        coverable += len(lines)
        covered += sum(1 for hits in lines.values() if hits > 0)
    return {
        "covered_lines": covered,
        "coverable_lines": coverable,
        "line_rate": coverage_rate(covered, coverable),
    }


def summarize_diff_coverage(coverage, changed_lines):
    coverable = 0
    covered = 0
    changed = 0
    uncovered = {}

    for source_file, lines in sorted(changed_lines.items()):
        changed += len(lines)
        file_coverage = coverage.get(source_file)
        if not file_coverage:
            continue

        for line in sorted(lines):
            if line not in file_coverage:
                continue
            coverable += 1
            if file_coverage[line] > 0:
                covered += 1
            else:
                uncovered.setdefault(source_file, []).append(line)

    return {
        "changed_lines": changed,
        "covered_lines": covered,
        "coverable_lines": coverable,
        "line_rate": coverage_rate(covered, coverable),
        "uncovered_lines": uncovered,
    }


def render_markdown(overall, diff, base_ref, head_ref):
    lines = [
        "# Coverage Summary",
        "",
        "| Scope | Covered lines | Coverable lines | Line coverage |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Overall | {overall['covered_lines']} | "
            f"{overall['coverable_lines']} | {format_rate(overall['line_rate'])} |"
        ),
        (
            f"| Changed lines | {diff['covered_lines']} | "
            f"{diff['coverable_lines']} | {format_rate(diff['line_rate'])} |"
        ),
        "",
        f"Diff base: `{base_ref or 'N/A'}`",
        f"Diff head: `{head_ref}`",
        f"Changed lines in diff: `{diff['changed_lines']}`",
    ]

    non_coverable = diff["changed_lines"] - diff["coverable_lines"]
    lines.append(f"Changed lines without LCOV data: `{non_coverable}`")

    if diff["uncovered_lines"]:
        lines.extend(["", "## Uncovered Changed Lines", ""])
        for source_file, uncovered_lines in sorted(diff["uncovered_lines"].items()):
            lines.append(f"- `{source_file}`: {format_line_ranges(uncovered_lines)}")

    lines.append("")
    return "\n".join(lines)


def write_outputs(output_dir, overall, diff, base_ref, head_ref):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall": overall,
        "diff": {
            "base_ref": base_ref,
            "head_ref": head_ref,
            **diff,
        },
    }
    (output_dir / "coverage-summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "coverage-summary.md").write_text(
        render_markdown(overall, diff, base_ref, head_ref),
        encoding="utf-8",
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcov", required=True, type=Path, help="Path to LCOV .dat/.info file")
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--base-ref", default="", help="Base ref for incremental coverage")
    parser.add_argument("--head-ref", default="HEAD", help="Head ref for incremental coverage")
    parser.add_argument("--output-dir", type=Path, default=Path("coverage"), help="Output directory")
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Only include files whose repository-relative path starts with this prefix",
    )
    parser.add_argument(
        "--fail-under-overall",
        type=float,
        default=None,
        help="Fail if overall line coverage is below this percentage",
    )
    parser.add_argument(
        "--fail-under-diff",
        type=float,
        default=None,
        help="Fail if changed-line coverage is below this percentage",
    )
    return parser.parse_args(argv)


def check_threshold(name, rate, threshold):
    if threshold is None or rate is None:
        return True
    return rate * 100 >= threshold


def main(argv):
    args = parse_args(argv)
    include_prefixes = [prefix.lstrip("./") for prefix in args.include_prefix]

    coverage = parse_lcov(args.lcov, args.workspace, include_prefixes)
    overall = summarize_coverage(coverage)

    diff_text = git_diff(args.base_ref, args.head_ref) if args.base_ref else ""
    changed_lines = parse_unified_diff(diff_text, include_prefixes)
    diff = summarize_diff_coverage(coverage, changed_lines)

    write_outputs(args.output_dir, overall, diff, args.base_ref, args.head_ref)

    print(render_markdown(overall, diff, args.base_ref, args.head_ref))

    ok = True
    if not check_threshold("overall", overall["line_rate"], args.fail_under_overall):
        print(
            f"overall coverage {format_rate(overall['line_rate'])} is below "
            f"{args.fail_under_overall:.2f}%",
            file=sys.stderr,
        )
        ok = False
    if not check_threshold("diff", diff["line_rate"], args.fail_under_diff):
        print(
            f"diff coverage {format_rate(diff['line_rate'])} is below "
            f"{args.fail_under_diff:.2f}%",
            file=sys.stderr,
        )
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
