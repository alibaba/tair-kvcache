#!/usr/bin/env python3

import tempfile
import textwrap
import unittest
from pathlib import Path

import coverage_report


class CoverageReportTest(unittest.TestCase):
    def test_lcov_and_diff_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            source = workspace / "kv_cache_manager" / "manager" / "cache_manager.cc"
            source.parent.mkdir(parents=True)
            source.write_text("placeholder\n", encoding="utf-8")

            lcov = workspace / "coverage.dat"
            lcov.write_text(
                textwrap.dedent(
                    f"""\
                    TN:
                    SF:{source}
                    DA:10,1
                    DA:11,0
                    DA:12,3
                    end_of_record
                    """
                ),
                encoding="utf-8",
            )

            coverage = coverage_report.parse_lcov(
                lcov,
                workspace,
                ["kv_cache_manager/"],
            )
            self.assertEqual(
                coverage,
                {"kv_cache_manager/manager/cache_manager.cc": {10: 1, 11: 0, 12: 3}},
            )

            diff = coverage_report.parse_unified_diff(
                textwrap.dedent(
                    """\
                    diff --git a/kv_cache_manager/manager/cache_manager.cc b/kv_cache_manager/manager/cache_manager.cc
                    --- a/kv_cache_manager/manager/cache_manager.cc
                    +++ b/kv_cache_manager/manager/cache_manager.cc
                    @@ -9,0 +10,3 @@
                    +line 10
                    +line 11
                    +line 12
                    diff --git a/docs/develop/README.md b/docs/develop/README.md
                    --- a/docs/develop/README.md
                    +++ b/docs/develop/README.md
                    @@ -1,0 +2,1 @@
                    +ignored
                    """
                ),
                ["kv_cache_manager/"],
            )
            self.assertEqual(
                diff,
                {"kv_cache_manager/manager/cache_manager.cc": {10, 11, 12}},
            )

            overall = coverage_report.summarize_coverage(coverage)
            self.assertEqual(overall["covered_lines"], 2)
            self.assertEqual(overall["coverable_lines"], 3)

            diff_summary = coverage_report.summarize_diff_coverage(coverage, diff)
            self.assertEqual(diff_summary["covered_lines"], 2)
            self.assertEqual(diff_summary["coverable_lines"], 3)
            self.assertEqual(
                diff_summary["uncovered_lines"],
                {"kv_cache_manager/manager/cache_manager.cc": [11]},
            )

    def test_empty_diff_rate_is_not_applicable(self):
        diff_summary = coverage_report.summarize_diff_coverage({}, {})
        self.assertIsNone(diff_summary["line_rate"])
        self.assertEqual(diff_summary["changed_lines"], 0)

    def test_lcov_duplicate_source_records_are_accumulated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            lcov = workspace / "coverage.dat"
            lcov.write_text(
                textwrap.dedent(
                    """\
                    TN:
                    SF:kv_cache_manager/common/env_util.cc
                    DA:7,1
                    DA:8,0
                    end_of_record
                    SF:kv_cache_manager/common/env_util.cc
                    DA:7,2
                    DA:8,3
                    end_of_record
                    """
                ),
                encoding="utf-8",
            )

            coverage = coverage_report.parse_lcov(lcov, workspace, ["kv_cache_manager/"])

        self.assertEqual(coverage["kv_cache_manager/common/env_util.cc"], {7: 3, 8: 3})

    def test_lcov_negative_line_hits_are_clamped_to_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            lcov = workspace / "coverage.dat"
            lcov.write_text(
                textwrap.dedent(
                    """\
                    TN:
                    SF:kv_cache_manager/common/env_util.cc
                    DA:7,-7
                    end_of_record
                    """
                ),
                encoding="utf-8",
            )

            coverage = coverage_report.parse_lcov(lcov, workspace, ["kv_cache_manager/"])

        self.assertEqual(coverage["kv_cache_manager/common/env_util.cc"], {7: 0})


if __name__ == "__main__":
    unittest.main()
