"""Evaluator unit tests: it must fail closed on every incomplete evidence."""
import copy
import os
import unittest

from integration_test.swarm import evaluator


class _Run(object):
    def __init__(self, report, exit_code=0, violations=None):
        self.report = report
        self.exit_code = exit_code
        self.violations = violations or []
        self.stdout = ""
        self.stderr = ""
        self.wall_seconds = 1.0

    def describe(self):
        return "stub"


def _report(status="PASS", checked=10, violations=0):
    return {
        "run": {"generator_saturated": False, "generator_saturation_reasons": [], "drain_complete": True,
                "quiesced": True, "metadata_only": True},
        "run_config": {}, "phases": {}, "runtime": {}, "behaviors": {},
        "rpc": {"total": 10, "success": 10,
                "by_api_phase": [{"api": "CheckHealth", "phase": "steady", "lane": "control", "total": 10,
                                  "success": 10, "latency": {"p99_ms": 5.0}}]},
        "transport": {}, "cache": {},
        "invariants": {"violations_log_failed": False, "violations_total": violations,
                       "checks": [{"check_name": "C5_health_probe_bounded_response", "status": status,
                                   "checked": checked, "violations": violations, "reason": "r",
                                   "counters": {"probes": checked}}]},
        "workload_shape": {}, "usage_observations": {}, "limitations": ["metadata-only"],
        "cleanup": {"preflight": {"passed": True, "failure_stage": "", "failure_detail": ""}},
    }


BASE_EXPECTATIONS = {
    "require_exit_code": 0,
    "require_report_sections": ["run", "invariants", "cleanup"],
    "require_apis_exercised": ["CheckHealth"],
    "min_rpc_samples": {"CheckHealth": 5},
    "min_success_rate": {"__all__": 1.0},
    "max_latency_p99_ms": {"CheckHealth": 100},
    "required_checks": {"C5_health_probe_bounded_response": {"min_checked": 5,
                                                             "min_counters": {"probes": 5}}},
    "non_gating_checks": [],
}


class EvaluatorTest(unittest.TestCase):
    def test_healthy_report_passes(self):
        evaluation = evaluator.evaluate(_Run(_report()), BASE_EXPECTATIONS)
        self.assertTrue(evaluation.ok, evaluation.describe())

    def test_empty_report_fails_closed(self):
        evaluation = evaluator.evaluate(_Run(None), BASE_EXPECTATIONS)
        self.assertFalse(evaluation.ok)
        self.assertIn("report is empty", evaluation.describe())

    def test_not_run_and_inconclusive_required_checks_fail(self):
        for status in ("NOT_RUN", "INCONCLUSIVE", "FAIL"):
            evaluation = evaluator.evaluate(_Run(_report(status=status)), BASE_EXPECTATIONS)
            self.assertFalse(evaluation.ok, status)
            self.assertIn(status, evaluation.describe())

    def test_insufficient_samples_and_counters_fail(self):
        evaluation = evaluator.evaluate(_Run(_report(checked=1)), BASE_EXPECTATIONS)
        self.assertFalse(evaluation.ok)
        self.assertIn("only checked 1 samples", evaluation.describe())
        self.assertIn("counter probes", evaluation.describe())

    def test_generator_saturation_invalidates_the_sample(self):
        report = _report()
        report["run"]["generator_saturated"] = True
        report["run"]["generator_saturation_reasons"] = ["session_admission_rejected"]
        evaluation = evaluator.evaluate(_Run(report), BASE_EXPECTATIONS)
        self.assertFalse(evaluation.ok)
        self.assertIn("not a valid KVCM capacity sample", evaluation.describe())
        # Explicitly non-gated saturation is only a note.
        relaxed = copy.deepcopy(BASE_EXPECTATIONS)
        relaxed["forbid_generator_saturation"] = False
        evaluation = evaluator.evaluate(_Run(report), relaxed)
        self.assertTrue(evaluation.ok, evaluation.describe())

    def test_missing_field_and_missing_section_fail(self):
        report = _report()
        del report["run"]["quiesced"]
        evaluation = evaluator.evaluate(_Run(report), BASE_EXPECTATIONS)
        self.assertFalse(evaluation.ok)
        self.assertIn("missing required field 'run.quiesced'", evaluation.describe())

        report = _report()
        del report["cleanup"]
        evaluation = evaluator.evaluate(_Run(report), BASE_EXPECTATIONS)
        self.assertFalse(evaluation.ok)
        self.assertIn("missing section 'cleanup'", evaluation.describe())

    def test_failed_violation_log_fails_the_scenario(self):
        report = _report()
        report["invariants"]["violations_log_failed"] = True
        evaluation = evaluator.evaluate(_Run(report), BASE_EXPECTATIONS)
        self.assertFalse(evaluation.ok)
        self.assertIn("violation log could not be written", evaluation.describe())

    def test_violation_entry_for_a_gated_contract_fails(self):
        run = _Run(_report(), violations=[{"check": "C5_health_probe_bounded_response",
                                           "detail": {"reason": "slow"}}])
        evaluation = evaluator.evaluate(run, BASE_EXPECTATIONS)
        self.assertFalse(evaluation.ok)
        self.assertIn("violation logged for gated contract", evaluation.describe())

    def test_non_gating_check_must_still_be_present(self):
        expectations = copy.deepcopy(BASE_EXPECTATIONS)
        expectations["non_gating_checks"] = ["C4_server_metric_cross_check"]
        evaluation = evaluator.evaluate(_Run(_report()), expectations)
        self.assertFalse(evaluation.ok)
        self.assertIn("non-gating contract C4_server_metric_cross_check is missing", evaluation.describe())

    def test_shipped_expectation_files_are_self_consistent(self):
        directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expectations")
        names = sorted(name for name in os.listdir(directory) if name.endswith(".json"))
        self.assertTrue(names)
        for name in names:
            expectations = evaluator.load_expectations(os.path.join(directory, name))
            self.assertIn("required_checks", expectations, name)
            self.assertTrue(expectations["required_checks"], name)
            for check, requirement in expectations["required_checks"].items():
                self.assertGreaterEqual(requirement.get("min_checked", 1), 1,
                                        "%s: %s must require a real sample" % (name, check))
            self.assertTrue(expectations.get("forbid_generator_saturation", True), name)
            self.assertEqual(expectations.get("require_exit_code", 0), 0, name)


if __name__ == "__main__":
    unittest.main()
