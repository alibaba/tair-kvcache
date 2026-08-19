"""Scenario evaluator: turns a Swarm report plus expectations into PASS/FAIL.

The evaluator lives outside the generator process. It consumes only the report
and the expectations file, and it fails closed: a missing field, an empty
report, a NOT_RUN or INCONCLUSIVE required check, or a saturated generator all
make the scenario fail rather than silently pass.
"""
import json


NON_GATING_STATUSES = ("PASS",)


class Evaluation(object):
    def __init__(self):
        self.failures = []
        self.notes = []

    @property
    def ok(self):
        return not self.failures

    def fail(self, message):
        self.failures.append(message)

    def note(self, message):
        self.notes.append(message)

    def describe(self):
        lines = []
        if self.failures:
            lines.append("FAILURES:")
            lines.extend("  - %s" % failure for failure in self.failures)
        if self.notes:
            lines.append("NOTES:")
            lines.extend("  - %s" % note for note in self.notes)
        return "\n".join(lines)


def load_expectations(path):
    with open(path) as handle:
        return json.load(handle)


def _get(report, path, evaluation):
    node = report
    for key in path.split("."):
        if isinstance(node, list):
            if not key.isdigit() or int(key) >= len(node):
                evaluation.fail("report is missing required field '%s'" % path)
                return None
            node = node[int(key)]
            continue
        if not isinstance(node, dict) or key not in node:
            evaluation.fail("report is missing required field '%s'" % path)
            return None
        node = node[key]
    return node


def evaluate(run, expectations):
    """`run` is a runner.SwarmRun; returns an Evaluation."""
    evaluation = Evaluation()

    expected_exit = expectations.get("require_exit_code", 0)
    if run.exit_code != expected_exit:
        evaluation.fail("generator exit code %s, expected %s" % (run.exit_code, expected_exit))
    if run.report is None:
        evaluation.fail("report is empty or missing, which is an execution failure")
        return evaluation

    report = run.report

    for section in expectations.get("require_report_sections", []):
        if section not in report:
            evaluation.fail("report is missing section '%s'" % section)

    if _get(report, "invariants.violations_log_failed", evaluation) is True:
        evaluation.fail("the violation log could not be written, so evidence is incomplete")

    if expectations.get("forbid_generator_saturation", True):
        saturated = _get(report, "run.generator_saturated", evaluation)
        if saturated:
            reasons = report.get("run", {}).get("generator_saturation_reasons", [])
            evaluation.fail("generator saturated (%s): this run is not a valid KVCM capacity sample"
                            % ", ".join(reasons))
    else:
        evaluation.note("generator saturation is not gated for this scenario")

    if expectations.get("require_preflight_pass", True):
        if not _get(report, "cleanup.preflight.passed", evaluation):
            evaluation.fail("preflight did not pass: stage=%s detail=%s"
                            % (report.get("cleanup", {}).get("preflight", {}).get("failure_stage"),
                               report.get("cleanup", {}).get("preflight", {}).get("failure_detail")))

    if expectations.get("require_drain_complete", True):
        if not _get(report, "run.drain_complete", evaluation):
            evaluation.fail("drain did not complete")
        if not _get(report, "run.quiesced", evaluation):
            evaluation.fail("the generator did not quiesce: asynchronous state was still live at report time")

    # ---- RPC level gates ----
    # One bucket per (behavior, api, phase, lane): collapsing them would hide a
    # failure that only happens on one behavior or one lane.
    aggregates = {}
    for entry in report.get("rpc", {}).get("by_api_phase", []):
        key = (entry.get("behavior_id", ""), entry["api"], entry["phase"], entry.get("lane", ""))
        aggregates[key] = entry

    def api_totals(api, phases=None):
        total = 0
        success = 0
        for key, entry in aggregates.items():
            if key[1] != api:
                continue
            if phases and key[2] not in phases:
                continue
            total += entry["total"]
            success += entry["success"]
        return total, success

    for api, minimum in expectations.get("min_rpc_samples", {}).items():
        total, _ = api_totals(api, expectations.get("rpc_sample_phases"))
        if total < minimum:
            evaluation.fail("API %s produced %d samples, expected at least %d" % (api, total, minimum))

    for api, threshold in expectations.get("min_success_rate", {}).items():
        if api == "__all__":
            total = report.get("rpc", {}).get("total", 0)
            success = report.get("rpc", {}).get("success", 0)
        else:
            total, success = api_totals(api)
        if total == 0:
            evaluation.fail("API %s has no sample, so its success rate cannot be judged" % api)
            continue
        rate = float(success) / float(total)
        if rate < threshold:
            detail = []
            for key, entry in sorted(aggregates.items()):
                if entry["success"] == entry["total"]:
                    continue
                if api != "__all__" and key[1] != api:
                    continue
                detail.append("%s %s/%s lane=%s %d/%d transport=%s service=%s"
                              % (key[0], key[1], key[2], key[3], entry["success"], entry["total"],
                                 entry.get("transport_errors"), entry.get("service_statuses")))
            evaluation.fail("API %s success rate %.4f below threshold %.4f (%d/%d); failing buckets: %s"
                            % (api, rate, threshold, success, total, "; ".join(detail) or "none"))

    for api, limit in expectations.get("max_latency_p99_ms", {}).items():
        worst = None
        for key, entry in aggregates.items():
            if key[1] != api:
                continue
            value = entry["latency"]["p99_ms"]
            worst = value if worst is None else max(worst, value)
        if worst is None:
            evaluation.fail("API %s has no latency sample" % api)
        elif worst > limit:
            evaluation.fail("API %s p99 latency %.2fms exceeds %.2fms" % (api, worst, limit))

    for api in expectations.get("require_apis_exercised", []):
        total, success = api_totals(api)
        if total == 0:
            evaluation.fail("API %s was never exercised on this transport" % api)
        elif success == 0:
            evaluation.fail("API %s never succeeded" % api)

    # ---- contract gates ----
    checks = {}
    for check in report.get("invariants", {}).get("checks", []):
        checks[check["check_name"]] = check
    for name, requirement in expectations.get("required_checks", {}).items():
        check = checks.get(name)
        if check is None:
            evaluation.fail("required contract %s is missing from the report" % name)
            continue
        if check["status"] != "PASS":
            evaluation.fail("required contract %s is %s (checked=%s violations=%s): %s"
                            % (name, check["status"], check["checked"], check["violations"], check["reason"]))
            continue
        if check["violations"] != 0:
            evaluation.fail("required contract %s reported %s violations" % (name, check["violations"]))
        minimum = requirement.get("min_checked", 1)
        if check["checked"] < minimum:
            evaluation.fail("required contract %s only checked %s samples, expected at least %s"
                            % (name, check["checked"], minimum))
        for counter, counter_min in (requirement.get("min_counters") or {}).items():
            value = check.get("counters", {}).get(counter)
            if value is None:
                evaluation.fail("contract %s is missing counter '%s'" % (name, counter))
            elif value < counter_min:
                evaluation.fail("contract %s counter %s is %s, expected at least %s"
                                % (name, counter, value, counter_min))

    for name in expectations.get("non_gating_checks", []):
        check = checks.get(name)
        if check is None:
            evaluation.fail("non-gating contract %s is missing from the report" % name)
        else:
            evaluation.note("non-gating contract %s: %s (%s)" % (name, check["status"], check["reason"]))

    # ---- behavior facts ----
    for path, minimum in expectations.get("min_report_values", {}).items():
        value = _get(report, path, evaluation)
        if value is None:
            continue
        if value < minimum:
            evaluation.fail("report value %s is %s, expected at least %s" % (path, value, minimum))
    for path, maximum in expectations.get("max_report_values", {}).items():
        value = _get(report, path, evaluation)
        if value is None:
            continue
        if value > maximum:
            evaluation.fail("report value %s is %s, expected at most %s" % (path, value, maximum))
    for path, expected in expectations.get("exact_report_values", {}).items():
        value = _get(report, path, evaluation)
        if value is None:
            continue
        if value != expected:
            evaluation.fail("report value %s is %s, expected %s" % (path, value, expected))

    if expectations.get("forbid_violation_log_entries", True):
        gated = set(expectations.get("required_checks", {}).keys())
        for entry in run.violations:
            if entry.get("check") in gated:
                evaluation.fail("violation logged for gated contract %s: %s"
                                % (entry.get("check"), json.dumps(entry.get("detail"))[:400]))

    return evaluation
