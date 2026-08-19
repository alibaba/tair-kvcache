"""Shared base for the Swarm integration tests.

Each test starts a real KVCM worker, creates an isolated instance group and
event-report storage, runs the real generator binary against it, evaluates the
report out of process and tears the environment down.
"""
import json
import logging
import os
import unittest

from testlib.test_base import TestBase

from integration_test.swarm import evaluator, fixture, runner


SCENARIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")
EXPECTATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expectations")


class SwarmScenarioTest(TestBase, unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.INFO)
        self.clean_workdir()
        self.prepare_test_resource(1)
        self.start_worker()
        env = self.envs[0]
        self.meta_http = "http://%s:%d" % (env.ip, int(env.http_port))
        self.meta_grpc = "%s:%d" % (env.ip, int(env.rpc_port))
        self.admin_http = "http://%s:%d" % (env.ip, int(env.admin_http_port))
        self.swarm_workdir = os.path.join(self.workdir, "swarm")
        self.fixture = None

    def tearDown(self):
        if self.fixture is not None:
            notes = self.fixture.teardown()
            logging.info("swarm fixture teardown: %s", notes)
        self.cleanup()

    def make_fixture(self, name_hint="swarm", quota_bytes=2 * 1024 * 1024 * 1024):
        self.fixture = fixture.SwarmFixture(self.meta_http,
                                            self.meta_grpc,
                                            self.admin_http,
                                            self.swarm_workdir,
                                            quota_bytes=quota_bytes,
                                            name_hint=name_hint)
        self.fixture.setup()
        return self.fixture

    def run_scenario(self, scenario, expectations=None, overrides=None, timeout_seconds=600,
                     name_hint=None, transport_override=None):
        fixture_instance = self.fixture or self.make_fixture(name_hint=name_hint or scenario)
        config_path = fixture_instance.render_config(
            os.path.join(SCENARIO_DIR, scenario + ".json"),
            overrides=overrides,
            transport_override=transport_override,
        )
        run = runner.run_swarm(config_path, timeout_seconds=timeout_seconds)
        logging.info("swarm run finished: %s", run.describe()[:4000])
        if expectations is None:
            return run
        expected = evaluator.load_expectations(os.path.join(EXPECTATION_DIR, expectations + ".json"))
        evaluation = evaluator.evaluate(run, expected)
        if not evaluation.ok:
            report_dump = json.dumps(run.report, indent=1)[:20000] if run.report else "<no report>"
            self.fail("scenario %s failed evaluation:\n%s\n\n--- run ---\n%s\n\n--- report ---\n%s"
                      % (scenario, evaluation.describe(), run.describe(), report_dump))
        logging.info("scenario %s evaluation notes:\n%s", scenario, evaluation.describe())
        return run
