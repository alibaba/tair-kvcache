import os
import unittest
from unittest.mock import patch

import requests

from integration_test.admin_service.http_interface_test import AdminServiceHttpClient
from integration_test.testlib.test_base import TestBase


class PrometheusCommonTagsTest(TestBase, unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {
            'KVCM_TEST_METRICS_WORKSPACE': 'workspace_a',
            'KVCM_TEST_METRICS_DEPLOYMENT': 'dep\\a"b\nc',
            'KVCM_TEST_METRICS_NODE': 'node_a',
            'KVCM_TEST_METRICS_EMPTY': '',
            'kvcm_metrics_env_tag_mappings': (
                'workspace=KVCM_TEST_METRICS_WORKSPACE,'
                'deployment_name=KVCM_TEST_METRICS_DEPLOYMENT,'
                'node.name=KVCM_TEST_METRICS_NODE,'
                'empty=KVCM_TEST_METRICS_EMPTY,missing=KVCM_TEST_METRICS_UNSET'
            ),
        })
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('KVCM_TEST_METRICS_UNSET', None)
        self.clean_workdir()
        self.prepare_test_resource(1)
        self.addCleanup(self.cleanup)
        self.start_worker(**{
            'kvcm.service.io_thread_num': 2,
            'kvcm.metrics.reporter_type': 'local',
            'kvcm.metrics.report_interval_ms': 100,
        })
        port = self.worker_manager.get_worker(0).env.admin_http_port
        self._url = f'http://localhost:{port}'
        self._client = AdminServiceHttpClient(self._url)
        self.addCleanup(self._client.close)

    def test_common_tags_are_exported_without_changing_admin_metrics(self):
        # A successful API call produces service metrics with their own tags.
        self._client.get_metrics({'trace_id': 'common_tags_first'})
        response = requests.get(self._url + '/metrics', timeout=5)
        response.raise_for_status()
        self.assertIn('text/plain', response.headers['Content-Type'])
        samples = [line for line in response.text.splitlines() if line and not line.startswith('#')]
        self.assertTrue(samples, response.text)
        self.assertTrue(any('api_name="GetMetrics"' in line for line in samples), response.text)
        for line in samples:
            self.assertIn('workspace="workspace_a"', line)
            self.assertIn(r'deployment_name="dep\\a\"b\nc"', line)
            self.assertIn('node_name="node_a"', line)
            self.assertNotIn('empty=', line)
            self.assertNotIn('missing=', line)

        metrics = self._client.get_metrics({'trace_id': 'common_tags_second'})['metrics']
        self.assertTrue(metrics)
        for metric in metrics:
            names = {tag['tag_key'] for tag in metric.get('metric_tags', [])}
            self.assertTrue(names.isdisjoint({
                'workspace', 'deployment_name', 'node.name', 'node_name', 'empty', 'missing'
            }), metric)


if __name__ == '__main__':
    unittest.main()
