# -*- coding: utf-8 -*-
"""Unit tests for SpectrumServiceDiscovery."""

import unittest
try:
    from unittest.mock import MagicMock, patch
except ImportError:
    from mock import MagicMock, patch
import time

from kv_cache_manager.py_connector.common.spectrum_service_discovery import (
    SpectrumServiceDiscovery, ServiceEndpoint,
    SPECTRUM_GATEWAY_BASE_URL,
)


VSID = "v-ad2d143d"
EXPECTED_URL = f"{SPECTRUM_GATEWAY_BASE_URL}/api/v1/discovery/virtual-services/{VSID}/instances"


def _make_mock_response(json_data, status_code=200):
    """创建 mock 响应"""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    return mock_resp


class TestSpectrumServiceDiscovery(unittest.TestCase):
    """Spectrum 服务发现测试"""

    @patch('kv_cache_manager.py_connector.common.spectrum_service_discovery.requests.Session')
    def test_init_and_refresh(self, mock_session_class):
        """测试初始化和刷新"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = _make_mock_response({
            'virtual_service_id': VSID,
            'instances': [
                {
                    'ip': '172.1.2.10', 'port': 8080,
                    'name': 'ds-abdedesd-ad2d-sded',
                    'physical_service_id': 'abdedesd',
                },
                {
                    'ip': '172.1.2.11', 'port': 8080,
                    'name': 'ds-anothername',
                    'physical_service_id': 'abdedesd',
                },
            ],
        })
        mock_session.get.return_value = mock_response

        client = SpectrumServiceDiscovery(VSID)

        endpoints = client.get_all_endpoints()
        self.assertEqual(len(endpoints), 2)
        self.assertEqual(endpoints[0].ip, '172.1.2.10')
        self.assertEqual(endpoints[0].port, 8080)
        self.assertEqual(endpoints[0].host, '172.1.2.10:8080')
        self.assertEqual(endpoints[0].weight, 100)

        # 校验 HTTP 调用走的是固定的 Spectrum 网关 URL
        called_url = mock_session.get.call_args[0][0]
        self.assertEqual(called_url, EXPECTED_URL)

        client.close()

    @patch('kv_cache_manager.py_connector.common.spectrum_service_discovery.requests.Session')
    def test_cache_ttl(self, mock_session_class):
        """测试缓存过期机制"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = _make_mock_response({
            'virtual_service_id': VSID,
            'instances': [
                {'ip': '172.1.2.10', 'port': 8080},
            ],
        })
        mock_session.get.return_value = mock_response

        client = SpectrumServiceDiscovery(VSID, cache_ttl=1)

        endpoints1 = client.get_all_endpoints()
        self.assertEqual(len(endpoints1), 1)

        client._cache_time = time.time() - 2

        endpoints2 = client.get_all_endpoints()
        self.assertEqual(len(endpoints2), 1)

        # 一次初始化 + 一次过期刷新 = 2 次
        self.assertEqual(mock_session.get.call_count, 2)

        client.close()

    @patch('kv_cache_manager.py_connector.common.spectrum_service_discovery.requests.Session')
    def test_get_one_endpoint(self, mock_session_class):
        """测试获取单个节点"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = _make_mock_response({
            'virtual_service_id': VSID,
            'instances': [
                {'ip': '172.1.2.10', 'port': 8080},
                {'ip': '172.1.2.11', 'port': 8080},
            ],
        })
        mock_session.get.return_value = mock_response

        client = SpectrumServiceDiscovery(VSID)

        endpoint = client.get_one_endpoint()
        self.assertIsNotNone(endpoint)
        self.assertIn(endpoint.ip, ['172.1.2.10', '172.1.2.11'])

        client.close()

    @patch('kv_cache_manager.py_connector.common.spectrum_service_discovery.requests.Session')
    def test_empty_instances(self, mock_session_class):
        """测试空实例列表"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = _make_mock_response({
            'virtual_service_id': VSID,
            'instances': [],
        })
        mock_session.get.return_value = mock_response

        client = SpectrumServiceDiscovery(VSID)

        endpoints = client.get_all_endpoints()
        self.assertEqual(len(endpoints), 0)

        endpoint = client.get_one_endpoint()
        self.assertIsNone(endpoint)

        client.close()

    @patch('kv_cache_manager.py_connector.common.spectrum_service_discovery.requests.Session')
    def test_refresh_failure(self, mock_session_class):
        """测试刷新失败"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_session.get.side_effect = Exception("Connection error")

        client = SpectrumServiceDiscovery(VSID)

        endpoints = client.get_all_endpoints()
        self.assertEqual(len(endpoints), 0)

        result = client.refresh()
        self.assertFalse(result)

        client.close()

    @patch('kv_cache_manager.py_connector.common.spectrum_service_discovery.requests.Session')
    def test_http_error(self, mock_session_class):
        """测试 HTTP 错误响应"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        import requests
        mock_session.get.side_effect = requests.HTTPError("404 Not Found")

        client = SpectrumServiceDiscovery(VSID)

        endpoints = client.get_all_endpoints()
        self.assertEqual(len(endpoints), 0)

        client.close()

    @patch('kv_cache_manager.py_connector.common.spectrum_service_discovery.requests.Session')
    def test_invalid_json_response(self, mock_session_class):
        """测试缺少 instances 字段的响应"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = _make_mock_response({
            'virtual_service_id': VSID,
        })
        mock_session.get.return_value = mock_response

        client = SpectrumServiceDiscovery(VSID)

        endpoints = client.get_all_endpoints()
        self.assertEqual(len(endpoints), 0)

        client.close()

    @patch('kv_cache_manager.py_connector.common.spectrum_service_discovery.requests.Session')
    def test_context_manager(self, mock_session_class):
        """测试 context manager 支持"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_response = _make_mock_response({
            'virtual_service_id': VSID,
            'instances': [
                {'ip': '172.1.2.10', 'port': 8080},
            ],
        })
        mock_session.get.return_value = mock_response

        with SpectrumServiceDiscovery(VSID) as client:
            endpoints = client.get_all_endpoints()
            self.assertEqual(len(endpoints), 1)

        mock_session.close.assert_called_once()

    @patch('kv_cache_manager.py_connector.common.spectrum_service_discovery.requests.Session')
    def test_manual_refresh(self, mock_session_class):
        """测试手动刷新"""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        call_count = [0]

        def mock_get(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_mock_response({
                    'virtual_service_id': VSID,
                    'instances': [
                        {'ip': '172.1.2.10', 'port': 8080},
                    ],
                })
            else:
                return _make_mock_response({
                    'virtual_service_id': VSID,
                    'instances': [
                        {'ip': '172.1.2.10', 'port': 8080},
                        {'ip': '172.1.2.11', 'port': 8080},
                    ],
                })

        mock_session.get.side_effect = mock_get

        client = SpectrumServiceDiscovery(VSID, auto_refresh=False)

        endpoints1 = client.get_all_endpoints()
        self.assertEqual(len(endpoints1), 1)

        result = client.refresh()
        self.assertTrue(result)

        endpoints2 = client.get_all_endpoints()
        self.assertEqual(len(endpoints2), 2)

        client.close()

    def test_empty_vsid_rejected(self):
        """空 virtual_service_id 应当直接报错"""
        with self.assertRaises(ValueError):
            SpectrumServiceDiscovery("")


if __name__ == '__main__':
    unittest.main()
