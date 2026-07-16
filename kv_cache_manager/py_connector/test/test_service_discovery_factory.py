# -*- coding: utf-8 -*-
"""Unit tests for service_discovery_factory.create_service_discovery."""

import sys
import unittest
from types import ModuleType
try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

from kv_cache_manager.py_connector.common.service_discovery_factory import (
    _parse_url,
    _get_int_param,
    create_service_discovery,
)


_SPECTRUM_MODULE = (
    "stub_source.kv_cache_manager.py_connector.common.spectrum_service_discovery"
)


class _FakeSpectrumServiceDiscovery:
    """Factory test double that does not depend on an internal implementation."""

    def __init__(
        self,
        virtual_service_id,
        *,
        cache_ttl=30,
        refresh_timeout=5,
        retry_count=0,
        custom_port=0,
    ):
        self.virtual_service_id = virtual_service_id
        self.cache_ttl = cache_ttl
        self.refresh_timeout = refresh_timeout
        self.retry_count = retry_count
        self.custom_port = custom_port

    def get_type(self):
        return "Spectrum"

    def close(self):
        return None


def _fake_spectrum_module():
    module = ModuleType(_SPECTRUM_MODULE)
    module.SpectrumServiceDiscovery = _FakeSpectrumServiceDiscovery
    return module


class TestParseUrl(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(_parse_url(""))

    def test_missing_scheme_returns_none(self):
        self.assertIsNone(_parse_url("not-a-url"))

    def test_empty_body_returns_none(self):
        self.assertIsNone(_parse_url("static://"))

    def test_static_no_query(self):
        scheme, body, params = _parse_url("static://10.0.0.1:8080,10.0.0.2:9090")
        self.assertEqual(scheme, "static")
        self.assertEqual(body, "10.0.0.1:8080,10.0.0.2:9090")
        self.assertEqual(params, {})

    def test_spectrum_with_query(self):
        scheme, body, params = _parse_url(
            "spectrum://v-ad2d143d?cache_time=30&retry_time=3&timeout=5000"
        )
        self.assertEqual(scheme, "spectrum")
        self.assertEqual(body, "v-ad2d143d")
        self.assertEqual(
            params,
            {"cache_time": "30", "retry_time": "3", "timeout": "5000"},
        )

    def test_vipserver_with_single_param(self):
        scheme, body, params = _parse_url("vipserver://my.domain.vipserver?timeout=10")
        self.assertEqual(scheme, "vipserver")
        self.assertEqual(body, "my.domain.vipserver")
        self.assertEqual(params, {"timeout": "10"})


class TestGetIntParam(unittest.TestCase):
    def test_default_when_missing(self):
        self.assertEqual(_get_int_param({}, "k", 7), 7)

    def test_default_when_empty(self):
        self.assertEqual(_get_int_param({"k": ""}, "k", 7), 7)

    def test_parse_int(self):
        self.assertEqual(_get_int_param({"k": "42"}, "k", 7), 42)

    def test_default_when_garbage(self):
        self.assertEqual(_get_int_param({"k": "abc"}, "k", 7), 7)


class TestCreateServiceDiscovery(unittest.TestCase):
    """工厂行为的端到端验证。"""

    def test_empty_url_returns_none(self):
        self.assertIsNone(create_service_discovery(""))

    def test_invalid_url_returns_none(self):
        self.assertIsNone(create_service_discovery("not-a-url"))

    def test_unknown_scheme_returns_none(self):
        self.assertIsNone(create_service_discovery("nacos://my.svc"))

    def test_vipserver_unimplemented(self):
        # py_connector 暂无 VIPServer 实现，应当返回 None 而不是抛异常
        self.assertIsNone(create_service_discovery("vipserver://pace.meta.vipserver"))

    def test_static_url_success(self):
        discovery = create_service_discovery("static://10.0.0.1:8080,10.0.0.2:9090")
        self.assertIsNotNone(discovery)
        self.assertEqual(discovery.get_type(), "Static")
        endpoints = discovery.get_all_endpoints()
        self.assertEqual(len(endpoints), 2)
        self.assertEqual(endpoints[0].host, "10.0.0.1:8080")
        self.assertEqual(endpoints[1].host, "10.0.0.2:9090")
        discovery.close()

    def test_static_url_malformed_returns_none(self):
        # 缺 port
        self.assertIsNone(create_service_discovery("static://10.0.0.1"))
        # port 非数字
        self.assertIsNone(create_service_discovery("static://10.0.0.1:abc"))

    def test_spectrum_url_passes_config_to_discovery(self):
        with patch.dict(sys.modules, {_SPECTRUM_MODULE: _fake_spectrum_module()}):
            discovery = create_service_discovery(
                "spectrum://v-ad2d143d?cache_time=10&retry_time=2&timeout=3000"
            )

        self.assertIsNotNone(discovery)
        self.assertEqual(discovery.get_type(), 'Spectrum')
        self.assertEqual(discovery.virtual_service_id, 'v-ad2d143d')
        self.assertEqual(discovery.cache_ttl, 10)
        self.assertEqual(discovery.retry_count, 2)
        self.assertEqual(discovery.refresh_timeout, 3)  # 3000 ms → 3 s
        discovery.close()

    def test_spectrum_empty_vsid_returns_none(self):
        # body 为空时 _parse_url 即拒绝
        self.assertIsNone(create_service_discovery("spectrum://"))

    def test_spectrum_url_passes_custom_port(self):
        with patch.dict(sys.modules, {_SPECTRUM_MODULE: _fake_spectrum_module()}):
            discovery = create_service_discovery(
                "spectrum://v-port?cache_time=10&retry_time=2&timeout=3000&port=12348"
            )

        self.assertIsNotNone(discovery)
        self.assertEqual(discovery.custom_port, 12348)
        discovery.close()

    def test_spectrum_body_is_forwarded_to_implementation(self):
        with patch.dict(sys.modules, {_SPECTRUM_MODULE: _fake_spectrum_module()}):
            discovery = create_service_discovery(
                "spectrum://virtual-service:opaque-suffix?cache_time=10"
            )

        self.assertIsNotNone(discovery)
        self.assertEqual(
            discovery.virtual_service_id,
            "virtual-service:opaque-suffix",
        )
        self.assertEqual(discovery.custom_port, 0)
        discovery.close()

    def test_spectrum_url_uses_default_port_override(self):
        with patch.dict(sys.modules, {_SPECTRUM_MODULE: _fake_spectrum_module()}):
            discovery = create_service_discovery(
                "spectrum://v-no-port?cache_time=10&retry_time=2&timeout=3000"
            )

        self.assertIsNotNone(discovery)
        self.assertEqual(discovery.custom_port, 0)
        discovery.close()


if __name__ == '__main__':
    unittest.main()
