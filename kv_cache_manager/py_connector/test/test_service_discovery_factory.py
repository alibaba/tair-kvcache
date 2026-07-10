# -*- coding: utf-8 -*-
"""Unit tests for the provider-neutral service discovery factory."""

import unittest

from kv_cache_manager.py_connector.common.service_discovery import (
    ServiceDiscovery,
    ServiceEndpoint,
)
from kv_cache_manager.py_connector.common.service_discovery_factory import (
    _parse_url,
    create_service_discovery,
    register_service_discovery_provider,
)


class FakeServiceDiscovery(ServiceDiscovery):
    def __init__(self, body, params):
        self.body = body
        self.params = params
        self.closed = False
        self.endpoint = ServiceEndpoint("127.0.0.1", 18080, "127.0.0.1:18080")

    def get_all_endpoints(self):
        return [self.endpoint]

    def get_one_endpoint(self):
        return self.endpoint

    def refresh(self):
        return True

    def get_type(self):
        return "Fake"

    def close(self):
        self.closed = True


class TestParseUrl(unittest.TestCase):
    def test_empty_returns_none(self):
        self.assertIsNone(_parse_url(""))

    def test_missing_scheme_returns_none(self):
        self.assertIsNone(_parse_url("not-a-url"))

    def test_empty_body_returns_none(self):
        self.assertIsNone(_parse_url("static://"))

    def test_static_no_query(self):
        scheme, body, params = _parse_url(
            "static://10.0.0.1:8080,10.0.0.2:9090"
        )
        self.assertEqual(scheme, "static")
        self.assertEqual(body, "10.0.0.1:8080,10.0.0.2:9090")
        self.assertEqual(params, {})

    def test_custom_provider_query(self):
        scheme, body, params = _parse_url(
            "custom://manager-service?timeout=5000&zone=test"
        )
        self.assertEqual(scheme, "custom")
        self.assertEqual(body, "manager-service")
        self.assertEqual(params, {"timeout": "5000", "zone": "test"})

    def test_malformed_query_returns_none(self):
        self.assertIsNone(_parse_url("custom://manager-service?timeout"))


class TestCreateServiceDiscovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_service_discovery_provider(
            "test-provider",
            lambda body, params: FakeServiceDiscovery(body, params),
        )

    def test_empty_url_returns_none(self):
        self.assertIsNone(create_service_discovery(""))

    def test_invalid_url_returns_none(self):
        self.assertIsNone(create_service_discovery("not-a-url"))

    def test_unknown_scheme_returns_none(self):
        self.assertIsNone(create_service_discovery("unknown://manager-service"))

    def test_static_url_success(self):
        discovery = create_service_discovery(
            "static://10.0.0.1:8080,10.0.0.2:9090"
        )
        self.assertIsNotNone(discovery)
        self.assertEqual(discovery.get_type(), "Static")
        endpoints = discovery.get_all_endpoints()
        self.assertEqual([endpoint.host for endpoint in endpoints], [
            "10.0.0.1:8080",
            "10.0.0.2:9090",
        ])

    def test_static_url_malformed_returns_none(self):
        self.assertIsNone(create_service_discovery("static://10.0.0.1"))
        self.assertIsNone(create_service_discovery("static://10.0.0.1:abc"))

    def test_registered_provider_receives_body_and_params(self):
        discovery = create_service_discovery(
            "test-provider://manager-service?timeout=5000"
        )
        self.assertIsInstance(discovery, FakeServiceDiscovery)
        self.assertEqual(discovery.body, "manager-service")
        self.assertEqual(discovery.params, {"timeout": "5000"})

    def test_duplicate_registration_is_rejected(self):
        with self.assertRaises(ValueError):
            register_service_discovery_provider(
                "test-provider",
                lambda body, params: FakeServiceDiscovery(body, params),
            )

    def test_invalid_registration_is_rejected(self):
        with self.assertRaises(ValueError):
            register_service_discovery_provider(
                "bad://scheme",
                lambda body, params: FakeServiceDiscovery(body, params),
            )


if __name__ == "__main__":
    unittest.main()
