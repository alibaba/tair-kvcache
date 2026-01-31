import abc
from typing import Dict
from testlib.test_base import TestBase
import unittest
import json

class DebugServiceClientBase(abc.ABC):
    @abc.abstractmethod
    def inject_fault(self, data, check_response=True) -> Dict:
        """inject fault with the service"""
        return {}
    @abc.abstractmethod
    def remove_fault(self, data, check_response=True) -> Dict:
        """remove fault with the service"""
        return {} 
    @abc.abstractmethod
    def clear_faults(self, data, check_response=True) -> Dict:
        """clear all faults with the service"""
        return {}

    @abc.abstractmethod
    def close(self):
        pass


class DebugServiceTestBase(abc.ABC, TestBase, unittest.TestCase):
    @abc.abstractmethod
    def _get_manager_client(self) -> DebugServiceClientBase:
        pass

    def setUp(self):
        self.init_default()
        # Default to HTTP client, but can be overridden in subclasses
        self._client: DebugServiceClientBase = self._get_manager_client()
        self._instance_id = "instance1"
        self._trace_id = "test_trace_id"

    def tearDown(self):
        self._client.close()
        self.cleanup()

    def test_basic_smoke(self):
        # InjectFault, ALWAYS
        inject_fault_data_1 = {
            "api_name": "StartWriteCache",
            "fault_type": "INTERNAL_ERROR",
            "fault_trigger_strategy": "ALWAYS",
        }
        inject_resp1 = self._client.inject_fault(inject_fault_data_1)

        self.assertIn("header", inject_resp1)
        self.assertEqual(inject_resp1["header"]["status"]["code"], "OK")
        print(inject_resp1)
        # InjectFault, ONCE
        inject_data_2 = {
            "api_name": "StartWriteCache",
            "fault_type": "INTERNAL_ERROR",
            "fault_trigger_strategy": "ONCE",
            "trigger_at_call":3,
        }
        inject_resp2 = self._client.inject_fault(inject_data_2)
        self.assertIn("header", inject_resp2)
        self.assertEqual(inject_resp2["header"]["status"]["code"], "OK")

        # RemoveFault
        remove_data_1 = {
            "api_name": "StartWriteCache",
        }
        remove_resp1 = self._client.remove_fault(remove_data_1)
        self.assertIn("header", remove_resp1)
        self.assertEqual(remove_resp1["header"]["status"]["code"], "OK")

        #ClearFaults
        clear_data = {}
        clear_resp = self._client.clear_faults(clear_data)
        self.assertIn("header", clear_resp)
        self.assertEqual(clear_resp["header"]["status"]["code"], "OK")



