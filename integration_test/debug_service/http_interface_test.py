import requests
import integration_test.debug_service.debug_interface_cases as cases


class DebugServiceHttpClient(cases.DebugServiceClientBase):
    """HTTP client for DebugService API endpoints"""

    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    def _make_request(self, method, endpoint, data=None):
        """Helper method to make HTTP requests to the service"""
        url = self.base_url + endpoint

        if method == 'POST':
            response = self.session.post(url, json=data, headers=self.headers)
        elif method == 'GET':
            response = self.session.get(url, params=data, headers=self.headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        return response

    def _make_api_request(self, endpoint, data=None, check_response=True):
        """Helper method to make POST requests to API endpoints and optionally validate response"""
        response = self._make_request('POST', endpoint, data)
        if response.status_code != 200:
            raise AssertionError(f"Request to {endpoint} failed with status code {response.status_code}")
        try:
            response_data = response.json()
        except ValueError as e:
            raise AssertionError(f"Response from {endpoint} is not valid JSON: {e}")
        if check_response:
            if 'header' not in response_data:
                raise AssertionError(f"Response from {endpoint} missing 'header' field")
            if response_data['header']['status']['code'] != "OK":
                raise AssertionError(
                    f"Request to {endpoint} failed with error: {response_data['header']['status']['message']}")
        return response_data

    def inject_fault(self, data, check_response=True):
        """inject fault with the service"""
        return self._make_api_request('/api/injectFault', data, check_response)

    def remove_fault(self, data, check_response=True):
        """remove fault with the service"""
        return self._make_api_request('/api/removeFault', data, check_response)

    def clear_faults(self, data, check_response=True):
        """clear all faults with the service"""
        return self._make_api_request('/api/clearFaults', data, check_response)

    def get_cache_meta_detail(self, data, check_response=True):
        return self._make_api_request('/api/getCacheMetaDetail', data, check_response)

    def close(self):
        """Close the HTTP session"""
        self.session.close()


class DebugServiceHttpTest(cases.DebugServiceTestBase):
    """HTTP version of the DebugService tests"""

    def _get_manager_client(self):
        self._http_port = self.worker_manager.get_worker(0).env.http_port 
        self._new_http_port = self._http_port + 3000
        self._http_url = "http://localhost:%d" % self._new_http_port # TODO: debug port is based on http + 3000（in service.cc）
        return DebugServiceHttpClient(self._http_url)

    def test_get_cache_meta_detail(self):
        meta_url = "http://localhost:%d" % self._http_port

        def meta_request(endpoint, data):
            response = requests.post(meta_url + endpoint, json=data)
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["header"]["status"]["code"], "OK", body)
            return body

        meta_request('/api/registerInstance', {
            "trace_id": self._trace_id,
            "instance_group": "default",
            "instance_id": self._instance_id,
            "block_size": 128,
            "model_deployment": {
                "model_name": "test_model",
                "dtype": "FP8",
                "use_mla": False,
                "tp_size": 1,
                "dp_size": 1,
                "pp_size": 1,
            },
            "location_spec_infos": [{"name": "tp0", "size": 1024}],
        })
        start = meta_request('/api/startWriteCache', {
            "trace_id": self._trace_id,
            "instance_id": self._instance_id,
            "block_keys": [123],
            "token_ids": [456],
            "write_timeout_seconds": 30,
        })
        meta_request('/api/finishWriteCache', {
            "trace_id": self._trace_id,
            "instance_id": self._instance_id,
            "write_session_id": start["write_session_id"],
            "success_blocks": {"bool_masks": {"values": [True]}},
        })

        detail_request = {
            "trace_id": self._trace_id,
            "instance_id": self._instance_id,
            "block_keys": [123, 999],
            "block_mask": {"offset": 0},
            "detail_level": 1,
        }
        detail = self._client.get_cache_meta_detail(detail_request)
        self.assertEqual(len(detail["items"]), 2)
        self.assertEqual([int(item["block_key"]) for item in detail["items"]], [123, 999])
        self.assertEqual(detail["items"][0]["status"]["code"], "OK")
        self.assertEqual(detail["items"][1]["status"]["code"], "OK")
        self.assertTrue(any(loc["status"] == "CLS_SERVING" for loc in detail["items"][0]["locations"]))
        self.assertEqual(detail["items"][1]["locations"][0]["status"], "CLS_NOT_FOUND")

        # The diagnostic route must not be exposed on the ordinary MetaService HTTP port.
        self.assertNotEqual(requests.post(meta_url + '/api/getCacheMetaDetail', json=detail_request).status_code, 200)


if __name__ == "__main__":
    import unittest
    unittest.main()
