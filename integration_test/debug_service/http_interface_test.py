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


if __name__ == "__main__":
    import unittest
    unittest.main()
