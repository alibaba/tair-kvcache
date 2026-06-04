"""HTTP integration tests for the Online Optimizer service."""

import requests

import integration_test.optimizer_test.optimizer_interface_cases as cases


class OptimizerServiceHttpClient(cases.OptimizerServiceClientBase):
    """HTTP client for Optimizer Service API endpoints."""

    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    def _make_api_request(self, endpoint, data=None, check_response=True):
        url = self.base_url + endpoint
        response = self.session.post(url, json=data, headers=self.headers)
        if response.status_code != 200:
            raise AssertionError(
                f"Request to {endpoint} failed with status code {response.status_code}: "
                f"{response.text[:500]}")
        try:
            response_data = response.json()
        except ValueError as e:
            raise AssertionError(f"Response from {endpoint} is not valid JSON: {e}")
        if check_response:
            if 'header' not in response_data:
                raise AssertionError(f"Response from {endpoint} missing 'header' field")
            if response_data['header']['status']['code'] != "OK":
                raise AssertionError(
                    f"Request to {endpoint} failed: "
                    f"{response_data['header']['status'].get('message', 'unknown error')}")
        return response_data

    def create_instance_group(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/createInstanceGroup', data, check_response)

    def update_instance_group(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/updateInstanceGroup', data, check_response)

    def remove_instance_group(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/removeInstanceGroup', data, check_response)

    def get_instance_group(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/getInstanceGroup', data, check_response)

    def list_instance_groups(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/listInstanceGroups', data, check_response)

    def register_instance(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/registerInstance', data, check_response)

    def remove_instance(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/removeInstance', data, check_response)

    def get_instance(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/getInstance', data, check_response)

    def trace_query(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/traceQuery', data, check_response)

    def list_instances(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/listInstances', data, check_response)

    def reset_stats(self, data, check_response=True):
        return self._make_api_request('/api/optimizer/resetStats', data, check_response)

    def close(self):
        self.session.close()


class OptimizerServiceHttpTest(cases.OptimizerServiceTestCases):
    """HTTP version of the optimizer service integration tests."""

    def _create_client(self):
        http_port = self._server_manager.http_port
        base_url = f"http://127.0.0.1:{http_port}"
        return OptimizerServiceHttpClient(base_url)


if __name__ == "__main__":
    import unittest
    unittest.main()
