import requests
import integration_test.admin_service.admin_interface_cases as cases


class AdminServiceHttpClient(cases.AdminServiceClientBase):
    """HTTP client for AdminService API endpoints"""

    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    def _make_request(self, method, endpoint, data=None):
        url = self.base_url + endpoint
        if method == 'POST':
            response = self.session.post(url, json=data, headers=self.headers)
        elif method == 'GET':
            response = self.session.get(url, params=data, headers=self.headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        return response

    def _make_api_request(self, endpoint, data=None, check_response=True):
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

    def add_storage(self, data, check_response=True):
        return self._make_api_request('/api/addStorage', data, check_response)

    def enable_storage(self, data, check_response=True):
        return self._make_api_request('/api/enableStorage', data, check_response)

    def disable_storage(self, data, check_response=True):
        return self._make_api_request('/api/disableStorage', data, check_response)

    def remove_storage(self, data, check_response=True):
        return self._make_api_request('/api/removeStorage', data, check_response)

    def update_storage(self, data, check_response=True):
        return self._make_api_request('/api/updateStorage', data, check_response)

    def list_storage(self, data, check_response=True):
        return self._make_api_request('/api/listStorage', data, check_response)

    def create_instance_group(self, data, check_response=True):
        return self._make_api_request('/api/createInstanceGroup', data, check_response)

    def update_instance_group(self, data, check_response=True):
        return self._make_api_request('/api/updateInstanceGroup', data, check_response)

    def remove_instance_group(self, data, check_response=True):
        return self._make_api_request('/api/removeInstanceGroup', data, check_response)

    def get_instance_group(self, data, check_response=True):
        return self._make_api_request('/api/getInstanceGroup', data, check_response)

    def get_cache_meta(self, data, check_response=True):
        return self._make_api_request('/api/getCacheMeta', data, check_response)

    def remove_cache(self, data, check_response=True):
        return self._make_api_request('/api/removeCache', data, check_response)

    def register_instance(self, data, check_response=True):
        return self._make_api_request('/api/registerInstance', data, check_response)

    def remove_instance(self, data, check_response=True):
        return self._make_api_request('/api/removeInstance', data, check_response)

    def get_instance_info(self, data, check_response=True):
        return self._make_api_request('/api/getInstanceInfo', data, check_response)

    def list_instance_info(self, data, check_response=True):
        return self._make_api_request('/api/listInstanceInfo', data, check_response)

    def add_account(self, data, check_response=True):
        return self._make_api_request('/api/addAccount', data, check_response)

    def delete_account(self, data, check_response=True):
        return self._make_api_request('/api/deleteAccount', data, check_response)

    def list_account(self, data, check_response=True):
        return self._make_api_request('/api/listAccount', data, check_response)

    def gen_config_snapshot(self, data, check_response=True):
        return self._make_api_request('/api/genConfigSnapshot', data, check_response)

    def load_config_snapshot(self, data, check_response=True):
        return self._make_api_request('/api/loadConfigSnapshot', data, check_response)

    def get_metrics(self, data, check_response=True):
        return self._make_api_request('/api/getMetrics', data, check_response)

    def check_health(self, data, check_response=True):
        return self._make_api_request('/api/checkHealth', data, check_response)

    def get_manager_cluster_info(self, data, check_response=True):
        return self._make_api_request('/api/getManagerClusterInfo', data, check_response)

    def leader_demote(self, data, check_response=True):
        return self._make_api_request('/api/leaderDemote', data, check_response)

    def get_leader_elector_config(self, data, check_response=True):
        return self._make_api_request('/api/getLeaderElectorConfig', data, check_response)

    def update_leader_elector_config(self, data, check_response=True):
        return self._make_api_request('/api/updateLeaderElectorConfig', data, check_response)

    def close(self):
        self.session.close()


class AdminServiceHttpTest(cases.AdminServiceTestBase):
    """HTTP version of the AdminService tests"""

    def _get_manager_client(self):
        self._http_port = self.worker_manager.get_worker(0).env.admin_http_port
        self._http_url = f"http://localhost:{self._http_port}"
        import logging
        logging.info(f"http url: {self._http_url}")
        return AdminServiceHttpClient(self._http_url)


class AdminServiceHttpLeaderElectionTest(cases.AdminServiceLeaderElectionTest):


    def _get_manager_client(self, worker_id: int):
        worker = self.worker_manager.get_worker(worker_id)
        http_port = worker.env.admin_http_port
        http_url = f"http://localhost:{http_port}"
        import logging
        logging.info(f"Connecting to worker {worker_id} at {http_url}")
        return AdminServiceHttpClient(http_url)


if __name__ == "__main__":
    import unittest
    unittest.main()
