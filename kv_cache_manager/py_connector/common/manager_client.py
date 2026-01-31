from typing import List

import requests

from kv_cache_manager.py_connector.common.logger import logger


class KvCacheManagerClient:
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
        response_data = response.json()
        if check_response:
            if response.status_code != 200:
                raise AssertionError(f"Request to {endpoint} failed with status code {response.status_code}")

            if 'header' not in response_data:
                raise AssertionError(f"Response from {endpoint} missing 'header' field")

            if response_data['header']['status']['code'] != "OK":
                raise AssertionError(
                    f"Request to {endpoint} failed with error: {response_data['header']['status']['message']}")

        return response_data

    def register_instance(self, data, check_response=True):
        """Register an instance with the service"""
        return self._make_api_request('/api/registerInstance', data, check_response)

    def get_instance_info(self, data, check_response=True):
        """Get information about a registered instance"""
        return self._make_api_request('/api/getInstanceInfo', data, check_response)

    def get_cache_location(self, data, check_response=True):
        """Get cache location for specified block keys"""
        return self._make_api_request('/api/getCacheLocation', data, check_response)

    def start_write_cache(self, data, check_response=True):
        """Start writing cache data"""
        return self._make_api_request('/api/startWriteCache', data, check_response)

    def finish_write_cache(self, data, check_response=True):
        """Finish writing cache data"""
        return self._make_api_request('/api/finishWriteCache', data, check_response)

    def remove_cache(self, data, check_response=True):
        """Remove cache data for specified block keys"""
        return self._make_api_request('/api/removeCache', data, check_response)

    def trim_cache(self, data, check_response=True):
        """Trim cache data based on specified strategy"""
        return self._make_api_request('/api/trimCache', data, check_response)

    def close(self):
        """Close the HTTP session"""
        self.session.close()
