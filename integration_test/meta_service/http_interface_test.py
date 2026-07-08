import requests
import integration_test.meta_service.meta_interface_cases as cases


class MetaServiceHttpClient(cases.MetaServiceClientBase):
    """HTTP client for MetaService API endpoints"""

    def __init__(self, base_url, admin_url=None):
        self.base_url = base_url
        self.admin_url = admin_url or base_url
        self.session = requests.Session()
        self.headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}

    def _make_request(self, method, endpoint, data=None, base_url=None):
        """Helper method to make HTTP requests to the service"""
        url = (base_url or self.base_url) + endpoint

        if method == 'POST':
            response = self.session.post(url, json=data, headers=self.headers)
        elif method == 'GET':
            response = self.session.get(url, params=data, headers=self.headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        return response

    def _make_api_request(self, endpoint, data=None, check_response=True, base_url=None):
        """Helper method to make POST requests to API endpoints and optionally validate response"""
        response = self._make_request('POST', endpoint, data, base_url=base_url)
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

    def register_instance(self, data, check_response=True):
        """Register an instance with the service"""
        return self._make_api_request('/api/registerInstance', data, check_response)

    def get_instance_info(self, data, check_response=True):
        """Get information about a registered instance"""
        return self._make_api_request('/api/getInstanceInfo', data, check_response)

    def get_cache_location(self, data, check_response=True):
        """Get cache location for specified block keys"""
        return self._make_api_request('/api/getCacheLocation', data, check_response)

    def get_cache_meta_detail(self, data, check_response=True):
        """Get full raw metadata detail for specified block keys"""
        return self._make_api_request('/api/getCacheMetaDetail', data, check_response)

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

    def get_cluster_info(self, data, check_response=True):
        """Get cluster info (leader discovery)"""
        return self._make_api_request('/api/getClusterInfo', data, check_response)

    def report_event(self, data, check_response=True):
        """Report EventReport node/block lifecycle events."""
        return self._make_api_request('/api/reportEvent', data, check_response)

    def get_cache_locations_by_backend(self, data, check_response=True):
        """Query locations with explicit backend selection."""
        return self._make_api_request('/api/getCacheLocationsByBackend', data, check_response)

    def _make_admin_api_request(self, endpoint, data=None, check_response=True):
        return self._make_api_request(
            endpoint,
            data,
            check_response,
            base_url=self.admin_url,
        )

    def add_storage(self, data, check_response=True):
        """Register storage through the admin HTTP endpoint."""
        return self._make_admin_api_request('/api/addStorage', data, check_response)

    def create_instance_group(self, data, check_response=True):
        """Create an instance group through the admin HTTP endpoint."""
        return self._make_admin_api_request('/api/createInstanceGroup', data, check_response)

    def close(self):
        """Close the HTTP session"""
        self.session.close()


class MetaServiceHttpTest(cases.MetaServiceTestBase):
    """HTTP version of the MetaService tests"""

    def _get_manager_client(self):
        worker_env = self.worker_manager.get_worker(0).env
        self._http_port = worker_env.http_port
        self._http_url = "http://localhost:%d" % self._http_port
        self._admin_http_url = "http://localhost:%d" % worker_env.admin_http_port
        return MetaServiceHttpClient(self._http_url, self._admin_http_url)

    @staticmethod
    def _event_report_storage(storage_name):
        return {
            "global_unique_name": storage_name,
            "storage_type": "ST_EVENT_REPORT_L2",
            "event_report": {
                "heartbeat_timeout_ms": 30000,
                "cleanup_grace_ms": 30000,
                "liveness_check_interval_ms": 1000,
            },
            "check_storage_available_when_open": False,
        }

    @staticmethod
    def _event_report_instance_group(group_name, storage_name):
        return {
            "name": group_name,
            "storage_candidates": ["nfs_01"],
            "global_quota_group_name": "default_quota_group",
            "max_instance_count": 10,
            "quota": {
                "capacity": 10737418240,
                "quota_config": [{"storage_type": 4, "capacity": 10737418240}],
            },
            "cache_config": {
                "reclaim_strategy": {
                    "storage_unique_name": "nfs_01",
                    "reclaim_policy": 1,
                    "trigger_strategy": {"used_size": 1073741824, "used_percentage": 0.8},
                    "trigger_period_seconds": 60,
                    "reclaim_step_size": 1073741824,
                    "reclaim_step_percentage": 10,
                },
                "data_storage_strategy": 2,
                "meta_indexer_config": {
                    "max_key_count": 10000,
                    "mutex_shard_num": 16,
                    "batch_key_size": 16,
                    "meta_storage_backend_config": {"storage_type": "local", "storage_uri": ""},
                    "meta_cache_policy_config": {"type": "LRU", "capacity": 10000},
                },
            },
            "event_report_storage_candidates": [storage_name],
            "version": 1,
        }

    @staticmethod
    def _report_events(instance_id, host, events, trace_id):
        return {
            "trace_id": trace_id,
            "instance_id": instance_id,
            "host_ip_port": host,
            "storage_type": "ST_EVENT_REPORT_L2",
            "events": events,
        }

    @staticmethod
    def _node_and_block_events(host, spec_name, block_keys):
        events = [{
            "event_type": "EVENT_NODE_REGISTER",
            "node_register": {"mediums": ["mem"]},
        }]
        events.extend({
            "event_type": "EVENT_BLOCK_ADD",
            "block_add": {
                "block_key": str(block_key),
                "medium": "mem",
                "specs": [{
                    "name": spec_name,
                    "uri": f"vineyard://{host}/mem",
                }],
            },
        } for block_key in block_keys)
        return events

    def test_event_report_requested_spec_filters_before_peer_selection(self):
        """Exercise ReportEvent -> HTTP query with the adversarial peer layout.

        The full-only peer has better raw coverage. Both cross-key strategies
        must nevertheless select the linear peer when linear_1 is requested.
        """
        storage_name = "event_report_l2_http_spec_filter"
        group_name = "event_report_http_spec_filter_group"
        instance_id = "event_report_http_spec_filter_instance"
        full_host = "10.10.0.1:9600"
        linear_host = "10.10.0.2:9600"
        block_keys = [82000, 82001]

        self._client.add_storage({
            "trace_id": "event_report_add_storage",
            "storage": self._event_report_storage(storage_name),
        })
        self._client.create_instance_group({
            "trace_id": "event_report_create_group",
            "instance_group": self._event_report_instance_group(group_name, storage_name),
        })
        self._client.register_instance({
            "trace_id": "event_report_register_instance",
            "instance_group": group_name,
            "instance_id": instance_id,
            "block_size": 128,
            "model_deployment": self._get_test_model_deployment(),
            "location_spec_infos": [
                {"name": "full_0", "size": 1024},
                {"name": "linear_1", "size": 1024},
            ],
            "location_spec_groups": [
                {"name": "full_0", "spec_names": ["full_0"]},
                {"name": "linear_1", "spec_names": ["linear_1"]},
            ],
        })
        self._client.report_event(self._report_events(
            instance_id,
            full_host,
            self._node_and_block_events(full_host, "full_0", block_keys),
            "event_report_full_peer",
        ))
        self._client.report_event(self._report_events(
            instance_id,
            linear_host,
            self._node_and_block_events(linear_host, "linear_1", block_keys[:1]),
            "event_report_linear_peer",
        ))

        for strategy in ("LSS_V6D_PREFIX", "LSS_V6D_COVERAGE"):
            response = self._client.get_cache_locations_by_backend({
                "trace_id": f"event_report_query_{strategy}",
                "instance_id": instance_id,
                "query_type": "QT_BATCH_GET",
                "block_keys": block_keys,
                "block_mask": {"offset": 0},
                "location_spec_names": ["linear_1"] * len(block_keys),
                "backend_selectors": [{
                    "backend_type": "ST_EVENT_REPORT_L2",
                    "strategy": strategy,
                }],
            })
            key_locations = response.get("key_locations", [])
            self.assertEqual(2, len(key_locations), response)
            first_locations = key_locations[0].get("locations", [])
            self.assertEqual(1, len(first_locations), response)
            first_specs = first_locations[0].get("location_specs", [])
            self.assertEqual(["linear_1"], [spec.get("name") for spec in first_specs], response)
            self.assertIn(linear_host, first_specs[0].get("uri", ""), response)
            self.assertEqual([], key_locations[1].get("locations", []), response)

        unknown_response = self._client.get_cache_locations_by_backend({
            "trace_id": "event_report_query_unknown_spec",
            "instance_id": instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": block_keys,
            "block_mask": {"offset": 0},
            "location_spec_names": ["unknown_spec"] * len(block_keys),
            "backend_selectors": [{
                "backend_type": "ST_EVENT_REPORT_L2",
                "strategy": "LSS_V6D_PREFIX",
            }],
        })
        self.assertEqual(
            [[], []],
            [item.get("locations", []) for item in unknown_response.get("key_locations", [])],
            unknown_response,
        )

        # Protobuf enums are open on the wire. 263 would truncate to the
        # internal uint8 value 7 (L1P5) if the service used a direct cast.
        invalid_selector = self._client.get_cache_locations_by_backend({
            "trace_id": "event_report_query_unknown_backend",
            "instance_id": instance_id,
            "query_type": "QT_BATCH_GET",
            "block_keys": block_keys,
            "block_mask": {"offset": 0},
            "backend_selectors": [{
                "backend_type": 263,
                "strategy": "LSS_WEIGHTED_RANDOM",
            }],
        }, check_response=False)
        self.assertEqual(
            "INVALID_ARGUMENT",
            invalid_selector.get("header", {}).get("status", {}).get("code"),
            invalid_selector,
        )


if __name__ == "__main__":
    import unittest
    unittest.main()
