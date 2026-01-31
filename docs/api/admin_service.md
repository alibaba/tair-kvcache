# KVCacheManager AdminService API curl Examples

## Basic Usage
1. Add Account
2. Add Storage
3. Create Instance Group
   1. add storage name into storage_candidates
4. Register Instance (Optional)
   1. can be done by meta service
   2. use instance group created above

Storage is globally shared. 
When creating an InstanceGroup, you can set which Storage is visible to the current InstanceGroup. 
Instances belong to a specific InstanceGroup.

## Add Storage
```bash
curl -g -vvv -X POST http://localhost:6492/api/addStorage \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_201",
    "storage": {
        "global_unique_name": "test_file_storage",
        "nfs": {
            "root_path": "/tmp/my_tmp_dir",
            "key_count_per_file": 8
        },
        "check_storage_available_when_open": true
    }
}'

curl -g -vvv -X POST http://127.0.0.1:6492/api/addStorage \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_3fs_01",
    "storage": {
        "global_unique_name": "3fs_01",
        "threefs": {
            "cluster_name": "",
            "mountpoint": "/data/",
            "root_dir": "kvcache/",
            "key_count_per_file": 8,
            "touch_file_when_create": false
        },
        "check_storage_available_when_open": false
    }
}'
```

## Enable Storage
```bash
curl -g -vvv -X POST http://localhost:6492/api/enableStorage \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_202",
    "storage_unique_name": "test_file_storage"
}'
```

## Disable Storage
```bash
curl -g -vvv -X POST http://localhost:6492/api/disableStorage \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_203",
    "storage_unique_name": "test_file_storage"
}'
```

## Remove Storage
```bash
curl -g -vvv -X POST http://localhost:6492/api/removeStorage \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_204",
    "storage_unique_name": "test_file_storage"
}'
```

## Update Storage
```bash
curl -g -vvv -X POST http://localhost:6492/api/updateStorage \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_205",
    "storage": {
        "global_unique_name": "test_file_storage",
        "nfs": {
            "root_dir": "data",
            "key_count_per_file": 8
        }
    },
    "check_storage_available_when_open": true,
    "force_update": false
}'
```

## List Storage
```bash
curl -g -vvv -X POST http://localhost:6492/api/listStorage \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_206"
}'
```

## Create Instance Group
```bash
curl -g -vvv -X POST http://localhost:6492/api/createInstanceGroup \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_207",
    "instance_group": {
        "name": "test_group",
        "storage_candidates": ["test_file_storage"],
        "global_quota_group_name": "default_quota",
        "max_instance_count": 100,
        "quota": {
            "capacity": 107374182400,
            "quota_config": [
                {
                    "storage_type": "ST_NFS",
                    "capacity": 107374182400
                }
            ]
        },
        "cache_config": {
            "reclaim_strategy": {
                "storage_unique_name": "test_file_storage",
                "reclaim_policy": "POLICY_LRU",
                "trigger_strategy": {
                    "used_size": 1073741824,
                    "used_percentage": 0.8
                },
                "trigger_period_seconds": 60,
                "reclaim_step_size": 1073741824,
                "reclaim_step_percentage": 10
            },
            "data_storage_strategy": "CPS_PREFER_3FS",
            "meta_indexer_config": {
                "max_key_count": 10000000,
                "mutex_shard_num": 1024,
                "meta_storage_backend_config": {
                    "storage_type": "local",
                    "local_path": "/tmp/meta"
                },
                "meta_cache_policy_config": {
                    "capacity": 10000,
                    "type": "lru"
                }
            }
        },
        "user_data": "test user data",
        "version": 1
    }
}'

curl -g -vvv -X POST http://localhost:6492/api/createInstanceGroup \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "abc",
    "instance_group": {
        "name": "hf3fs_group",
        "storage_candidates": ["3fs_01"],
        "global_quota_group_name": "default_quota",
        "max_instance_count": 100,
        "quota": {
            "capacity": 107374182400,
            "quota_config": [
                {
                    "storage_type": "ST_3FS",
                    "capacity": 107374182400
                }
            ]
        },
        "cache_config": {
            "reclaim_strategy": {
                "storage_unique_name": "3fs_01",
                "reclaim_policy": "POLICY_LRU",
                "trigger_strategy": {
                    "used_size": 107374182400,
                    "used_percentage": 0.8
                },
                "trigger_period_seconds": 60,
                "reclaim_step_size": 1073741824,
                "reclaim_step_percentage": 10,
                "delay_before_delete_ms": 1000
            },
            "data_storage_strategy": "CPS_PREFER_3FS",
            "meta_indexer_config": {
                "max_key_count": 10000,
                "mutex_shard_num": 1024,
                "meta_storage_backend_config": {
                    "storage_type": "local",
                    "storage_uri": ""
                },
                "meta_cache_policy_config": {
                    "type": "LRU",
                    "capacity": 100000,
                    "cache_shard_bits": 0,
                    "high_pri_pool_ratio": 0.0
                }
            }
        },
        "user_data": "test user data",
        "version": 1
    }
}'
```

## Update Instance Group
```bash
curl -g -vvv -X POST http://localhost:6492/api/updateInstanceGroup \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_208",
    "instance_group": {
        "name": "test_group",
        "storage_candidates": ["test_file_storage"],
        "global_quota_group_name": "default_quota",
        "max_instance_count": 100,
        "quota": {
            "capacity": 107374182400,
            "quota_config": [
                {
                    "storage_type": "ST_NFS",
                    "capacity": 107374182400
                }
            ]
        },
        "cache_config": {
            "reclaim_strategy": {
                "storage_unique_name": "test_file_storage",
                "reclaim_policy": "POLICY_LRU",
                "trigger_strategy": {
                    "used_size": 1073741824,
                    "used_percentage": 0.8
                },
                "trigger_period_seconds": 60,
                "reclaim_step_size": 1073741824,
                "reclaim_step_percentage": 10,
                "delay_before_delete_ms": 1000
            },
            "data_storage_strategy": "CPS_PREFER_3FS",
            "meta_indexer_config": {
                "max_key_count": 10000000,
                "mutex_shard_num": 1024,
                "meta_storage_backend_config": {
                    "storage_type": "redis",
                    "storage_uri": "redis://xxxx:xxxx@r-xxxx.redis.zhangbei.rds.aliyuncs.com:6379"
                },
                "meta_cache_policy_config": {
                    "capacity": 10000,
                    "type": "lru"
                }
            }
        },
        "user_data": "test user data",
        "version": 2
    },
    "current_version": 1
}'
```

## Remove Instance Group
```bash
curl -g -vvv -X POST http://localhost:6492/api/removeInstanceGroup \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_209",
    "name": "test_group"
}'
```

## Get Instance Group
```bash
curl -g -vvv -X POST http://localhost:6492/api/getInstanceGroup \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_210",
    "name": "test_group"
}'
```

## Get Cache Meta
```bash
curl -g -vvv -X POST http://localhost:6492/api/getCacheMeta \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_211",
    "instance_id": "instance1",
    "block_keys": [123],
    "token_ids": [456],
    "block_mask": {
        "offset": 0
    },
    "detail_level": 1
}'
```

## Remove Cache
```bash
curl -g -vvv -X POST http://localhost:6492/api/removeCache \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_212",
    "instance_id": "instance1",
    "block_keys": [123],
    "token_ids": [456],
    "block_mask": {
        "offset": 0
    }
}'
```

## Register Instance
```bash
curl -g -vvv -X POST http://localhost:6492/api/registerInstance \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_213",
    "instance_group": "test_group",
    "instance_id": "instance1",
    "model_deployment": {
        "model_name": "test_model",
        "dtype": "FP8",
        "use_mla": false,
        "tp_size": 1,
        "dp_size": 1,
        "pp_size": 1,
        "extra": "extra_info",
        "user_data": "user_data"
    },
    "block_size": 128,
    "location_spec_infos": [
        {"name": "tp0", "size": 1024}
    ]
}'
```

## Remove Instance
```bash
curl -g -vvv -X POST http://localhost:6492/api/removeInstance \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_214",
    "instance_group": "test_group",
    "instance_id": "taobao_keman_test_instance_id",
"
}'
```

## Get Instance Info
```bash
curl -g -vvv -X POST http://localhost:6492/api/getInstanceInfo \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_215",
    "instance_id": "instance1"
}'
```

## List Instance Info
```bash
curl -g -vvv -X POST http://localhost:6492/api/listInstanceInfo \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_216",
    "instance_group_name": "test_group"
}'
```

## Add Account
```bash
curl -g -vvv -X POST http://localhost:6492/api/addAccount \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_217",
    "user_name": "test_user",
    "password": "test_password",
    "role": "ROLE_ADMIN"
}'
```

## Delete Account
```bash
curl -g -vvv -X POST http://localhost:6492/api/deleteAccount \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_218",
    "user_name": "test_user"
}'
```

## List Account
```bash
curl -g -vvv -X POST http://localhost:6492/api/listAccount \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_219"
}'
```

## Get Metrics
```bash
curl -g -vvv -X POST http://localhost:6492/api/getMetrics \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_220"
}'
```

## Check Health
Check service health status, including whether it's the Leader node and elector status.

```bash
curl -g -vvv -X POST http://localhost:6492/api/checkHealth \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_223"
}'
```

## Get Manager Cluster Info
Get Manager cluster information, including Leader node info, lease expiration time, etc.

```bash
curl -g -vvv -X POST http://localhost:6492/api/getManagerClusterInfo \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_224"
}'
```

## Leader Demote
Actively demote the current Leader node to trigger a new Leader election.

```bash
curl -g -vvv -X POST http://localhost:6492/api/leaderDemote \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_225"
}'
```

## Get Leader Elector Config
Get Leader elector configuration information.

```bash
curl -g -vvv -X POST http://localhost:6492/api/getLeaderElectorConfig \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_226"
}'
```

## Update Leader Elector Config
Update Leader elector configuration, such as campaign delay time.

```bash
curl -g -vvv -X POST http://localhost:6492/api/updateLeaderElectorConfig \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "trace_id": "trace_id_227",
    "campaign_delay_time_ms": 100000
}'
```