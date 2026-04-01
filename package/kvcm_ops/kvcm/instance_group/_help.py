HELP_MESSAGE = '''
instance_group module:
    get instance_group info:
        python3 -m kvcm_ops get_instance_group --help
        python3 -m kvcm_ops get_instance_group -n default
        python3 -m kvcm_ops get_instance_group -H http://localhost:56040 -n default
    create new instance_group:
        python3 -m kvcm_ops create_instance_group --help
        python3 -m kvcm_ops create_instance_group -n test_group -s nfs_01
        python3 -m kvcm_ops create_instance_group -H http://localhost:56040 -n test_group -s nfs_01
    delete instance_group:
        python3 -m kvcm_ops remove_instance_group --help
        python3 -m kvcm_ops remove_instance_group -n test_group -s nfs_01
        python3 -m kvcm_ops remove_instance_group -H http://localhost:56040 -n test_group
    update existed instance_group(version of instance_group will plus one automatically):
        python3 -m kvcm_ops update_instance_group --help
        python3 -m kvcm_ops update_instance_group -n test_group -u new_1 --max_key_count 99999
        python3 -m kvcm_ops update_instance_group -n test_group -u new_2 --max_key_count 99999 --quota_configs ST_TAIRMEMPOOL,8888888888888
'''