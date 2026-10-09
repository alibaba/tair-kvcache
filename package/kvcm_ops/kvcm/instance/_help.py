HELP_MESSAGE = '''
instance module:
    list instances (all groups by default; -n filters by instance_group):
        python3 -m kvcm_ops list_instance --help
        python3 -m kvcm_ops list_instance
        python3 -m kvcm_ops list_instance -H http://localhost:56040
        python3 -m kvcm_ops list_instance -n default
        python3 -m kvcm_ops list_instance -H http://localhost:56040 -n default
    get instance info:
        python3 -m kvcm_ops get_instance --help
        python3 -m kvcm_ops get_instance -i test_instance_id
        python3 -m kvcm_ops get_instance -H http://localhost:56040 -i test_instance_id
    delete instance:
        python3 -m kvcm_ops remove_instance --help
        python3 -m kvcm_ops remove_instance -i test_instance_id
        python3 -m kvcm_ops remove_instance -H http://localhost:56040 -i test_instance_id
    register instance manually(only for test):
        python3 -m kvcm_ops register_instance --help
        python3 -m kvcm_ops register_instance -g default -i test_instance_id
        python3 -m kvcm_ops register_instance -H http://localhost:56040 -g default -i test_instance_id
'''
