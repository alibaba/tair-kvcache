HELP_MESSAGE = '''
trace module:
    python3 -m kvcm_ops trace_key --help
    python3 -m kvcm_ops trace_key --path path/to/kvcm_log -i 3219140058766880711 -k 1513637688269809021 --need_uri

    python3 -m kvcm_ops trace_uri --help
    python3 -m kvcm_ops trace_uri --path path/to/kvcm_log -i 14784965019911564610 --uri 'pace://common_pace_storage/689702818545664?media_type=2&node_id=2883&range_id=0&size=25165824'
    python3 -m kvcm_ops trace_uri --path path/to/kvcm_log --uri 'pace://common_pace_storage/689702818545664?media_type=2&node_id=2883&range_id=0&size=25165824'
'''