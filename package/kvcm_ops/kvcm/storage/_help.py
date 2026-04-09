HELP_MESSAGE = '''
storage module:
    list all storages:
        python3 -m kvcm_ops list_storage --help
        python3 -m kvcm_ops list_storage
        python3 -m kvcm_ops list_storage -H http://localhost:56040
    add new storage:
        python3 -m kvcm_ops add_storage --help
        python3 -m kvcm_ops add_storage nfs --help
        python3 -m kvcm_ops add_storage pace --help
        python3 -m kvcm_ops add_storage 3fs --help

        # add nfs storage, given "global_unique_name, root_path, key_kount_per_file"
        python3 -m kvcm_ops add_storage -u test_nfs_1 nfs -r /home/zhaotaonan.ztn/temp -k 16

        # add pace storage, given "global_unique_name, domain, timeout"
        python3 -m kvcm_ops add_storage -u common_pace_storage pace -d 'http://pace-meta-service-bj-1.alibaba-inc.com' -t 30

        # add 3fs storage, given "global_unique_name, cluster_name, mountpoint, root_dir, key_count_per_file, touch_file_when_create(default true)"
        python3 -m kvcm_ops add_storage -u common_3fs_storage 3fs -c '' -m '/3fs/stage/3fs/' -r "common_3fs" -k 16
        python3 -m kvcm_ops add_storage -u common_3fs_storage 3fs -c '' -m '/3fs/stage/3fs/' -r "common_3fs" -k 16 --not_touch_file_when_create
    update storage:
        python3 -m kvcm_ops update_storage --help
        python3 -m kvcm_ops update_storage nfs --help
        python3 -m kvcm_ops update_storage pace --help
        python3 -m kvcm_ops update_storage 3fs --help
        python3 -m kvcm_ops update_storage -u test_nfs_1 nfs -r /home/zhaotaonan.ztn/temp -k 64
    enable/disable storage:
        python3 -m kvcm_ops enable_storage --help
        python3 -m kvcm_ops enable_storage -u test_nfs_1
        python3 -m kvcm_ops disable_storage -u test_nfs_1
'''