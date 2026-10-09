```shell
python3 -m kvcm_ops --help
```

List instances from all instance groups (omit `-n`):

```shell
python3 -m kvcm_ops list_instance -H http://localhost:6492
```

List instances from a specific group:

```shell
python3 -m kvcm_ops list_instance -H http://localhost:6492 -n default
```

The JSON output contains an `instance_info` list, with `instance_group_name`
identifying the group each instance belongs to. Listing all instances queries
the groups first, then queries each group and merges the results. An empty
result is shown as `"instance_info": []`; if any query fails, the command reports
an error instead of displaying a partial list.
