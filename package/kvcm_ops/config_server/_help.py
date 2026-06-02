HELP_MESSAGE = '''
config_server module (use: python3 -m kvcm_ops config_server <subcommand> ...):

    === hpnzone lifecycle ===

    List all hpnzones:
        python3 -m kvcm_ops config_server list-hpnzones
        python3 -m kvcm_ops config_server list-hpnzones --url http://10.0.0.1:9101

    Create a hpnzone:
        python3 -m kvcm_ops config_server create-hpnzone --hpnzone prod_zone_a

    Delete a hpnzone:
        python3 -m kvcm_ops config_server delete-hpnzone --hpnzone prod_zone_a
        python3 -m kvcm_ops config_server delete-hpnzone --hpnzone prod_zone_a --yes

    === common ===

    Detect server routing mode:
        python3 -m kvcm_ops config_server server_capability
        python3 -m kvcm_ops config_server server_capability --url http://10.0.0.1:9101 --json

    === instance_pin mode (cell / group pin / instance pin operations) ===

    cell management:
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a list-cells
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a register-cell --cell-url url://kvcm-cell-0
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a unregister-cell --cell-url url://kvcm-cell-0

    group pin management:
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a list-group-pins
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a register-group-pin --group groupA --cell-url url://kvcm-cell-1
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a reassign-group-pin --group groupA --cell-url url://kvcm-cell-2
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a unregister-group-pin --group groupA

    instance management:
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a register-instance --instance model-x:dep-007 --group groupA
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a register-instance --instance model-x:dep-007 --group groupA --pin-url url://kvcm-cell-1
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a resolve-instance --instance model-x:dep-007 --group groupA
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a reassign-instance --instance model-x:dep-007 --cell-url url://kvcm-cell-2
        python3 -m kvcm_ops config_server instance_pin --hpnzone prod_zone_a unregister-instance --instance model-x:dep-007
'''
