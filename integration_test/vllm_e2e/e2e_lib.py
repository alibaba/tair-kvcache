"""Compatibility facade over the split harness modules.

Pure helpers live in lib_utils.py, process lifecycle (manager binary,
vLLM server, ScenarioEnv) in servers.py; this module re-exports the
original surface so existing scenario files keep their imports.
"""

from lib_utils import (  # noqa: F401
    is_hybrid_model, is_mla_model, is_multigroup_model, is_swa_model,
    block_tables_from_logs, has_full_attention_group, loaded_block_count,
    model_fingerprint, skipped_window_tokens, window_kind, window_tokens,
    window_tail_blocks, scenario_tp_size, _runfiles_root,
    find_repo_root, find_manager_binary, find_python, free_port, wait_http,
    tokenize, get_manager_block_size, block_token_hash, full_block_hashes,
    wait_for_prefix_cached, send_completions, count_captures,
    wait_for_captures, compare_captures, loaded_capture_keys,
    assert_loaded_blocks, assert_loaded_nonempty, assert_pinned_ok,
    assert_report_ok, make_base_prompts,
    shared_token_prefix_len,
)
from servers import (  # noqa: F401
    ManagerProcess, VllmServer, ScenarioEnv, run_e2e, pinned_budget_gib,
)
