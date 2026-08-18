"""Compatibility facade over the split harness modules.

Pure helpers live in lib_utils.py, process lifecycle (manager binary,
vLLM server, ScenarioEnv) in servers.py; this module re-exports the
original surface so existing scenario files keep their imports.
"""

from lib_utils import (  # noqa: F401
    is_hybrid_model, _runfiles_root, find_repo_root, find_manager_binary,
    find_python, free_port, wait_http, tokenize, get_manager_block_size,
    block_token_hash, full_block_hashes, wait_for_prefix_cached,
    send_completions, count_captures, wait_for_captures, compare_captures,
    assert_report_ok, make_base_prompts, shared_token_prefix_len,
)
from servers import (  # noqa: F401
    ManagerProcess, VllmServer, ScenarioEnv, run_e2e,
)
