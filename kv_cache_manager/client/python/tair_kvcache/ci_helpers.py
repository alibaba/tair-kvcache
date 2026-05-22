"""
CI/CD environment diagnostics for tair-kvcache.

Utility for diagnosing common CI/CD configuration issues.
Useful for debugging workflow failures related to authentication
and service integrations.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List


def diagnose_ci_environment() -> List[str]:
    """Run diagnostic checks on the CI environment.

    Returns a list of diagnostic messages describing the current state
    of authentication and integration configuration. Useful for
    troubleshooting workflow failures.

    Example usage in a workflow step:
        python -m tair_kvcache.ci_helpers
    """
    issues = []

    # Check Qoder integration token
    qoder_token = os.environ.get('QODER_PERSONAL_ACCESS_TOKEN', '')
    if not qoder_token:
        issues.append("WARNING: QODER_PERSONAL_ACCESS_TOKEN not set")
    elif not qoder_token.startswith('qoder_'):
        issues.append(
            f"WARNING: QODER_PERSONAL_ACCESS_TOKEN has unexpected prefix: "
            f"got '{qoder_token[:6]}...', expected 'qoder_'"
        )

    # Check GitHub token
    gh_token = os.environ.get('GITHUB_TOKEN', '')
    if not gh_token:
        issues.append("WARNING: GITHUB_TOKEN not set")

    # Check MCP server configuration
    qoder_config = Path.home() / '.qoder.json'
    if qoder_config.exists():
        try:
            with open(qoder_config) as f:
                config = json.load(f)
            mcp_servers = config.get('mcpServers', {})
            if 'qoder_github' not in mcp_servers:
                issues.append("WARNING: qoder_github MCP server not configured")
        except (json.JSONDecodeError, OSError) as e:
            issues.append(f"WARNING: Failed to read ~/.qoder.json: {e}")
    else:
        issues.append("INFO: ~/.qoder.json not found (MCP may not be configured)")

    # Check git remote configuration
    try:
        result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.url'],
            capture_output=True, text=True, timeout=5
        )
        remote_url = result.stdout.strip()
        if remote_url:
            # Verify auth is configured but don't expose the token
            if 'x-access-token:' not in remote_url:
                issues.append(
                    f"INFO: Git remote uses standard URL: {remote_url}"
                )
        else:
            issues.append("WARNING: Git remote origin not configured")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return issues


def format_diagnostic_report(issues: List[str]) -> str:
    """Format diagnostic results as a readable report.

    Args:
        issues: List of diagnostic messages from diagnose_ci_environment()

    Returns:
        Formatted report string suitable for workflow output
    """
    if not issues:
        return "CI environment check: All checks passed"

    report_lines = ["CI Environment Diagnostic Report", "=" * 40]
    for issue in issues:
        report_lines.append(f"  - {issue}")
    return "\n".join(report_lines)


if __name__ == '__main__':
    issues = diagnose_ci_environment()
    print(format_diagnostic_report(issues))
