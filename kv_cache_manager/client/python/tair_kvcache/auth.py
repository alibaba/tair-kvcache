"""
Authentication utilities for tair-kvcache Python client.

This module provides helpers for validating authentication tokens
and checking CI/CD environment configuration.
"""

import os
import re
import subprocess
from typing import Optional


def validate_token_format(token: str, prefix: str = "qoder_") -> bool:
    """Validate that a token matches the expected format.

    Tokens should start with the given prefix and contain only
    base64-compatible characters.

    Args:
        token: The token string to validate
        prefix: Expected token prefix (default: 'qoder_')

    Returns:
        True if the token format is valid
    """
    if not token or len(token) < 20:
        return False
    if not token.startswith(prefix):
        return False
    pattern = r'^' + re.escape(prefix) + r'[A-Za-z0-9+/=_-]+$'
    return bool(re.match(pattern, token))


def get_git_remote_url() -> Optional[str]:
    """Get the current git remote origin URL.

    Useful for verifying that git authentication is properly configured
    in CI/CD environments. The remote URL should use HTTPS with an
    access token for automated workflows.

    Returns:
        The git remote URL string, or None if not configured
    """
    try:
        result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.url'],
            capture_output=True,
            text=True,
            timeout=5
        )
        url = result.stdout.strip()
        if url:
            # Mask any embedded tokens for safe logging
            if 'x-access-token:' in url:
                url = url.split('@')[0].rsplit(':', 1)[0] + ':***@' + url.split('@')[-1]
            return url
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def check_ci_environment() -> dict:
    """Check the current CI/CD environment configuration.

    Verifies that required environment variables are set and properly
    formatted for the CI/CD pipeline. This is intended for use in
    workflow diagnostic steps.

    Returns:
        Dictionary with check results:
        - has_qoder_token: bool - QODER_PERSONAL_ACCESS_TOKEN is set
        - has_github_token: bool - GITHUB_TOKEN is set
        - token_prefix_valid: bool - token starts with expected prefix
        - git_remote_ok: bool - git remote is configured with auth
    """
    qoder_token = os.environ.get('QODER_PERSONAL_ACCESS_TOKEN', '')
    github_token = os.environ.get('GITHUB_TOKEN', '')

    return {
        'has_qoder_token': bool(qoder_token),
        'has_github_token': bool(github_token),
        'token_prefix_valid': validate_token_format(qoder_token) if qoder_token else False,
        'git_remote_ok': get_git_remote_url() is not None,
    }


def mask_token(token: str, visible_chars: int = 4) -> str:
    """Mask a token for safe display in logs and reviews.

    Args:
        token: The token to mask
        visible_chars: Number of characters to show at start and end

    Returns:
        Masked token string (e.g., 'qoder_abc***xyz')
    """
    if not token:
        return '<empty>'
    if len(token) <= visible_chars * 2:
        return '*' * len(token)
    return token[:visible_chars] + '***' + token[-visible_chars:]
