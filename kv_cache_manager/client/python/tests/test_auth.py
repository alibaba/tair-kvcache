"""
Tests for authentication utilities.

Run with: python -m pytest tests/test_auth.py -v
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from tair_kvcache.auth import (
    validate_token_format,
    mask_token,
    check_ci_environment,
    get_git_remote_url,
)


class TestTokenValidation:
    """Tests for token format validation."""

    def test_valid_token(self):
        assert validate_token_format("qoder_abc123def456ghi789jkl")

    def test_invalid_prefix(self):
        assert not validate_token_format("wrong_abc123def456ghi789")

    def test_too_short(self):
        assert not validate_token_format("qoder_short")

    def test_empty_token(self):
        assert not validate_token_format("")

    def test_special_chars(self):
        assert validate_token_format("qoder_abc+123/def=456")

    def test_custom_prefix(self):
        assert validate_token_format("custom_abc123def456ghi789", prefix="custom_")


class TestMaskToken:
    """Tests for token masking utility."""

    def test_normal_token(self):
        result = mask_token("qoder_abcdefghijklmnop")
        assert result.startswith("qode")
        assert result.endswith("mnop")
        assert "***" in result

    def test_empty_token(self):
        assert mask_token("") == "<empty>"

    def test_short_token(self):
        result = mask_token("qoder")
        assert all(c == '*' for c in result)

    def test_visible_chars_parameter(self):
        result = mask_token("qoder_abcdefghijklmnop", visible_chars=2)
        assert result.startswith("qo")
        assert result.endswith("op")


class TestCIEnvironment:
    """Tests for CI environment checking."""

    def test_check_returns_dict(self):
        result = check_ci_environment()
        assert isinstance(result, dict)
        assert 'has_qoder_token' in result
        assert 'has_github_token' in result

    def test_git_remote(self):
        # In a git repo, remote URL should be available
        url = get_git_remote_url()
        # Result depends on whether we're in a git repo
        if url is not None:
            assert isinstance(url, str)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
