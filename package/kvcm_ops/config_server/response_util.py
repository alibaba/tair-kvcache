"""Shared helpers for ConfigServer HTTP JSON responses."""

import json


def check_response_header(obj: dict) -> None:
    """Raise SystemExit if the protobuf response header carries a non-OK status."""
    header = obj.get("header") or {}
    st = header.get("status") or {}
    code = st.get("code")
    if code is None:
        code = "OK"
    code_str = code if isinstance(code, str) else str(code)
    if code_str not in ("OK", "1"):
        msg = st.get("message", "")
        raise SystemExit(f"[ERROR] server returned {code_str}: {msg}")


def check_response_header_text(resp_text: str) -> None:
    """Parse JSON when possible and reject non-OK protobuf header.status."""
    try:
        obj = json.loads(resp_text)
    except (json.JSONDecodeError, TypeError):
        return
    check_response_header(obj)
