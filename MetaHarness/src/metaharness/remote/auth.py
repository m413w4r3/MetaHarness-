"""Bearer-token primitives for the remote gateway.

Token values are handled opaquely: no function in this module returns or
raises a message that repeats the token it was given.
"""

from __future__ import annotations

import secrets
from pathlib import Path

DEFAULT_MAX_TOKEN_BYTES = 4096
_BEARER_SCHEME = "bearer"


class TokenFileError(ValueError):
    """A token file cannot be used, described without its content."""


def load_token_file(path: str | Path, *, max_bytes: int = DEFAULT_MAX_TOKEN_BYTES) -> str:
    """Return the token stored in *path*.

    The file must be UTF-8, at most *max_bytes* bytes, and hold a single
    token optionally terminated by one ``\\n`` or ``\\r\\n``.  Failure
    messages never include the file content.
    """

    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    target = Path(path).expanduser()
    with target.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise TokenFileError(f"token file {str(target)!r} exceeds {max_bytes} bytes")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise TokenFileError(f"token file {str(target)!r} is not valid UTF-8") from None
    token = _without_final_newline(text)
    if not token:
        raise TokenFileError(f"token file {str(target)!r} is empty")
    if "\n" in token or "\r" in token:
        raise TokenFileError(f"token file {str(target)!r} contains an embedded newline")
    return token


def bearer_token_from_header(value: str | None) -> str | None:
    """Return the credential of a ``Bearer`` authorization header."""

    if not isinstance(value, str):
        return None
    fields = value.split(None, 1)
    if len(fields) != 2 or fields[0].casefold() != _BEARER_SCHEME:
        return None
    return fields[1].strip() or None


def token_matches(candidate: str | None, expected: str) -> bool:
    """Compare two tokens in constant time."""

    if not candidate or not expected:
        return False
    return secrets.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def _without_final_newline(text: str) -> str:
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text


__all__ = [
    "DEFAULT_MAX_TOKEN_BYTES",
    "TokenFileError",
    "bearer_token_from_header",
    "load_token_file",
    "token_matches",
]
