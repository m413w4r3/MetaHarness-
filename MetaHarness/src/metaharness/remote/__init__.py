"""Remote gateway primitives.

Only authentication helpers live here for now; no server is defined yet.
"""

from .auth import (
    DEFAULT_MAX_TOKEN_BYTES,
    TokenFileError,
    bearer_token_from_header,
    load_token_file,
    token_matches,
)

__all__ = [
    "DEFAULT_MAX_TOKEN_BYTES",
    "TokenFileError",
    "bearer_token_from_header",
    "load_token_file",
    "token_matches",
]
