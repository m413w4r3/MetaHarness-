"""Remote gateway primitives.

The package holds authentication helpers and the local control client; no
server is defined here.
"""

from .auth import (
    DEFAULT_MAX_TOKEN_BYTES,
    TokenFileError,
    bearer_token_from_header,
    load_token_file,
    token_matches,
)
from .client import (
    LOCAL_HOST,
    MAX_RESPONSE_BYTES,
    LocalMetaHarnessClient,
    LocalMetaHarnessError,
)

__all__ = [
    "DEFAULT_MAX_TOKEN_BYTES",
    "LOCAL_HOST",
    "MAX_RESPONSE_BYTES",
    "LocalMetaHarnessClient",
    "LocalMetaHarnessError",
    "TokenFileError",
    "bearer_token_from_header",
    "load_token_file",
    "token_matches",
]
