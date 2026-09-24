"""Remote gateway primitives.

The package holds the token helpers, the local control client, and the
gateway that exposes the observation routes and the five control mutations
of the local MetaHarness API to a remote client.
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
from .server import (
    DEFAULT_METAHARNESS_PORT,
    create_remote_gateway,
    serve_remote_gateway,
)

__all__ = [
    "DEFAULT_MAX_TOKEN_BYTES",
    "DEFAULT_METAHARNESS_PORT",
    "LOCAL_HOST",
    "MAX_RESPONSE_BYTES",
    "LocalMetaHarnessClient",
    "LocalMetaHarnessError",
    "TokenFileError",
    "bearer_token_from_header",
    "create_remote_gateway",
    "load_token_file",
    "serve_remote_gateway",
    "token_matches",
]
