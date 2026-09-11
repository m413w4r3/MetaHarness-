"""Local read-only observation UI and plan approval endpoint."""

from .server import create_server, serve

__all__ = ["create_server", "serve"]
