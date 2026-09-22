"""Injectable GitHub metadata port.

The bootstrap intentionally contains no GitHub transport.  Applications may
inject a client implementing this protocol, while the default client is a
strict no-network null object.  GitHub content is not part of any MetaHarness
authority contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class GitHubIntegrationError(RuntimeError):
    """A requested GitHub metadata operation could not be completed."""

    code = "GITHUB_WORKSTREAM_FAILURE"


class GitHubWorkstreamError(GitHubIntegrationError):
    """A requested workstream metadata operation failed explicitly."""

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


@dataclass(frozen=True)
class GitHubIssue:
    """Minimal issue result; body text is intentionally not persisted."""

    number: int
    title: str = ""
    body: str = ""


@dataclass(frozen=True)
class GitHubPullRequest:
    """Minimal pull-request result."""

    number: int
    title: str = ""


class GitHubWorkstreamClient(Protocol):
    """Port for optional GitHub workstream metadata operations."""

    def read_issue(self, issue_number: int) -> GitHubIssue | None:
        """Read one issue, returning ``None`` when it does not exist."""

    def create_issue(self, title: str, body: str) -> GitHubIssue | int:
        """Create one issue and return its identifier."""

    def create_pull_request(
        self,
        title: str,
        body: str,
        base_branch: str,
        head_branch: str,
    ) -> GitHubPullRequest | int:
        """Create one PR from the exact reviewed head branch."""


class NullGitHubWorkstreamClient:
    """Default client: it never performs network I/O."""

    def read_issue(self, issue_number: int) -> None:
        del issue_number
        raise GitHubIntegrationError("GitHub client is not configured")

    def create_issue(self, title: str, body: str) -> None:
        del title, body
        raise GitHubIntegrationError("GitHub client is not configured")

    def create_pull_request(
        self,
        title: str,
        body: str,
        base_branch: str,
        head_branch: str,
    ) -> None:
        del title, body, base_branch, head_branch
        raise GitHubIntegrationError("GitHub client is not configured")


__all__ = [
    "GitHubIntegrationError",
    "GitHubWorkstreamError",
    "GitHubIssue",
    "GitHubPullRequest",
    "GitHubWorkstreamClient",
    "NullGitHubWorkstreamClient",
]
