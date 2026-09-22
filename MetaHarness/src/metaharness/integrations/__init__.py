"""Optional integrations that never participate in core authority."""

from .github import (
    GitHubIntegrationError,
    GitHubIssue,
    GitHubPullRequest,
    GitHubWorkstreamError,
    GitHubWorkstreamClient,
    NullGitHubWorkstreamClient,
)

__all__ = [
    "GitHubIntegrationError",
    "GitHubWorkstreamError",
    "GitHubIssue",
    "GitHubPullRequest",
    "GitHubWorkstreamClient",
    "NullGitHubWorkstreamClient",
]
