"""Claude Code runtime integration for the MetaHarness reviser."""

from .agent import (
    ClaudeAgentError,
    ClaudeCodeAgent,
    ClaudeCommittedError,
    ClaudeResult,
    build_claude_environment,
    build_revision_prompt,
    classify_claude_failure,
)
from .auth import ClaudeAuthStatus, check_claude_authentication
from .runtime import ClaudeRuntimeError, prepare_claude_home

__all__ = [
    "ClaudeAgentError",
    "ClaudeAuthStatus",
    "ClaudeCodeAgent",
    "ClaudeCommittedError",
    "ClaudeResult",
    "ClaudeRuntimeError",
    "build_claude_environment",
    "build_revision_prompt",
    "check_claude_authentication",
    "classify_claude_failure",
    "prepare_claude_home",
]
