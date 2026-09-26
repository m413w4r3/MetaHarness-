"""Planning authorities: protocol, validation, artifacts, planners and repair.

The package deliberately exposes no flat façade.  Import each authority from
its own module; only the transport type shared by several authorities lives
here.
"""

from __future__ import annotations

from typing import Protocol

from ..llm.chat import TextLLMResult


class TextCompletionClient(Protocol):
    """The transport one planner transaction needs: prompt in, text out."""

    def complete(self, prompt: str) -> TextLLMResult | str: ...


__all__ = ["TextCompletionClient"]
