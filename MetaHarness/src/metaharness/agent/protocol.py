"""Backend-neutral worker protocol messages."""

from __future__ import annotations

CONTRACT_MISMATCH_HEADER = "META CONTRACT MISMATCH v1"
DEFERRED_VERIFY_HEADER = "DEFERRED VERIFY DEPENDENCY"


def deferred_verify_dependency(final_message: str) -> str | None:
    """Return the worker's deferred verification dependency note, if any."""

    if not isinstance(final_message, str):
        raise TypeError("worker final message must be a string")
    lines = final_message.splitlines()
    for index, line in enumerate(lines):
        if line.strip().rstrip(":").casefold() != DEFERRED_VERIFY_HEADER.casefold():
            continue
        body = "\n".join(lines[index + 1:]).strip()
        return body or None
    return None


def contract_mismatch_explanation(final_message: str) -> str | None:
    """Return the protocol exception, if it is the first content line."""

    if not isinstance(final_message, str):
        raise TypeError("worker final message must be a string")
    lines = final_message.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        if line != CONTRACT_MISMATCH_HEADER:
            return None
        return "\n".join(lines[index + 1:]).strip()
    return None


__all__ = [
    "CONTRACT_MISMATCH_HEADER",
    "DEFERRED_VERIFY_HEADER",
    "contract_mismatch_explanation",
    "deferred_verify_dependency",
]
