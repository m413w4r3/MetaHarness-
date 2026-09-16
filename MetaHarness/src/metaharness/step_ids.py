"""Authoritative META PLAN v2 step capacity and step-ID helpers.

This module has no MetaHarness imports on purpose: the parser, the execution
selection, resume checkpoints, usage accounting and the web layer all share
this one bound without any import cycle.  No other module may spell its own
step-ID character class or step-count literal: they import from here.
"""

from __future__ import annotations

import re
from typing import Any

PROTOCOL_MAX_STEPS = 99
# Protocol syntax bound, not a recommended execution size.
MAX_STEPS = PROTOCOL_MAX_STEPS

# ``S01`` .. ``S99``: an explicit alternation, so the bound is exactly
# MAX_STEPS and never an accidental character-class range.
STEP_ID_PATTERN = "S(?:" + "|".join(f"{number:02d}" for number in range(1, MAX_STEPS + 1)) + ")"
STEP_ID_RE = re.compile(STEP_ID_PATTERN)


def is_step_id(value: Any) -> bool:
    """True only for one exact step ID ``S01`` .. ``S{MAX_STEPS:02d}``."""

    return isinstance(value, str) and STEP_ID_RE.fullmatch(value) is not None


def step_ids(count: int) -> tuple[str, ...]:
    """The canonical contiguous IDs ``S01`` .. ``S{count:02d}``."""

    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_STEPS:
        raise ValueError(f"step count must be between 1 and {MAX_STEPS}")
    return tuple(f"S{number:02d}" for number in range(1, count + 1))


ALL_STEP_IDS = step_ids(MAX_STEPS)
LAST_STEP_ID = ALL_STEP_IDS[-1]


__all__ = [
    "ALL_STEP_IDS", "LAST_STEP_ID", "MAX_STEPS", "PROTOCOL_MAX_STEPS",
    "STEP_ID_PATTERN", "STEP_ID_RE",
    "is_step_id", "step_ids",
]
