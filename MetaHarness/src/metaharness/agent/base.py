"""Types communs aux agents d'implémentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentResult:
    """Résultat observable d'une exécution d'agent."""

    exit_code: int
    timed_out: bool
    final_message: str
    usage: dict[str, int]
    stderr_tail: str


class AgentError(RuntimeError):
    """Erreur empêchant de produire un résultat d'agent."""
