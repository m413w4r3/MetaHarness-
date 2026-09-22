"""Removal of configured secret values from persisted text."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from .models import HarnessConfig

REDACTED = "[REDACTED]"
# Very short values cannot be redacted without destroying unrelated text; they
# are not realistic API keys either.
_MIN_SECRET_LENGTH = 8


def secret_values(
    env_names: Iterable[str | None],
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return values from *environment*; ``None`` preserves old callers."""

    source = os.environ if environment is None else environment
    values: set[str] = set()
    for name in env_names:
        if not name:
            continue
        value = source.get(name)
        if value and len(value) >= _MIN_SECRET_LENGTH:
            values.add(value)
    return tuple(sorted(values, key=len, reverse=True))


def config_secret_values(
    config: HarnessConfig,
    environment: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Secrets referenced by all configured execution profiles."""

    names = [
        config.planner.api_key_env,
        config.reviewer.api_key_env,
        config.github.api_key_env,
    ]
    names.extend(
        profile.api_key_env for profile in config.model_profiles.values()
    )
    if environment is None and config.runtime_environment:
        environment = config.runtime_environment
    return secret_values(names, environment)


def redact(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return text


def contains_secret(text: str, secrets: Iterable[str]) -> bool:
    return any(secret and secret in text for secret in secrets)


def redact_file(path: Path, secrets: tuple[str, ...]) -> None:
    """Rewrite *path* atomically if it contains a secret value.

    Secrets never contain line breaks (the transport rejects such keys), so a
    line-by-line scan is exact and keeps memory bounded by one line.
    """

    if not secrets or not path.is_file():
        return
    encoded = [secret.encode("utf-8") for secret in secrets]
    with path.open("rb") as source:
        if not any(any(secret in line for secret in encoded) for line in source):
            return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as target, path.open("rb") as source:
            for line in source:
                for secret in encoded:
                    line = line.replace(secret, REDACTED.encode("ascii"))
                target.write(line)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


__all__ = [
    "REDACTED",
    "config_secret_values",
    "contains_secret",
    "redact",
    "redact_file",
    "secret_values",
]
