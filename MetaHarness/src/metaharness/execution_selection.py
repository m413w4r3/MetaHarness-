"""Durable, allowlisted execution selection for one run."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import ExecutionRole, ExecutionSelection, SelectedProfile
from .profiles import profile_for_role
from .models import HarnessConfig


_FILENAME = "execution_selection.json"


def _selected(profile: Any) -> SelectedProfile:
    return SelectedProfile(
        profile_id=profile.id,
        driver=profile.driver.value,
        model=profile.model,
        selection_mode=profile.selection_mode.value,
        effort=profile.effort,
        sandbox=profile.sandbox,
    )


def resolve_execution_selection(
    config: HarnessConfig,
    *,
    planner_profile_id: str,
    implementer_profile_id: str,
    reviewer_profile_id: str,
) -> ExecutionSelection:
    return ExecutionSelection(
        schema_version=1,
        planner=_selected(profile_for_role(config, planner_profile_id, ExecutionRole.PLANNER)),
        implementer=_selected(
            profile_for_role(config, implementer_profile_id, ExecutionRole.IMPLEMENTER)
        ),
        reviewer=_selected(profile_for_role(config, reviewer_profile_id, ExecutionRole.REVIEWER)),
    )


def _payload(selection: ExecutionSelection) -> dict[str, Any]:
    if not isinstance(selection, ExecutionSelection) or selection.schema_version != 1:
        raise ValueError("execution selection schema_version must be 1")
    return {"schema_version": 1, **{
        name: asdict(getattr(selection, name))
        for name in ("planner", "implementer", "reviewer")
    }}


def write_execution_selection(
    run_dir: Path,
    selection: ExecutionSelection,
) -> None:
    directory = Path(run_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / _FILENAME
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=directory
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_payload(selection), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise ValueError(f"could not write execution selection: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _selected_from(value: Any, name: str) -> SelectedProfile:
    if not isinstance(value, dict):
        raise ValueError(f"execution selection {name} must be an object")
    required = {"profile_id", "driver", "model", "selection_mode", "effort", "sandbox"}
    if set(value) != required:
        raise ValueError(f"execution selection {name} has invalid fields")
    strings = ("profile_id", "driver", "model", "selection_mode")
    if any(not isinstance(value[key], str) for key in strings):
        raise ValueError(f"execution selection {name} has invalid metadata")
    for key in ("effort", "sandbox"):
        if value[key] is not None and not isinstance(value[key], str):
            raise ValueError(f"execution selection {name}.{key} is invalid")
    return SelectedProfile(**value)


def read_execution_selection(
    run_dir: Path,
) -> ExecutionSelection:
    path = Path(run_dir).expanduser().resolve() / _FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("execution selection is missing or invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("execution selection schema_version is unsupported")
    required = {"schema_version", "planner", "implementer", "reviewer"}
    if set(payload) != required:
        raise ValueError("execution selection has invalid fields")
    return ExecutionSelection(
        schema_version=1,
        planner=_selected_from(payload["planner"], "planner"),
        implementer=_selected_from(payload["implementer"], "implementer"),
        reviewer=_selected_from(payload["reviewer"], "reviewer"),
    )


__all__ = [
    "resolve_execution_selection",
    "write_execution_selection",
    "read_execution_selection",
]
