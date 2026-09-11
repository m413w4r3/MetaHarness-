"""Local, non-shell environment file loading for MetaHarness."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence


class EnvironmentFileError(ValueError):
    pass


_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse the deliberately small MetaHarness ``.env`` format."""

    path = Path(path).expanduser()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EnvironmentFileError(f"cannot read environment file {path}") from exc
    except UnicodeError as exc:
        raise EnvironmentFileError(f"environment file is not valid UTF-8: {path}") from exc

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export"):
            if not line.startswith("export "):
                raise EnvironmentFileError(f"invalid environment assignment at {path}:{line_number}")
            line = line[7:].lstrip()
        if "=" not in line:
            raise EnvironmentFileError(f"invalid environment assignment at {path}:{line_number}")
        raw_name, raw_value = line.split("=", 1)
        name = raw_name.strip()
        if _NAME.fullmatch(name) is None:
            raise EnvironmentFileError(f"invalid environment variable name at {path}:{line_number}")
        if name in values:
            raise EnvironmentFileError(
                f"duplicate environment variable {name!r} at {path}:{line_number}"
            )
        value = raw_value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise EnvironmentFileError(f"invalid quoted value at {path}:{line_number}")
            value = value[1:-1]
        elif value[-1:] in {"'", '"'}:
            raise EnvironmentFileError(f"invalid quoted value at {path}:{line_number}")
        values[name] = value
    return values


def build_runtime_environment(
    files: Sequence[Path],
    process_environment: Mapping[str, str],
) -> dict[str, str]:
    """Merge env files, then process variables, with later sources winning."""

    result: dict[str, str] = {}
    for path in files:
        result.update(parse_env_file(Path(path)))
    result.update(process_environment)
    return result


__all__ = [
    "EnvironmentFileError",
    "build_runtime_environment",
    "parse_env_file",
]
