"""Optional parallel test runner: one Python process per test module.

The canonical command stays ``python -m unittest discover -s tests -v``.  This
runner only lowers wall-clock time: it discovers ``tests/test_*.py``, runs each
module as ``python -m unittest tests.<module>`` in its own process (no global
state is shared between modules) through a bounded pool, then reports every
module's result and prints the complete output of each failing module.

Usage::

    python scripts/test_parallel.py            # min(cpu_count, 8) workers
    python scripts/test_parallel.py -j 4
    python scripts/test_parallel.py test_state test_procutil

The exit status is non-zero as soon as one module fails.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"

_RAN = re.compile(r"^Ran (\d+) tests? in ", re.MULTILINE)
_COUNTS = re.compile(r"(failures|errors|skipped|expected failures|unexpected successes)=(\d+)")


@dataclass(frozen=True)
class ModuleResult:
    module: str
    returncode: int
    duration: float
    stdout: str
    stderr: str

    @property
    def summary_line(self) -> str:
        lines = [line for line in self.stderr.splitlines() if line.startswith(("OK", "FAILED"))]
        return lines[-1] if lines else "(no unittest summary)"

    @property
    def tests_run(self) -> int:
        match = _RAN.search(self.stderr)
        return int(match.group(1)) if match else 0

    def count(self, kind: str) -> int:
        return sum(int(n) for k, n in _COUNTS.findall(self.summary_line) if k == kind)


def discover(selected: list[str]) -> list[str]:
    modules = sorted(TESTS.glob("test_*.py"), key=lambda path: path.stat().st_size, reverse=True)
    names = [path.stem for path in modules]
    if selected:
        wanted = {name.removeprefix("tests.").removesuffix(".py") for name in selected}
        unknown = wanted - set(names)
        if unknown:
            raise SystemExit(f"unknown test modules: {', '.join(sorted(unknown))}")
        names = [name for name in names if name in wanted]
    # Largest files first: long modules start early and do not end the run alone.
    return names


def run_module(module: str, verbose: bool) -> ModuleResult:
    argv = [sys.executable, "-m", "unittest", f"tests.{module}"]
    if verbose:
        argv.append("-v")
    started = time.monotonic()
    completed = subprocess.run(
        argv, cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True,
    )
    return ModuleResult(
        module, completed.returncode, time.monotonic() - started,
        completed.stdout, completed.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("modules", nargs="*", help="test modules to run (default: all)")
    parser.add_argument("-j", "--jobs", type=int, default=min(os.cpu_count() or 1, 8),
                        help="maximum number of worker processes")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="pass -v to unittest and print every module's output")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    modules = discover(args.modules)
    workers = min(args.jobs, len(modules)) or 1
    print(f"running {len(modules)} test modules with {workers} workers", flush=True)
    started = time.monotonic()
    results: list[ModuleResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_module, module, args.verbose) for module in modules]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            status = "ok  " if result.returncode == 0 else "FAIL"
            print(f"{status} {result.module:40s} {result.duration:6.2f}s  {result.summary_line}",
                  flush=True)
    wall = time.monotonic() - started

    failed = sorted((r for r in results if r.returncode != 0), key=lambda r: r.module)
    for result in sorted(results, key=lambda r: r.module):
        if args.verbose or result.returncode != 0:
            print(f"\n{'=' * 70}\n{result.module} (exit {result.returncode})\n{'=' * 70}")
            sys.stdout.write(result.stdout)
            sys.stdout.write(result.stderr)

    totals = {kind: sum(r.count(kind) for r in results) for kind in ("failures", "errors", "skipped")}
    print(f"\n{'-' * 70}")
    print(f"Ran {sum(r.tests_run for r in results)} tests in {len(results)} modules; "
          f"wall-clock {wall:.2f}s (sum of module times {sum(r.duration for r in results):.2f}s)")
    print(f"failures={totals['failures']} errors={totals['errors']} skipped={totals['skipped']}")
    if failed:
        print(f"FAILED modules ({len(failed)}): {', '.join(r.module for r in failed)}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
