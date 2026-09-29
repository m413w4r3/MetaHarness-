from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unittest.mock import patch

from metaharness.cli import main
from metaharness.models import INTERRUPTED_REASON, RunStatus
from metaharness.result import RunResult


class CLIStatusTests(unittest.TestCase):
    def test_interruption_keeps_the_shell_interrupt_exit_code(self) -> None:
        result = RunResult(
            Path("run"), RunStatus.FAILED,
            {"failure": {"reason": INTERRUPTED_REASON}},
        )
        with contextlib.redirect_stdout(io.StringIO()), patch(
            "metaharness.cli.load_config", return_value=object(),
        ), patch("metaharness.cli.run_orchestrator", return_value=result):
            self.assertEqual(
                main(["run", "--config", "config.toml", "--spec", "spec.md"]), 130,
            )

    def test_old_status_spellings_are_projected_when_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "state.json").write_text(
                '{"run_id":"old-run","status":"implementing","phase":"implement_step"}',
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(["status", "--run", str(run_dir)]), 0)
        self.assertIn(f"status: {RunStatus.RUNNING.value}", output.getvalue())


if __name__ == "__main__":
    unittest.main()
