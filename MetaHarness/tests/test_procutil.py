import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaharness.procutil import read_capped, run_bounded  # noqa: E402


class BoundedProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_code(self, code: str, timeout: float, **kwargs):
        stdout_path = self.root / "stdout"
        stderr_path = self.root / "stderr"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            started = time.monotonic()
            result = run_bounded(
                [sys.executable, "-c", code],
                cwd=self.root,
                timeout_seconds=timeout,
                stdout=stdout,
                stderr=stderr,
                **kwargs,
            )
        return result, time.monotonic() - started

    def test_timeout_kills_the_whole_group_even_without_output(self) -> None:
        marker = self.root / "grandchild-survived"
        code = (
            "import subprocess, time\n"
            f"subprocess.Popen(['sh', '-c', 'sleep 3; touch {marker}'])\n"
            "time.sleep(30)\n"
        )
        (exit_code, timed_out), elapsed = self.run_code(code, 0.5, grace_seconds=0.2)
        self.assertTrue(timed_out)
        self.assertNotEqual(exit_code, 0)
        self.assertLess(elapsed, 5)
        time.sleep(3.5)
        self.assertFalse(marker.exists())

    def test_process_ignoring_sigterm_and_sigint_is_killed(self) -> None:
        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "time.sleep(30)\n"
        )
        (_, timed_out), elapsed = self.run_code(code, 0.3, grace_seconds=0.3)
        self.assertTrue(timed_out)
        self.assertLess(elapsed, 5)

    def test_background_child_does_not_outlive_a_successful_command(self) -> None:
        marker = self.root / "late-write"
        code = (
            "import subprocess\n"
            f"subprocess.Popen(['sh', '-c', 'sleep 1; touch {marker}'])\n"
        )
        (exit_code, timed_out), elapsed = self.run_code(code, 10)
        self.assertEqual((exit_code, timed_out), (0, False))
        self.assertLess(elapsed, 5)
        time.sleep(1.5)
        self.assertFalse(marker.exists())

    def test_huge_output_never_blocks_and_stdin_is_closed(self) -> None:
        code = (
            "import sys\n"
            "assert sys.stdin.read() == ''\n"
            "sys.stdout.write('x' * 5_000_000)\n"
            "sys.stderr.write('y' * 5_000_000)\n"
        )
        (exit_code, timed_out), _ = self.run_code(code, 20)
        self.assertEqual((exit_code, timed_out), (0, False))
        self.assertEqual((self.root / "stdout").stat().st_size, 5_000_000)
        text, truncated = read_capped(self.root / "stdout", 1000)
        self.assertTrue(truncated)
        self.assertTrue(text.endswith("x" * 1000))

    def test_missing_executable_raises_oserror(self) -> None:
        with (self.root / "o").open("wb") as out, (self.root / "e").open("wb") as err:
            with self.assertRaises(OSError):
                run_bounded(
                    [str(self.root / "missing")],
                    cwd=self.root,
                    timeout_seconds=1,
                    stdout=out,
                    stderr=err,
                )


if __name__ == "__main__":
    unittest.main()
